"""FM-ready probing — evaluate whether a representation encodes the GENERATIVE FACTORS.

Classical Algorithms recover structure from a single recording (`estimate(obs, system)`).
A foundation model is evaluated differently: freeze it, embed many simulated recordings
whose generative factor (coupling, delay, dynamical regime, wavenumber, frequency, flow
direction) is varied and KNOWN, and linearly probe the embeddings for that factor. On
simulated data the answer is known, so this asks a question downstream accuracy cannot:
*does the model's representation encode the actual physics, under realistic sensor
confounds (volume conduction + 1/f noise)?*

Bring-your-own model: pass any `embed_fn(obs) -> vector` (mean-pooled features of a frozen
FM). `braindecode_embed` adapts a braindecode / HuggingFace-style model (input shape
`(batch, n_chans, n_times)`); `bandpower_embed` is a dependency-light spectral baseline an
FM must beat. Complementary to real-data identity audits (e.g. fmscope; Tang et al. 2026):
here the confounds and the ground truth are known by construction.
"""
from __future__ import annotations

import numpy as np


def factor_dataset(system_factory, factor, values, n_per=8, seed0=0, space="sensor", **fixed):
    """Simulate `n_per` recordings at each value of a generative `factor`, in sensor space.
    Returns [(obs (n_ch, n_times), value), ...] — the labeled probe set."""
    data = []
    for v in values:
        for r in range(n_per):
            sysm = system_factory(space=space, seed_struct=seed0 + r, **{factor: v}, **fixed)
            obs, _ = sysm.simulate(seed=seed0 + r)
            data.append((obs, v))
    return data


def representation_probe(embed_fn, dataset, task="regress", cv=5):
    """Cross-validated linear probe of the factor from embeddings. Regression -> Pearson r
    and R^2; classification -> ROC-AUC. `embed_fn(obs)` returns a feature vector."""
    X = np.stack([np.asarray(embed_fn(o), float).ravel() for o, _ in dataset])
    y = np.array([v for _, v in dataset])
    from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if task == "regress":
        from sklearn.linear_model import RidgeCV
        est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pred = cross_val_predict(est, X, y, cv=KFold(cv, shuffle=True, random_state=0))
        ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-30)
        return {"metric": "r", "r": float(np.corrcoef(y, pred)[0, 1]),
                "r2": 1.0 - ss_res / ss_tot, "n": len(y)}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    yb = y.astype(int)
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(est, X, yb, cv=StratifiedKFold(cv, shuffle=True, random_state=0),
                              method="predict_proba")
    auc = (roc_auc_score(yb, proba[:, 1]) if proba.shape[1] == 2
           else roc_auc_score(yb, proba, multi_class="ovr"))
    return {"metric": "auc", "auc": float(auc), "n": len(y)}


def probe_factor(embed_fn, system_factory, factor, values, task="regress",
                 n_per=8, space="sensor", **fixed):
    """Convenience: build a factor dataset and probe it in one call."""
    return representation_probe(embed_fn, factor_dataset(
        system_factory, factor, values, n_per=n_per, space=space, **fixed), task=task)


def bandpower_embed(obs, fs=250.0, bands=((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))):
    """Dependency-light spectral baseline: log band-power per channel. The floor an FM
    representation should clear to claim it has learned anything beyond the spectrum."""
    from scipy.signal import welch
    f, P = welch(np.asarray(obs, float), fs=fs, nperseg=min(256, obs.shape[1]), axis=1)
    return np.concatenate([np.log(P[:, (f >= lo) & (f < hi)].mean(1) + 1e-20) for lo, hi in bands])


def braindecode_embed(model, obs, device="cpu"):
    """Adapter for a braindecode / HuggingFace-style EEG model (frozen). Feeds one recording
    as `(1, n_chans, n_times)` and returns the mean-pooled output as the embedding. Pass a
    feature-extractor model (classifier head removed) for representation probing. Requires
    the optional `[fm]` extra (torch + braindecode)."""
    import torch
    model.eval()
    x = torch.as_tensor(np.asarray(obs, dtype="float32"))[None]     # (1, n_ch, n_times)
    with torch.no_grad():
        out = model(x.to(device))
    return np.asarray(out.detach().cpu()).ravel()
