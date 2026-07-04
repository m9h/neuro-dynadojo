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


def cv_auc(F, y, probe="linear", seed=0, n_splits=5):
    """Cross-validated binary-classification AUC of features `F` for label `y`, with a
    SELECTABLE probe: 'linear' (LogReg, the platform's default -- decodability under a linear
    boundary), 'kernel' (RBF-SVM -- decodability under a non-linear boundary the frozen
    representation's geometry alone determines), or 'mlp' (small 2-layer MLP -- a learned
    non-linear boundary). Added in response to the review critique that a linear probe measures
    linear decodability only: 'the FM does not expose factor X' should mean under ANY reasonable
    probe, not just a linear one. Same StandardScaler + StratifiedKFold protocol for all three,
    so results differ only in the probe's capacity, not the evaluation procedure."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    if probe == "linear":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
    elif probe == "kernel":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
    elif probe == "mlp":
        from sklearn.neural_network import MLPClassifier
        # early_stopping=True carves a validation split off small CV folds and can halt near-
        # random init on tiny samples; rely on max_iter as the training budget instead.
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"unknown probe {probe!r}: use 'linear', 'kernel', or 'mlp'")
    est = make_pipeline(StandardScaler(), clf)
    pred = cross_val_predict(est, F, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pred))


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


def bendr_embed(model, obs, device="cpu"):
    """Embed with BENDR on its EXACT pretraining montage. `obs` = the 19 BENDR EEG channels in
    order (generator `montage="bendr"`). The 19 EEG channels are per-channel z-scored and the
    20th 'SCALE' channel is appended as a CONSTANT bounded relative-amplitude (dn3's
    MappingDeep1010 SCALE is a per-recording scalar, not a time series and not re-normalised)."""
    x = np.asarray(obs, float)
    xe = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)   # z-score EEG
    scale_val = float(np.tanh(np.log(np.sqrt((x ** 2).mean()) + 1e-8)))      # bounded scalar
    scale = np.full((1, x.shape[1]), scale_val)
    return braindecode_embed(model, np.vstack([xe, scale]), device=device, standardize=False)


def braindecode_embed(model, obs, device="cpu", standardize=True):
    """Adapter for a braindecode / HuggingFace-style EEG foundation model (frozen). Feeds one
    recording as `(1, n_chans, n_times)` and returns the encoder EMBEDDING (via
    `model(x, return_features=True)` where supported, else the raw output). Per-channel
    z-scoring by default (standard EEG-FM input convention). Requires the `[fm]` extra."""
    import torch
    x = np.asarray(obs, dtype="float32")
    if standardize:
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    model.eval()
    xt = torch.as_tensor(x)[None].to(device)                        # (1, n_ch, n_times)
    with torch.no_grad():
        try:
            out = model(xt, return_features=True)
        except TypeError:
            out = model(xt)
    if isinstance(out, dict):                                       # {'features': ..., 'cls_token': ...}
        out = out.get("features", out.get("cls_token", next(iter(out.values()))))
    return np.asarray(out.detach().cpu()).ravel()
