"""System-ID and classical dynamics contenders — neuro-dynadojo compares the whole method
landscape, not just foundation models. Each is a feature extractor `(N, C, T) -> (N, D)` so
it slots into the same probe/scoring as band-power and FM embeddings:

  sindy   sparse identification of nonlinear dynamics (PySINDy) — the governing-equation coeffs
  dmd     dynamic mode decomposition (Koopman) — operator eigenvalues (magnitude + frequency)
  dysco   Dynamic Symmetric Connectivity (Rabuffo et al. 2025) — eigenvalue-spectrum dynamics
          of sliding-window connectivity (metastability, reconfiguration speed)
  hmm     Gaussian HMM state-space (the osl-dynamics/DyNeMo family) — state occupancy + means

All run per-recording on a PCA-reduced trajectory; pure NumPy/sklearn + pysindy/hmmlearn.
"""
from __future__ import annotations

import numpy as np


def _pca(x, k):
    from sklearn.decomposition import PCA
    return PCA(k, random_state=0).fit_transform(x.T)              # (T, k) trajectory


def sindy_features(X, k=5, dt=1 / 250.0):
    """PySINDy governing-equation coefficients of each recording's PCA-reduced dynamics."""
    from pysindy import SINDy
    from pysindy.optimizers import STLSQ
    from pysindy.feature_library import PolynomialLibrary
    feats, D = [], None
    for x in X:
        z = _pca(x, k)
        try:
            m = SINDy(optimizer=STLSQ(threshold=0.05),
                      feature_library=PolynomialLibrary(degree=2)).fit(z, t=dt)
            c = m.coefficients().ravel(); D = D or c.size; feats.append(c)
        except Exception:
            feats.append(None)
    D = D or 1
    return np.array([f if f is not None and f.size == D else np.zeros(D) for f in feats])


def dmd_features(X, k=6):
    """DMD/Koopman operator eigenvalues (|lambda| and angle) of the PCA-reduced dynamics."""
    feats = []
    for x in X:
        z = _pca(x, k).T                                         # (k, T)
        A = z[:, 1:] @ np.linalg.pinv(z[:, :-1])
        ev = np.linalg.eigvals(A)
        feats.append(np.concatenate([np.abs(ev), np.angle(ev)]))
    return np.array(feats)


def dysco_features(X, win=125, step=25, n_eig=3):
    """DySCo: eigenvalue spectrum of sliding-window connectivity over time -> its mean,
    metastability (temporal std), and reconfiguration speed (leading-eigenvalue drift)."""
    feats = []
    for x in X:
        evs = []
        for s in range(0, x.shape[1] - win, step):
            C = np.corrcoef(x[:, s:s + win])
            evs.append(np.sort(np.linalg.eigvalsh(C))[::-1][:n_eig])
        evs = np.array(evs) if evs else np.zeros((1, n_eig))
        feats.append(np.concatenate([evs.mean(0), evs.std(0),
                                     [np.abs(np.diff(evs[:, 0])).mean() if len(evs) > 1 else 0.0]]))
    return np.array(feats)


def hmm_features(X, n_states=4, k=6):
    """Gaussian-HMM state-space (osl-dynamics/DyNeMo family): state occupancy + state means."""
    from hmmlearn.hmm import GaussianHMM
    feats, D = [], n_states + n_states * k
    for x in X:
        z = _pca(x, k)
        try:
            h = GaussianHMM(n_states, covariance_type="diag", n_iter=15, random_state=0).fit(z)
            st = h.predict(z)
            occ = np.bincount(st, minlength=n_states) / len(st)
            feats.append(np.concatenate([occ, h.means_.ravel()]))
        except Exception:
            feats.append(np.zeros(D))
    return np.array([f if f.size == D else np.zeros(D) for f in feats])


DYNAMICS = {"SINDy": sindy_features, "DMD": dmd_features, "DySCo": dysco_features, "HMM": hmm_features}
