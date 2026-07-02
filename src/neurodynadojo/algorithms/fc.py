"""Functional-connectivity Algorithms for the electrophysiological FSLNets bench.

Each returns an (n, n) score matrix (higher = stronger putative edge); the Challenge is
`edge_recovery_auc` against the true structural adjacency. The set is chosen to expose the
VOLUME-CONDUCTION axis: zero-lag measures (correlation, partial correlation, PLV) are
fooled by instantaneous source leakage, whereas imaginary coherence and wPLI discard the
zero-lag component by construction and should survive it (Nolte 2004; Vinck 2011). Wrapped
as bench Algorithms (`.name`, `.estimate(obs, system)`) so they feed run_benchmark /
adversarial_search directly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from ..challenges.recovery import edge_recovery_auc


def _analytic(X, fs, band):
    b, a = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    return hilbert(filtfilt(b, a, X, axis=1), axis=1)


def correlation_fc(X, **_):
    return np.abs(np.corrcoef(X))


def partialcorr_fc(X, reg=1e-2, **_):
    C = np.corrcoef(X)
    P = np.linalg.inv(C + reg * np.eye(C.shape[0]))
    d = np.sqrt(np.abs(np.diag(P)))
    return np.abs(-P / np.outer(d, d))


def coherency(X, fs, band):
    """Complex coherency from the band-limited analytic signal."""
    Z = _analytic(X, fs, band)
    S = Z @ Z.conj().T / Z.shape[1]
    d = np.sqrt(np.abs(np.diag(S)))
    return S / (np.outer(d, d) + 1e-12)


def imag_coherence_fc(X, fs=1000.0, band=(8.0, 12.0), **_):
    """|Im coherency| -- ignores the zero-lag component => leakage-robust (Nolte 2004)."""
    return np.abs(np.imag(coherency(X, fs, band)))


def plv_fc(X, fs=1000.0, band=(8.0, 12.0), **_):
    Z = _analytic(X, fs, band)
    ph = np.angle(Z)
    d = ph[:, None, :] - ph[None, :, :]
    return np.abs(np.mean(np.exp(1j * d), axis=2))


def wpli_fc(X, fs=1000.0, band=(8.0, 12.0), **_):
    """Weighted phase-lag index -- sign-weighted imaginary cross-spectrum, leakage-robust
    (Vinck 2011)."""
    Z = _analytic(X, fs, band)
    n = Z.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        cs = Z[i][None, :] * np.conj(Z)            # cross-spectra (n, T)
        im = np.imag(cs)
        W[i] = np.abs(np.mean(im, axis=1)) / (np.mean(np.abs(im), axis=1) + 1e-12)
    return W


class _FC:
    def __init__(self, fn, name):
        self.fn, self.name = fn, name

    def estimate(self, obs, system):
        return self.fn(obs, fs=getattr(system, "fs", 1000.0),
                       band=getattr(system, "band", (8.0, 12.0)))


def fc_algorithms():
    """The FSLNets-style contender set (zero-lag vs leakage-robust)."""
    return [
        _FC(correlation_fc, "correlation"),
        _FC(partialcorr_fc, "partial corr"),
        _FC(plv_fc, "PLV"),
        _FC(imag_coherence_fc, "imag coherence"),
        _FC(wpli_fc, "wPLI"),
    ]


def edge_auc(score, true_adj):
    """Challenge: AUC of recovering true structural edges from a score matrix."""
    return edge_recovery_auc(score, true_adj)
