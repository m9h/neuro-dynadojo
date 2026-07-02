"""Challenges — recovery metrics that score an Algorithm's estimate against the known
ground truth a Generator emits. Undirected / directed edge recovery (AUC) and wavenumber
consistency. `directed_edge_auc` is re-exported from the directed algorithms so the whole
metric set lives under one namespace.
"""
from __future__ import annotations

import numpy as np

from ..algorithms.directed import directed_edge_auc  # noqa: F401 (unified challenges namespace)


def edge_recovery_auc(score, adjacency):
    """AUC of recovering undirected structural edges (upper triangle) from a score matrix."""
    from sklearn.metrics import roc_auc_score
    adjacency = np.asarray(adjacency); iu = np.triu_indices(len(adjacency), 1)
    y = adjacency[iu].astype(int); s = np.asarray(score)[iu]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def wavenumber_consistency(x, fs=250.0, band=(8.0, 12.0)):
    """Circular concentration of the per-timepoint dominant wavenumber along channel order
    (1.0 = a single stable spatial wave; ~0 = none). The wave-regime ground truth."""
    from scipy.signal import butter, filtfilt, hilbert
    from ..generators.waves import dominant_wavenumber
    b, a = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    ph = np.angle(hilbert(filtfilt(b, a, x, axis=1), axis=1))
    ks = np.array([dominant_wavenumber(ph[:, t]) for t in range(0, ph.shape[1], 3)])
    return float(np.abs(np.mean(np.exp(1j * 2 * np.pi * ks / x.shape[0]))))
