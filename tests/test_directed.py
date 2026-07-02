"""TDD for smni_cmi.directed — directed methods recover the traveling-wave FLOW that
undirected FC cannot, and Granger (zero-lag-confounded) collapses under volume conduction
while DMD is more robust.
"""
import numpy as np

from neurodynadojo.generators.hopf import RingWaveSystem
from neurodynadojo.algorithms.directed import granger_bivariate, dmd_transition, directed_edge_auc


def _wave_dir_adj(n, m):
    """Directed ground truth = edges oriented along the wave (toward decreasing index)."""
    A = np.zeros((n, n))
    for i in range(n):
        for d in range(1, m + 1):
            A[i, (i - d) % n] = 1.0
    return A


def test_directed_methods_recover_wave_flow_in_source_space():
    g, d = [], []
    for s in range(4):
        sysm = RingWaveSystem(seed_struct=s, space="source", alpha=0.9, m=2)
        x, _ = sysm.simulate(seed=s)
        DA = _wave_dir_adj(sysm.n, sysm.m)
        g.append(directed_edge_auc(granger_bivariate(x, 4), DA))
        d.append(directed_edge_auc(dmd_transition(x, 8), DA))
    assert np.mean(g) > 0.8 and np.mean(d) > 0.8          # directionality recovers the wave


def test_granger_collapses_under_leakage_more_than_dmd():
    gl, dl = [], []
    for s in range(4):
        sysm = RingWaveSystem(seed_struct=s, space="leaked", leak=0.8, alpha=0.9, m=2)
        x, _ = sysm.simulate(seed=s)
        DA = _wave_dir_adj(sysm.n, sysm.m)
        gl.append(directed_edge_auc(granger_bivariate(x, 4), DA))
        dl.append(directed_edge_auc(dmd_transition(x, 8), DA))
    assert np.mean(gl) < np.mean(dl)                      # zero-lag mixing hurts Granger more
