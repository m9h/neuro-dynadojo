"""TDD for smni_cmi.netsim — the Smith-2011 confound battery ported to electrophysiology.
Baseline recovers the directed connectome; the netsim-classic confounds bite in the right
places (shared global input specifically collapses DIRECTED recovery).
"""
import numpy as np

from neurodynadojo.generators.netsim import NetsimSystem, directed_modular_adjacency
from neurodynadojo.algorithms.fc import correlation_fc, partialcorr_fc, edge_auc
from neurodynadojo.algorithms.directed import dmd_transition, directed_edge_auc


def test_directed_connectome_and_truths():
    rng = np.random.default_rng(0)
    C, _ = directed_modular_adjacency(30, 3, 0.6, 0.0, 0.3, rng)
    assert C.shape == (30, 30) and not np.allclose(C, C.T)     # asymmetric (directed)
    sysm = NetsimSystem(seed_struct=0, back=0.3)
    assert sysm.directed_truth().sum() > 0
    assert sysm.undirected_truth().sum() >= sysm.directed_truth().sum()


def test_baseline_recovers_structure():
    cu, du = [], []
    for s in range(3):
        sysm = NetsimSystem(seed_struct=s, back=0.3)
        x, _ = sysm.simulate(seed=s)
        cu.append(max(edge_auc(correlation_fc(x), sysm.undirected_truth()),
                      edge_auc(partialcorr_fc(x), sysm.undirected_truth())))
        du.append(directed_edge_auc(dmd_transition(x, 8), sysm.directed_truth()))
    assert np.mean(cu) > 0.85 and np.mean(du) > 0.75


def test_external_connectome_thresholds_truth_and_runs():
    """The real-connectome path: a dense weighted SC yields a density-matched binary edge
    truth and a runnable simulation (data-independent synthetic dense matrix)."""
    rng = np.random.default_rng(0)
    W = np.abs(rng.standard_normal((40, 40))); W = (W + W.T) / 2; np.fill_diagonal(W, 0)
    sysm = NetsimSystem(connectome=W, edge_density=0.25, seed_struct=0)
    assert sysm.n == 40
    dens = sysm.undirected_truth()[np.triu_indices(40, 1)].mean()
    assert 0.15 < dens < 0.4                                  # ~edge_density after symmetrising
    x, _ = sysm.simulate(seed=0)
    assert x.shape[0] == 40 and np.all(np.isfinite(x))


def test_shared_input_collapses_directed_not_undirected():
    """The netsim common-input confound: a shared global driver wrecks directionality but
    partial correlation still recovers the undirected edges."""
    du, uu = [], []
    for s in range(3):
        sysm = NetsimSystem(seed_struct=s, back=0.3, shared=0.5)
        x, _ = sysm.simulate(seed=s)
        du.append(directed_edge_auc(dmd_transition(x, 8), sysm.directed_truth()))
        uu.append(edge_auc(partialcorr_fc(x), sysm.undirected_truth()))
    assert np.mean(uu) > 0.8                                   # undirected survives
    assert np.mean(du) < np.mean(uu)                          # directed collapses relative to it
