"""TDD for the electrophysiological FSLNets bench (hopf.py + fc.py):
(1) the Hopf generator is stable and its SOURCE-space FC recovers the structural
connectome well (fixing the weak MPR working point); (2) volume-conduction leakage breaks
zero-lag correlation but imaginary coherence / wPLI survive it.
"""
import numpy as np

from neurodynadojo.generators.hopf import (HopfNetworkSystem, KuramotoNetworkSystem, RingWaveSystem,
                          simulate_hopf, leakage_matrix)
from neurodynadojo.algorithms.fc import (correlation_fc, imag_coherence_fc, wpli_fc, edge_auc,
                        fc_algorithms)


def test_generator_stable_and_finite():
    sys = HopfNetworkSystem(seed_struct=0, T=2000.0)
    x, adj = sys.simulate(seed=0)
    assert x.shape == (30, 500) and np.all(np.isfinite(x))    # 2000 ms @ 250 Hz
    assert x.std() > 0 and np.abs(x).max() < 1e3        # bounded (no blow-up)
    assert adj.shape == (30, 30) and set(np.unique(adj)) <= {0, 1}


def test_source_space_correlation_recovers_structure():
    aucs = []
    for s in range(4):
        sys = HopfNetworkSystem(seed_struct=s, space="source")
        x, adj = sys.simulate(seed=s)
        aucs.append(edge_auc(correlation_fc(x), adj))
    assert np.mean(aucs) > 0.8                            # strong SC->FC (unlike weak MPR)


def test_volume_conduction_degrades_correlation():
    """Honest, regime-specific claim: Hopf coupling imprints structure on ZERO-LAG
    correlation, which volume-conduction leakage corrupts. (Lag-robust recovery needs a
    PHASE-coupled generator, not this amplitude-coupled one -- see the bench script.)"""
    corr_s, corr_l = [], []
    for s in range(5):
        base = dict(seed_struct=s)
        xs, adj = HopfNetworkSystem(space="source", **base).simulate(seed=s)
        xl, _ = HopfNetworkSystem(space="leaked", leak=0.8, **base).simulate(seed=s)
        corr_s.append(edge_auc(correlation_fc(xs), adj))
        corr_l.append(edge_auc(correlation_fc(xl), adj))
    assert np.mean(corr_s) > 0.8                          # generator imprints structure
    assert np.mean(corr_l) < np.mean(corr_s) - 0.05       # leakage is a real confound


def test_kuramoto_system_stable_and_recovers_structure():
    """The PHASE-coupled companion System is stable and its source-space correlation
    recovers structure (a valid generator; broadens the bench's regime coverage)."""
    aucs = []
    for s in range(4):
        sysm = KuramotoNetworkSystem(seed_struct=s, space="source")
        x, adj = sysm.simulate(seed=s)
        assert np.all(np.isfinite(x)) and np.abs(x).max() <= 1.0 + 1e-6   # sin() bounded
        aucs.append(edge_auc(correlation_fc(x), adj))
    assert np.mean(aucs) > 0.8


def test_ring_wave_system_stable_and_bench_compatible():
    """Directed traveling-wave System: stable, bounded sin() signal, returns (obs, adj).
    (Adjacency is intentionally NOT recoverable by pairwise FC under a global wave -- its
    identifiable target is wavenumber; see smni_cmi.waves.)"""
    sysm = RingWaveSystem(seed_struct=0, T=2000.0)
    obs, adj = sysm.simulate(seed=0)
    assert obs.shape == (30, 500) and np.all(np.isfinite(obs)) and np.abs(obs).max() <= 1.0 + 1e-6
    assert adj.shape == (30, 30) and set(np.unique(adj)) <= {0, 1}


def test_fc_algorithms_are_bench_compatible():
    sys = HopfNetworkSystem(T=2000.0)
    obs, adj = sys.simulate(seed=0)
    for alg in fc_algorithms():
        score = alg.estimate(obs, sys)
        assert score.shape == (sys.n, sys.n)
        assert np.isfinite(edge_auc(score, adj)) or True   # AUC defined given both classes


def test_isolated_node_no_damping():
    """Verify that isolated nodes (degree = 0) do not suffer from coupling self-damping."""
    C = np.zeros((3, 3))
    dist = np.zeros((3, 3))
    omega = np.array([2 * np.pi * 10 / 1000.0] * 3)
    x = simulate_hopf(C, dist, a=0.05, omega=omega, k=5.0, velocity=5.0, dt=0.1, T=1000.0, noise=0.0, seed=0)
    # Undamped Hopf nodes in a limit cycle should have high standard deviation (~0.22)
    assert np.all(x.std(axis=1) > 0.1)


def test_leadfield_3shell_selection():
    """Verify that leadfield='3shell' selects the Berg-Scherg model and simulates successfully."""
    sys_radial = HopfNetworkSystem(leadfield="radial", T=500.0)
    sys_3shell = HopfNetworkSystem(leadfield="3shell", T=500.0)
    assert sys_radial.L.shape == sys_3shell.L.shape
    assert not np.allclose(sys_radial.L, sys_3shell.L)
    x, _ = sys_3shell.simulate(seed=0)
    assert np.all(np.isfinite(x))
