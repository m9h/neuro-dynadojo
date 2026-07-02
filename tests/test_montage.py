"""Realistic 10-20 montage generators — real channel names/positions for FM-fair sensor
space. Skipped when the optional `[mne]` extra is absent (keeps core CI green)."""
import numpy as np
import pytest

pytest.importorskip("mne")

from neurodynadojo import HopfNetworkSystem
from neurodynadojo.generators.montage import resolve_montage


def test_resolve_montage_centered_and_scaled():
    names, pos = resolve_montage("1020_20")
    assert len(names) == 20 and pos.shape == (20, 3)
    assert np.allclose(pos.mean(0), 0.0, atol=1e-6)          # centred at centroid
    assert 60.0 < np.linalg.norm(pos, axis=1).mean() < 130.0  # ~mm head scale


def test_system_uses_real_montage():
    sysm = HopfNetworkSystem(space="sensor", montage="1020_20", T=2000.0)
    assert sysm.n_ch == 20
    assert sysm.ch_names[:3] == ["Fp1", "Fp2", "F7"]
    obs, _ = sysm.simulate(seed=0)
    assert obs.shape == (20, 500) and np.all(np.isfinite(obs))  # real-montage sensor recording
