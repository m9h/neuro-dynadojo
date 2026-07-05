import os
import numpy as np
import pytest
from neurodynadojo import HopfNetworkSystem
from neurodynadojo.generators.montage import resolve_montage


def test_resolve_sci_montages():
    if not os.path.exists("/home/mhough/data/sci.utah.edu/~datasets/SCI_headmodel/EEG.zip"):
        pytest.skip("SCI Head Model EEG.zip not available")
    
    for spec in ("sci128", "sci256"):
        names, pos = resolve_montage(spec)
        n_ch = 128 if spec == "sci128" else 256
        assert len(names) == n_ch and pos.shape == (n_ch, 3)
        assert np.allclose(pos.mean(0), 0.0, atol=1e-5)
        assert 70.0 < np.linalg.norm(pos, axis=1).mean() < 100.0


def test_system_uses_sci_montage():
    if not os.path.exists("/home/mhough/data/sci.utah.edu/~datasets/SCI_headmodel/EEG.zip"):
        pytest.skip("SCI Head Model EEG.zip not available")
        
    sysm = HopfNetworkSystem(space="sensor", montage="sci128", leadfield="radial", T=200.0)
    assert sysm.n_ch == 128
    assert sysm.ch_names[:3] == ["EEG001", "EEG002", "EEG003"]
    obs, _ = sysm.simulate(seed=0)
    assert obs.shape == (128, 50) and np.all(np.isfinite(obs))
