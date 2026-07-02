"""The FM-ready probe harness: a bring-your-own embedding is linearly probed for a known
generative factor of the simulated sensor recordings. Uses the dependency-light spectral
baseline as a stand-in 'FM' — it must recover the oscillation frequency (which band-power
trivially encodes) and produce a sensor-shaped labeled dataset.
"""
import numpy as np

from neurodynadojo import HopfNetworkSystem
from neurodynadojo.probes import factor_dataset, probe_factor, bandpower_embed


def test_factor_dataset_is_sensor_shaped():
    ds = factor_dataset(HopfNetworkSystem, "k", [0.5, 1.5], n_per=3, space="sensor", T=2000.0)
    assert len(ds) == 6
    obs, v = ds[0]
    assert obs.shape[0] == 64                 # n_ch sensor projection (braindecode input shape)
    assert obs.shape[1] == 500                # 2000 ms @ 250 Hz
    assert v in (0.5, 1.5)


def test_bendr_exact_montage():
    """BENDR's exact 19-EEG pretraining montage (hardcoded standard_1005 positions, no MNE)."""
    from neurodynadojo.generators.montage import resolve_montage
    names, pos = resolve_montage("bendr")
    assert names[0] == "FP1" and names[-1] == "O2" and len(names) == 19
    assert pos.shape == (19, 3) and np.allclose(pos.mean(0), 0.0, atol=1e-6)
    obs, _ = HopfNetworkSystem(space="sensor", montage="bendr", T=2000.0).simulate(seed=0)
    assert obs.shape[0] == 19 and np.all(np.isfinite(obs))


def test_bandpower_probe_recovers_frequency():
    res = probe_factor(lambda o: bandpower_embed(o, fs=250.0), HopfNetworkSystem,
                       "f0", [6.0, 9.0, 12.0, 15.0, 18.0], n_per=6, T=3000.0)
    assert res["metric"] == "r"
    assert res["r"] > 0.8                     # spectral baseline trivially encodes frequency
