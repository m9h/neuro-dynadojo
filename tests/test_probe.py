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


def test_bandpower_probe_recovers_frequency():
    res = probe_factor(lambda o: bandpower_embed(o, fs=250.0), HopfNetworkSystem,
                       "f0", [6.0, 9.0, 12.0, 15.0, 18.0], n_per=6, T=3000.0)
    assert res["metric"] == "r"
    assert res["r"] > 0.8                     # spectral baseline trivially encodes frequency
