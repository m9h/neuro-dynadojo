"""Probe a representation for the GENERATIVE FACTORS of simulated M/EEG — the foundation-
model evaluation neuro-dynadojo adds to the classical bench. Here the 'model' is the
dependency-light spectral baseline (`bandpower_embed`); swap in `braindecode_embed(model, .)`
for a real frozen FM. Each factor is probed in SENSOR space under volume conduction + noise,
so the score reflects whether the representation encodes the physics through the confounds a
real recording imposes.

  python examples/fm_probe.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurodynadojo import HopfNetworkSystem, KuramotoNetworkSystem, RingWaveSystem
from neurodynadojo.probes import probe_factor, bandpower_embed

EMBED = lambda o: bandpower_embed(o, fs=250.0)          # <- replace with braindecode_embed(model, o)


def main():
    print("Representation probe (embedding = spectral baseline) — recovery of generative factors")
    print("from SENSOR-space recordings under volume conduction (leak 0.8) + 6 dB pink noise.\n")
    factors = [
        ("frequency f0 (Hz)",   HopfNetworkSystem,   "f0",       [6, 9, 12, 15, 18], "regress"),
        ("coupling k",          HopfNetworkSystem,   "k",        [0.3, 0.7, 1.1, 1.5], "regress"),
        ("conduction velocity", HopfNetworkSystem,   "velocity", [3, 5, 8, 12],       "regress"),
        ("phase-lag alpha",     RingWaveSystem,      "alpha",    [0.3, 0.6, 0.9, 1.2], "regress"),
    ]
    print(f"  {'generative factor':22s} {'system':18s} {'probe r':>8s}")
    for label, Sys, factor, values, task in factors:
        res = probe_factor(EMBED, Sys, factor, values, task=task, n_per=8,
                           space="sensor", leak=0.8, snr=6.0)
        print(f"  {label:22s} {Sys.__name__:18s} {res.get('r', res.get('auc')):8.2f}")
    print("\n  A representation that has learned the dynamics should recover these factors ABOVE")
    print("  the spectral baseline (frequency is trivially spectral; coupling / velocity / phase-lag")
    print("  are the interesting ones). Swap EMBED for a frozen braindecode FM to score it.")


if __name__ == "__main__":
    main()
