"""Probe a *braindecode* model — the FM-loading path (requires the `[fm]` extra:
`pip install -e .[fm]`). Here we use a braindecode architecture as a frozen feature
extractor; swap in real pretrained weights via braindecode's HuggingFace-style loading
(`Model.from_pretrained(...)`, e.g. BENDR / LUNA) to score an actual foundation model.

The point: the same probe harness that scores classical estimators scores a frozen FM.
A random-init architecture recovers little (that is the honest floor); a model that has
learned the dynamics should recover the generative factors ABOVE the spectral baseline
(see examples/fm_probe.py), in sensor space, under volume conduction + noise.

  pip install -e .[fm]
  python examples/fm_probe_braindecode.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo import HopfNetworkSystem, RingWaveSystem
from neurodynadojo.probes import braindecode_embed, probe_factor

try:
    from braindecode.models import EEGNet
except ImportError:
    raise SystemExit("Install the FM extra:  pip install -e .[fm]")

T_MS, FS = 3000.0, 250.0
N_CH, N_T = 64, int(T_MS / 1000 * FS)


def load_model():
    """Return a frozen (n_chans, n_times) -> features model.
    Replace with a pretrained foundation model, e.g.:
        from braindecode.models import Labram
        model = Labram.from_pretrained("...hf-hub-id...")   # HuggingFace-style loading
    """
    model = EEGNet(n_chans=N_CH, n_outputs=16, n_times=N_T)   # random-init stand-in
    model.eval()
    return model


def main():
    model = load_model()
    embed = lambda o: braindecode_embed(model, o)
    print(f"Probing a braindecode model ({model.__class__.__name__}) — sensor space, leak 0.8, 6 dB pink.")
    print("(random-init architecture here; load pretrained weights to score a real FM)\n")
    factors = [
        ("frequency f0",  HopfNetworkSystem, "f0",       [6, 10, 14, 18]),
        ("coupling k",    HopfNetworkSystem, "k",        [0.3, 0.7, 1.1, 1.5]),
        ("velocity",      HopfNetworkSystem, "velocity", [3, 5, 8, 12]),
        ("phase-lag",     RingWaveSystem,    "alpha",    [0.3, 0.6, 0.9, 1.2]),
    ]
    print(f"  {'factor':14s} {'probe r':>8s}")
    for label, Sys, factor, values in factors:
        res = probe_factor(embed, Sys, factor, values, task="regress", n_per=6,
                           space="sensor", leak=0.8, snr=6.0, T=T_MS)
        print(f"  {label:14s} {res['r']:8.2f}")
    print("\n  Compare these to examples/fm_probe.py (spectral baseline). A foundation model")
    print("  worth its weights should meet or beat the baseline on the non-trivial factors.")


if __name__ == "__main__":
    main()
