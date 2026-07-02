"""Scaled BENDR-vs-baseline probe — a more robust table across more seeds and factors.
Each factor's dataset is simulated ONCE (shared by both embedders), BENDR is BATCH-embedded
on GPU, and six generative factors are probed on BENDR's exact montage with realistic 1/f
statistics. Faster integration (dt=0.5) keeps the simulation from dominating.

  pip install -e .[fm,mne]
  python examples/fm_probe_bendr_scaled.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
torch.backends.cudnn.enabled = False                 # avoids a cuDNN version mismatch on some boxes
from neurodynadojo import HopfNetworkSystem, RingWaveSystem
from neurodynadojo.probes import factor_dataset, representation_probe, bandpower_embed

try:
    from braindecode.models import BENDR
except ImportError:
    raise SystemExit("Install the FM extras:  pip install -e .[fm,mne]")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PER, DT, BG = 20, 0.5, 2.0
COMMON = dict(space="sensor", montage="bendr", background=BG, leak=0.8, snr=6.0, T=4000.0, dt=DT)
FACTORS = [
    ("frequency",  HopfNetworkSystem, "f0",       [6, 9, 12, 15, 18]),
    ("coupling",   HopfNetworkSystem, "k",        [0.3, 0.6, 0.9, 1.2, 1.5]),
    ("delay(vel)", HopfNetworkSystem, "velocity", [3, 5, 7, 10, 13]),
    ("regime(a)",  HopfNetworkSystem, "a",        [-0.05, -0.02, 0.0, 0.02, 0.05]),
    ("phase-lag",  RingWaveSystem,    "alpha",    [0.3, 0.6, 0.9, 1.2, 1.5]),
    ("wavenumber", RingWaveSystem,    "m",        [1, 2, 3, 4]),
]


def bendr_batch(model, obs_list):
    """Batch-embed a list of (19, T) recordings: z-score EEG + constant SCALE -> one forward."""
    X = []
    for o in obs_list:
        x = np.asarray(o, float)
        xe = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
        sc = np.full((1, x.shape[1]), float(np.tanh(np.log(np.sqrt((x ** 2).mean()) + 1e-8))))
        X.append(np.vstack([xe, sc]))
    xt = torch.as_tensor(np.stack(X), dtype=torch.float32).to(DEV)     # (N, 20, T)
    with torch.no_grad():
        out = model(xt, return_features=True)
    f = out["features"] if isinstance(out, dict) else out
    return np.asarray(f.detach().cpu()).reshape(len(obs_list), -1)


def main():
    print(f"Scaled BENDR probe: device={DEV}, n_per={N_PER}, dt={DT}, background={BG}, exact montage.")
    model = BENDR.from_pretrained("braindecode/braindecode-bendr", n_outputs=2).to(DEV).eval()
    print(f"\n  {'factor':12s} {'BENDR |r|':>10s} {'baseline |r|':>13s} {'delta':>7s}  (n)")
    for label, Sys, factor, values in FACTORS:
        t0 = time.time()
        ds = factor_dataset(Sys, factor, values, n_per=N_PER, **COMMON)   # simulate ONCE
        obs = [o for o, _ in ds]
        Xb = bendr_batch(model, obs)
        rb = abs(representation_probe(lambda v: v, list(zip(Xb, [v for _, v in ds])))["r"])
        rs = abs(representation_probe(lambda o: bandpower_embed(o, fs=250.0), ds)["r"])
        print(f"  {label:12s} {rb:10.2f} {rs:13.2f} {rb - rs:+7.2f}  ({len(ds)})  [{time.time()-t0:.0f}s]")
    print("\n  |r| (sign removed). With realistic 1/f statistics and BENDR's exact montage, the")
    print("  spectral baseline is the floor to beat; delta>0 means the frozen FM embedding carries")
    print("  factor information beyond the spectrum. This is the fair, scaled mechanistic probe.")


if __name__ == "__main__":
    main()
