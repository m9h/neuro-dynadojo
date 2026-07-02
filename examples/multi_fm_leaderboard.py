"""Multi-FM leaderboard — score pretrained EEG foundation models on neuro-dynadojo's
ground-truth generative factors, using the **emeg-fm / fmscope** machinery.

Three-project interop (all open, complementary):
  * neuro-dynadojo (this repo) supplies the labeled synthetic recordings — each generative
    factor binarised low/high — the *known ground truth* real EEG can't provide.
  * fmscope (github.com/Jimmy110101013/fmscope) scores each frozen embedding with its
    canonical LP (`eval_seed`: per-window LogReg -> pooled balanced accuracy) — the same
    probe it uses to audit FMs for identity confound on real data; here we audit *mechanistic
    content* on synthetic data.
  * emeg-fm (the NeuroTechX Atlas) provides the ONE montage-agnostic extractor that loads the
    whole EEG-FM zoo (BENDR/EEGPT/LaBraM/BIOT/CBraMod/LUNA/REVE) uniformly, via
    `Interpolated*.from_pretrained(hf_id, chs_info=...)` + a `final_layer` forward-pre-hook.

Any FM wrapped as an `(N,C,T)->(N,D)` callable drops in. Below, BENDR loads directly in a
stock braindecode; the rest of the zoo loads via **emeg-fm's container extractor** (NGC 26.06
+ braindecode 1.5.2; stock 1.6.x hits Interpolated ctor / chans_id mismatches). Run this
bridge inside that container with `emeg_fm.eeg_fm` extractors to fill the whole board.
Baseline = log band-power (the floor to beat).

  pip install -e .[fm,mne]     # + fmscope on PYTHONPATH; full zoo needs emeg-fm's container
  python examples/multi_fm_leaderboard.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/home/mhough/dev/emeg-fm/fmscope")   # fmscope (other lab; interop)

import numpy as np
import torch
torch.backends.cudnn.enabled = False
from neurodynadojo import HopfNetworkSystem, RingWaveSystem
from neurodynadojo.probes import factor_dataset, bandpower_embed

try:
    from fmscope.training.lp import eval_seed
    from braindecode.models import BENDR, EEGPT, Labram
except ImportError as e:
    raise SystemExit(f"Need fmscope + braindecode: {e}")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PER, DT, BG, SEEDS = 20, 0.5, 2.0, (0, 1, 2)
FACTORS = [
    ("frequency",  HopfNetworkSystem, "f0",       [6, 9, 12, 15, 18]),
    ("coupling",   HopfNetworkSystem, "k",        [0.3, 0.6, 0.9, 1.2, 1.5]),
    ("delay",      HopfNetworkSystem, "velocity", [3, 5, 7, 10, 13]),
    ("regime",     HopfNetworkSystem, "a",        [-0.05, -0.02, 0.0, 0.02, 0.05]),
    ("phase-lag",  RingWaveSystem,    "alpha",    [0.3, 0.6, 0.9, 1.2, 1.5]),
    ("wavenumber", RingWaveSystem,    "m",        [1, 2, 3, 4]),
]


def batch_embed(model, W, scale_ch=False):
    """(N, C, T) recordings -> (N, D) frozen features. z-score channels; optionally append a
    constant SCALE channel (BENDR). Uses model(x, return_features=True) where available."""
    X = (W - W.mean(2, keepdims=True)) / (W.std(2, keepdims=True) + 1e-8)
    if scale_ch:
        sc = np.tanh(np.log(np.sqrt((W ** 2).mean(axis=(1, 2))) + 1e-8))     # (N,)
        X = np.concatenate([X, np.broadcast_to(sc[:, None, None], (X.shape[0], 1, X.shape[2]))], 1)
    xt = torch.as_tensor(X, dtype=torch.float32).to(DEV)
    with torch.no_grad():
        out = model(xt, return_features=True)
    f = out.get("features", out.get("cls_token", next(iter(out.values())))) if isinstance(out, dict) else out
    return np.asarray(f.detach().cpu()).reshape(len(W), -1)


def load_fms():
    """Return [(name, montage, embed_fn)] for the FMs that load; skip the rest."""
    fms = []
    try:
        m = BENDR.from_pretrained("braindecode/braindecode-bendr", n_outputs=2).to(DEV).eval()
        fms.append(("BENDR", "bendr", lambda W, mm=m: batch_embed(mm, W, scale_ch=True)))
        print("[ok] BENDR")
    except Exception as e:
        print("[skip] BENDR:", repr(e)[:80])
    try:
        from braindecode.models.eegpt import EEGPT_CHANNELS
        chs = [c for c in EEGPT_CHANNELS if c not in ("SCALE",)]
        m = EEGPT.from_pretrained("braindecode/eegpt-pretrained", n_chans=len(chs),
                                  n_times=1000, sfreq=250).to(DEV).eval()
        fms.append(("EEGPT", list(chs), lambda W, mm=m: batch_embed(mm, W)))
        print(f"[ok] EEGPT ({len(chs)} ch)")
    except Exception as e:
        print("[skip] EEGPT:", repr(e)[:80])
    try:
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER
        chs = [c for c in LABRAM_CHANNEL_ORDER if c not in ("SCALE",)]
        m = Labram.from_pretrained("braindecode/labram-pretrained", n_chans=len(chs),
                                   n_times=1000, sfreq=250).to(DEV).eval()
        fms.append(("LaBraM", list(chs), lambda W, mm=m: batch_embed(mm, W)))
        print(f"[ok] LaBraM ({len(chs)} ch)")
    except Exception as e:
        print("[skip] LaBraM:", repr(e)[:80])
    return fms


def score(feats, labels):
    """fmscope canonical LP: per-window LogReg -> pooled balanced accuracy (1 window/rec)."""
    n = len(labels)
    ba = [eval_seed(feats, np.arange(n), np.asarray(labels), np.arange(n), s,
                    cv="stratified-kfold", n_splits=5)[0] for s in SEEDS]
    return float(np.mean(ba))


def main():
    print(f"Multi-FM leaderboard via fmscope LP.  device={DEV}, n_per={N_PER}, factors={len(FACTORS)}\n")
    fms = load_fms()
    if not fms:
        raise SystemExit("no FMs loaded")
    rows = {name: {} for name, _, _ in fms}
    rows["baseline (bandpower)"] = {}
    print(f"\n  probing {len(FACTORS)} factors ...")
    for label, Sys, factor, values in FACTORS:
        for name, montage, embed in fms:
            try:
                ds = factor_dataset(Sys, factor, values, n_per=N_PER, space="sensor",
                                    montage=montage, background=BG, leak=0.8, snr=6.0, T=4000.0, dt=DT)
            except Exception as e:
                rows[name][label] = np.nan; continue
            W = np.stack([o for o, _ in ds]); y = np.array([v for _, v in ds])
            yb = (y > np.median(y)).astype(int)
            rows[name][label] = score(embed(W), yb)
            if name == fms[0][0]:                                        # baseline on the first FM's montage
                rows["baseline (bandpower)"][label] = score(
                    np.stack([bandpower_embed(o, fs=250.0) for o in W]), yb)
    facs = [f[0] for f in FACTORS]
    print(f"\n  {'model':22s} " + " ".join(f"{f[:5]:>6s}" for f in facs) + f" {'MEAN':>7s}")
    order = sorted(rows, key=lambda r: -np.nanmean([rows[r].get(f, np.nan) for f in facs]))
    for r in order:
        vals = [rows[r].get(f, np.nan) for f in facs]
        print(f"  {r:22s} " + " ".join(f"{v:6.2f}" if v == v else "   nan" for v in vals) +
              f" {np.nanmean(vals):7.2f}")
    print("\n  Balanced accuracy (chance 0.50) recovering each generative factor from frozen")
    print("  embeddings. Baseline = band-power. Scored with fmscope's canonical LP (eval_seed).")


if __name__ == "__main__":
    main()
