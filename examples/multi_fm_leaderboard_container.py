"""Full EEG-FM-zoo leaderboard — runs INSIDE emeg-fm's NGC container (braindecode 1.5.2).
Loads the zoo with emeg-fm's validated recipe (Interpolated*/native `from_pretrained` +
`final_layer` forward-pre-hook = pooled embedding), scores each on neuro-dynadojo's synthetic
generative factors with fmscope's canonical LP. Three-project interop.

Launch (from the host):
  bash examples/run_leaderboard_container.sh
"""
import sys
sys.path.insert(0, "/ndd/src")                    # neuro-dynadojo (mounted)

import numpy as np, torch, mne
import braindecode.models as bm
from scipy.signal import resample
from fmscope.training.lp import eval_seed
from neurodynadojo import HopfNetworkSystem, RingWaveSystem
from neurodynadojo.probes import factor_dataset, bandpower_embed

mne.set_log_level("error")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PER, DT, BG, SEEDS = 20, 0.5, 2.0, (0, 1, 2)
# (Interpolated/native class, hf id, sfreq, n_times, extra_from_pretrained_kwargs, needs_lazy_init)
ZOO = {
    "BIOT":    ("InterpolatedBIOT",  "braindecode/biot-pretrained-six-datasets-18chs", 200.0, 1000, {}, False),
    "CBraMod": ("CBraMod",           "braindecode/cbramod-pretrained", 200.0, 1000, {}, True),
    "LUNA":    ("LUNA",              "PulpBio/LUNA", 200.0, 1000, {"filename": "LUNA_base.safetensors"}, True),
    "LaBraM":  ("InterpolatedLaBraM","braindecode/labram-pretrained", 200.0, 1000, {}, False),
    "BENDR":   ("InterpolatedBENDR", "braindecode/braindecode-bendr", 250.0, 1000, {}, False),
}
CH32 = ["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
        "CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO3","PO4","O1","Oz","O2","AF3","AF4"]
FACTORS = [
    ("frequency",  HopfNetworkSystem, "f0",       [6, 9, 12, 15, 18]),
    ("coupling",   HopfNetworkSystem, "k",        [0.3, 0.6, 0.9, 1.2, 1.5]),
    ("delay",      HopfNetworkSystem, "velocity", [3, 5, 7, 10, 13]),
    ("regime",     HopfNetworkSystem, "a",        [-0.05, -0.02, 0.0, 0.02, 0.05]),
    ("phase-lag",  RingWaveSystem,    "alpha",    [0.3, 0.6, 0.9, 1.2, 1.5]),
    ("wavenumber", RingWaveSystem,    "m",        [1, 2, 3, 4]),
]


def build(cls_name, mid, sfreq, win, extra, lazy_init):
    cls = getattr(bm, cls_name)
    if cls_name.startswith("Interpolated"):                 # montage-adapt via chs_info
        info = mne.create_info(CH32, sfreq, "eeg"); info.set_montage("standard_1005", on_missing="ignore")
        kw = dict(chs_info=info["chs"])
    else:                                                   # native model takes n_chans
        kw = dict(n_chans=len(CH32))
    m = cls.from_pretrained(mid, n_outputs=2, n_times=win, sfreq=sfreq, **kw, **extra).to(DEV).eval()
    if lazy_init:                                           # CBraMod/LUNA: init lazy modules with a dummy forward
        with torch.no_grad():
            m(torch.zeros(1, len(CH32), win, device=DEV))
    cap = {}
    m.final_layer.register_forward_pre_hook(lambda mod, a: cap.__setitem__("z", a[0].detach()))
    return m, cap


def embed(m, cap, W, sfreq_out, win, bs=64):
    out = []
    for i in range(0, len(W), bs):
        b = W[i:i + bs].astype(np.float64)
        if abs(250.0 - sfreq_out) > 1e-6:
            b = resample(b, int(round(b.shape[-1] * sfreq_out / 250.0)), axis=-1)
        L = b.shape[-1]
        if L != win:
            b = (b[..., (L - win) // 2:(L - win) // 2 + win] if L > win
                 else np.pad(b, [(0, 0), (0, 0), ((win - L) // 2, win - L - (win - L) // 2)], mode="edge"))
        b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
        with torch.no_grad():
            m(torch.tensor(b, dtype=torch.float32, device=DEV))
        z = cap["z"]
        out.append((z.mean(1) if z.ndim == 3 else z).cpu().numpy().reshape(len(b), -1))
    return np.concatenate(out)


def adapter_embed(adapter, loaded, W, ch_names, sfreq_out, win, bs=64):
    """Embed via an emeg-fm hand adapter (REVE/LaBraM/ZUNA): resample 250->sfreq_out, fit
    `win` (None=flexible), z-score+clamp, then adapter.extract_features with its input dict."""
    b = W.astype(np.float64)
    if abs(250.0 - sfreq_out) > 1e-6:
        b = resample(b, int(round(b.shape[-1] * sfreq_out / 250.0)), axis=-1)
    if win:
        L = b.shape[-1]
        b = (b[..., (L - win) // 2:(L - win) // 2 + win] if L > win
             else np.pad(b, [(0, 0), (0, 0), ((win - L) // 2, win - L - (win - L) // 2)], mode="edge"))
    b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
    out = []
    for i in range(0, len(b), bs):
        out.append(np.asarray(adapter.extract_features(
            loaded, {"eeg": b[i:i + bs], "ch_names": list(ch_names), "electrode_names": list(ch_names)})))
    return np.concatenate(out).reshape(len(b), -1)


def score(feats, y):
    n = len(y)
    return float(np.mean([eval_seed(feats, np.arange(n), np.asarray(y), np.arange(n), s,
                                    cv="stratified-kfold", n_splits=5)[0] for s in SEEDS]))


def main():
    print(f"Full-zoo leaderboard (emeg-fm container).  device={DEV}, n_per={N_PER}\n")
    # simulate every factor dataset once (32-ch montage) -> reuse across all FMs
    data = {}
    for label, Sys, factor, values in FACTORS:
        ds = factor_dataset(Sys, factor, values, n_per=N_PER, space="sensor", montage=CH32,
                            background=BG, leak=0.8, snr=6.0, T=4000.0, dt=DT)
        W = np.stack([o for o, _ in ds]); y = np.array([v for _, v in ds])
        data[label] = (W, (y > np.median(y)).astype(int))
    facs = [f[0] for f in FACTORS]
    rows = {}
    rows["baseline (bandpower)"] = {f: score(np.stack([bandpower_embed(o, fs=250.0) for o in data[f][0]]),
                                             data[f][1]) for f in facs}
    for name, (cls_name, mid, sf, win, extra, lazy) in ZOO.items():
        try:
            m, cap = build(cls_name, mid, sf, win, extra, lazy)
            rows[name] = {f: score(embed(m, cap, data[f][0], sf, win), data[f][1]) for f in facs}
            print(f"[ok] {name}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {name}: {repr(e)[:100]}")
    # emeg-fm hand-adapter models (REVE flexible; LaBraM 200 Hz / 3000 samp / 10-20 vocab)
    try:
        from emeg_fm.eeg_fm import (REVEAdapter, LaBraMAdapter, REVE_BASE_ID, LABRAM_DEFAULT_ID)
        ADP = [("REVE", REVEAdapter, REVE_BASE_ID, 200.0, None),
               ("LaBraM", LaBraMAdapter, LABRAM_DEFAULT_ID, 200.0, 3000)]
    except Exception as e:
        ADP = []; print(f"[skip] adapters unavailable: {repr(e)[:80]}")
    for name, Acls, hf, sfo, win in ADP:
        try:
            ad = Acls(); loaded = ad.load_model(hf)
            rows[name] = {f: score(adapter_embed(ad, loaded, data[f][0], CH32, sfo, win), data[f][1]) for f in facs}
            print(f"[ok] {name}")
        except Exception as e:
            print(f"[skip] {name}: {repr(e)[:100]}")
    print(f"\n  {'model':22s} " + " ".join(f"{f[:5]:>6s}" for f in facs) + f" {'MEAN':>7s}")
    for r in sorted(rows, key=lambda r: -np.nanmean([rows[r].get(f, np.nan) for f in facs])):
        v = [rows[r].get(f, np.nan) for f in facs]
        print(f"  {r:22s} " + " ".join(f"{x:6.2f}" for x in v) + f" {np.nanmean(v):7.2f}")
    print("\n  Balanced accuracy (chance 0.50) recovering each generative factor from frozen FM")
    print("  embeddings, scored with fmscope's canonical LP. Baseline = band-power.")


if __name__ == "__main__":
    main()
