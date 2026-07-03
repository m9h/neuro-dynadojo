"""Scenario battery x method matrix — the netsim table for the EEG-FM era. Runs the HBN-
grounded scenarios against classical features (band-power, phase-connectivity) AND the FM zoo
(BIOT/CBraMod/LUNA + REVE/LaBraM), scored with fmscope's canonical LP. Each scenario is
engineered to favor a different method, so the matrix should have DIFFERENT winners per row.

  bash examples/run_leaderboard_container.sh /ndd/examples/scenario_benchmark.py
"""
import sys
sys.path.insert(0, "/ndd/src")

import numpy as np, torch, mne
import braindecode.models as bm
from scipy.signal import resample, welch, hilbert, butter, filtfilt
from fmscope.training.lp import eval_seed
from neurodynadojo.scenarios import SCENARIOS, CH32

mne.set_log_level("error")
torch.backends.cudnn.enabled = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PER, SEEDS = 60, (0, 1, 2)
BD = {"BIOT": ("InterpolatedBIOT", "braindecode/biot-pretrained-six-datasets-18chs", 200., 1000, {}, False),
      "CBraMod": ("CBraMod", "braindecode/cbramod-pretrained", 200., 1000, {}, True),
      "LUNA": ("LUNA", "PulpBio/LUNA", 200., 1000, {"filename": "LUNA_base.safetensors"}, True)}


def bandpower(X, bands=((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))):
    f, P = welch(X.astype(np.float64), fs=250., nperseg=256, axis=2)
    return np.concatenate([np.log(P[:, :, (f >= lo) & (f < hi)].mean(2) + 1e-20) for lo, hi in bands], 1)


def phaseconn(X):
    b, a = butter(4, [8 / 125., 12 / 125.], btype="band"); Z = hilbert(filtfilt(b, a, X, axis=2), axis=2)
    return np.stack([np.imag(z @ z.conj().T / z.shape[1]).ravel() for z in Z])


def info32():
    info = mne.create_info(CH32, 200., "eeg"); info.set_montage("standard_1005", on_missing="ignore")
    return info


def build_bd(cls, mid, sf, win, extra, lazy, info):
    C = getattr(bm, cls)
    kw = dict(chs_info=info["chs"]) if cls.startswith("Interpolated") else dict(n_chans=32)
    m = C.from_pretrained(mid, n_outputs=2, n_times=win, sfreq=sf, **kw, **extra).to(DEV).eval()
    if lazy:
        with torch.no_grad(): m(torch.zeros(1, 32, win, device=DEV))
    cap = {}; m.final_layer.register_forward_pre_hook(lambda mod, a: cap.__setitem__("z", a[0].detach()))
    return m, cap


def embed_bd(m, cap, X, sf, win, bs=128):
    out = []
    for i in range(0, len(X), bs):
        b = resample(X[i:i + bs].astype(np.float64), int(round(1000 * sf / 250.)), axis=-1)
        L = b.shape[-1]
        if L != win:
            b = b[..., (L - win) // 2:(L - win) // 2 + win] if L > win else np.pad(
                b, [(0, 0), (0, 0), ((win - L) // 2, win - L - (win - L) // 2)], mode="edge")
        b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
        with torch.no_grad(): m(torch.tensor(b, dtype=torch.float32, device=DEV))
        z = cap["z"]; out.append((z.mean(1) if z.ndim == 3 else z).cpu().numpy().reshape(len(b), -1))
    return np.concatenate(out)


def embed_adp(ad, loaded, X, sf, win, bs=64):
    b = resample(X.astype(np.float64), int(round(1000 * sf / 250.)), axis=-1)
    if win and b.shape[-1] != win:
        L = b.shape[-1]; b = b[..., (L - win) // 2:(L - win) // 2 + win] if L > win else np.pad(
            b, [(0, 0), (0, 0), ((win - L) // 2, win - L - (win - L) // 2)], mode="edge")
    b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
    out = [np.asarray(ad.extract_features(loaded, {"eeg": b[i:i + bs], "ch_names": CH32,
           "electrode_names": CH32})).reshape(len(b[i:i + bs]), -1) for i in range(0, len(b), bs)]
    return np.concatenate(out)


def score(F, y):
    n = len(y)
    return float(np.mean([eval_seed(F, np.arange(n), np.asarray(y), np.arange(n), s,
                                    cv="stratified-kfold", n_splits=5)[0] for s in SEEDS]))


def main():
    print(f"Scenario battery x method matrix (fmscope LP balanced acc, chance .50). device={DEV}\n")
    data = {name: fn(N_PER, 0) for name, fn in SCENARIOS.items()}      # (X, y, ch) per scenario
    scen = list(SCENARIOS)
    rows = {"band-power": {}, "phase-conn": {}}
    for s in scen:
        X, y, _ = data[s]
        rows["band-power"][s] = score(bandpower(X), y)
        rows["phase-conn"][s] = score(phaseconn(X), y)
    info = info32()
    for name, (cls, mid, sf, win, extra, lazy) in BD.items():
        try:
            m, cap = build_bd(cls, mid, sf, win, extra, lazy, info)
            rows[name] = {s: score(embed_bd(m, cap, data[s][0], sf, win), data[s][1]) for s in scen}
            print(f"[ok] {name}"); del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {name}: {repr(e)[:80]}")
    try:
        from emeg_fm.eeg_fm import REVEAdapter, LaBraMAdapter, REVE_BASE_ID, LABRAM_DEFAULT_ID
        for name, Acls, hf, sf, win in [("REVE", REVEAdapter, REVE_BASE_ID, 200., None),
                                        ("LaBraM", LaBraMAdapter, LABRAM_DEFAULT_ID, 200., 3000)]:
            try:
                ad = Acls(); loaded = ad.load_model(hf)
                rows[name] = {s: score(embed_adp(ad, loaded, data[s][0], sf, win), data[s][1]) for s in scen}
                print(f"[ok] {name}")
            except Exception as e:
                print(f"[skip] {name}: {repr(e)[:80]}")
    except Exception as e:
        print(f"[skip] adapters: {repr(e)[:80]}")
    print(f"\n  {'method':14s} " + " ".join(f"{s[:6]:>7s}" for s in scen))
    for r in rows:
        print(f"  {r:14s} " + " ".join(f"{rows[r][s]:7.2f}" for s in scen))
    print("\n  Each scenario favors a different method (band-power->spectral, waveform/FM->evoked,")
    print("  phase->wave). A good FM should win the evoked/wave/naturalistic rows band-power can't.")


if __name__ == "__main__":
    main()
