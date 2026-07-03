"""FM-vs-classical on REAL HBN ERP data — the real-data complement to the synthetic
leaderboard. 998 HBN children, SurroundSupp condition-ERPs (111-ch GSN-HydroCel, from
/data), embedded with the EEG-FM zoo (emeg-fm recipe, in-container) and ridge/logistic-CV
to brain-age and sex, vs a band-power baseline. Runs inside emeg-fm's NGC container.

  bash examples/run_leaderboard_container.sh /ndd/examples/hbn_erp_fm.py
"""
import glob, os
import numpy as np, pandas as pd, torch, mne
import braindecode.models as bm
from scipy.signal import resample, welch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score

mne.set_log_level("error")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE, HBN = "/data/datasets/hbn-prep-evoked", "/data/datasets/hbn-eeg"
ZOO = {  # (class, hf id, sfreq, n_times, extra, lazy_init)
    "BIOT":    ("InterpolatedBIOT", "braindecode/biot-pretrained-six-datasets-18chs", 200., 1000, {}, False),
    "CBraMod": ("CBraMod",          "braindecode/cbramod-pretrained", 200., 1000, {}, True),
    "LUNA":    ("LUNA",             "PulpBio/LUNA", 200., 1000, {"filename": "LUNA_base.safetensors"}, True),
}


def load_data():
    names = list(np.load(f"{CACHE}/ch_names.npy"))
    fs = [f for f in sorted(glob.glob(f"{CACHE}/*.npy")) if "ch_names" not in f]
    ids = [os.path.basename(f)[:-4] for f in fs]
    X = np.stack([np.load(f) for f in fs]).astype(np.float32)          # (N, 111, 600)
    p = pd.read_csv(f"{HBN}/participants.tsv", sep="\t")
    p["pid"] = p.participant_id.str.replace("sub-", "", regex=False)
    lab = p.set_index("pid")
    age = np.array([float(lab.loc[i, "age"]) for i in ids])
    sex = np.array([1 if str(lab.loc[i, "sex"]).strip() in ("M", "Male", "1") else 0 for i in ids])
    info = mne.create_info(names, 100., "eeg")
    info.set_montage("GSN-HydroCel-129", on_missing="ignore", match_case=False)
    return X, age, sex, info


def build(cls_name, mid, sfreq, win, extra, lazy, info):
    cls = getattr(bm, cls_name)
    kw = dict(chs_info=info["chs"]) if cls_name.startswith("Interpolated") else dict(n_chans=len(info["ch_names"]))
    m = cls.from_pretrained(mid, n_outputs=2, n_times=win, sfreq=sfreq, **kw, **extra).to(DEV).eval()
    if lazy:
        with torch.no_grad():
            m(torch.zeros(1, len(info["ch_names"]), win, device=DEV))
    cap = {}
    m.final_layer.register_forward_pre_hook(lambda mod, a: cap.__setitem__("z", a[0].detach()))
    return m, cap


def embed(m, cap, X, win, bs=32):
    out = []
    for i in range(0, len(X), bs):
        b = resample(X[i:i + bs].astype(np.float64), win, axis=-1)     # (b, C, win)
        b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
        with torch.no_grad():
            m(torch.tensor(b, dtype=torch.float32, device=DEV))
        z = cap["z"]
        out.append((z.mean(1) if z.ndim == 3 else z).cpu().numpy().reshape(len(b), -1))
    return np.concatenate(out)


def bandpower(X, fs=100.0, bands=((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))):
    f, P = welch(X.astype(np.float64), fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    return np.concatenate([np.log(P[:, :, (f >= lo) & (f < hi)].mean(-1) + 1e-20) for lo, hi in bands], 1)


def reg_r(F, y):
    pred = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 4, 13))), F, y,
                             cv=KFold(5, shuffle=True, random_state=0))
    return pearsonr(y, pred)[0], r2_score(y, pred)


def clf_auc(F, y):
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), F, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, pr)


def main():
    X, age, sex, info = load_data()
    print(f"HBN ERP: {len(X)} subjects, {X.shape[1]}ch x {X.shape[2]}, age {age.min():.0f}-{age.max():.0f}, "
          f"sex {sex.mean():.2f}.  device={DEV}\n")
    rows = {}
    r, r2 = reg_r(bandpower(X), age); rows["baseline (bandpower)"] = (r, clf_auc(bandpower(X), sex))
    for name, (cls, mid, sf, win, extra, lazy) in ZOO.items():
        try:
            m, cap = build(cls, mid, sf, win, extra, lazy, info)
            F = embed(m, cap, X, win)
            r, _ = reg_r(F, age); rows[name] = (r, clf_auc(F, sex))
            print(f"[ok] {name}  (embed dim {F.shape[1]})")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {name}: {repr(e)[:100]}")
    print(f"\n  {'model':22s} {'brain-age r':>12s} {'sex AUC':>9s}")
    for k in sorted(rows, key=lambda k: -rows[k][0]):
        print(f"  {k:22s} {rows[k][0]:12.3f} {rows[k][1]:9.3f}")
    print("\n  Real HBN ERP (SurroundSupp evoked). FM embeddings vs band-power, ridge/logistic 5-fold CV.")


if __name__ == "__main__":
    main()
