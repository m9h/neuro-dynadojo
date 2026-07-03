"""Single-trial SurroundSupp CONDITION decoding on real HBN EEG — the FM-favorable regime.
Decode stimulus condition (event code '4' vs '8') from single trials, WITHIN each subject
(identity-free: no subject-identity shortcut), FMs vs band-power. Reads the cleaned .mat
(111-ch GSN-HydroCel @500 Hz), extracts 0.6 s epochs, embeds each trial with the zoo
(emeg-fm recipe), 5-fold logistic AUC per subject, averaged. Runs in emeg-fm's container.

  bash examples/run_leaderboard_container.sh /ndd/examples/hbn_surroundsupp_decode.py
"""
import glob, os
import numpy as np, torch, mne
import braindecode.models as bm
from scipy.io import loadmat
from scipy.signal import resample, welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

mne.set_log_level("error")
# default CPU (the shared GB10 is often contended by sibling agents -> exit 144); NDD_DEV=cuda to override
DEV = os.environ.get("NDD_DEV", "cpu")
if DEV == "cuda":
    torch.backends.cudnn.enabled = False
SRC, CACHE = "/data/datasets/hbn-prep-mat", "/data/datasets/hbn-prep-evoked"
SR, PRE, POST, CODES = 500, 50, 250, ("4", "8")
N_SUBJ, MIN_TR = int(os.environ.get("NDD_NSUBJ", 40)), 40                                   # subjects; min trials (both conds) per subject
ZOO = {} if os.environ.get("NDD_SKIP_ZOO") else {
    "BIOT":    ("InterpolatedBIOT", "braindecode/biot-pretrained-six-datasets-18chs", 200., 1000, {}, False),
    "CBraMod": ("CBraMod",          "braindecode/cbramod-pretrained", 200., 1000, {}, True),
    "LUNA":    ("LUNA",             "PulpBio/LUNA", 200., 1000, {"filename": "LUNA_base.safetensors"}, True),
}


def subject_trials(sub_dir):
    X, y = [], []
    for blk in ("SurroundSupp_Block1", "SurroundSupp_Block2"):
        f = os.path.join(sub_dir, blk + ".mat")
        if not os.path.exists(f):
            continue
        m = loadmat(f, squeeze_me=True, struct_as_record=False)["result"]
        d = np.asarray(m.data, float)
        for e in np.atleast_1d(m.event):
            s, t = int(float(e.sample)), str(e.type).strip()
            if t in CODES and s - PRE >= 0 and s + POST <= d.shape[1]:
                seg = d[:, s - PRE:s + POST]
                X.append((seg - seg[:, :PRE].mean(1, keepdims=True)).astype(np.float32))
                y.append(CODES.index(t))
    return (np.array(X), np.array(y)) if X else (np.empty((0, 111, PRE + POST)), np.empty(0))


def build(cls_name, mid, sfreq, win, extra, lazy, info):
    cls = getattr(bm, cls_name)
    kw = dict(chs_info=info["chs"]) if cls_name.startswith("Interpolated") else dict(n_chans=111)
    m = cls.from_pretrained(mid, n_outputs=2, n_times=win, sfreq=sfreq, **kw, **extra).to(DEV).eval()
    if lazy:
        with torch.no_grad():
            m(torch.zeros(1, 111, win, device=DEV))
    cap = {}
    m.final_layer.register_forward_pre_hook(lambda mod, a: cap.__setitem__("z", a[0].detach()))
    return m, cap


def embed(m, cap, X, win, bs=256):
    out = []
    for i in range(0, len(X), bs):
        b = resample(X[i:i + bs].astype(np.float64), win, axis=-1)
        b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
        with torch.no_grad():
            m(torch.tensor(b, dtype=torch.float32, device=DEV))
        z = cap["z"]
        out.append((z.mean(1) if z.ndim == 3 else z).cpu().numpy().reshape(len(b), -1))
    return np.concatenate(out)


def adapter_embed(adapter, loaded, X, ch_names, sfreq_out=200.0, bs=256):
    """Embed HBN trials via an emeg-fm hand adapter (REVE, montage-flexible)."""
    b = resample(X.astype(np.float64), int(round(X.shape[-1] * sfreq_out / SR)), axis=-1)
    b = np.clip((b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-8), -15, 15)
    out = []
    for i in range(0, len(b), bs):
        out.append(np.asarray(adapter.extract_features(
            loaded, {"eeg": b[i:i + bs], "ch_names": list(ch_names),
                     "electrode_names": list(ch_names)})).reshape(len(b[i:i + bs]), -1))
    return np.concatenate(out)


def bandpower(X, fs=500., bands=((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))):
    f, P = welch(X.astype(np.float64), fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    return np.concatenate([np.log(P[:, :, (f >= lo) & (f < hi)].mean(-1) + 1e-20) for lo, hi in bands], 1)


def within_subj_auc(F, y, sid):
    aucs = []
    for s in np.unique(sid):
        m = sid == s
        if y[m].sum() < 8 or (1 - y[m]).sum() < 8:
            continue
        pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                               F[m], y[m], cv=StratifiedKFold(5, shuffle=True, random_state=0),
                               method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y[m], pr))
    return float(np.mean(aucs)), len(aucs)


def main():
    names = list(np.load(f"{CACHE}/ch_names.npy"))
    info = mne.create_info(names, 100., "eeg"); info.set_montage("GSN-HydroCel-129", on_missing="ignore", match_case=False)
    subs = sorted(glob.glob(f"{SRC}/*/SurroundSupp_Block1.mat"))
    Xs, ys, sids = [], [], []
    for i, sf in enumerate(subs):
        if len(sids) >= N_SUBJ:
            break
        X, y = subject_trials(os.path.dirname(sf))
        if len(y) >= MIN_TR and y.sum() >= 15 and (1 - y).sum() >= 15:
            Xs.append(X); ys.append(y); sids.append(np.full(len(y), len(sids)))
    X = np.concatenate(Xs); y = np.concatenate(ys); sid = np.concatenate(sids)
    print(f"Single-trial SurroundSupp 4-vs-8: {len(np.unique(sid))} subjects, {len(y)} trials "
          f"({y.mean():.2f} balance).  device={DEV}\n")
    rows = {}
    a, n = within_subj_auc(bandpower(X), y, sid); rows["baseline (bandpower)"] = a
    for name, (cls, mid, sf, win, extra, lazy) in ZOO.items():
        try:
            m, cap = build(cls, mid, sf, win, extra, lazy, info)
            a, n = within_subj_auc(embed(m, cap, X, win), y, sid); rows[name] = a
            print(f"[ok] {name}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {name}: {repr(e)[:100]}")
    try:                                                     # REVE (montage-flexible, Atlas's top cognitive FM)
        from emeg_fm.eeg_fm import REVEAdapter, REVE_BASE_ID
        ad = REVEAdapter(); loaded = ad.load_model(REVE_BASE_ID)
        a, n = within_subj_auc(adapter_embed(ad, loaded, X, names), y, sid); rows["REVE"] = a
        print("[ok] REVE")
    except Exception as e:
        print(f"[skip] REVE: {repr(e)[:120]}")
    print(f"\n  {'model':22s} {'within-subj cond-decode AUC':>28s}")
    for k in sorted(rows, key=lambda k: -rows[k]):
        print(f"  {k:22s} {rows[k]:28.3f}")
    print("\n  Identity-free single-trial condition decoding (chance 0.50). FM embeddings vs band-power.")


if __name__ == "__main__":
    main()
