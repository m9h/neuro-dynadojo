"""Per-seed AUC distributions on the evolved `cfc_pac` scenario, for a raincloud plot.

A raincloud needs a DISTRIBUTION per method, so we regenerate `cfc_pac` across many seeds and score
every method (one consistent metric per run: 5-fold AUC under a SELECTABLE probe, chance .50) once
per seed. Three passes accumulate into one JSON (`NDD_JSON`): the venv pass (classical / system-ID /
latent), the container pass (FM zoo, `NDD_FMS=1`), and the oracle-osl pass (`NDD_OSL=1`). Then
plot_cfc_pac_raincloud.py draws it.

`NDD_PROBE` selects the probe (default `linear`, i.e. LogReg -- the platform default everywhere
else): `kernel` (RBF-SVM) or `mlp` (small 2-layer net) test whether a method/FM that fails under a
LINEAR probe is nonetheless NON-LINEARLY decodable -- the review's Critique D (a linear probe shows
linear decodability only). Same StandardScaler+StratifiedKFold protocol for all three probes
(`neurodynadojo.probes.cv_auc`), so results differ only in probe capacity.

  # venv:       classical + sysid + CEBRA
  PYTHONPATH=src NDD_JSON=out.json python examples/cfc_pac_seeds.py
  PYTHONPATH=src NDD_PROBE=kernel NDD_JSON=out_kernel.json python examples/cfc_pac_seeds.py
  # container:  FMs   (via run_leaderboard_container.sh -> this script with NDD_FMS=1)
  # oracle-osl: TDE-HMM (via run_osl_container.sh -> this script with NDD_OSL=1)
"""
import json
import os
import sys

sys.path.insert(0, "/ndd/src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.scenarios import SCENARIOS

SEEDS = int(os.environ.get("NDD_SEEDS", "12"))
NPER = int(os.environ.get("NDD_NPER", "40"))
JSON = os.environ["NDD_JSON"]
PROBE = os.environ.get("NDD_PROBE", "linear")


def auc(F, y):
    from neurodynadojo.probes import cv_auc
    return cv_auc(F, y, probe=PROBE)


def classical_methods():
    from neurodynadojo.probes import bandpower_embed
    m = {"band-power": lambda X: np.stack([bandpower_embed(x, fs=250.0) for x in X])}
    try:
        from neurodynadojo.algorithms.dynamics import DYNAMICS
        m.update(DYNAMICS)                                   # SINDy, DMD, DySCo, HMM (lazy imports)
    except Exception:
        pass
    try:
        from neurodynadojo.algorithms.latent import cebra_features
        m["CEBRA"] = lambda X: cebra_features(X, dim=8, iters=200)
    except Exception:
        pass
    return m


def fm_methods():
    """FM embedders (container only): each returns an (N,C,T)->(N,D) closure."""
    import torch
    from scenario_benchmark import BD, build_bd, embed_bd, embed_adp, info32
    torch.backends.cudnn.enabled = False
    info = info32()
    fns = {}
    for name, (cls, mid, sf, win, extra, lazy) in BD.items():
        m, cap = build_bd(cls, mid, sf, win, extra, lazy, info)
        fns[name] = (lambda mm, cc, s, w: (lambda X: embed_bd(mm, cc, X, s, w)))(m, cap, sf, win)
    from emeg_fm.eeg_fm import REVEAdapter, LaBraMAdapter, REVE_BASE_ID, LABRAM_DEFAULT_ID
    for name, Acls, hf, sf, win in [("REVE", REVEAdapter, REVE_BASE_ID, 200.0, None),
                                    ("LaBraM", LaBraMAdapter, LABRAM_DEFAULT_ID, 200.0, 3000)]:
        ad = Acls(); loaded = ad.load_model(hf)
        fns[name] = (lambda a, l, s, w: (lambda X: embed_adp(a, l, X, s, w)))(ad, loaded, sf, win)
    return fns


def osl_methods():
    from neurodynadojo.algorithms.osl_dynamics import osl_hmm_features
    return {"osl-TDE-HMM": osl_hmm_features}


def main():
    if os.environ.get("NDD_FMS"):
        methods = fm_methods()
    elif os.environ.get("NDD_OSL"):
        methods = osl_methods()
    else:
        methods = classical_methods()
    print(f"cfc_pac seed sweep: {list(methods)} x {SEEDS} seeds (n_per={NPER})", flush=True)

    results = {}
    if os.path.exists(JSON):
        with open(JSON) as fh:
            results = json.load(fh)
    for name, fn in methods.items():
        vals = []
        for s in range(SEEDS):
            X, y, _ = SCENARIOS["cfc_pac"](NPER, s)
            try:
                vals.append(auc(fn(X), y))
            except Exception as e:
                print(f"  {name} seed {s} FAIL {repr(e)[:60]}", flush=True)
        results[name] = vals
        print(f"  {name:14s} median={np.median(vals):.3f}  ({len(vals)} seeds)", flush=True)
        with open(JSON, "w") as fh:                          # write incrementally
            json.dump(results, fh, indent=0)
    print(f"wrote {JSON}", flush=True)


if __name__ == "__main__":
    main()
