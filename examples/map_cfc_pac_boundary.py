"""Phase 1.1 (reply to the reviewer): map the boundary of the `cfc_pac` foundation-model blind
spot, rather than reporting it as a single evolved point.

`cfc_pac` (scenarios.cfc_pac) is one specific parameterisation: a sharp (exponent-4) 6->40 Hz
phase gate on a matched-power background (strength 0.6). This script generalises the SAME
mechanism into a parameterised family, `cfc_pac_param(n_per, seed, gate_exp, bg_strength, f_lo,
f_hi)`, and sweeps two axes:

  Grid A  gate_exp x bg_strength   (5x5) at the canonical 6/40 Hz pair
          -- how SHARP must the nonlinear coupling be, and how much background NOISE can it
          tolerate, before the blind spot appears/disappears?
  Grid B  (f_lo, f_hi) frequency pairs   at the canonical gate_exp=4, bg_strength=0.6
          -- does the blind spot generalise across bands, or is it specific to theta-gamma?

Each grid cell is scored with band-power (spectral baseline) and DMD (linear/Koopman baseline)
vs. the two methods that read the canonical scenario, SINDy (system-ID) and CEBRA (self-supervised
latent embedding) -- the same zoo as figures/cfc_pac_raincloud.png, one consistent LogReg-AUC
metric. Writes results/cfc_pac_boundary_grid.csv and results/cfc_pac_boundary_freq.csv; plot with
examples/plot_cfc_pac_boundary.py.

  PYTHONPATH=src python examples/map_cfc_pac_boundary.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from neurodynadojo.scenarios import _fixed_sensors, _project, _bg_field, _stack, sphere_points, T, FS
from neurodynadojo.algorithms.dynamics import sindy_features, dmd_features
from neurodynadojo.algorithms.latent import cebra_features
from neurodynadojo.probes import bandpower_embed

N_PER = int(os.environ.get("NDD_NPER", "30"))
N_SEEDS_FAST = int(os.environ.get("NDD_SEEDS_FAST", "5"))     # band-power / SINDy / DMD
N_SEEDS_CEBRA = int(os.environ.get("NDD_SEEDS_CEBRA", "3"))   # CEBRA (slower per-fit)
GATE_EXP = [0.5, 1.0, 2.0, 4.0, 8.0]
BG_STRENGTH = [0.2, 0.4, 0.6, 0.9, 1.3]
FREQ_PAIRS = [(4.0, 20.0), (6.0, 20.0), (6.0, 40.0), (8.0, 60.0), (10.0, 30.0)]   # (6,40) = canonical


def cfc_pac_param(n_per, seed, gate_exp=4.0, bg_strength=0.6, f_lo=6.0, f_hi=40.0):
    """The `cfc_pac` mechanism (matched-power cross-frequency phase-gating), parameterised.
    `gate_exp` controls how sharply the f_hi burst concentrates around the f_lo gating phase
    (nonlinearity of the coupling); `bg_strength` is the 1/f background's relative RMS (inverse
    SNR); `(f_lo, f_hi)` are the coupled bands. Same anti-leakage controls as the canonical
    scenario: high-pass the gated HF component, renormalise HF power, match channel variance."""
    rng = np.random.default_rng(seed); sens = _fixed_sensors(seed); t = np.arange(T) / FS
    src_lo = sphere_points(3, 60.0, np.random.default_rng(7000 + seed))
    src_hi = sphere_points(3, 60.0, np.random.default_rng(7500 + seed))
    f = np.fft.rfftfreq(T, 1 / FS); f[0] = f[1]
    hp_cut = min(f_lo * 2.5, f_hi * 0.6)                      # scales the canonical 20 Hz cutoff
    recs = []
    for lab in (0, 1):
        cphase = 0.0 if lab == 0 else np.pi
        for _ in range(n_per):
            ph0 = rng.uniform(0, 6.28)
            lo = np.sin(2 * np.pi * f_lo * t + ph0)
            gate = (0.5 * (1 + np.cos(2 * np.pi * f_lo * t + ph0 - cphase))) ** gate_exp
            hi = gate * np.sin(2 * np.pi * f_hi * t + rng.uniform(0, 6.28))
            Hi = np.fft.rfft(hi); Hi[f < hp_cut] = 0
            hi = np.fft.irfft(Hi, n=T); hi = hi / (hi.std() + 1e-9)
            lo = lo / (lo.std() + 1e-9)
            S_lo = lo[None, :] * (1 + 0.1 * rng.standard_normal((3, 1)))
            S_hi = hi[None, :] * (1 + 0.1 * rng.standard_normal((3, 1)))
            x = (_project(sens, src_lo, S_lo, 1.0) + _project(sens, src_hi, S_hi, 1.0) +
                 _bg_field(sens, T, bg_strength, rng))
            x = x / (x.std(1, keepdims=True) + 1e-9)
            recs.append((x, lab))
    return _stack(recs, rng)


def auc(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                           F, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pr))


def score_point(gate_exp, bg_strength, f_lo, f_hi):
    """Mean AUC per method over N_SEEDS_FAST (band-power/SINDy/DMD) and N_SEEDS_CEBRA (CEBRA)."""
    fast = {"band-power": [], "SINDy": [], "DMD": []}
    for s in range(N_SEEDS_FAST):
        X, y, _ = cfc_pac_param(N_PER, s, gate_exp, bg_strength, f_lo, f_hi)
        bp = np.stack([bandpower_embed(x, fs=FS) for x in X])
        fast["band-power"].append(auc(bp, y))
        fast["SINDy"].append(auc(sindy_features(X), y))
        fast["DMD"].append(auc(dmd_features(X), y))
    cebra_vals = []
    for s in range(N_SEEDS_CEBRA):
        X, y, _ = cfc_pac_param(N_PER, s, gate_exp, bg_strength, f_lo, f_hi)
        cebra_vals.append(auc(cebra_features(X, dim=8, iters=150), y))
    return {n: float(np.mean(v)) for n, v in fast.items()} | {"CEBRA": float(np.mean(cebra_vals))}


def main():
    print(f"Mapping the cfc_pac blind-spot boundary (n_per={N_PER}, "
          f"seeds fast={N_SEEDS_FAST}/CEBRA={N_SEEDS_CEBRA}).\n")

    print("=== Grid A: gate_exp x bg_strength (canonical 6/40 Hz) ===")
    rows_a = []
    for ge in GATE_EXP:
        for bg in BG_STRENGTH:
            r = score_point(ge, bg, 6.0, 40.0)
            rows_a.append({"gate_exp": ge, "bg_strength": bg, **r})
            print(f"  gate_exp={ge:>4.1f} bg={bg:>4.1f}  " +
                  " ".join(f"{k}={v:.2f}" for k, v in r.items()), flush=True)
    with open("results/cfc_pac_boundary_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a[0].keys())); w.writeheader(); w.writerows(rows_a)

    print("\n=== Grid B: frequency pairs (canonical gate_exp=4, bg=0.6) ===")
    rows_b = []
    for f_lo, f_hi in FREQ_PAIRS:
        r = score_point(4.0, 0.6, f_lo, f_hi)
        rows_b.append({"f_lo": f_lo, "f_hi": f_hi, **r})
        print(f"  f_lo={f_lo:>4.1f} f_hi={f_hi:>4.1f}  " +
              " ".join(f"{k}={v:.2f}" for k, v in r.items()), flush=True)
    with open("results/cfc_pac_boundary_freq.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_b[0].keys())); w.writeheader(); w.writerows(rows_b)

    print("\nwrote results/cfc_pac_boundary_grid.csv, results/cfc_pac_boundary_freq.csv")


if __name__ == "__main__":
    main()
