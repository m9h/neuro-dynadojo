"""Critique B follow-up: does spatially-embedding the connectome (structural coupling collinear
with volume-conduction leakage, instead of orthogonal to it) make FC recovery harder, and does it
hurt leakage-vulnerable zero-lag correlation more than leakage-robust phase measures?

Netsim/DynaDojo-style generators draw the connectome independently of node position, so a true
structural edge and a leakage-induced spurious edge (short physical distance -> lead-field overlap
-> zero-lag correlation) are UNCORRELATED events. An independent review pointed out that real
cortex is not like this: wiring cost favours short-range connections, so structural coupling and
volume-conduction leakage are COLLINEAR -- a true edge is *more* likely to also be a leakage
artefact, which should make source separation harder than the orthogonal-by-construction default.

`wiring_length` (added in response to that review) lets us dial this in: 0 = position-independent
(netsim default), decreasing values -> increasingly short-range, spatially-embedded wiring. We
measure (1) structural-leakage collinearity directly (top-leakage-strength pairs vs true edges,
odds ratio) and (2) FC-recovery AUC for a leakage-vulnerable measure (zero-lag correlation) and two
leakage-robust measures (imaginary coherence, wPLI), under `space="leaked"`, across wiring_length
in {0 (random), 60, 30, 15} mm and several seeds.

  python examples/wiring_geometry_study.py
"""
import sys

sys.path.insert(0, "src")

import numpy as np

from neurodynadojo.generators.hopf import HopfNetworkSystem, structural_leakage_collinearity
from neurodynadojo.algorithms.fc import correlation_fc, imag_coherence_fc, wpli_fc, edge_auc

SEEDS = range(8)
# calibrated to the r_src=70mm sphere (mean pairwise distance ~92mm) so every setting keeps a
# usable edge count (~150 -> ~10 edges/30 nodes as wiring_length shrinks; see docstring below)
WIRING_LENGTHS = [0.0, 90.0, 45.0, 25.0]          # 0 = netsim-default (position-independent)
METHODS = [("correlation (leakage-vulnerable)", correlation_fc),
           ("imag-coherence (leakage-robust)", imag_coherence_fc),
           ("wPLI (leakage-robust)", wpli_fc)]


def sweep(leak):
    """Run the wiring_length sweep at a fixed leakage strength (0.0 = a no-leakage control that
    isolates whether an effect is about LEAKAGE-collinearity or just 'short edges are dynamically
    easier', e.g. shorter conduction delay -> stronger zero-lag synchrony regardless of leakage).
    The collinearity metric is only meaningful when leak>0 (M is trivially the identity otherwise)."""
    rows = []
    for wl in WIRING_LENGTHS:
        coll, aucs = [], {name: [] for name, _ in METHODS}
        for s in SEEDS:
            space = "leaked" if leak > 0 else "source"
            sysm = HopfNetworkSystem(seed_struct=s, wiring_length=wl, leak=leak, space=space)
            if leak > 0:
                coll.append(structural_leakage_collinearity(sysm.C, sysm.M))
            x, adj = sysm.simulate(seed=s)
            for name, fn in METHODS:
                try:
                    aucs[name].append(edge_auc(fn(x, fs=sysm.fs, band=sysm.band), adj))
                except Exception:
                    pass                                   # too few/degenerate edges some seeds
        rows.append((wl, np.nanmean(coll) if coll else None,
                    {n: np.nanmean(v) if v else float("nan") for n, v in aucs.items()}))
    return rows


def _print(rows, title):
    print(f"\n{title}")
    print(f"{'wiring_length':>14s} {'collinearity':>13s} " +
          " ".join(f"{name.split(' ')[0]:>12s}" for name, _ in METHODS))
    for wl, coll, aucs in rows:
        coll_s = f"{coll:>13.2f}" if coll is not None else f"{'n/a':>13s}"
        print(f"{wl:>14.0f} {coll_s} " + " ".join(f"{aucs[n]:>12.3f}" for n, _ in METHODS))


def main():
    print("Critique B: FC recovery as structural coupling and volume-conduction leakage go from "
          "ORTHOGONAL (wiring_length=0, netsim default) to COLLINEAR (small wiring_length).")
    _print(sweep(leak=0.8), "WITH volume-conduction leakage (space='leaked', leak=0.8):")
    _print(sweep(leak=0.0), "NO-LEAKAGE CONTROL (space='source', leak=0.0) -- isolates a pure "
                            "wiring-geometry effect from a leakage-collinearity effect:")
    print("\nCollinearity ~1 = structure/leakage independent (netsim default); >1 = a structural "
          "edge is also disproportionately likely to be a leakage artefact (real-cortex-like).\n"
          "Read the two tables together: if correlation AUC moves with wiring_length similarly in "
          "BOTH tables, the effect is intrinsic to short-range wiring (delay/synchrony), not "
          "leakage-collinearity specifically; if it moves only (or much more) in the leaked table, "
          "that is the leakage-confound effect Critique B raised.")


if __name__ == "__main__":
    main()
