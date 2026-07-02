"""Netsim-parity confound table (Smith et al. 2011, ported to electrophysiology).

A directed modular Hopf network is recovered under each confound in turn -- volume
conduction, sensor noise, shared/global input, non-stationarity, per-node gain+latency
jitter (the HRF-variability analog), short session -- and a combined 'realistic' row.
Columns: undirected edge AUC (correlation, partial corr) and DIRECTED edge AUC (Granger,
DMD). This is the electrophysiological analog of Smith's simulation table.

  .venv/bin/python scripts/netsim_table.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.generators.netsim import NetsimSystem
from neurodynadojo.algorithms.fc import correlation_fc, partialcorr_fc, edge_auc
from neurodynadojo.algorithms.directed import granger_bivariate, dmd_transition, directed_edge_auc

NSEED = 4
ROWS = [
    ("baseline (clean source)",       {}),
    ("+ volume conduction",           dict(leak=0.8)),
    ("+ sensor noise (6 dB pink)",    dict(leak=0.8, snr=6.0)),
    ("+ shared global input",         dict(shared=0.4)),
    ("+ non-stationary coupling",     dict(nonstat=0.5)),
    ("+ gain/latency jitter (HRF)",   dict(jitter_gain=0.5, jitter_lat=3)),
    ("+ short session (1.5 s)",       dict(T=1500.0)),
    ("realistic (all sensor-side)",   dict(leak=0.8, snr=6.0, shared=0.3,
                                          jitter_gain=0.4, jitter_lat=2)),
]


def row_scores(kw):
    corr, part, gr, dm = [], [], [], []
    for s in range(NSEED):
        sysm = NetsimSystem(seed_struct=s, back=0.3, **kw)
        x, _ = sysm.simulate(seed=s)
        UT, DT = sysm.undirected_truth(), sysm.directed_truth()
        corr.append(edge_auc(correlation_fc(x), UT))
        part.append(edge_auc(partialcorr_fc(x), UT))
        gr.append(directed_edge_auc(granger_bivariate(x, 4), DT))
        dm.append(directed_edge_auc(dmd_transition(x, 8), DT))
    return (np.nanmean(corr), np.nanmean(part), np.nanmean(gr), np.nanmean(dm))


def main():
    print("Netsim-parity confound table -- directed modular Hopf network, edge-recovery AUC.")
    print("30 nodes, 4 seeds, alpha band. Undirected vs undirected truth; directed vs forward edges.\n")
    print(f"  {'confound':30s} {'corr':>6s} {'partial':>8s} | {'Granger':>8s} {'DMD':>6s}")
    print("  " + "-" * 66)
    for label, kw in ROWS:
        c, p, g, d = row_scores(kw)
        print(f"  {label:30s} {c:6.2f} {p:8.2f} | {g:8.2f} {d:6.2f}")
    print("\n  READING:")
    print("  * Undirected recovery (partial corr) is robust to shared input & non-stationarity")
    print("    but degrades under volume conduction + sensor noise -- the mixing confounds.")
    print("  * DIRECTED recovery is fragile in the netsim-classic way: a SHARED GLOBAL INPUT")
    print("    collapses Granger to chance (common-input confound), and leakage/noise hit both")
    print("    Granger (zero-lag-confounded) and DMD. Directionality needs clean, unmixed data.")
    print("  * gain/latency jitter (the HRF-variability analog) mainly costs correlation; the")
    print("    'realistic' row is the honest operating point where every confound stacks.")


if __name__ == "__main__":
    main()
