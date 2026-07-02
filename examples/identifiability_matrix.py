"""Which GROUND TRUTH is identifiable in which REGIME? One matrix tying the FC/adjacency
bench to the wavenumber bench: the same generator family, three candidate ground truths
(undirected adjacency, directed flow, spatial wavenumber), scored by the best method in
each cell, in source space and under volume-conduction leakage.

  .venv/bin/python scripts/identifiability_matrix.py   # prints table + writes JSON
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.generators.hopf import HopfNetworkSystem, KuramotoNetworkSystem, RingWaveSystem
from neurodynadojo.algorithms.fc import correlation_fc, partialcorr_fc, edge_auc
from neurodynadojo.algorithms.directed import granger_bivariate, dmd_transition, directed_edge_auc
from neurodynadojo.generators.waves import dominant_wavenumber
from scipy.signal import butter, filtfilt, hilbert

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "out", "identifiability_matrix.json")
NSEED = 4
LEAK_SNR = 6.0        # realistic sensor operating point: leak + 6 dB pink+white noise
SNR_AXIS = [float("inf"), 10.0, 6.0, 3.0, 0.0]


def _wave_dir_adj(n, m):
    A = np.zeros((n, n))
    for i in range(n):
        for d in range(1, m + 1):
            A[i, (i - d) % n] = 1.0
    return A


def _wavenumber_consistency(x, fs=250.0, band=(8.0, 12.0)):
    """Concentration of the per-timepoint dominant wavenumber along ring order (1.0 = a
    single stable wavenumber; ~0 = no consistent spatial wave)."""
    b, a = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    ph = np.angle(hilbert(filtfilt(b, a, x, axis=1), axis=1))
    ks = np.array([dominant_wavenumber(ph[:, t]) for t in range(0, ph.shape[1], 3)])
    return float(np.abs(np.mean(np.exp(1j * 2 * np.pi * ks / x.shape[0]))))  # circular concentration


def undirected_adjacency(SystemCls, space, leak, snr=float("inf")):
    v = []
    for s in range(NSEED):
        sysm = SystemCls(seed_struct=s, space=space, leak=leak, snr=snr)
        x, adj = sysm.simulate(seed=s)
        v.append(max(edge_auc(correlation_fc(x), adj), edge_auc(partialcorr_fc(x), adj)))
    return float(np.nanmean(v))


def directed_flow(space, leak, snr=float("inf")):     # only defined for the directed wave
    g, d = [], []
    for s in range(NSEED):
        sysm = RingWaveSystem(seed_struct=s, space=space, leak=leak, snr=snr)
        x, _ = sysm.simulate(seed=s)
        DA = _wave_dir_adj(sysm.n, sysm.m)
        g.append(directed_edge_auc(granger_bivariate(x, 4), DA))
        d.append(directed_edge_auc(dmd_transition(x, 8), DA))
    return float(np.nanmean(g)), float(np.nanmean(d))


def wavenumber(space, leak, snr=float("inf")):        # only meaningful for the ring wave
    v = [_wavenumber_consistency(RingWaveSystem(seed_struct=s, space=space, leak=leak, snr=snr)
                                 .simulate(seed=s)[0]) for s in range(NSEED)]
    return float(np.nanmean(v))


def main():
    # cells: "source" = clean source (ceiling); "leaked" = leak 0.8 + LEAK_SNR dB pink noise (realistic)
    R = {}
    for name, cls in [("Hopf (amplitude)", HopfNetworkSystem),
                      ("Kuramoto (phase)", KuramotoNetworkSystem),
                      ("Ring wave (directed)", RingWaveSystem)]:
        R[name] = {"undirected_adjacency": {
            "source": undirected_adjacency(cls, "source", 0.0),
            "leaked": undirected_adjacency(cls, "leaked", 0.8, LEAK_SNR)}}
    for sp, lk, sn, key in [("source", 0.0, float("inf"), "source"),
                            ("leaked", 0.8, LEAK_SNR, "leaked")]:
        gs, ds = directed_flow(sp, lk, sn)
        R["Ring wave (directed)"].setdefault("directed_flow_granger", {})[key] = gs
        R["Ring wave (directed)"].setdefault("directed_flow_dmd", {})[key] = ds
        R["Ring wave (directed)"].setdefault("wavenumber", {})[key] = wavenumber(sp, lk, sn)

    # SNR-robustness strip (leaked, sweeping sensor SNR) for the representative cells
    strip = {"snr": ["inf" if not np.isfinite(s) else s for s in SNR_AXIS],
             "Hopf undir (corr)": [], "Wave dir (DMD)": [], "Wave wavenumber": []}
    for sn in SNR_AXIS:
        strip["Hopf undir (corr)"].append(undirected_adjacency(HopfNetworkSystem, "leaked", 0.8, sn))
        g, d = directed_flow("leaked", 0.8, sn)
        strip["Wave dir (DMD)"].append(d)
        strip["Wave wavenumber"].append(wavenumber("leaked", 0.8, sn))
    R["_snr_strip"] = strip

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2)
    print("REGIME x CHALLENGE identifiability (source / leaked; best method per cell):\n")
    print(f"  {'regime':22s} {'undir-adj':>18s} {'dir-flow (G/DMD)':>22s} {'wavenumber':>16s}")
    for name in R:
        if name.startswith("_"):
            continue
        ua = R[name]["undirected_adjacency"]
        df = R[name].get("directed_flow_dmd")
        dg = R[name].get("directed_flow_granger")
        wv = R[name].get("wavenumber")
        ua_s = f"{ua['source']:.2f}/{ua['leaked']:.2f}"
        df_s = (f"{dg['source']:.2f},{df['source']:.2f}/{dg['leaked']:.2f},{df['leaked']:.2f}"
                if df else "n/a")
        wv_s = f"{wv['source']:.2f}/{wv['leaked']:.2f}" if wv else "n/a"
        print(f"  {name:22s} {ua_s:>18s} {df_s:>22s} {wv_s:>16s}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
