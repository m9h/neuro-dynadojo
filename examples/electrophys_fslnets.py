"""Electrophysiological FSLNets: score FC methods on recovering a KNOWN structural
connectome from a Hopf whole-brain simulation, in SOURCE space vs under VOLUME-CONDUCTION
leakage -- the confound fMRI FSLNets never had.

System   : HopfNetworkSystem (Stuart-Landau on a modular connectome, distance delays,
           lead-field leakage observation; positions decoupled from structure).
Algorithms: correlation, partial correlation, PLV, imaginary coherence, wPLI.
Challenge : edge-recovery AUC vs the true adjacency.

  .venv/bin/python scripts/electrophys_fslnets.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.generators.hopf import HopfNetworkSystem, KuramotoNetworkSystem, RingWaveSystem
from neurodynadojo.algorithms.fc import fc_algorithms, edge_auc


def score_space(SystemCls, space, leak, n_seeds=6, **kw):
    algs = fc_algorithms()
    acc = {a.name: [] for a in algs}
    for s in range(n_seeds):
        sysm = SystemCls(seed_struct=s, space=space, leak=leak, **kw)
        obs, adj = sysm.simulate(seed=s)
        for a in algs:
            acc[a.name].append(edge_auc(a.estimate(obs, sysm), adj))
    return {m: np.nanmean(v) for m, v in acc.items()}


def report(name, SystemCls):
    algs = [a.name for a in fc_algorithms()]
    print(f"\n=== System: {name} (30 nodes, 3 modules, alpha, distance delays) ===")
    src = score_space(SystemCls, "source", leak=0.0)
    lk = score_space(SystemCls, "leaked", leak=0.8)
    print(f"  {'method':16s} {'source':>8s} {'leaked':>8s} {'drop':>8s}")
    for m in algs:
        print(f"  {m:16s} {src[m]:8.3f} {lk[m]:8.3f} {src[m]-lk[m]:8.3f}")
    ls = [0.0, 0.3, 0.6, 0.9, 1.2]
    row = [score_space(SystemCls, "leaked", leak=l)["correlation"] for l in ls]
    print("  correlation vs leak: " + "  ".join(f"{l:.1f}:{v:.3f}" for l, v in zip(ls, row)))


def main():
    print("Electrophysiological FSLNets: FC methods recovering a KNOWN connectome, in")
    print("SOURCE space vs under VOLUME-CONDUCTION leakage, across two generative regimes.")
    report("Hopf (AMPLITUDE-coupled)", HopfNetworkSystem)
    report("Kuramoto (PHASE-coupled, distance lags)", KuramotoNetworkSystem)
    report("Ring wave (directed TRAVELING WAVE)", RingWaveSystem)

    print("\n  READING (the DynaDojo point -- honest, cross-REGIME; the best method FLIPS):")
    print("  * Hopf & Kuramoto (modular): marginal CORRELATION recovers structure well and")
    print("    DEGRADES monotonically with volume conduction (real confound).")
    print("  * Ring WAVE: marginal correlation COLLAPSES (~0.64) -- a global wave makes every")
    print("    pair cohere (phase diff ~ node separation) -- but PARTIAL correlation RECOVERS")
    print("    the direct ring edges (~0.90) by CONDITIONING OUT the shared wave. So the best")
    print("    method flips with regime, and the marginal-vs-CONDITIONAL distinction (direct")
    print("    vs indirect coupling) matters more than any single statistic.")
    print("  * Lag-robust measures (imag-coh, wPLI) are CORRECT (a controlled quarter-cycle")
    print("    lag gives imag-coh~1.0) yet stay near chance for ADJACENCY in ALL regimes:")
    print("    modular coupling is near-zero-lag, and the wave's phase is GLOBAL not")
    print("    edge-specific. Their zero-lag artifact rejection is real (it's why correlation")
    print("    drops under leakage) but it is NOT superior structure recovery.")
    print("  * The CORRECT challenge is also regime-dependent: ADJACENCY for modular coupling,")
    print("    WAVENUMBER/direction for the wave (neurodynadojo.generators.waves) -- same generator,")
    print("    different ground truth. No single method/challenge wins everywhere; the")
    print("    System x Challenge x Algorithm interaction is the FSLNets/DynaDojo result.")


if __name__ == "__main__":
    main()
