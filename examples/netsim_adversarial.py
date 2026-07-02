"""DynaDojo-v2 adversarial search over the netsim confound grid: automatically find the
confound COMBINATIONS that most break the methods (`hardest`) or make them most DISAGREE
(`disagreement`). Wires NetsimSystem into bench.adversarial_search. All four methods are
scored against the UNDIRECTED edge truth (directed scores symmetrised), so one challenge
ranks the whole panel.

  .venv/bin/python scripts/netsim_adversarial.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.generators.netsim import NetsimSystem
from neurodynadojo.algorithms.fc import correlation_fc, partialcorr_fc, edge_recovery_auc
from neurodynadojo.algorithms.directed import granger_bivariate, dmd_transition
from neurodynadojo.bench import adversarial_search


class _Alg:
    def __init__(self, fn, name, directed=False):
        self.fn, self.name, self.directed = fn, name, directed

    def estimate(self, obs, system):
        S = self.fn(obs, fs=getattr(system, "fs", 250.0), band=getattr(system, "band", (8, 12))) \
            if not self.directed else self.fn(obs)
        return np.maximum(S, S.T) if self.directed else S      # symmetrise directed for undirected truth


class _NS:
    """Adapter: NetsimSystem whose simulate() returns (obs, undirected truth) for the bench."""
    def __init__(self, **p):
        self.s = NetsimSystem(back=0.3, **p)
        self.fs, self.band = self.s.fs, self.s.band

    def simulate(self, seed=0):
        obs, _ = self.s.simulate(seed)
        return obs, self.s.undirected_truth()


ALGS = [_Alg(correlation_fc, "correlation"), _Alg(partialcorr_fc, "partial"),
        _Alg(lambda x: granger_bivariate(x, 4), "Granger", directed=True),
        _Alg(dmd_transition, "DMD", directed=True)]

GRID = {"leak": [0.0, 0.8], "snr": [float("inf"), 6.0], "shared": [0.0, 0.4]}


def show(title, res, n=4):
    print(f"\n  {title}")
    for params, means, obj in res[:n]:
        p = ", ".join(f"{k}={'inf' if v == float('inf') else v}" for k, v in params.items())
        sc = " ".join(f"{k[:4]}={v:.2f}" for k, v in means.items())
        print(f"    [obj {obj:+.3f}] {p:38s} | {sc}")


def main():
    print("Adversarial confound search (NetsimSystem x {corr,partial,Granger,DMD} vs undirected truth).")
    hardest = adversarial_search(_NS, ALGS, GRID, edge_recovery_auc, n_seeds=3, objective="hardest")
    disagree = adversarial_search(_NS, ALGS, GRID, edge_recovery_auc, n_seeds=3, objective="disagreement")
    show("HARDEST (min best-method AUC first) -- the confound cocktails that break everything:", hardest)
    show("MOST DISAGREEMENT (max cross-method spread) -- the informative rows:", disagree)
    print("\n  'Hardest' surfaces the combinations where even the best method is near chance;")
    print("  'disagreement' surfaces where the panel splits (e.g. shared input: partial holds,")
    print("  directed methods collapse). This is the auto-discovered version of the confound table.")


if __name__ == "__main__":
    main()
