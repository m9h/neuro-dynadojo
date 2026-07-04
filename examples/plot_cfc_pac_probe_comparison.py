"""Phase 1.2 (reply to the reviewer): does a non-linear probe recover cfc_pac for any method the
LINEAR probe called blind? Critique D is right that a linear probe only shows linear
decodability -- 'the FM does not represent X' should mean under ANY reasonable probe, not just a
linear one. This reads the three probe-type JSONs (linear/kernel/mlp; see cfc_pac_seeds.py's
NDD_PROBE) and renders a method x probe heatmap.

  python examples/plot_cfc_pac_probe_comparison.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBES = ["linear", "kernel", "mlp"]


def main(paths=("figures/cfc_pac_seeds.json", "results/cfc_pac_probe_kernel.json",
                "results/cfc_pac_probe_mlp.json"),
        out="figures/cfc_pac_probe_comparison.png"):
    data = [json.load(open(p)) for p in paths]
    methods = list(dict.fromkeys([m for d in data for m in d]))     # preserve first-seen order
    grid = np.full((len(methods), 3), np.nan)
    for j, d in enumerate(data):
        for i, m in enumerate(methods):
            if m in d:
                grid[i, j] = np.median(d[m])

    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(methods) + 1.6))
    fig.patch.set_facecolor("white")
    im = ax.imshow(grid, cmap="magma", vmin=0.35, vmax=1.0, aspect="auto")
    for i in range(len(methods)):
        for j in range(3):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#999")
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v < 0.72 else "black",
                    fontweight="bold" if v >= 0.8 else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(PROBES, fontsize=10)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=10)
    ax.set_title("cfc_pac under three probes (chance = 0.50)\n"
                 "does a non-linear probe recover it for any linear-blind method?",
                fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="median AUC (12 seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
