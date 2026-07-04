"""Method x scenario landscape as a heatmap — the diagnostic object the battery exists to produce.

Reads results/landscape_matrix.csv (decoding score per method per scenario; chance = 0.50) and
renders a heatmap diverging around chance, so winners (bright) and blind spots (pale/dark) are
visible at a glance. Rows are grouped and colour-tagged by method family; the `cfc_pac` column —
the LLaMEA-evolved scenario only SINDy and CEBRA read — stands out on the right.

  python examples/plot_landscape_heatmap.py results/landscape_matrix.csv figures/landscape_heatmap.png
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_COLOUR = {"spectral": "#8c8c8c", "connectivity": "#b0a08c", "system-ID": "#e0902f",
                 "dynamic-FC": "#4c9f70", "state-space": "#2f6f9f", "latent": "#9b5fb0",
                 "foundation-model": "#c0413b"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/landscape_matrix.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/landscape_heatmap.png"
    rows = list(csv.DictReader(open(src)))
    scen = [c for c in rows[0] if c not in ("method", "family", "metric", "n")]
    M = np.array([[float(r[s]) for s in scen] for r in rows])
    methods = [r["method"] for r in rows]
    fams = [r["family"] for r in rows]

    fig, ax = plt.subplots(figsize=(1.05 * len(scen) + 4.0, 0.52 * len(methods) + 1.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(M, cmap="magma", vmin=0.4, vmax=1.0, aspect="auto")

    for i in range(len(methods)):
        for j in range(len(scen)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v < 0.72 else "black",
                    fontweight="bold" if v >= 0.85 else "normal")

    ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    for tick, fam in zip(ax.get_yticklabels(), fams):
        tick.set_color(FAMILY_COLOUR.get(fam, "#333"))
        tick.set_fontweight("bold")
    # family separators
    for i in range(1, len(methods)):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    ax.axvline(len(scen) - 1.5, color="#00e5ff", lw=2.2, alpha=0.7)   # mark the cfc_pac column
    ax.set_title("Method × scenario landscape — decoding of the generative label (chance = 0.50)\n"
                 "every scenario crowns a different family; cfc_pac (evolved) is an FM blind spot",
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("decoding score (AUC / balanced-acc)", fontsize=9)
    ax.set_xlabel("scenario", fontsize=10)
    fig.text(0.01, 0.01, "rows coloured by family • bold = strong (≥0.85) • see docs/TECHNICAL_REPORT.md for metric notes",
             fontsize=7.5, color="#666")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
