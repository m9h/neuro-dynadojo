"""Plot the cfc_pac blind-spot boundary maps produced by map_cfc_pac_boundary.py.

Two figures:
  1. Heatmaps of gate_exp x bg_strength for band-power, SINDy, CEBRA, and a "blind-spot margin"
     panel (best dynamics-method AUC minus best spectral-method AUC) -- where the margin is high
     (bright), the FM/spectral blind spot holds; where it fades to ~0, the mechanism has become
     either too weak (soft gate) or too buried (high background) to be a blind spot at all.
  2. A line plot of AUC vs. frequency pair, showing whether the blind spot generalises across bands.

  python examples/plot_cfc_pac_boundary.py
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    return list(csv.DictReader(open(path)))


def plot_grid(path="results/cfc_pac_boundary_grid.csv", out="figures/cfc_pac_boundary_grid.png"):
    rows = _read(path)
    ge = sorted({float(r["gate_exp"]) for r in rows})
    bg = sorted({float(r["bg_strength"]) for r in rows})
    grids = {m: np.zeros((len(ge), len(bg))) for m in ("band-power", "SINDy", "DMD", "CEBRA")}
    for r in rows:
        i, j = ge.index(float(r["gate_exp"])), bg.index(float(r["bg_strength"]))
        for m in grids:
            grids[m][i, j] = float(r[m])
    margin = np.maximum(grids["SINDy"], grids["CEBRA"]) - np.maximum(grids["band-power"], grids["DMD"])

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    fig.patch.set_facecolor("white")
    panels = [("band-power", grids["band-power"], "magma", 0.4, 1.0),
              ("SINDy", grids["SINDy"], "magma", 0.4, 1.0),
              ("CEBRA", grids["CEBRA"], "magma", 0.4, 1.0),
              ("blind-spot margin\n(max(SINDy,CEBRA) - max(band-power,DMD))", margin, "RdBu_r", -0.5, 0.5)]
    for ax, (title, grid, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(ge)):
            for j in range(len(bg)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.22 else "black")
        ax.set_xticks(range(len(bg))); ax.set_xticklabels([f"{b:g}" for b in bg])
        ax.set_yticks(range(len(ge))); ax.set_yticklabels([f"{g:g}" for g in ge])
        ax.set_xlabel("background strength (inverse SNR)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("gate sharpness (exponent)", fontsize=9)
    fig.suptitle("cfc_pac blind-spot boundary — gate sharpness x background strength (chance = 0.50)",
                fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def plot_freq(path="results/cfc_pac_boundary_freq.csv", out="figures/cfc_pac_boundary_freq.png"):
    rows = _read(path)
    labels = [f"{r['f_lo']}→{r['f_hi']} Hz" for r in rows]
    methods = ["band-power", "DMD", "SINDy", "CEBRA"]
    colours = {"band-power": "#8c8c8c", "DMD": "#e0902f", "SINDy": "#c0413b", "CEBRA": "#9b5fb0"}
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor("white")
    for m in methods:
        vals = [float(r[m]) for r in rows]
        ax.plot(x, vals, marker="o", label=m, color=colours[m], lw=2)
    ax.axhline(0.5, color="#333", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC (chance = 0.50)", fontsize=10)
    ax.set_title("Does the cfc_pac blind spot generalise across frequency bands?\n"
                 "(gate sharpness=4, background=0.6 held at canonical values)",
                fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    plot_grid()
    plot_freq()
