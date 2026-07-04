"""Raincloud of per-seed AUC on the evolved `cfc_pac` scenario — the LLaMEA scenario that defeats
every EEG-FM. Reads the JSON accumulated by cfc_pac_seeds.py (venv + container + oracle-osl passes)
and draws a half-violin + jittered strip + IQR box per method, coloured by family, with the chance
line at 0.50. Raincloud (not a bar plot) so the full per-seed distribution is visible.

  python examples/plot_cfc_pac_raincloud.py results.json figures/cfc_pac_raincloud.png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

FAMILY = {  # method -> (family, colour)
    "band-power": ("spectral", "#8c8c8c"), "phase-conn": ("connectivity", "#b0a08c"),
    "DMD": ("system-ID", "#e2b13c"), "SINDy": ("system-ID", "#e07b39"),
    "DySCo": ("dynamic-FC", "#4c9f70"), "HMM": ("state-space", "#7bb6d6"),
    "osl-TDE-HMM": ("state-space", "#2f6f9f"), "CEBRA": ("latent", "#9b5fb0"),
    "BIOT": ("foundation model", "#c0413b"), "CBraMod": ("foundation model", "#c0413b"),
    "LUNA": ("foundation model", "#c0413b"), "REVE": ("foundation model", "#c0413b"),
    "LaBraM": ("foundation model", "#c0413b"),
}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "cfc_pac_seeds.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/cfc_pac_raincloud.png"
    with open(src) as fh:
        data = {k: np.asarray(v, float) for k, v in json.load(fh).items() if v}

    order = sorted(data, key=lambda k: np.median(data[k]))    # worst (bottom) -> best (top)
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(order) + 1.4))
    fig.patch.set_facecolor("white")

    for i, name in enumerate(order):
        v = data[name]
        colour = FAMILY.get(name, ("other", "#666"))[1]
        # half-violin (KDE) above the row
        if len(v) > 1 and v.std() > 1e-6:
            kde = gaussian_kde(v)
            xs = np.linspace(max(0.3, v.min() - 0.05), min(1.0, v.max() + 0.05), 200)
            dens = kde(xs); dens = 0.36 * dens / dens.max()
            ax.fill_between(xs, i + 0.06, i + 0.06 + dens, color=colour, alpha=0.55, lw=0)
        # jittered strip below the row
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.22
        ax.scatter(v, i - 0.18 + jit, s=18, color=colour, alpha=0.8, edgecolor="white", lw=0.4, zorder=3)
        # IQR box + median tick on the row
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i - 0.02, i - 0.02], color=colour, lw=6, alpha=0.85, solid_capstyle="round")
        ax.plot([med, med], [i - 0.11, i + 0.07], color="white", lw=1.6, zorder=4)
        ax.text(1.005, i, f"{med:.2f}", va="center", ha="left", fontsize=9,
                color=colour, fontweight="bold", transform=ax.get_yaxis_transform())

    ax.axvline(0.5, color="#333", ls="--", lw=1, alpha=0.7)
    ax.text(0.5, len(order) - 0.35, "chance", rotation=90, va="top", ha="right",
            fontsize=8, color="#333", alpha=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=10)
    ax.set_ylim(-0.7, len(order) - 0.2)
    ax.set_xlim(0.35, 1.06)
    ax.set_xlabel("cross-validated AUC on cfc_pac  (per seed; chance = 0.50)", fontsize=10)
    n_seed = max(len(v) for v in data.values())
    ax.set_title(f"cfc_pac — the LLaMEA-evolved scenario no EEG-FM can read\n"
                 f"per-seed AUC across {n_seed} scenario realizations", fontsize=12, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.35)

    fams = dict.fromkeys(FAMILY[k][0] for k in order if k in FAMILY)
    handles = [Patch(facecolor=next(c for m, (f2, c) in FAMILY.items() if f2 == f), label=f)
               for f in fams]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, title="family",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
