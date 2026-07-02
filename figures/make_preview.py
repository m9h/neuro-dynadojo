"""Render figures/preview.png for the README — the netsim confound battery (which method
survives which confound) and the SNR scaling law. Data are the benchmark outputs; regenerate
with the example scripts. No bar charts (a heatmap + line plot).

  python figures/make_preview.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["correlation", "partial", "Granger", "DMD"]
ROWS = [
    ("baseline",              [.99, 1.00, .84, .89]),
    ("+ volume conduction",   [.85, .81, .70, .76]),
    ("+ sensor noise 6 dB",   [.79, .64, .69, .59]),
    ("+ shared input",        [1.00, 1.00, .44, .48]),
    ("+ non-stationary",      [.99, 1.00, .84, .89]),
    ("+ gain/latency jitter", [.89, .81, .89, .84]),
    ("+ short session",       [.98, .96, .84, .88]),
    ("realistic (all)",       [.51, .52, .50, .51]),
]
SNR_X = ["clean", "10", "6", "3", "0"]
SNR_PARTIAL = [.92, .78, .72, .69, .66]
SNR_DMD = [.87, .73, .64, .62, .61]

fig, (axh, axs) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

M = np.array([r[1] for r in ROWS])
im = axh.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
axh.set_xticks(range(len(METHODS))); axh.set_xticklabels(METHODS, rotation=20, ha="right")
axh.set_yticks(range(len(ROWS))); axh.set_yticklabels([r[0] for r in ROWS])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        axh.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                 color="white" if M[i, j] < 0.8 else "black", fontsize=8)
axh.set_title("Netsim confound battery — edge-recovery AUC", fontsize=11)
fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04, label="AUC")

x = range(len(SNR_X))
axs.plot(x, SNR_PARTIAL, "-o", color="#0f8a8a", lw=2.2, label="partial corr (undirected)")
axs.plot(x, SNR_DMD, "-o", color="#bd5540", lw=2.2, label="DMD (directed)")
axs.axhline(0.5, ls="--", color="#999", lw=1, label="chance")
axs.set_xticks(list(x)); axs.set_xticklabels(SNR_X)
axs.set_xlabel("sensor SNR (dB)"); axs.set_ylabel("edge-recovery AUC")
axs.set_ylim(0.45, 1.0); axs.set_title("Scaling with sensor noise", fontsize=11)
axs.legend(fontsize=8, frameon=False)
axs.spines[["top", "right"]].set_visible(False)

fig.suptitle("neuro-dynadojo — no method wins across confounds", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("figures/preview.png", dpi=120, bbox_inches="tight")
print("wrote figures/preview.png")
