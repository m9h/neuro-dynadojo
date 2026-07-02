"""Real 10-20 electrode montages, so simulated sensor recordings carry genuine channel
names and positions — the in-distribution input a pretrained EEG foundation model expects.
Positions come from MNE's `standard_1020` (metres -> millimetres, centred at the electrode
centroid so they share the source coordinate frame). Requires the `[mne]` extra.
"""
from __future__ import annotations

# Classic 10-20 sets (all present in MNE standard_1020).
MONTAGE_1020_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                   "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
MONTAGE_1020_20 = MONTAGE_1020_19 + ["Oz"]

PRESETS = {"1020_19": MONTAGE_1020_19, "1020_20": MONTAGE_1020_20}


def resolve_montage(spec):
    """`spec`: a list of 10-20 channel names, or a preset key ("1020_19", "1020_20").
    Returns (ch_names, positions_mm) with positions centred at the electrode centroid."""
    import numpy as np
    import mne
    names = PRESETS[spec] if isinstance(spec, str) else list(spec)
    cp = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    missing = [n for n in names if n not in cp]
    if missing:
        raise ValueError(f"channels not in standard_1020: {missing}")
    P = np.array([cp[n] for n in names], float) * 1000.0     # m -> mm
    return names, P - P.mean(0)
