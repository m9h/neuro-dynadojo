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

# BENDR's EXACT pretraining montage: the 19 EEG channels (order + standard_1005 positions,
# metres) taken verbatim from braindecode.models.bendr (dn3 To1020.EEG_20_div). The 20th
# 'SCALE' relative-amplitude channel is appended at embed time (see probes.bendr_embed).
BENDR_CHS = [
    ("FP1", (-0.0294367, 0.0839171, -0.0069900)), ("FP2", (0.0298723, 0.0848959, -0.0070800)),
    ("F7", (-0.0702629, 0.0424743, -0.0114200)),  ("F3", (-0.0502438, 0.0531112, 0.0421920)),
    ("FZ", (0.0003122, 0.0585120, 0.0664620)),    ("F4", (0.0518362, 0.0543048, 0.0408140)),
    ("F8", (0.0730431, 0.0444217, -0.0120000)),   ("T7", (-0.0841611, -0.0160187, -0.0093460)),
    ("C3", (-0.0653581, -0.0116317, 0.0643580)),  ("CZ", (0.0004009, -0.0091670, 0.1002440)),
    ("C4", (0.0671179, -0.0109003, 0.0635800)),   ("T8", (0.0850799, -0.0150203, -0.0094900)),
    ("T5", (-0.0724343, -0.0734527, -0.0024870)), ("P3", (-0.0530073, -0.0787878, 0.0559400)),
    ("PZ", (0.0003247, -0.0811150, 0.0826150)),   ("P4", (0.0556667, -0.0785602, 0.0565610)),
    ("T6", (0.0730557, -0.0730683, -0.0025400)),  ("O1", (-0.0294134, -0.1124490, 0.0088390)),
    ("O2", (0.0298426, -0.1121560, 0.0088000)),
]


def resolve_montage(spec):
    """`spec`: a list of 10-20 channel names, or a preset key ("1020_19", "1020_20",
    "bendr" = BENDR's exact 19-EEG pretraining montage, "sci128" = SCI 128-channel montage,
    "sci256" = SCI 256-channel montage). Returns (ch_names, positions_mm)
    centred at the electrode centroid. Only the standard_1020 presets require MNE."""
    import numpy as np
    import os
    if spec == "bendr":
        names = [c for c, _ in BENDR_CHS]
        P = np.array([p for _, p in BENDR_CHS], float) * 1000.0      # m -> mm
        return names, P - P.mean(0)
    elif spec in ("sci128", "sci256"):
        import zipfile
        import scipy.io
        import io
        # NDD_SCI_HEADMODEL_ZIP overrides the path (matches NDD_CONNECTOME's convention for
        # external data files) -- the hardcoded default only exists on the machine this was
        # authored on and is not portable/reproducible elsewhere without an override.
        zip_path = os.environ.get(
            "NDD_SCI_HEADMODEL_ZIP",
            "/home/mhough/data/sci.utah.edu/~datasets/SCI_headmodel/EEG.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"SCI Head Model EEG.zip not found at {zip_path}. Set NDD_SCI_HEADMODEL_ZIP to "
                f"its location, or download it from sci.utah.edu's SCI_headmodel dataset.")
        
        n_ch = 128 if spec == "sci128" else 256
        mat_internal_path = (
            "EEG/128 channel/electrodes_scirun_128.mat"
            if spec == "sci128"
            else "EEG/256 channel/electrodes_scirun.mat"
        )
        
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(mat_internal_path) as f:
                mat_data = scipy.io.loadmat(io.BytesIO(f.read()))
                
        node = mat_data["Field1"][0, 0]["node"]
        names = [f"EEG{i+1:03d}" for i in range(n_ch)]
        P = np.asarray(node[:n_ch], float)
        return names, P - P.mean(0)
    import mne
    names = PRESETS[spec] if isinstance(spec, str) else list(spec)
    cp = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    missing = [n for n in names if n not in cp]
    if missing:
        raise ValueError(f"channels not in standard_1020: {missing}")
    P = np.array([cp[n] for n in names], float) * 1000.0     # m -> mm
    return names, P - P.mean(0)
