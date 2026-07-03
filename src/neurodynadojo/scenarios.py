"""HBN-grounded simulation SCENARIOS — the netsim battery for the EEG-FM era.

As HCP's connectome grounded FSLNets/netsim, real HBN statistics ground these scenarios:
the 1/f aperiodic exponent (~-1.13, measured on HBN), oscillatory peak bands, and evoked
waveform latency (~0.28 s) come from the HBN cohort. Each scenario is a labeled binary
decoding task in sensor space (fixed montage) with a realistic 1/f background, and each is
engineered so a DIFFERENT method family wins — the whole point of a benchmark battery:

  spectral      resting-like: 1/f + an oscillatory peak whose BAND is the label      -> band-power
  evoked        SurroundSupp-like: two ERP waveforms (same power spectrum, different
                SHAPE/latency) on 1/f; the label is which                            -> foundation models
  wave          a directed traveling wave; the label is its DIRECTION                -> phase / directed
  naturalistic  a slow latent driving cross-frequency multiband coupling             -> foundation models
  burst         intermittent gamma bursts; the label is the burst RATE               -> band-power / FMs

Each `scenario(n_per, seed)` returns (X (2*n_per, n_ch, T), y (2*n_per,), ch_names). The
sensor montage is FIXED across recordings (channel identity is stable); only the sources and
noise vary per trial.
"""
from __future__ import annotations

import numpy as np

from .generators.hopf import sphere_points, leadfield_radial, _pink

CH32 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3",
        "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3",
        "PO4", "O1", "Oz", "O2", "AF3", "AF4"]
N_CH, T, FS, BG_EXP = 32, 1000, 250.0, 1.13    # 4 s @ 250 Hz; 1/f exponent from HBN


def _fixed_sensors(seed):
    """One stable electrode montage per scenario (channel identity constant across trials)."""
    return sphere_points(N_CH, 90.0, np.random.default_rng(1000 + seed))


def _bg_field(sens, n_t, strength, rng, n_src=40):
    """Spatially-correlated 1/f background at the HBN exponent on the FIXED montage `sens`."""
    L = leadfield_radial(sphere_points(n_src, 60.0, rng), sens)
    p = _pink((n_src, n_t), rng)
    p = np.fft.irfft(np.fft.rfft(p, axis=1) *
                     (np.fft.rfftfreq(n_t) + 1e-6)[None, :] ** (-(BG_EXP - 1) / 2), n=n_t, axis=1)
    bg = L @ p
    return strength * bg / (bg.std() + 1e-12)


def _project(sens, src_pos, S, strength=1.0):
    """Project source activity to sensors and normalise to `strength` x unit RMS (the lead
    field's absolute scale is tiny, so signal must be scaled explicitly vs the background)."""
    x = leadfield_radial(src_pos, sens) @ S
    return strength * x / (x.std() + 1e-12)


def _stack(pos_neg, rng):
    X = np.stack([r for r, _ in pos_neg]); y = np.array([l for _, l in pos_neg])
    idx = rng.permutation(len(y))
    return X[idx], y[idx], CH32


def spectral(n_per, seed):
    """1/f + an oscillatory peak in a LOW (6 Hz) vs HIGH (11 Hz) band -> band-power wins."""
    rng = np.random.default_rng(seed); sens = _fixed_sensors(seed); t = np.arange(T) / FS
    src = sphere_points(3, 60.0, np.random.default_rng(2000 + seed))              # FIXED topography
    recs = []
    for lab in (0, 1):
        f0 = 6.0 if lab == 0 else 11.0
        for _ in range(n_per):
            osc = np.array([np.sin(2 * np.pi * (f0 + 0.3 * rng.standard_normal()) * t + rng.uniform(0, 6.28))
                            for _ in range(3)])
            recs.append((_project(sens, src, osc, 1.0) + _bg_field(sens, T, 0.5, rng), lab))
    return _stack(recs, rng)


def evoked(n_per, seed):
    """A 10 Hz Gabor burst at the HBN latency (~0.28 s) with PHASE 0 vs pi/2 — the SAME power
    spectrum, so band-power is blind; only the WAVEFORM/phase differs -> FMs / raw-waveform."""
    rng = np.random.default_rng(seed); sens = _fixed_sensors(seed); t = np.arange(T) / FS
    src = sphere_points(3, 60.0, np.random.default_rng(2000 + seed))              # fixed topography
    env = np.exp(-((t - 0.28) ** 2) / (2 * 0.05 ** 2))
    recs = []
    for lab in (0, 1):
        phase = 0.0 if lab == 0 else np.pi / 2
        for _ in range(n_per):
            amp = 1.0 + 0.3 * rng.standard_normal(3)
            w = env * np.sin(2 * np.pi * 10 * t + phase)
            recs.append((_project(sens, src, amp[:, None] * w[None, :], 1.2) +
                         _bg_field(sens, T, 0.4, rng), lab))
    return _stack(recs, rng)


def wave(n_per, seed):
    """A directed traveling wave; label is its DIRECTION (forward vs backward phase ramp)
    -> phase/directed methods; a pure power spectrum is blind to direction."""
    rng = np.random.default_rng(seed); sens = _fixed_sensors(seed); t = np.arange(T) / FS
    src = sphere_points(8, 60.0, np.random.default_rng(4000 + seed))             # fixed ring geometry
    recs = []
    for lab in (0, 1):
        ph = (1.0 if lab == 0 else -1.0) * 2 * np.pi * np.arange(8) / 8.0
        for _ in range(n_per):
            S = np.array([np.sin(2 * np.pi * 10 * t + ph[i] + rng.uniform(0, 0.3)) for i in range(8)])
            recs.append((_project(sens, src, S, 1.0) + _bg_field(sens, T, 0.4, rng), lab))
    return _stack(recs, rng)


def naturalistic(n_per, seed):
    """A slow latent modulates theta-gamma coupling; label = coupled vs uncoupled -> rich
    broadband/cross-frequency structure, foundation-model-favored."""
    rng = np.random.default_rng(seed); sens = _fixed_sensors(seed); t = np.arange(T) / FS
    src = sphere_points(4, 60.0, np.random.default_rng(5000 + seed))
    recs = []
    for lab in (0, 1):
        for _ in range(n_per):
            theta = np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6.28))
            amp = (1 + (theta if lab == 1 else 0.0)) / 2
            S = np.array([theta + amp * np.sin(2 * np.pi * 40 * t + rng.uniform(0, 6.28))
                          for _ in range(4)]) * (0.6 + 0.2 * rng.standard_normal((4, 1)))
            recs.append((_project(sens, src, S, 1.0) + _bg_field(sens, T, 0.5, rng), lab))
    return _stack(recs, rng)


def burst(n_per, seed):
    """Intermittent gamma bursts on 1/f; label = burst RATE (few vs many) -> band-power / FMs."""
    rng = np.random.default_rng(seed); sens = _fixed_sensors(seed); t = np.arange(T) / FS
    src = sphere_points(3, 60.0, np.random.default_rng(6000 + seed))
    recs = []
    for lab in (0, 1):
        rate = 3 if lab == 0 else 10
        for _ in range(n_per):
            env = np.zeros(T)
            for c in rng.uniform(0, 4, rate):
                env += np.exp(-((t - c) ** 2) / 5e-3)
            S = np.array([env * np.sin(2 * np.pi * 40 * t + rng.uniform(0, 6.28)) for _ in range(3)])
            recs.append((_project(sens, src, S, 1.0) + _bg_field(sens, T, 0.4, rng), lab))
    return _stack(recs, rng)


SCENARIOS = {"spectral": spectral, "evoked": evoked, "wave": wave,
             "naturalistic": naturalistic, "burst": burst}
