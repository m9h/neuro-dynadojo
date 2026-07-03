# LLaMEA-evolved adversarial scenario — CONFOUND-AWARE (targeted) fitness.
#
# Run: Claude (claude-opus-4-8) backend, (1+1) elitist evolution, budget=30, seed 0,
#      NDD_MODE=targeted. Fitness = margin of the best GENUINE-DYNAMICS method (SINDy/DySCo/HMM)
#      over the best SPECTRAL/amplitude method (band-power/DMD). A scenario scores only if it
#      beats EVERY spectral method — so a leaked power/amplitude confound is penalized, not rewarded.
#
# Reproduced per-method AUC (chance .50), robust across seeds (NOT seed-overfit):
#     seed 0:  band-power 0.53  SINDy 1.00  DMD 0.51  DySCo 0.46  HMM 0.53   margin +0.47
#     seed 7:  band-power 0.50  SINDy 0.99  DMD 0.40  DySCo 0.57  HMM 0.49   margin +0.49
#
# This is the payoff of the confound-aware objective. Under raw disagreement, Opus produced a
# "PAC" scenario whose envelope-scramble LEAKED gamma power, so band-power aced it (see
# evolved_scenario_opus.py). Re-run under the targeted margin, the SAME model was forced to build
# a GENUINELY spectrum-matched cross-frequency phase-coupling contrast: it added explicit
# anti-leakage steps — high-pass the coupled HF component so no slow envelope bleeds into the low
# band, renormalize HF power, and match per-channel variance — so band-power AND DMD sit at chance
# (~.50) while only nonlinear system-ID (SINDy) recovers the 6->40 Hz gating-phase label. Fitness
# shaping turned a confound-gamed win into a legitimate, generalizing ground-truth scenario.
import numpy as np


class Scenario:
    """6-40 Hz phase-amplitude coupling at gating-phase 0 vs pi — matched per-band power, PAC-only label."""
    def generate(self, n_per, seed):
        rng = np.random.default_rng(seed); C, T, fs = 32, 1000, 250.0
        t = np.arange(T) / fs
        geo = np.random.default_rng(1000).standard_normal((C, 6))       # FIXED lead field
        topo_lo = np.random.default_rng(2000).standard_normal(6)
        topo_hi = np.random.default_rng(3000).standard_normal(6)
        f = np.fft.rfftfreq(T, 1 / fs); f[0] = f[1]
        X, y = [], []
        for lab in (0, 1):
            cphase = 0.0 if lab == 0 else np.pi                          # gating phase offset
            for _ in range(n_per):
                ph0 = rng.uniform(0, 2 * np.pi)
                lo = np.sin(2 * np.pi * 6 * t + ph0)
                # sharp gate concentrates HF into a narrow phase window (strong nonlinear coupling)
                g = 0.5 * (1 + np.cos(2 * np.pi * 6 * t + ph0 - cphase))
                gate = g ** 4
                hf = np.sin(2 * np.pi * 40 * t + rng.uniform(0, 2 * np.pi))
                hi = gate * hf
                # remove any slow envelope leakage into low band, then normalize HF power
                Hi = np.fft.rfft(hi); Hi[f < 20] = 0
                hi = np.fft.irfft(Hi, n=T)
                hi = hi / (hi.std() + 1e-9)
                lo = lo / (lo.std() + 1e-9)
                src = np.outer(topo_lo, lo) + np.outer(topo_hi, hi)
                sig = geo @ src
                w = rng.standard_normal((6, T))
                W = np.fft.rfft(w, axis=1) / f
                bg = geo @ np.fft.irfft(W, n=T, axis=1)
                x = sig / (sig.std() + 1e-9) + 0.8 * bg / (bg.std() + 1e-9)
                x = x / (x.std(axis=1, keepdims=True) + 1e-9)           # match channel variance
                X.append(x); y.append(lab)
        X = np.stack(X); idx = rng.permutation(len(y))
        return X[idx], np.array(y)[idx], [f"E{i}" for i in range(C)]
