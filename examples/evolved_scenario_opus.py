# LLaMEA-evolved adversarial scenario — Opus backend, longer budget.
#
# Run: Claude (claude-opus-4-8) backend, (1+1) elitist evolution, budget=30, seed 0.
# Fitness = cross-method disagreement (std of per-method AUC over the classical+sysid zoo).
# Champion found at generation 2, held through all 30 generations; fitness (disagreement std)
# = 0.258 — HIGHER than the sonnet run (0.095) AND the hand-tuned battery (~0.21). Opus climbs
# the disagreement objective far more effectively than sonnet.
#
# Reproduced per-method AUC (n_per=50, seed 0; chance .50):
#     band-power 1.00   SINDy 0.44   DMD 1.00   DySCo 0.61   HMM 0.42
#
# HONEST NUANCE (a real adversarial-benchmark lesson): the model *intended* a pure theta-gamma
# phase-amplitude-coupling contrast that band-power cannot see (its docstring says so). But the
# way it built class 0 — phase-SCRAMBLING the gamma envelope — leaked a gamma-band POWER
# difference, so band-power (1.00) and DMD (1.00) trivially ace it while the nonlinearity-oriented
# methods (SINDy 0.44, HMM 0.42) fail. The disagreement-maximizing fitness happily rewarded this
# spectral confound. This is exactly the failure mode ground-truth adversarial design must guard
# against: a high-"disagreement" scenario can be won by an unintended easy feature. It argues for
# a confound-penalizing fitness (e.g. reward margin of the INTENDED family over a named baseline)
# — see the note in examples/llamea_evolve_scenarios.py. Kept verbatim as an instructive artifact.
import numpy as np


class Scenario:
    """Theta-gamma phase-amplitude coupling in class 1 only; matched power spectrum defeats band-power, nonlinearity defeats DMD/SINDy."""
    def generate(self, n_per, seed):
        rng = np.random.default_rng(seed); C, T, fs = 32, 1000, 250.0
        t = np.arange(T) / fs
        geo = np.random.default_rng(1000).standard_normal((C, 6))   # FIXED lead field
        ft = 6.0; fg = 40.0
        freqs = np.fft.rfftfreq(T, 1 / fs); freqs[0] = 1.0
        X, y = [], []
        for lab in (0, 1):
            for _ in range(n_per):
                pt = rng.uniform(0, 2 * np.pi); pg = rng.uniform(0, 2 * np.pi)
                theta = np.sin(2 * np.pi * ft * t + pt)
                gph = 2 * np.pi * fg * t + pg
                if lab == 1:
                    # gamma amplitude modulated by theta phase -> CFC
                    env = 0.5 * (1 + theta)
                else:
                    # same mean gamma power but modulation scrambled (no phase locking)
                    shuf = rng.permutation(T)
                    env = 0.5 * (1 + theta[shuf])
                env = env / (env.std() + 1e-9)
                gamma = env * np.sin(gph)
                sig = theta / (theta.std() + 1e-9) + gamma / (gamma.std() + 1e-9)
                # spatial mixing (fixed) so all channels share the coupling
                w = np.random.default_rng(2000).standard_normal(C)
                wave = np.outer(w, sig)
                # spatially-correlated 1/f background
                nz = rng.standard_normal((6, T))
                spec = np.fft.rfft(nz, axis=1) / freqs[None, :]
                bg = geo @ np.fft.irfft(spec, n=T, axis=1)
                x = wave / (wave.std() + 1e-9) + 0.8 * bg / (bg.std() + 1e-9)
                X.append(x); y.append(lab)
        X = np.stack(X); idx = rng.permutation(len(y))
        return X[idx], np.array(y)[idx], [f"E{i}" for i in range(C)]
