# LLaMEA-evolved adversarial scenario — the ACTUAL output of examples/llamea_evolve_scenarios.py.
#
# Run: Claude (claude-sonnet-5) backend, (1+1) elitist evolution, budget=12, seed 0.
# Fitness = cross-method disagreement (std of per-method AUC over the classical+sysid zoo).
# Champion found at generation 3, held under elitism; final fitness (disagreement std) = 0.095.
#
# Reproduced per-method AUC (n_per=50, seed 0; chance .50):
#     band-power 0.52   SINDy 0.66   DMD 0.39   DySCo 0.45   HMM 0.41
#
# What the LLM discovered: a traveling wave whose propagation DIRECTION is the label, carried
# purely by cross-channel phase-lag with a MATCHED power spectrum + 1/f background — so band-power
# is blind (0.52) while system-ID (SINDy) reads the directed dynamics (0.66). It rediscovered, on
# its own, the mechanism behind the hand-authored `wave` scenario: spectral methods cannot see
# propagation direction. (The hand-tuned battery still reaches higher disagreement, ~0.21; 12
# generations of (1+1) on a single model did not beat it — an honest baseline for the engine.)
import numpy as np


class Scenario:
    """Broadband delayed traveling-wave direction (matched power spectrum) encodes label via cross-channel phase-lag only."""
    def generate(self, n_per, seed):
        rng = np.random.default_rng(seed)
        C, T, fs = 32, 1000, 250.0
        t = np.arange(T) / fs
        freqs = np.fft.rfftfreq(T, d=1 / fs)
        pos = np.linspace(0, 1, C)  # fixed sensor geometry
        n_src = 6
        geo = np.random.default_rng(1000).standard_normal((C, n_src))
        geo /= np.linalg.norm(geo, axis=0, keepdims=True)
        env = np.exp(-((t - 0.5) ** 2) / (2 * 0.25 ** 2))
        max_delay = 0.02  # s, fixed max propagation delay across array
        band_center, band_width = 10.0, 2.0
        band_mask = np.exp(-0.5 * ((freqs - band_center) / band_width) ** 2)

        def make_bg():
            spec = 1.0 / np.maximum(freqs, 1.0)
            src = np.zeros((n_src, T))
            for s in range(n_src):
                phases = rng.uniform(0, 2 * np.pi, len(freqs))
                F = spec * np.exp(1j * phases)
                sig = np.fft.irfft(F, n=T)
                src[s] = sig / (sig.std() + 1e-9)
            return geo @ src

        def make_wave(direction):
            phases = rng.uniform(0, 2 * np.pi, len(freqs))
            base_spec = band_mask * np.exp(1j * phases)
            wave = np.zeros((C, T))
            for c in range(C):
                delay = direction * max_delay * pos[c]
                shifted = base_spec * np.exp(-1j * 2 * np.pi * freqs * delay)
                wave[c] = np.fft.irfft(shifted, n=T)
            wave *= env
            return wave

        X, y = [], []
        for lab in (0, 1):
            direction = 1.0 if lab == 0 else -1.0
            for _ in range(n_per):
                amp_jit = 1.0 + 0.05 * rng.standard_normal()
                wave = make_wave(direction) * amp_jit
                bg = make_bg()
                sig = wave / (wave.std() + 1e-9)
                noise = 0.7 * bg / (bg.std() + 1e-9)
                x = 1.3 * sig + noise
                X.append(x)
                y.append(lab)
        X = np.stack(X)
        idx = rng.permutation(len(y))
        ch_names = [f"E{i}" for i in range(C)]
        return X[idx], np.array(y)[idx], ch_names
