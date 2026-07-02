"""A DynaDojo/FSLNets-style comparison harness for cortical-dynamics methods.

The dynamics literature too often proposes a technique without testing it against
alternatives on common, ground-truth-anchored footing. Two templates fix that:
  - FSLNets (Smith et al., NeuroImage 54:875, 2011): judge many methods against
    SIMULATED data with KNOWN ground truth (recovery, not just plausibility).
  - DynaDojo (Bhamidipaty, Bruzzese, Tran, Mrad & Kanwal, NeurIPS 2023): an extensible
    `System` x `Algorithm` x `Challenge` API with SCALING diagnostics (vs #samples,
    system complexity, target error).

This module adopts that structure for our facets:
  - `System`     : a generative model that emits a signal + its ground truth (here the
                   Kuramoto/Budzinski traveling-wave system; extend with Hopf/TVB,
                   geometric eigenmodes, AR/linear-Gaussian nulls).
  - `Algorithm`  : a method that estimates the target from the signal (phase-flow,
                   naive snapshot, or the STRUCTURE-based Budzinski predictor that uses
                   only connectivity).
  - `Challenge`  : a recovery metric in [0,1].
  - `run_benchmark`: sweeps a scaling axis (noise / length / complexity), averaging over
                   seeds -> a {method: [score per scaling value]} table.

The point is comparability: a structure-based predictor (noise-immune but needs the
connectome) and data-based estimators (recover from signal but degrade with noise) are
scored on the SAME ground truth, the same axis.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .generators.waves import (wave_modes, predicted_phase_map, dominant_wavenumber,
                            kuramoto_simulate)

__all__ = ["KuramotoWaveSystem", "StructurePredictor", "PhaseFlowRecovery",
           "NaiveSnapshot", "wavenumber_recovery", "run_benchmark",
           "adversarial_search"]


# -----------------------------------------------------------------------------
# Systems (generative, ground-truth)
# -----------------------------------------------------------------------------
@dataclass
class KuramotoWaveSystem:
    """Kuramoto traveling-wave system on a (directed) ring with distance-dependent delays.
    `simulate` returns (noisy phase observation (n, t), true realized wavenumber)."""
    n: int = 48
    m: int = 2
    velocity: float = 6.0
    omega0: float = 1.0
    K: float = 2.5
    T: float = 60.0
    directed: bool = True

    def __post_init__(self):
        A = np.zeros((self.n, self.n)); D = np.zeros((self.n, self.n))
        for i in range(self.n):
            for d in range(1, self.m + 1):
                A[i, (i + d) % self.n] = 1.0
                if not self.directed:
                    A[i, (i - d) % self.n] = 1.0
            for j in range(self.n):
                dd = abs(i - j); D[i, j] = min(dd, self.n - dd)
        self.A, self.D = A, D

    @property
    def complexity(self) -> int:
        return self.n

    def simulate(self, noise=0.0, seed=0):
        lag = self.omega0 * self.D / self.velocity
        th = kuramoto_simulate(self.A, self.omega0, self.K, lag=lag, T=self.T, seed=seed)
        true_k = dominant_wavenumber(th[:, -1])
        rng = np.random.default_rng(1000 + seed)
        obs = th + noise * rng.standard_normal(th.shape)
        return obs, true_k


# -----------------------------------------------------------------------------
# Algorithms (estimate the target wavenumber)
# -----------------------------------------------------------------------------
class StructurePredictor:
    """Budzinski-Muller: predict the wave from CONNECTIVITY ONLY (dominant coupling-operator
    eigenmode). Ignores the signal -> immune to observation noise, but needs the connectome."""
    name = "structure (Budzinski)"

    def estimate(self, obs, system):
        _, V = wave_modes(system.A, distances=system.D, omega0=system.omega0,
                          velocity=system.velocity)
        return dominant_wavenumber(predicted_phase_map(V[:, 0]))


class PhaseFlowRecovery:
    """Data-based: the instantaneous-phase field's dominant wavenumber, MEDIAN over the
    second half of the trajectory (the phase-flow facet's robustness to transient noise)."""
    name = "phase-flow (data)"

    def estimate(self, obs, system):
        t0 = obs.shape[1] // 2
        ks = [dominant_wavenumber(obs[:, t]) for t in range(t0, obs.shape[1])]
        return int(np.round(np.median(ks)))


class NaiveSnapshot:
    """Data-based baseline: wavenumber from a single final snapshot (no temporal pooling)."""
    name = "naive snapshot"

    def estimate(self, obs, system):
        return dominant_wavenumber(obs[:, -1])


# -----------------------------------------------------------------------------
# Challenges (recovery metrics) + the benchmark loop
# -----------------------------------------------------------------------------
def wavenumber_recovery(k_est, k_true):
    """1.0 if the (unsigned) wavenumber is recovered exactly, else 0.0."""
    return float(abs(int(k_est)) == abs(int(k_true)))


def run_benchmark(system, algorithms, scaling_values, axis="noise", n_seeds=5,
                  challenge=wavenumber_recovery):
    """Sweep a scaling axis, averaging recovery over seeds. Returns a dict
    {method_name: [mean score per scaling value]} plus the axis values under 'axis'.
    axis='noise' varies observation noise; 'length' scales the trajectory T."""
    table = {alg.name: [] for alg in algorithms}
    for val in scaling_values:
        acc = {alg.name: [] for alg in algorithms}
        for s in range(n_seeds):
            if axis == "length":
                system.T = float(val); obs, k_true = system.simulate(noise=0.0, seed=s)
            else:
                obs, k_true = system.simulate(noise=float(val), seed=s)
            for alg in algorithms:
                acc[alg.name].append(challenge(alg.estimate(obs, system), k_true))
        for alg in algorithms:
            table[alg.name].append(float(np.mean(acc[alg.name])))
    table["axis"] = list(scaling_values)
    return table


# -----------------------------------------------------------------------------
# DynaDojo-v2: adversarial System search (find the regimes that break methods)
# -----------------------------------------------------------------------------
def adversarial_search(system_factory, algorithms, param_grid, challenge,
                       n_seeds=3, objective="disagreement"):
    """Search a System-parameter grid for the most ADVERSARIAL setting. For each grid
    point, build the System, score every algorithm (mean over seeds), and rank by an
    adversarial objective:
      - 'disagreement': maximise cross-method spread (std of scores) — the regimes where
        the field's methods diverge are the most informative rows of the benchmark.
      - 'hardest':      minimise the BEST method's score — where every method struggles.
    Returns a list of (params, {method: score}, objective_value), best first. This is the
    grid/active-learning form; with vbjax's autodiff the same objective can be optimised by
    gradient over continuous System params (and by evolution over discrete structure)."""
    results = []
    keys = list(param_grid.keys())
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        system = system_factory(**params)
        per = {a.name: [] for a in algorithms}
        for s in range(n_seeds):
            obs, gt = system.simulate(seed=s)
            for a in algorithms:
                per[a.name].append(challenge(a.estimate(obs, system), gt))
        means = {k: float(np.nanmean(v)) for k, v in per.items()}
        vals = [v for v in means.values() if np.isfinite(v)]
        if not vals:
            obj = float("-inf")
        elif objective == "hardest":
            obj = -max(vals)
        else:
            obj = float(np.std(vals))
        results.append((params, means, float(obj)))
    results.sort(key=lambda r: -r[2])
    return results
