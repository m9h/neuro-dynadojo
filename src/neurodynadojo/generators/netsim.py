"""Netsim-parity generator: the Smith et al. (2011) confound battery, ported to
electrophysiology. Their 28 simulations varied, beyond noise and #nodes/duration:
shared/global inputs, NON-STATIONARY connection strengths, BACKWARD/directed connections,
and per-node HRF-delay variability (SD 0.5 s). This module adds the electrophysiological
analogs on top of the Hopf/Stuart-Landau core + lead-field leakage + pink measurement noise
already in neurodynadojo.generators.hopf:

  - directed connectome (forward-dominant edges + weak backward) -> a DIRECTED ground truth;
  - shared/global common input (false-edge confound);
  - non-stationary connectivity (edge strengths cycle over segments);
  - per-node GAIN + LATENCY jitter (the HRF-variability analog: variable node gain & delay);
  - (inherited) volume-conduction leakage + pink/white sensor noise at a target SNR.

Everything is off by default, so the baseline reproduces the clean Hopf modular recovery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hopf import (sphere_points, leadfield_radial, leakage_matrix, measurement_noise)


def directed_modular_adjacency(n, n_mod, p_within, p_between, back, rng):
    """Directed modular connectome: each selected pair gets a forward edge (weight 1) and a
    weak backward edge (weight `back`). Returns C (asymmetric). Directed truth = C>0.5
    (forward), undirected truth = (C+C.T)>0."""
    lab = np.repeat(np.arange(n_mod), int(np.ceil(n / n_mod)))[:n]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = p_within if lab[i] == lab[j] else p_between
            if i < j and rng.random() < p:
                if rng.random() < 0.5:
                    C[i, j], C[j, i] = 1.0, back
                else:
                    C[j, i], C[i, j] = 1.0, back
    return C, lab


def _ou(nT, rng, tau=0.99):
    """Unit-variance smooth common driver (Ornstein-Uhlenbeck)."""
    s = np.zeros(nT)
    for t in range(1, nT):
        s[t] = tau * s[t - 1] + np.sqrt(1 - tau * tau) * rng.standard_normal()
    return s / (s.std() + 1e-12)


def simulate_netsim(Clist, a, omega, k, dt, T, noise, seed, fs_out=250.0,
                    shared=0.0, seg=None, gain=None, latency=None):
    """Stuart-Landau network with netsim confounds. `Clist` = list of connectomes cycled
    every `seg` integration steps (non-stationarity; len 1 = stationary). `shared` injects a
    common global driver. `gain`/`latency` (per-node) apply output amplitude/delay jitter."""
    rng = np.random.default_rng(seed)
    n = Clist[0].shape[0]
    Cn = [C / (C.sum(1, keepdims=True) + 1e-9) for C in Clist]
    nT = int(T / dt)
    seg = seg or nT
    ss = _ou(nT, rng) if shared else None
    z = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    step = max(1, int(round((1000.0 / fs_out) / dt)))
    out = np.empty((n, nT // step)); oi = 0
    sq = np.sqrt(dt)
    for t in range(nT):
        C = Cn[(t // seg) % len(Cn)]
        coupling = k * (C @ z - z)
        dz = (a + 1j * omega - np.abs(z) ** 2) * z + coupling
        if shared:
            dz = dz + shared * ss[t]                         # global common input
        z = z + dt * dz + noise * sq * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        if t % step == 0 and oi < out.shape[1]:
            out[:, oi] = z.real; oi += 1
    x = out[:, :oi]
    if gain is not None:
        x = x * gain[:, None]
    if latency is not None:
        x = np.stack([np.roll(x[i], int(latency[i])) for i in range(n)])
    return x


@dataclass
class NetsimSystem:
    """Netsim-parity System: directed modular Hopf network with the Smith-2011 confound
    battery. `simulate(seed)` -> (obs, C) where C is the (directed) connectome; use
    `undirected_truth`/`directed_truth` for the two ground truths."""
    n: int = 30
    n_mod: int = 3
    p_within: float = 0.6
    p_between: float = 0.0
    back: float = 0.0                # backward-edge weight (0 = purely feedforward)
    connectome: object = None        # optional external weighted SC (overrides modular graph)
    edge_density: float = 0.3        # strongest fraction of SC pairs taken as ground-truth edges
    a: float = 0.02
    k: float = 1.0
    f0: float = 10.0
    fsigma: float = 2.0
    dt: float = 0.1
    fs_out: float = 250.0
    T: float = 6000.0
    noise: float = 0.05              # process noise
    leak: float = 0.0                # volume conduction (0 = source space)
    snr: float = float("inf")        # sensor SNR (dB)
    pink: float = 1.0
    shared: float = 0.0              # global common-input strength
    nonstat: float = 0.0             # non-stationarity (0 = stationary; >0 rewires strengths)
    jitter_gain: float = 0.0         # per-node gain SD (HRF-amplitude analog)
    jitter_lat: float = 0.0          # per-node latency SD in samples (HRF-delay analog)
    r_src: float = 70.0
    r_sens: float = 90.0
    n_ch: int = 64
    seed_struct: int = 0
    band: tuple = (8.0, 12.0)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed_struct)
        if self.connectome is not None:                      # REAL structural connectome
            W = np.asarray(self.connectome, float).copy()
            np.fill_diagonal(W, 0.0)
            self.n = W.shape[0]
            self.C = W / (W.max() + 1e-12)                   # raw-normalised weighted coupling
            #  (strongest streamline counts dominate -> SC imprints on FC; log flattens to chance)
            iu = np.triu_indices(self.n, 1)
            pos_w = W[iu][W[iu] > 0]
            thr = np.quantile(pos_w, 1 - self.edge_density) if pos_w.size else np.inf
            A = (W >= thr).astype(float); np.fill_diagonal(A, 0.0)
            self._utruth = ((A + A.T) > 0).astype(int)
            self.labels = np.zeros(self.n, int)
        else:
            self.C, self.labels = directed_modular_adjacency(
                self.n, self.n_mod, self.p_within, self.p_between, self.back, rng)
            self._utruth = ((self.C + self.C.T) > 0).astype(int)
        # non-stationarity: a second connectome, same edges, re-weighted
        self.Clist = [self.C]
        if self.nonstat:
            C2 = self.C * (1.0 + self.nonstat * rng.standard_normal(self.C.shape)) * (self.C > 0)
            self.Clist = [self.C, np.abs(C2)]
        self.pos = sphere_points(self.n, self.r_src, rng)
        self.sens = sphere_points(self.n_ch, self.r_sens, rng)
        self.L = leadfield_radial(self.pos, self.sens)
        self.M = leakage_matrix(self.L, self.leak) if self.leak else np.eye(self.n)
        self.omega = 2 * np.pi * (self.f0 + self.fsigma * rng.standard_normal(self.n)) / 1000.0
        self.gain = (1.0 + self.jitter_gain * rng.standard_normal(self.n)) if self.jitter_gain else None
        self.latency = (self.jitter_lat * rng.standard_normal(self.n)).round() if self.jitter_lat else None
        self.fs = self.fs_out
        self.seg = int(0.5 * (self.T / self.dt)) if self.nonstat else None

    @property
    def complexity(self):
        return self.n

    def undirected_truth(self):
        return self._utruth

    def directed_truth(self):
        if self.connectome is not None:
            return self._utruth                     # real SC is symmetric: no directed truth
        return (self.C > 0.5).astype(int)           # forward (strong) edges only

    def simulate(self, seed=0):
        x = simulate_netsim(self.Clist, self.a, self.omega, self.k, self.dt, self.T,
                            self.noise, seed, fs_out=self.fs_out, shared=self.shared,
                            seg=self.seg, gain=self.gain, latency=self.latency)
        obs = self.M @ x if self.leak else x
        obs = measurement_noise(obs, self.snr, self.pink, np.random.default_rng(7919 * seed + 3))
        return obs, self.C
