"""Stuart-Landau (Hopf) whole-brain generator for the electrophysiological FSLNets bench.

The netsim/FSLNets ground-truth task (Smith et al. 2011) simulated DCM-BOLD; the dynamics
version needs an electrophysiologically realistic generator with the three things fMRI
never had: (1) tunable dynamical REGIME, (2) conduction DELAYS, (3) an OBSERVATION model
with VOLUME CONDUCTION. This module supplies all three, pure NumPy (no vbjax/GPU), so it
is fast, deterministic and testable, replacing the fragile MPR working point.

Model (Deco et al. 2017, standard whole-brain Hopf):
    dz_j/dt = (a + i.omega_j - |z_j|^2) z_j  +  k * sum_i C_ji (z_i(t-tau_ji) - z_j) + noise
    signal  x_j = Re(z_j)
`a` sweeps the regime: a<0 damped/noise-driven (near-critical resting rhythm), a>0 limit
cycle. tau_ji = dist_ji / velocity are distance-dependent conduction delays (ring buffer).

CRUCIAL design choice: node POSITIONS are random on the head sphere and INDEPENDENT of the
modular structural connectome. So geometry (which drives conduction delays AND the
volume-conduction leakage of the observation model) is decoupled from structure -- making
leakage a GENUINE confound (spurious zero-lag coupling between structurally-unconnected but
spatially-near nodes), exactly the EEG/MEG failure mode absent from fMRI FSLNets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sphere_points(n, radius, rng):
    """`n` points on a sphere of given radius (random directions)."""
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return radius * v


def modular_adjacency(n, n_mod, p_within, p_between, rng):
    """Symmetric binary modular connectome (module = contiguous index block), zero diagonal.
    Structure is by index, INDEPENDENT of spatial position."""
    lab = np.repeat(np.arange(n_mod), int(np.ceil(n / n_mod)))[:n]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_within if lab[i] == lab[j] else p_between
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A, lab


def leadfield_radial(src_pos, sens_pos):
    """Lead field (n_ch, n_src): radial dipole at each source, infinite-medium potential."""
    ori = src_pos / (np.linalg.norm(src_pos, axis=1, keepdims=True) + 1e-12)
    cols = []
    for j in range(len(src_pos)):
        d = sens_pos - src_pos[j][None, :]
        cols.append((d @ ori[j]) / ((np.sum(d * d, 1) ** 1.5) + 1e-9))
    return np.stack(cols, 1)


def leakage_matrix(L, strength):
    """Zero-lag source-leakage mixing M (n_src,n_src): y = M @ x. Off-diagonal =
    lead-field topography overlap (point-spread / cross-talk), scaled by `strength`.
    Spatially-overlapping sources bleed into each other with NO phase lag -- the
    volume-conduction confound that inflates zero-lag correlation."""
    Ln = L / (np.linalg.norm(L, axis=0, keepdims=True) + 1e-12)
    Ovl = np.abs(Ln.T @ Ln)
    np.fill_diagonal(Ovl, 0.0)
    return np.eye(L.shape[1]) + strength * Ovl


def _pink(shape, rng):
    """Unit-variance 1/f (pink) noise per channel via spectral shaping."""
    n, T = shape
    F = np.fft.rfft(rng.standard_normal((n, T)), axis=1)
    f = np.fft.rfftfreq(T); f[0] = f[1] if len(f) > 1 else 1.0
    p = np.fft.irfft(F / np.sqrt(f)[None, :], n=T, axis=1)
    return p / (p.std(axis=1, keepdims=True) + 1e-12)


def measurement_noise(y, snr_db, pink, rng):
    """Add sensor MEASUREMENT noise to an observation at total-power SNR `snr_db` (dB); a
    fraction `pink` of it is 1/f (the dominant EEG background), the rest white. snr_db=inf
    -> unchanged. This is the netsim-style thermal/background noise the leakage mix lacked."""
    if not np.isfinite(snr_db):
        return y
    Ps = float(np.mean(y ** 2)) + 1e-30
    wn = rng.standard_normal(y.shape); wn /= (wn.std(axis=1, keepdims=True) + 1e-12)
    nz = pink * _pink(y.shape, rng) + (1.0 - pink) * wn
    sigma = np.sqrt(Ps / (10.0 ** (snr_db / 10.0)) / (float(np.mean(nz ** 2)) + 1e-30))
    return y + sigma * nz


def cortical_background(sens, n_times, strength, rng, n_src=40, r_src=60.0):
    """Spatially-correlated 1/f EEG background: many random cortical dipoles each emitting
    pink noise, projected through the lead field. Real EEG is a strong 1/f field with
    oscillatory PEAKS on top -- this supplies the aperiodic broadband structure a foundation
    model trained on real EEG expects. Returns (n_ch, n_times) at `strength` x unit RMS."""
    pos = sphere_points(n_src, r_src, rng)
    L = leadfield_radial(pos, sens)                    # (n_ch, n_src)
    bg = L @ _pink((n_src, n_times), rng)              # spatially-correlated 1/f
    return strength * bg / (bg.std() + 1e-12)


def simulate_hopf(C, dist, a, omega, k, velocity, dt, T, noise, seed, fs_out=250.0):
    """Euler-Maruyama Stuart-Landau network with distance delays, integrated at `dt` (ms)
    then decimated to `fs_out` Hz. DIFFUSIVE (row-normalised) coupling keeps the term
    bounded regardless of node degree -> stable. Returns Re(z) (n, n_out)."""
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    Cn = C / (C.sum(1, keepdims=True) + 1e-9)          # row-normalised: mean-field coupling
    Dind = np.maximum(1, np.round(dist / velocity / dt).astype(int))   # delay in samples
    np.fill_diagonal(Dind, 1)
    buf = int(Dind.max()) + 2
    hist = np.zeros((n, buf), dtype=complex)
    z = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    hist[:, 0] = z
    nT = int(T / dt)
    step = max(1, int(round((1000.0 / fs_out) / dt)))  # decimation stride
    out = np.empty((n, nT // step))
    oi = 0
    Iidx = np.tile(np.arange(n), (n, 1))               # Iidx[j,i] = i  (source index)
    sq2dt = np.sqrt(dt)
    for t in range(nT):
        p = t % buf
        read = (p - Dind) % buf                        # read[j,i] delayed sample index
        zdel = hist[Iidx, read]                         # z_i(t - tau_ji), shape (n,n)
        coupling = k * (np.sum(Cn * zdel, 1) - z)       # diffusive, bounded
        dz = (a + 1j * omega - np.abs(z) ** 2) * z + coupling
        z = z + dt * dz + noise * sq2dt * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        hist[:, (t + 1) % buf] = z
        if t % step == 0 and oi < out.shape[1]:
            out[:, oi] = z.real; oi += 1
    return out[:, :oi]


@dataclass
class HopfNetworkSystem:
    """Electrophysiological FSLNets System: Hopf network on a modular connectome with
    distance delays, observed with optional volume-conduction leakage. `simulate(seed)`
    returns (obs (n, nT), true adjacency (n, n)); `space` selects source vs leaked obs.
    Compatible with bench.run_benchmark / adversarial_search."""
    n: int = 30
    n_mod: int = 3
    p_within: float = 0.6
    p_between: float = 0.0
    a: float = 0.02                  # regime: <0 noise-driven near-critical, >0 limit cycle
    k: float = 1.0                   # global coupling (validated SC->FC working point)
    velocity: float = 5.0            # conduction speed (mm/ms)
    f0: float = 10.0                 # centre frequency (Hz)
    fsigma: float = 2.0
    dt: float = 0.1                  # ms integration step
    fs_out: float = 250.0            # Hz output sampling (for FC)
    T: float = 6000.0                # ms
    leak: float = 0.6                # volume-conduction strength
    noise: float = 0.05
    snr: float = float("inf")        # measurement SNR (dB); inf = no sensor noise
    pink: float = 1.0                # fraction of measurement noise that is 1/f
    space: str = "source"            # 'source' (x) or 'leaked' (M @ x)
    r_src: float = 70.0
    r_sens: float = 90.0
    n_ch: int = 64
    montage: object = None       # real 10-20 montage (name list or preset) -> FM-fair sensor space
    background: float = 0.0      # 1/f cortical background strength (x signal RMS); realistic EEG statistics
    seed_struct: int = 0
    band: tuple = (8.0, 12.0)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed_struct)
        self.C, self.labels = modular_adjacency(self.n, self.n_mod,
                                                self.p_within, self.p_between, rng)
        self.pos = sphere_points(self.n, self.r_src, rng)
        if self.montage is not None:
            from .montage import resolve_montage
            self.ch_names, self.sens = resolve_montage(self.montage)
            self.n_ch = len(self.ch_names)
        else:
            self.ch_names = None
            self.sens = sphere_points(self.n_ch, self.r_sens, rng)
        self.dist = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        self.L = leadfield_radial(self.pos, self.sens)
        self.M = leakage_matrix(self.L, self.leak)
        self.omega = 2 * np.pi * (self.f0 + self.fsigma * rng.standard_normal(self.n)) / 1000.0
        self.fs = self.fs_out

    @property
    def complexity(self) -> int:
        return self.n

    def simulate(self, seed=0):
        x = simulate_hopf(self.C, self.dist, self.a, self.omega, self.k, self.velocity,
                          self.dt, self.T, self.noise, seed, fs_out=self.fs_out)
        if self.space == "sensor":
            obs = self.L @ x
            if self.background:
                obs = obs / (obs.std() + 1e-12) + cortical_background(
                    self.sens, obs.shape[1], self.background, np.random.default_rng(4242 * seed + 7))
        elif self.space == "leaked":
            obs = self.M @ x
        else:
            obs = x
        obs = measurement_noise(obs, self.snr, self.pink, np.random.default_rng(7919 * seed + 11))
        return obs, (self.C > 0).astype(int)


def ring_adjacency(n, m, rng=None):
    """Symmetric ring: node i connects to i +/- 1..m (mod n). Zero diagonal."""
    C = np.zeros((n, n))
    for i in range(n):
        for d in range(1, m + 1):
            C[i, (i + d) % n] = C[i, (i - d) % n] = 1.0
    return C


def simulate_kuramoto(C, lag, omega, K, noise, dt, T, seed, fs_out=250.0):
    """Noisy Sakaguchi-Kuramoto on connectome C with per-edge phase lags `lag`. Noise keeps
    it in PARTIAL synchrony (full lock => degenerate FC). Returns signal sin(theta) (n, n_out)."""
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    deg = np.maximum(C.sum(1), 1e-9)
    th = rng.uniform(0, 2 * np.pi, n)
    nT = int(T / dt)
    step = max(1, int(round((1000.0 / fs_out) / dt)))
    out = np.empty((n, nT // step)); oi = 0
    sq = np.sqrt(dt)
    for t in range(nT):
        dphi = th[None, :] - th[:, None] - lag           # theta_j - theta_i - lag_ij
        coup = (C * np.sin(dphi)).sum(1) / deg
        th = th + dt * (omega + K * coup) + noise * sq * rng.standard_normal(n)
        if t % step == 0 and oi < out.shape[1]:
            out[:, oi] = np.sin(th); oi += 1
    return out[:, :oi]


@dataclass
class KuramotoNetworkSystem:
    """PHASE-coupled companion to HopfNetworkSystem: Sakaguchi-Kuramoto on the same modular
    connectome, with distance delays entering as consistent per-edge PHASE LAGS. Here true
    coupling is LAGGED, so lag-based measures (imag-coherence, wPLI) can recover structure
    AND resist zero-lag volume-conduction leakage -- the regime the Hopf (amplitude-coupled)
    System cannot provide. Same geometry / lead-field leakage observation."""
    n: int = 30
    n_mod: int = 3
    p_within: float = 0.6
    p_between: float = 0.0
    K: float = 1.5                   # coupling (partial-sync regime)
    f0: float = 10.0
    fsigma: float = 2.0
    velocity: float = 5.0            # mm/ms -> phase lag = omega0 * dist/velocity
    dt: float = 1.0                  # ms
    fs_out: float = 250.0
    T: float = 6000.0
    leak: float = 0.6
    noise: float = 0.3               # phase noise (keeps partial synchrony)
    snr: float = float("inf")        # measurement SNR (dB)
    pink: float = 1.0
    space: str = "source"
    r_src: float = 70.0
    r_sens: float = 90.0
    n_ch: int = 64
    montage: object = None       # real 10-20 montage (name list or preset) -> FM-fair sensor space
    background: float = 0.0      # 1/f cortical background strength (x signal RMS); realistic EEG statistics
    seed_struct: int = 0
    band: tuple = (8.0, 12.0)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed_struct)
        self.C, self.labels = modular_adjacency(self.n, self.n_mod,
                                                self.p_within, self.p_between, rng)
        self.pos = sphere_points(self.n, self.r_src, rng)
        if self.montage is not None:
            from .montage import resolve_montage
            self.ch_names, self.sens = resolve_montage(self.montage)
            self.n_ch = len(self.ch_names)
        else:
            self.ch_names = None
            self.sens = sphere_points(self.n_ch, self.r_sens, rng)
        self.dist = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=2)
        self.L = leadfield_radial(self.pos, self.sens)
        self.M = leakage_matrix(self.L, self.leak)
        omega0 = 2 * np.pi * self.f0 / 1000.0
        self.lag = omega0 * self.dist / self.velocity        # per-edge phase lag (rad)
        self.omega = 2 * np.pi * (self.f0 + self.fsigma * rng.standard_normal(self.n)) / 1000.0
        self.fs = self.fs_out

    @property
    def complexity(self) -> int:
        return self.n

    def simulate(self, seed=0):
        x = simulate_kuramoto(self.C, self.lag, self.omega, self.K, self.noise,
                              self.dt, self.T, seed, fs_out=self.fs_out)
        if self.space == "sensor":
            obs = self.L @ x
            if self.background:
                obs = obs / (obs.std() + 1e-12) + cortical_background(
                    self.sens, obs.shape[1], self.background, np.random.default_rng(4242 * seed + 7))
        elif self.space == "leaked":
            obs = self.M @ x
        else:
            obs = x
        obs = measurement_noise(obs, self.snr, self.pink, np.random.default_rng(7919 * seed + 11))
        return obs, (self.C > 0).astype(int)


@dataclass
class RingWaveSystem:
    """TRAVELING-WAVE regime that completes the volume-conduction dissociation. A directed
    ring (i~i+-1..m) is driven with a UNIFORM phase frustration so connected neighbours lock
    at a CONSISTENT non-zero phase lag (a traveling wave) -- the one regime where lag-based
    measures (imag-coherence, wPLI) can recover structure AND resist zero-lag leakage.
    CRUCIAL: node 3-D positions are RANDOM (decoupled from ring order), so the lead-field
    leakage is a genuine zero-lag confound, while the wave's lags follow the ring, not space."""
    n: int = 30
    m: int = 2                       # ring neighbourhood (edges = i +- 1..m)
    K: float = 1.2
    alpha: float = 0.9               # phase frustration (rad) -> consistent neighbour lag
    f0: float = 10.0
    fsigma: float = 0.5
    dt: float = 1.0
    fs_out: float = 250.0
    T: float = 6000.0
    leak: float = 0.6
    noise: float = 0.15
    snr: float = float("inf")        # measurement SNR (dB)
    pink: float = 1.0
    space: str = "source"
    r_src: float = 70.0
    r_sens: float = 90.0
    n_ch: int = 64
    montage: object = None       # real 10-20 montage (name list or preset) -> FM-fair sensor space
    background: float = 0.0      # 1/f cortical background strength (x signal RMS); realistic EEG statistics
    seed_struct: int = 0
    band: tuple = (8.0, 12.0)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed_struct)
        self.C = ring_adjacency(self.n, self.m)
        self.lag = np.zeros((self.n, self.n))                # DIRECTED frustration -> traveling wave
        for i in range(self.n):
            for d in range(1, self.m + 1):
                self.lag[i, (i + d) % self.n] = self.alpha   # ahead
                self.lag[i, (i - d) % self.n] = -self.alpha  # behind
        self.pos = sphere_points(self.n, self.r_src, rng)    # RANDOM, decoupled from ring
        if self.montage is not None:
            from .montage import resolve_montage
            self.ch_names, self.sens = resolve_montage(self.montage)
            self.n_ch = len(self.ch_names)
        else:
            self.ch_names = None
            self.sens = sphere_points(self.n_ch, self.r_sens, rng)
        self.L = leadfield_radial(self.pos, self.sens)
        self.M = leakage_matrix(self.L, self.leak)
        self.omega = 2 * np.pi * (self.f0 + self.fsigma * rng.standard_normal(self.n)) / 1000.0
        self.fs = self.fs_out

    @property
    def complexity(self) -> int:
        return self.n

    def simulate(self, seed=0):
        x = simulate_kuramoto(self.C, self.lag, self.omega, self.K, self.noise,
                              self.dt, self.T, seed, fs_out=self.fs_out)
        if self.space == "sensor":
            obs = self.L @ x
            if self.background:
                obs = obs / (obs.std() + 1e-12) + cortical_background(
                    self.sens, obs.shape[1], self.background, np.random.default_rng(4242 * seed + 7))
        elif self.space == "leaked":
            obs = self.M @ x
        else:
            obs = x
        obs = measurement_noise(obs, self.snr, self.pink, np.random.default_rng(7919 * seed + 11))
        return obs, (self.C > 0).astype(int)
