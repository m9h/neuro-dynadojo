"""Structure -> traveling-wave prediction (Budzinski-Muller) + Kuramoto ground truth.

The generative counterpart of the phase-flow facet: instead of *measuring* traveling
waves in EEG, we *predict* which waves a connectivity structure produces. Budzinski,
Muller et al. (Chaos 32:031104, 2022; Phys. Rev. Research 5:013159, 2023) show that the
spatiotemporal patterns of a Kuramoto network are organized by the eigenvectors of a
coupling operator: with distance-dependent conduction delays the operator becomes complex
(W_ij = A_ij e^{-i omega0 tau_ij}), and its complex eigen-spectrum selects the specific
traveling waves and their direction. This module implements that central mechanism plus a
direct Kuramoto simulator, so the analytic prediction can be validated against the
simulation (the ground-truth arm of our FSLNets-style methods comparison) and compared to
the routing modes measured by `phaseflow` on real EEG.

This is a faithful reference implementation of the framework's core (eigenmodes of the
[delay-augmented] coupling operator predict traveling-wave patterns), validated by direct
simulation in the tests; it is not a bit-exact reproduction of every result in the papers.
"""
from __future__ import annotations

import numpy as np

__all__ = ["coupling_operator", "wave_modes", "predicted_phase_map", "order_parameter",
           "dominant_wavenumber", "kuramoto_simulate"]


# -----------------------------------------------------------------------------
# Analytic structure -> wave prediction
# -----------------------------------------------------------------------------
def coupling_operator(A, distances=None, omega0=1.0, velocity=None):
    """Budzinski-Muller coupling operator from an adjacency matrix A.

    No delay (distances/velocity unset) -> A itself (real). With distance-dependent
    conduction delays tau_ij = distances_ij / velocity, the operator gains a per-edge
    phase factor W_ij = A_ij * exp(-i * omega0 * tau_ij); the resulting complex spectrum
    is what breaks left/right symmetry and selects a traveling direction (PRR 2023)."""
    A = np.asarray(A, float)
    if distances is None or velocity is None:
        return A.astype(complex)
    tau = np.asarray(distances, float) / velocity
    return A * np.exp(-1j * omega0 * tau)


def wave_modes(A, distances=None, omega0=1.0, velocity=None):
    """Eigen-decompose the (delay-augmented) coupling operator. Returns (eigvals, modes)
    ordered by descending real part (least-stable / dominant wave first). Each mode is a
    complex eigenvector: angle(mode) is the predicted spatial phase pattern of a wave."""
    W = coupling_operator(A, distances, omega0, velocity)
    if np.max(np.abs(W.imag)) < 1e-12:                  # real-symmetric -> standing modes
        w, V = np.linalg.eigh(W.real)
        w = w.astype(complex); V = V.astype(complex)
    else:
        w, V = np.linalg.eig(W)
    order = np.argsort(w.real)[::-1]
    return w[order], V[:, order]


def predicted_phase_map(mode):
    """The 'function on eigenvectors': a wave's spatial phase pattern = arg(eigenvector)."""
    return np.angle(np.asarray(mode))


# -----------------------------------------------------------------------------
# Diagnostics shared with the phase-flow facet
# -----------------------------------------------------------------------------
def order_parameter(theta):
    """Kuramoto order parameter r in [0,1] (global phase coherence). theta is (n,) or
    (n, t) -> scalar or (t,)."""
    theta = np.asarray(theta)
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


def dominant_wavenumber(phase_on_ring):
    """Integer spatial wavenumber of a phase pattern sampled around a ring: the index of
    the largest non-DC component of the spatial FFT of exp(i*phase). Signed (direction)."""
    z = np.exp(1j * np.asarray(phase_on_ring, float))
    F = np.fft.fft(z)
    F[0] = 0.0
    k = int(np.argmax(np.abs(F)))
    n = len(z)
    return k - n if k > n // 2 else k                   # map to signed wavenumber


# -----------------------------------------------------------------------------
# Kuramoto ground-truth simulator
# -----------------------------------------------------------------------------
def kuramoto_simulate(A, omega=0.0, K=2.0, lag=None, dt=0.01, T=30.0,
                      theta0=None, seed=0):
    """Integrate the (Sakaguchi-)Kuramoto model on adjacency A (Euler):

        dtheta_i/dt = omega_i + (K/deg_i) * sum_j A_ij sin(theta_j - theta_i - lag_ij)

    `lag` is the per-edge phase lag matrix (e.g. omega0 * tau_ij for distance-dependent
    delays under the frozen-frequency reduction that yields `coupling_operator`). Returns
    theta (n, n_steps). The ground-truth generator for the structure->dynamics comparison."""
    A = np.asarray(A, float)
    n = A.shape[0]
    deg = np.maximum(A.sum(1), 1e-9)
    omega = np.broadcast_to(np.asarray(omega, float), (n,)).astype(float)
    L = np.zeros((n, n)) if lag is None else np.asarray(lag, float)
    if theta0 is None:
        theta0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    th = np.asarray(theta0, float).copy()
    nt = int(T / dt)
    out = np.empty((n, nt))
    for t in range(nt):
        dphi = th[None, :] - th[:, None] - L            # theta_j - theta_i - lag_ij
        coup = (A * np.sin(dphi)).sum(1) / deg
        th = th + dt * (omega + K * coup)
        out[:, t] = th
    return out
