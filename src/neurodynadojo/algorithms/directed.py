"""Directed-connectivity Algorithms for the bench — does DIRECTIONALITY recover structure
where symmetric FC could not? The traveling-wave regime carries its information in
consistent phase LAGS, which undirected FC (correlation, imag-coh) cannot turn into an
adjacency; a directed method should read out the wave's DIRECTION of flow.

  granger_bivariate : pairwise Granger causality G[i,j] = influence i->j (log residual-
                      variance ratio of a VAR with vs without i's past).
  dmd_transition    : |A| of the reduced DMD/VAR(1) operator x_{t+1}=A x_t -- a directed,
                      dynamics-based influence score.
Scored by `directed_edge_auc` against the DIRECTED ground truth (edges oriented with the
wave). `waves.dominant_wavenumber` supplies the orthogonal WAVENUMBER challenge.
"""
from __future__ import annotations

import numpy as np


def _ar_resid_var(y, preds, p):
    """Residual variance of AR(p): y[t] ~ lagged `preds` (list of series). LS fit."""
    T = len(y)
    rows = range(p, T)
    Y = y[p:]
    X = [np.ones(len(rows))]
    for s in preds:
        for L in range(1, p + 1):
            X.append(s[p - L:T - L])
    X = np.stack(X, 1)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(np.var(Y - X @ beta)) + 1e-30


def granger_bivariate(X, order=4):
    """Pairwise Granger causality matrix G (n,n): G[i,j] = ln(Vr/Vf) >= 0, the reduction in
    j's prediction error when i's past is added (influence i->j)."""
    n, T = X.shape
    G = np.zeros((n, n))
    vr = [_ar_resid_var(X[j], [X[j]], order) for j in range(n)]     # j from its own past
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            vf = _ar_resid_var(X[j], [X[j], X[i]], order)           # j from j & i past
            G[i, j] = max(0.0, np.log(vr[j] / vf))
    return G


def dmd_transition(X, rank=8):
    """|A| of the rank-`rank` DMD/VAR(1) operator (x_{t+1}=A x_t) in a PCA subspace, mapped
    back to nodes -- a directed, dynamics-based influence score (n, n)."""
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    Q = U[:, :rank]
    Z = Q.T @ X
    Z0, Z1 = Z[:, :-1], Z[:, 1:]
    A = Z1 @ np.linalg.pinv(Z0)
    An = Q @ A @ Q.T                      # x_{t+1}=An x_t: An[a,b] = influence b->a
    return np.abs(An).T                   # transpose so score[i,j] = influence i->j (Granger convention)


def directed_edge_auc(score, dir_adj):
    """AUC of recovering DIRECTED edges (ordered off-diagonal pairs) from a directed score."""
    from sklearn.metrics import roc_auc_score
    n = dir_adj.shape[0]
    off = ~np.eye(n, dtype=bool)
    y = np.asarray(dir_adj)[off].astype(int)
    s = np.asarray(score)[off]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))
