"""CEBRA latent-embedding contender produces finite per-recording features and, trained purely
self-supervised (no labels), still separates the traveling-wave scenario under a linear probe —
a genuine third method family beside classical FC and system-ID. Needs the `[latent]` extra
(cebra); skipped otherwise. (osl-dynamics TDE-HMM is validated separately in the oracle-osl
container — it can't run in the plain test venv.)"""
import numpy as np
import pytest

pytest.importorskip("cebra")

from neurodynadojo.scenarios import SCENARIOS
from neurodynadojo.algorithms.latent import cebra_features


def _auc(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                           F, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    return roc_auc_score(y, pr)


def test_cebra_features_finite_and_reads_wave():
    X, y, _ = SCENARIOS["wave"](20, 0)
    F = cebra_features(X, dim=8, iters=150)
    assert F.shape == (len(X), 16) and np.all(np.isfinite(F))
    assert _auc(F, y) > 0.7            # self-supervised embedding still carries wave dynamics
