"""System-ID / classical dynamics contenders (SINDy, DMD, DySCo, HMM) produce finite
per-recording features and specialise: SINDy recovers the traveling-wave dynamics, the HMM
state-space recovers the evoked transition — where band-power is blind. Needs the `[sysid]`
extra (pysindy, hmmlearn); skipped otherwise.
"""
import numpy as np
import pytest

pytest.importorskip("pysindy")
pytest.importorskip("hmmlearn")

from neurodynadojo.scenarios import SCENARIOS
from neurodynadojo.algorithms.dynamics import DYNAMICS, sindy_features, hmm_features


def _auc(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                           F, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    return roc_auc_score(y, pr)


def test_feature_shapes_finite():
    X, y, _ = SCENARIOS["wave"](12, 0)
    for name, fn in DYNAMICS.items():
        F = fn(X)
        assert F.shape[0] == len(X) and F.ndim == 2 and np.all(np.isfinite(F))


def test_sindy_wave_and_hmm_evoked():
    Xw, yw, _ = SCENARIOS["wave"](40, 0)
    Xe, ye, _ = SCENARIOS["evoked"](40, 0)
    assert _auc(sindy_features(Xw), yw) > 0.7      # sysid recovers wave dynamics
    assert _auc(hmm_features(Xe), ye) > 0.7        # state-space recovers evoked transition
