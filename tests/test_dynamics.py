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


def test_cfc_pac_is_spectral_blind_but_sindy_reads_it():
    """The LLaMEA-evolved PAC scenario: nonlinear system-ID recovers the gating-phase label while
    band-power stays at chance (matched marginal power) — the confound-aware target it was bred for."""
    from neurodynadojo.probes import bandpower_embed
    X, y, _ = SCENARIOS["cfc_pac"](40, 0)
    bp = np.stack([bandpower_embed(x, fs=250.0) for x in X])
    assert _auc(sindy_features(X), y) > 0.85       # SINDy reads the cross-frequency phase coupling
    assert _auc(bp, y) < 0.65                       # band-power is blind (power spectrum matched)
