"""Negative controls — the first thing a skeptical reviewer should check. If the harness leaked
the label through anything other than the intended signal (row order, class-balanced artefacts,
a probe that memorises indices), a decode would survive label shuffling. It must not: permuting y
must drop every method to chance, and a pure-noise scenario must be undecodable by everyone. These
guard against the failure mode where an impressive AUC is an evaluation bug, not a real effect."""
import numpy as np
import pytest

pytest.importorskip("pysindy")

from neurodynadojo.scenarios import SCENARIOS
from neurodynadojo.algorithms.dynamics import sindy_features


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


def test_shuffled_label_kills_the_decode():
    """SINDy reads cfc_pac (AUC > .8); with y permuted it must collapse to chance (< .6)."""
    X, y, _ = SCENARIOS["cfc_pac"](40, 0)
    F = sindy_features(X)
    assert _auc(F, y) > 0.8                                   # the real, headline decode
    rng = np.random.default_rng(0)
    shuffled = [_auc(F, rng.permutation(y)) for _ in range(5)]
    assert np.mean(shuffled) < 0.6                            # no signal survives label shuffling


def test_pure_noise_is_undecodable():
    """A scenario that is only 1/f background with a random label must be at chance for band-power."""
    from neurodynadojo.probes import bandpower_embed
    from neurodynadojo.scenarios import _fixed_sensors, _bg_field, _stack, N_CH, T
    rng = np.random.default_rng(0); sens = _fixed_sensors(0)
    recs = [( _bg_field(sens, T, 1.0, rng), int(i % 2 == 0)) for i in range(80)]  # label independent of signal
    X, y, _ = _stack(recs, rng)
    bp = np.stack([bandpower_embed(x, fs=250.0) for x in X])
    assert 0.35 < _auc(bp, y) < 0.65                          # chance: nothing to decode
