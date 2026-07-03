"""The HBN-grounded scenario battery must have DISTINCT winners: band-power decodes the
`spectral` scenario but is blind to `evoked` (a phase contrast with the same power spectrum),
while a waveform/phase feature decodes `evoked`. That dissociation is the point of the battery.
"""
import numpy as np

from neurodynadojo.scenarios import SCENARIOS
from neurodynadojo.probes import bandpower_embed


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


def test_scenarios_shape_and_labels():
    for name, fn in SCENARIOS.items():
        X, y, ch = fn(20, 0)
        assert X.shape == (40, 32, 1000) and set(np.unique(y)) == {0, 1} and len(ch) == 32
        assert np.all(np.isfinite(X))


def test_bandpower_wins_spectral_but_is_blind_to_evoked():
    Xs, ys, _ = SCENARIOS["spectral"](50, 0)
    Xe, ye, _ = SCENARIOS["evoked"](50, 0)
    bp = lambda X: np.stack([bandpower_embed(x, fs=250.0) for x in X])
    raw = lambda X: X[:, :, ::4].reshape(len(X), -1)
    assert _auc(bp(Xs), ys) > 0.85                      # spectral is band-power's home turf
    assert _auc(bp(Xe), ye) < 0.65                      # evoked is a phase contrast -> band-power blind
    assert _auc(raw(Xe), ye) > 0.85                     # ...but the waveform carries it
