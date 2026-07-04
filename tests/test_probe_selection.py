"""Probe-selection (Critique D: a linear probe measures linear decodability only). `cv_auc`
supports 'linear' (LogReg, the platform default), 'kernel' (RBF-SVM), and 'mlp' (small net) --
same StandardScaler+StratifiedKFold protocol for all three, so a method's failure isn't an
artefact of the probe's capacity unless we check with a more expressive one too."""
import numpy as np
import pytest

from neurodynadojo.probes import cv_auc
from neurodynadojo.scenarios import SCENARIOS
from neurodynadojo.probes import bandpower_embed


def test_cv_auc_probes_agree_on_linearly_separable_data():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 40)
    F = np.stack([rng.standard_normal(5) + (3.0 if yi else -3.0) for yi in y])  # trivially linear
    for probe in ("linear", "kernel", "mlp"):
        assert cv_auc(F, y, probe=probe) > 0.95


def test_cv_auc_kernel_recovers_xor_linear_cannot():
    """A probe-selection sanity check: an XOR-style non-linear boundary should be near chance
    under 'linear' but clearly above chance under 'kernel'/'mlp' -- proof the harness actually
    exercises probe capacity, not just re-running the same effective classifier three times."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.uniform(-1, 1, n); x2 = rng.uniform(-1, 1, n)
    y = ((x1 * x2) > 0).astype(int)                          # XOR-like: not linearly separable
    F = np.stack([x1, x2], axis=1)
    assert cv_auc(F, y, probe="linear") < 0.65
    assert cv_auc(F, y, probe="kernel") > 0.85
    assert cv_auc(F, y, probe="mlp") > 0.75


@pytest.mark.parametrize("probe", ["linear", "kernel", "mlp"])
def test_band_power_stays_at_chance_on_cfc_pac_under_every_probe(probe):
    """The headline cfc_pac result must not be an artefact of probe choice: band-power should
    stay near chance whether probed linearly or non-linearly."""
    X, y, _ = SCENARIOS["cfc_pac"](40, 0)
    bp = np.stack([bandpower_embed(x, fs=250.0) for x in X])
    assert cv_auc(bp, y, probe=probe) < 0.65
