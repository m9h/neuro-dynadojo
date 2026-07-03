"""osl-dynamics contender — the Oxford OSL state-space family, PROPER (not the hmmlearn stand-in).

`hmm_features` in dynamics.py uses a plain Gaussian HMM on a PCA trajectory as a lightweight proxy
for the osl-dynamics / DyNeMo family. This module runs the real thing: osl-dynamics' signature
**TDE-HMM** (time-delay-embedded Hidden Markov Model), the method behind the OSL M/EEG dynamics
papers. TDE embedding lets each state model cross-channel *spectral/phase* structure (not just
instantaneous covariance), so it can pick up dynamics a Gaussian HMM on raw PCA cannot.

It needs osl-dynamics (TensorFlow) + its `fsl` dependency, which only the `neurojax/oracle-osl`
container provides — so this runs via `examples/run_osl_container.sh`, NOT the main venv. The
`fsl` import (used only for parcellation/plotting we never touch) is stubbed so the sensor-space
HMM path imports cleanly.

Contract matches the rest of the zoo: `(N, C, T) -> (N, D)`  (per-recording state fractional
occupancy + switching rate).
"""
from __future__ import annotations

import sys
import types

import numpy as np


def _stub_fsl():
    """osl-dynamics imports `fsl` for parcellation/plotting; stub it (we stay in sensor space)."""
    for m in ("fsl", "fsl.wrappers", "fsl.utils", "fsl.utils.image", "fsl.data", "fsl.data.image"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["fsl"].wrappers = sys.modules["fsl.wrappers"]


def osl_hmm_features(X, n_states=6, n_embeddings=15, n_pca=20, n_epochs=10, fs=250.0):
    """Group TDE-HMM over all recordings; per-recording feature = state fractional occupancy plus
    switching rate. Runs inside the oracle-osl container (TensorFlow + osl-dynamics)."""
    _stub_fsl()
    from osl_dynamics.data import Data
    from osl_dynamics.models.hmm import Config, Model

    data = Data([np.ascontiguousarray(x.T.astype("float32")) for x in X],  # list of (T, C)
                sampling_frequency=fs, use_tfrecord=False)
    data.prepare({"tde_pca": {"n_embeddings": n_embeddings, "n_pca_components": n_pca},
                  "standardize": {}})
    config = Config(n_states=n_states, n_channels=data.n_channels, sequence_length=100,
                    learn_means=False, learn_covariances=True, batch_size=16,
                    learning_rate=0.01, n_epochs=n_epochs)
    model = Model(config)
    model.fit(data)
    alphas = model.get_alpha(data)                                  # list of (T_i, n_states)
    if not isinstance(alphas, list):
        alphas = [alphas]
    feats = []
    for a in alphas:
        a = np.asarray(a)
        occ = a.mean(0)                                             # fractional occupancy
        state = a.argmax(1)
        switch = np.mean(state[1:] != state[:-1]) if len(state) > 1 else 0.0
        feats.append(np.concatenate([occ, [switch]]))
    return np.asarray(feats)


OSL = {"osl-TDE-HMM": osl_hmm_features}
