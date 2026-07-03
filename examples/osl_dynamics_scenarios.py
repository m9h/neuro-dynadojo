"""osl-dynamics TDE-HMM across the scenario battery — runs in the oracle-osl container.

Scores osl-dynamics' real TDE-HMM (the OSL M/EEG state-space method) on every scenario, with the
same cross-validated linear probe as the rest of the zoo, so its row is directly comparable to
band-power / SINDy / DySCo / the hmmlearn Gaussian-HMM stand-in / the FMs.

  bash examples/run_osl_container.sh
"""
import sys

sys.path.insert(0, "/ndd/src")

import numpy as np
from neurodynadojo.scenarios import SCENARIOS
from neurodynadojo.algorithms.osl_dynamics import osl_hmm_features


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


def main():
    n_per = int(__import__("os").environ.get("NDD_NPER", "40"))
    print(f"osl-dynamics TDE-HMM x scenario battery (LogReg AUC, chance .50; n_per={n_per})\n")
    for s in SCENARIOS:
        X, y, _ = SCENARIOS[s](n_per, 0)
        try:
            F = osl_hmm_features(X)
            print(f"  {s:14s} osl-TDE-HMM  {_auc(F, y):.3f}", flush=True)
        except Exception as e:
            print(f"  {s:14s} FAILED: {repr(e)[:90]}", flush=True)


if __name__ == "__main__":
    main()
