"""Adversarial scenario evolution with LLaMEA — the difficulty/variation engine.

neuro-dynadojo is an adversarial-development project: rather than hand-write scenarios, let an
LLM MUTATE scenario-generator code and keep the ones that most DISCRIMINATE the method zoo.
LLaMEA (Large Language Model Evolutionary Algorithm, van Stein & Bäck; github.com/XAI-liacs/
LLaMEA, MIT) invents Python code in a (1+1) loop from an evaluation function that returns
(feedback, fitness, error). Here the "algorithm" LLaMEA evolves is a SCENARIO GENERATOR, and
the fitness is how ADVERSARIAL it is — cross-method disagreement (the regime where the field's
methods diverge is the most informative row of the benchmark), optionally targeting a method to
break. This is the LLM-driven form of `bench.adversarial_search`.

Requires: pip install llamea + an LLM key (OPENAI_API_KEY / GEMINI / Ollama).
  python examples/llamea_evolve_scenarios.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.probes import bandpower_embed
from neurodynadojo.algorithms.dynamics import DYNAMICS

FS = 250.0


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


def _methods():
    """Classical + sysid contenders (FMs added when run in emeg-fm's container)."""
    m = {"band-power": lambda X: np.stack([bandpower_embed(x, fs=FS) for x in X])}
    m.update(DYNAMICS)
    return m


def evaluate_scenario(code: str):
    """LLaMEA eval contract -> (feedback, fitness, error). `code` must define
    `scenario(n_per, seed) -> (X (2n,32,T), y, ch_names)`. Fitness = cross-method DISAGREEMENT
    (std of per-method AUC): scenarios where methods diverge are the most informative/adversarial.
    """
    ns = {}
    try:
        exec(code, {"np": np}, ns)
        X, y, _ = ns["scenario"](50, 0)
        assert X.ndim == 3 and set(np.unique(y)) == {0, 1}
    except Exception as e:
        return f"generation failed: {e}", -1.0, str(e)
    aucs = {}
    for name, fn in _methods().items():
        try:
            aucs[name] = _auc(fn(X), y)
        except Exception:
            aucs[name] = 0.5
    spread = float(np.std(list(aucs.values())))               # disagreement = adversarial value
    best = max(aucs, key=aucs.get)
    fb = (f"AUCs: {{{', '.join(f'{k}:{v:.2f}' for k, v in aucs.items())}}}. "
          f"Best={best} ({aucs[best]:.2f}); disagreement(std)={spread:.3f}. "
          f"Make a scenario where methods DISAGREE MORE (one family wins, others fail).")
    return fb, spread, ""


SEED_CODE = '''
import numpy as np
def scenario(n_per, seed):
    """A traveling wave whose DIRECTION is the label (phase methods win, band-power blind)."""
    rng = np.random.default_rng(seed); t = np.arange(1000)/250.0; C=32
    sens = np.random.default_rng(1000+seed).standard_normal((C,3))
    recs=[]
    for lab in (0,1):
        ph=(1.0 if lab==0 else -1.0)*2*np.pi*np.arange(8)/8.0
        for _ in range(n_per):
            S=np.array([np.sin(2*np.pi*10*t+ph[i]) for i in range(8)])
            L=np.random.default_rng().standard_normal((C,8))
            x=L@S; x=x/(x.std()+1e-9) + 0.4*rng.standard_normal((C,1000))
            recs.append((x, lab))
    X=np.stack([r for r,_ in recs]); y=np.array([l for _,l in recs])
    idx=rng.permutation(len(y)); return X[idx], y[idx], [f"E{i}" for i in range(C)]
'''


def main():
    print("LLaMEA adversarial scenario evolution. fitness = cross-method disagreement (std of AUC).")
    fb, fit, err = evaluate_scenario(SEED_CODE)                # sanity: score the seed scenario
    print(f"\n  seed scenario -> fitness={fit:.3f}\n  {fb}")
    try:
        from llamea import LLaMEA
    except ImportError:
        print("\n  [scaffold] `pip install llamea` + set OPENAI_API_KEY to run the evolution loop:")
        print("    opt = LLaMEA(f=evaluate_scenario, api_key=os.environ['OPENAI_API_KEY'],")
        print("                 budget=40, n_parents=1, n_offspring=1)   # (1+1) ES over scenario code")
        print("    best_code, best_fitness = opt.run()")
        print("  -> yields scenario generators the current method zoo most disagrees on (the")
        print("     hardest, most-informative rows to add to the battery).")
        return
    opt = LLaMEA(f=evaluate_scenario, api_key=os.environ["OPENAI_API_KEY"],
                 budget=40, n_parents=1, n_offspring=1)
    best_code, best_fitness = opt.run()
    print(f"\n  evolved scenario fitness={best_fitness:.3f}\n{best_code}")


if __name__ == "__main__":
    main()
