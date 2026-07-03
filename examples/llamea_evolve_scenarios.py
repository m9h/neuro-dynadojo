"""Adversarial scenario evolution with LLaMEA — the difficulty/variation engine.

neuro-dynadojo is an adversarial-development project: rather than hand-write scenarios, let an
LLM MUTATE scenario-generator code and keep the ones that most DISCRIMINATE the method zoo.
LLaMEA (Large Language Model Evolutionary Algorithm, van Stein & Bäck; github.com/XAI-liacs/
LLaMEA, MIT) runs a (mu+lambda) loop that writes/refines a Python CLASS from an evaluation
function. Here the evolved "algorithm" is a SCENARIO GENERATOR and the fitness is how
ADVERSARIAL it is — cross-method DISAGREEMENT (std of per-method AUC): the regime where the
field's methods most diverge is the most informative row of the benchmark. The winner is a
new, harder scenario to fold into scenarios.py.

Backends: LLaMEA ships Gemini/OpenAI/Ollama. This file adds a tiny `Anthropic_LLM` so the loop
runs on ANTHROPIC_API_KEY. Requires: pip install llamea anthropic.

  python examples/llamea_evolve_scenarios.py            # real (1+1) evolution, Claude backend
  NDD_BUDGET=8 NDD_OUT=winner.py python examples/llamea_evolve_scenarios.py

Real runs (both saved verbatim, both reproduce):
  - claude-sonnet-5, budget 12  -> `examples/evolved_scenario.py`  (fitness 0.095): a traveling-
    wave-DIRECTION scenario carried purely by cross-channel phase-lag with a matched power
    spectrum — band-power blind (AUC 0.52), SINDy reads it (0.66). Clean single-winner.
  - claude-opus-4-8, budget 30  -> `examples/evolved_scenario_opus.py` (fitness 0.258, BEATS the
    hand-tuned battery's ~0.21): a theta-gamma PAC contrast. Opus climbs the objective far better,
    but the win exposes a lesson — its class-0 phase-scramble leaked a gamma-power confound, so
    band-power/DMD ace it (1.00) rather than the intended nonlinearity-only methods. See below.

Eval runs are wrapped in a hard SIGALRM timeout (LLaMEA's SequentialBackend ignores its own
eval_timeout), so a pathological generated `generate()` can't hang the loop.

Two fitness MODES (env NDD_MODE): `disagree` = raw cross-method disagreement (std of AUC), and
`targeted` (default) = margin of the best GENUINE-DYNAMICS method (SINDy/DySCo/HMM) over the best
SPECTRAL method (band-power/DMD). The targeted mode is CONFOUND-AWARE: a scenario scores only if it
beats EVERY spectral method, so a leaked power/amplitude feature is penalized rather than rewarded.

  NDD_MODE=targeted NDD_MODEL=claude-opus-4-8 NDD_BUDGET=30 python examples/llamea_evolve_scenarios.py

A targeted Opus run (`examples/evolved_scenario_targeted.py`, margin +0.47) is the payoff: under
raw disagreement Opus produced a "PAC" scenario whose envelope-scramble LEAKED gamma power so
band-power aced it; re-run under the targeted margin the SAME model was forced to build a genuinely
spectrum-matched 6->40 Hz phase-coupling contrast (high-pass the coupled HF so no envelope bleeds
into the low band, renormalize HF power, match channel variance) — band-power AND DMD at chance
(~.50), only SINDy recovers it (1.00), and it generalizes to held-out seeds. Fitness shaping turned
a confound-gamed win into a legitimate ground-truth scenario.

Eval runs are wrapped in a hard SIGALRM timeout (LLaMEA's SequentialBackend ignores its own
eval_timeout), so a pathological generated `generate()` can't hang the loop.
"""
import os, sys, textwrap, signal, contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neurodynadojo.probes import bandpower_embed
from neurodynadojo.algorithms.dynamics import DYNAMICS

FS, N_CH, T = 250.0, 32, 1000


# ── the method zoo the evolved scenario must split (classical + sysid; FMs live in the container) ──
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
    m = {"band-power": lambda X: np.stack([bandpower_embed(x, fs=FS) for x in X])}
    m.update(DYNAMICS)                                    # SINDy, DMD, DySCo, HMM
    return m


@contextlib.contextmanager
def _time_limit(seconds):
    """Hard wall-clock cap on LLM-authored code (SequentialBackend ignores LLaMEA's eval_timeout)."""
    def _raise(signum, frame):
        raise TimeoutError(f"eval exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _run_scenario(code, n_per=50, seed=0):
    """Exec generated code, find a Scenario class (or a `scenario` fn), return (X, y)."""
    ns = {}
    exec(code, {"np": np, "__builtins__": __builtins__}, ns)
    if "Scenario" in ns:
        X, y, _ = ns["Scenario"]().generate(n_per, seed)
    elif "scenario" in ns:
        X, y, _ = ns["scenario"](n_per, seed)
    else:
        raise ValueError("code defines neither `class Scenario` nor `def scenario`")
    X, y = np.asarray(X, float), np.asarray(y).ravel()
    assert X.shape[1:] == (N_CH, T) and set(np.unique(y)) == {0, 1}, f"bad shapes {X.shape}"
    return X, y


# ── fitness modes ──────────────────────────────────────────────────────────────────────────────
# The Opus run exposed that raw cross-method disagreement is gamed by an UNINTENDED easy feature
# (a "PAC-only" scenario whose class-0 scramble leaked gamma power → band-power aced it). The
# targeted objective is confound-proof: reward the margin of the best GENUINE-DYNAMICS method over
# the best SPECTRAL/AMPLITUDE method, so a scenario only scores if it beats EVERY spectral method
# (band-power AND DMD — both leaked last time), not just one.
SPECTRAL = ("band-power", "DMD")                          # amplitude/linear baselines that must stay near chance
DYNAMICS_FAMILY = ("SINDy", "DySCo", "HMM")               # genuine nonlinear / state-space / dynamic-FC


def _fitness(aucs):
    """Return (score, mode_label, guidance). Mode from NDD_MODE: 'targeted' (default) or 'disagree'."""
    mode = os.environ.get("NDD_MODE", "targeted")
    if mode == "disagree":
        return float(np.std(list(aucs.values()))), "disagreement std", (
            "make ONE method win decisively while the others stay near chance (raises the std)")
    best_dyn = max(DYNAMICS_FAMILY, key=lambda m: aucs.get(m, 0.5))
    best_spec = max(SPECTRAL, key=lambda m: aucs.get(m, 0.5))
    margin = aucs.get(best_dyn, 0.5) - aucs.get(best_spec, 0.5)
    guidance = (f"TARGETED: a genuine-dynamics method ({best_dyn}={aucs.get(best_dyn):.2f}) must beat "
                f"EVERY spectral/amplitude method (best spectral {best_spec}={aucs.get(best_spec):.2f}). "
                f"margin={margin:+.2f}. To improve: keep band-power AND DMD at chance (~.50) — do NOT "
                f"let any power/amplitude/linear feature leak the label (watch envelope scrambles, "
                f"band-power imbalance) — while making the label readable ONLY through nonlinear phase / "
                f"state-transition / dynamic-connectivity structure (e.g. cross-frequency phase coupling "
                f"with matched marginal power, a state-sequence contrast, directed phase flow).")
    return float(margin), f"dynamics-minus-spectral margin ({best_dyn}−{best_spec})", guidance


def evaluate(individual, logger=None):
    """LLaMEA v1.2 contract: score `individual.code`, set fitness/feedback/error, return it."""
    try:
        with _time_limit(45):                            # cap arbitrary LLM-authored generate()
            X, y = _run_scenario(individual.code)
    except Exception as e:
        individual.set_scores(-1.0, f"scenario generation/shape failed: {e!r}", e)
        return individual
    aucs = {}
    for name, fn in _methods().items():
        try:
            with _time_limit(30):                         # cap each method's feature extraction
                aucs[name] = float(_auc(fn(X), y))
        except Exception:
            aucs[name] = 0.5
    score, label, guidance = _fitness(aucs)
    fb = (f"Per-method AUC (chance .50): { {k: round(v, 2) for k, v in aucs.items()} }. "
          f"{label}={score:.3f}. {guidance}")
    individual.set_scores(score, fb, None)
    return individual


ROLE_PROMPT = ("You are a computational-neuroscience benchmark designer. You write NumPy code that "
               "synthesises labelled EEG-like recordings engineered so that ONE family of analysis "
               "methods can decode the label and the others cannot.")

TASK_PROMPT = textwrap.dedent(f"""
    Design an adversarial EEG scenario generator for the neuro-dynadojo method-comparison battery.

    Write a Python class named `Scenario` with a method
        def generate(self, n_per, seed):
    that returns (X, y, ch_names) where:
      - X is a float ndarray of shape (2*n_per, {N_CH}, {T})  ({N_CH} channels, {T} samples @ {FS:.0f} Hz),
      - y is an int array of {2}*n_per binary labels (0/1), balanced,
      - ch_names is a list of {N_CH} channel-name strings.
    Use only numpy (imported as np). Seed all randomness from `seed` (np.random.default_rng).

    The recordings are scored by a fixed zoo of methods: band-power and DMD (SPECTRAL / amplitude /
    linear), and SINDy, DySCo, HMM (genuine DYNAMICS: nonlinear system-ID, dynamic connectivity,
    state-space). Your goal (TARGETED): make a genuine-dynamics method decode the label while EVERY
    spectral/amplitude method — band-power AND DMD — stays at chance (~.50). The label must live
    ONLY in nonlinear-phase / state-transition / dynamic-connectivity structure, with MATCHED
    marginal power between classes. Guard against leakage: any per-class difference in band power,
    total amplitude, or channel variance hands the win to band-power/DMD and FAILS the objective (a
    common trap: scrambling an envelope changes its power). Prefer mechanisms like cross-frequency
    PHASE coupling with identical band powers, a hidden state-SEQUENCE contrast, or directed phase
    flow. Add realistic spatially-correlated 1/f background, and keep channel identity FIXED across
    trials (a fixed sensor geometry / lead field), or all methods collapse to chance.

    Return ONLY the class in a single ```python code block.""")

EXAMPLE_PROMPT = textwrap.dedent('''
    An example of the required response format (a same-spectrum PHASE contrast — band-power blind,
    waveform/state-space methods win):

    ```python
    import numpy as np
    class Scenario:
        """Evoked 10 Hz Gabor at phase 0 vs pi/2 — identical power spectrum, different waveform."""
        def generate(self, n_per, seed):
            rng = np.random.default_rng(seed); C, T, fs = 32, 1000, 250.0
            t = np.arange(T) / fs
            geo = np.random.default_rng(1000).standard_normal((C, 6))      # FIXED lead field
            S_topo = np.random.default_rng(2000).standard_normal(6)
            X, y = [], []
            for lab in (0, 1):
                phase = 0.0 if lab == 0 else np.pi / 2
                for _ in range(n_per):
                    gabor = np.exp(-((t - 0.28) ** 2) / (2 * 0.04 ** 2)) * np.sin(2*np.pi*10*t + phase)
                    src = np.outer(S_topo, gabor)                          # (6, T)
                    sig = geo @ src
                    bg = geo @ (0.5 * rng.standard_normal((6, T)))          # spatially-correlated noise
                    x = sig / (sig.std() + 1e-9) + bg / (bg.std() + 1e-9)
                    X.append(x); y.append(lab)
            X = np.stack(X); idx = rng.permutation(len(y))
            return X[idx], np.array(y)[idx], [f"E{i}" for i in range(C)]
    ```''')


from llamea import LLM as _LLM


class Anthropic_LLM(_LLM):
    """Minimal LLaMEA LLM backend over the Anthropic Messages API (Claude). Inherits the base
    regex code/name/description extractors and sample_solution; only `query` is implemented."""
    def __init__(self, api_key, model="claude-sonnet-5", temperature=0.9, max_tokens=12000):
        import anthropic
        super().__init__(api_key, model)                   # sets .code_pattern/.name_pattern/etc.
        self.client = anthropic.Anthropic(api_key=api_key)
        self.temperature, self.max_tokens = temperature, max_tokens

    def query(self, session):
        sys_txt = "\n\n".join(m["content"] for m in session if m.get("role") == "system")
        msgs, last = [], None
        for m in session:                                  # fold system into the conversation
            if m.get("role") == "system":
                continue
            role = "assistant" if m.get("role") == "assistant" else "user"
            if role == last and msgs:                      # merge consecutive same-role turns
                msgs[-1]["content"] += "\n\n" + m["content"]
            else:
                msgs.append({"role": role, "content": m["content"]}); last = role
        if not msgs or msgs[0]["role"] != "user":          # Anthropic requires a leading user turn
            msgs.insert(0, {"role": "user", "content": "Begin."})
        if sys_txt:                                        # prepend system as a top text block
            msgs[0]["content"] = sys_txt + "\n\n" + msgs[0]["content"]
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, messages=msgs)
        return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")


def main():
    from llamea import LLaMEA
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("set ANTHROPIC_API_KEY")
    budget = int(os.environ.get("NDD_BUDGET", "12"))
    llm = Anthropic_LLM(api_key=key, model=os.environ.get("NDD_MODEL", "claude-sonnet-5"))
    print(f"LLaMEA adversarial scenario evolution — Claude backend ({llm.model}), budget={budget}.")
    print("fitness = cross-method disagreement (std of per-method AUC); (1+1) evolution.\n")
    fmt = ("Respond concisely: a one-line description then the class in ONE ```python block. "
           "Keep the docstring to one line and the class under ~60 lines so it is never truncated.\n"
           "# Description: <one line>\n# Code:\n```python\n<code>\n```")
    opt = LLaMEA(f=evaluate, llm=llm, role_prompt=ROLE_PROMPT, task_prompt=TASK_PROMPT,
                 example_prompt=EXAMPLE_PROMPT, output_format_prompt=fmt,
                 n_parents=1, n_offspring=1, elitism=True,
                 budget=budget, minimization=False, log=False, experiment_name="ndd-scenario")
    best = opt.run()
    print("\n" + "=" * 78 + f"\nBEST scenario  fitness(disagreement std)={best.fitness:.3f}\n")
    print(best.feedback + "\n\n" + best.code)
    out = os.environ.get("NDD_OUT")
    if out and best.code:
        with open(out, "w") as fh:
            fh.write(f"# LLaMEA-evolved scenario. fitness(disagreement std)={best.fitness:.3f}\n"
                     f"# {best.feedback}\n\n{best.code}\n")
        print(f"\nwrote winner -> {out}")


if __name__ == "__main__":
    main()
