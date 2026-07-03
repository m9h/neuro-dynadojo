# neuro-dynadojo

[![tests](https://github.com/m9h/neuro-dynadojo/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/neuro-dynadojo/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A ground-truth benchmark for M/EEG dynamics — connectivity recovery, directed flow,
traveling waves, and foundation-model probing — under realistic sensor confounds.**

![Netsim confound battery and SNR scaling: no method wins across confounds](figures/preview.png)

Real M/EEG has no known connectivity ground truth, so a method's claim to *recover
structure* can only be validated on simulation. [FSLNets/netsim](https://www.fmrib.ox.ac.uk/datasets/netsim/)
(Smith et al. 2011) established that discipline for fMRI; [DynaDojo](https://github.com/DynaDojo/dynadojo)
(Bhamidipaty et al. 2023) generalised it into an extensible `System × Algorithm × Challenge`
platform with scaling diagnostics. **neuro-dynadojo ports that discipline to
electrophysiology** and adds the three things fMRI never had — a tunable dynamical
**regime**, conduction **delays**, and an observation model with **volume conduction** —
then extends the `Algorithm` slot so a frozen **EEG foundation model** can be evaluated the
same way as a classical estimator.

## Why

- **Methods are proposed without comparison.** neuro-dynadojo scores correlation, partial
  correlation, coherence / imaginary coherence / wPLI / PLV, Granger, and DMD on the *same*
  ground truth under the *same* confounds — so "which method wins" is a table, not an assertion.
- **Foundation models are evaluated on downstream accuracy, not mechanistic content.** A model
  that classifies seizures at 95% may not encode *coupling direction* or tell a *traveling wave*
  from independent oscillators. On simulated data the answer is known: freeze the model, embed
  the recording, and linearly probe the embedding for the generative factor. This complements
  real-data identity audits ([fmscope](https://github.com/Jimmy110101013/fmscope), *The Identity
  Trap in EEG Foundation Models*; Tang et al. 2026) — there the confound is unknown; here it is
  known by construction.
- **One generative family overfits.** netsim used a single linear-DCM+balloon model; methods
  overfit it. neuro-dynadojo ships amplitude-coupled, phase-coupled, and traveling-wave regimes,
  plus a **real diffusion-MRI connectome** (Desikan-68), so a method can't win by memorising one physics.

## Install

```bash
git clone https://github.com/m9h/neuro-dynadojo
cd neuro-dynadojo
pip install -e .            # core: numpy, scipy, scikit-learn
pip install -e .[fm]        # + torch, braindecode  (load & probe a foundation model)
```

## Quickstart — recover a connectome

```python
import neurodynadojo as ndd

sys = ndd.NetsimSystem(seed_struct=0, back=0.3, leak=0.8, snr=6.0)   # directed net, volume conduction + noise
obs, C = sys.simulate(seed=0)                                        # (n_nodes, n_times), ground-truth connectome
score = ndd.partialcorr_fc(obs)
auc   = ndd.edge_recovery_auc(score, sys.undirected_truth())        # how well structure is recovered
```

## Quickstart — probe a foundation model

```python
import neurodynadojo as ndd
from neurodynadojo.probes import probe_factor, braindecode_embed, bandpower_embed

# bring your own frozen FM (e.g. a braindecode model, classifier head removed):
#   embed = lambda obs: braindecode_embed(model, obs)          # (n_ch, n_times) -> vector
embed = lambda obs: bandpower_embed(obs, fs=250.0)             # spectral baseline (the floor to beat)

# does the representation encode conduction DELAY, in sensor space, under volume conduction?
res = probe_factor(embed, ndd.HopfNetworkSystem, factor="velocity",
                   values=[3, 5, 8, 12], task="regress", space="sensor", leak=0.8, snr=6.0)
print(res)   # {'metric': 'r', 'r': ..., 'r2': ..., 'n': ...}
```

Generators emit `(n_ch, n_times)` sensor recordings (braindecode's input shape) with known
factors — coupling, delay, regime, wavenumber, frequency, flow direction — so any FM can be
scored on whether its embedding recovers the physics. See
[`examples/fm_probe.py`](examples/fm_probe.py) (spectral baseline — the floor to beat),
[`examples/fm_probe_braindecode.py`](examples/fm_probe_braindecode.py) (any braindecode model),
and [`examples/fm_probe_bendr.py`](examples/fm_probe_bendr.py) (a **real pretrained FM**, BENDR).

### First real-FM result (BENDR) — and an honest caveat

Probing a frozen, pretrained **BENDR** (loaded from the HF Hub via braindecode) for the
generative factors of our synthetic sensor recordings, factor-recovery |r| vs the spectral
baseline — across three montages, incl. BENDR's **exact** 19-EEG pretraining montage
(`montage="bendr"`, standard_1005 positions + the SCALE channel):

| factor | BENDR (random) | BENDR (10-20) | BENDR (exact) | spectral baseline |
|---|---|---|---|---|
| frequency | 0.23 | 0.59 | 0.12 | **0.98** |
| coupling | 0.30 | 0.03 | 0.18 | **0.93** |
| velocity | 0.32 | 0.43 | 0.22 | **0.97** |
| phase-lag | 0.15 | 0.21 | 0.30 | **0.44** |
| *embedding responsiveness* | *0.56* | *1.06* | *0.86* | — |

On a **generic** montage BENDR's embedding is nearly *invariant* to the input (an
**out-of-distribution collapse**, not a verdict on the model). A **real montage** lifts the
collapse (responsiveness 0.56 → 1.06) — but even BENDR's **exact** pretraining montage does **not**
close the gap, and more real spatial channels (generic-20) beat the exact-19. So the montage is not
the bottleneck.

Adding **realistic signal statistics** — a spatially-correlated **1/f cortical background**
(`background=`) — poses the *fair* question. The scaled probe (**n=100 per factor**, six generative
factors, on BENDR's exact montage with 1/f background, GPU-batched) gives the robust verdict:

| factor | BENDR \|r\| | spectral baseline \|r\| | Δ |
|---|---|---|---|
| frequency | 0.32 | 0.49 | −0.17 |
| coupling | 0.16 | 0.58 | −0.43 |
| delay (velocity) | 0.30 | 0.92 | −0.62 |
| regime | 0.25 | 0.77 | −0.52 |
| phase-lag | 0.04 | 0.58 | −0.54 |
| wavenumber | 0.38 | 0.66 | −0.27 |

**BENDR's frozen embedding trails a trivial band-power baseline on every one of six generative
factors.** It carries *some* signal (|r| 0.04–0.38, above chance for most) but consistently and
substantially less than the spectrum — even on its exact montage with in-distribution 1/f input.
So the fair, scaled, mechanistic probe reaches a clean conclusion: *an off-the-shelf frozen EEG-FM
embedding does not encode these generative dynamics factors as well as band-power does.* That is
the finding the benchmark exists to produce. Run
[`examples/fm_probe_bendr_scaled.py`](examples/fm_probe_bendr_scaled.py) (`pip install -e .[fm,mne]`).

## What the benchmark shows (the honest through-line)

- **No method or challenge wins everywhere.** The identifiable ground truth *and* the winning
  method shift with regime: correlation recovers adjacency under amplitude/phase coupling; a
  traveling wave hands adjacency to **partial correlation**, exposes **directed flow** (DMD, Granger),
  and makes the **wavenumber** trivially readable.
- **Lag-robust ≠ better recovery.** Imaginary coherence / wPLI are correct (a controlled
  quarter-cycle lag → ≈1.0) yet sit near chance for structure recovery, because node-coupled
  dynamics imprint connectivity on the *zero-lag* component.
- **Volume conduction and shared input are the dominant confounds, and they interact** — leakage
  defeats partial correlation's immunity to a shared global input. An adversarial search over the
  confound grid rediscovers this automatically.
- **One thing survives a real recording:** the **wavenumber** is recovered at ceiling across
  leakage *and* every sensor SNR down to 0 dB.
- **Scaling laws:** recovery falls as node count grows, rises then saturates with duration, climbs with SNR.
- **Real connectomes are much harder than synthetic.** On the Desikan-68 diffusion-MRI graph,
  clean recovery is only ~0.67 (matching empirical SC–FC) and **volume conduction alone collapses it
  to chance** — the synthetic modular graphs (0.9+) systematically overstate recoverability.

An interactive figure of the regime × challenge identifiability matrix, confound battery, and
scaling laws is in [`figures/`](figures/).

### Multi-FM leaderboard (interop with emeg-fm + fmscope)

A three-project pipeline scores foundation models on the *known* generative factors: **emeg-fm**
(the NeuroTechX Atlas) loads the EEG-FM zoo through one montage-agnostic extractor (in its NGC
container, braindecode 1.5.2; hand adapters for REVE/LaBraM); **neuro-dynadojo** supplies the labeled
synthetic recordings; **fmscope** scores each frozen embedding with its canonical linear probe
(`eval_seed`, balanced accuracy). Five real pretrained FMs vs band-power (chance = 0.50):

| model | freq | coupling | delay | regime | phase | wavenum | mean |
|---|---|---|---|---|---|---|---|
| **REVE** | **0.67** | **0.73** | **0.83** | **0.95** | **0.60** | 0.50 | **0.71** |
| baseline (band-power) | 0.57 | 0.68 | 0.80 | 0.83 | 0.53 | 0.33 | 0.62 |
| CBraMod | 0.43 | 0.72 | 0.54 | 0.90 | 0.60 | 0.47 | 0.61 |
| BIOT | 0.58 | 0.62 | 0.71 | 0.88 | 0.38 | 0.34 | 0.58 |
| LaBraM | 0.39 | 0.61 | 0.45 | 0.90 | 0.63 | **0.51** | 0.58 |
| LUNA | 0.55 | 0.56 | 0.51 | 0.92 | 0.52 | 0.41 | 0.58 |

**REVE (0.71) is the standout — the one FM that clearly beats band-power (0.62), and it wins on all
six factors, including the spectral ones (frequency, delay) where every other FM loses.** REVE is the
Atlas's top *cognitive* FM; here it's confirmed on known mechanistic ground truth. The rest are
factor-dependent: **every model beats band-power on `regime`** (0.88–0.95 vs 0.83), and CBraMod/LaBraM
also beat it on the dynamics/spatial factors (coupling, phase-lag, wavenumber), while band-power holds
the pure-spectral factors against all but REVE. This is the mechanistic form of the Atlas's finding —
**classical wins spectral, FMs win dynamics/cognition** — with REVE the clear leader. Run inside
emeg-fm's container: [`examples/run_leaderboard_container.sh`](examples/run_leaderboard_container.sh).
The stock-venv bridge [`examples/multi_fm_leaderboard.py`](examples/multi_fm_leaderboard.py) scores
whatever loads directly.

### HBN-grounded scenario battery (netsim for the EEG-FM era)

As HCP's connectome grounded FSLNets/netsim, real **HBN** statistics (1/f exponent −1.13, evoked
latency ~0.28 s, band structure) ground a battery of simulation scenarios — each modeled on an HBN
task regime and *engineered so a different method family wins*. Every method (classical + the FM
zoo) is scored on every scenario (fmscope LP balanced accuracy, chance 0.50):

| method | spectral | evoked | wave | naturalistic | burst |
|---|---|---|---|---|---|
| band-power | **1.00** | 0.42 | 0.50 | 0.61 | 0.47 |
| phase-conn | 0.66 | 0.41 | **1.00** | 0.49 | 0.46 |
| BIOT | 1.00 | 0.47 | 0.55 | **0.99** | **0.98** |
| CBraMod | 1.00 | 1.00 | 1.00 | 0.64 | 0.86 |
| LUNA | 1.00 | 1.00 | 0.88 | 0.72 | 0.88 |
| REVE | 1.00 | **1.00** | **1.00** | 0.67 | 0.82 |
| LaBraM | 1.00 | 1.00 | 1.00 | 0.54 | 0.52 |

This is a **method-comparison platform, not an FM comparator** — the contenders span the whole
neural-dynamics landscape: classical spectral/connectivity, **system-ID** (SINDy, DMD/Koopman),
**dynamic connectivity** (DySCo), **state-space** (HMM — the osl-dynamics/DyNeMo family), and FMs.
Adding those rows (`[sysid]` extra) shows every scenario crowns a *different method family*:

| method (family) | spectral | evoked | wave | naturalistic | burst |
|---|---|---|---|---|---|
| band-power (spectral) | **1.00** | 0.46 | 0.51 | 0.62 | 0.56 |
| DMD (Koopman) | 0.86 | 0.49 | 0.43 | 0.56 | 0.33 |
| SINDy (system-ID) | 0.41 | 0.47 | **0.97** | 0.33 | 0.41 |
| DySCo (dynamic-FC) | 0.68 | 0.54 | 0.56 | 0.59 | **0.90** |
| HMM (state-space) | 0.48 | **0.98** | 0.59 | 0.56 | 0.63 |
| phase-conn | 0.38 | 0.43 | **1.00** | 0.46 | 0.50 |
| BIOT (FM) | 1.00 | 0.47 | 0.55 | **0.99** | **0.98** |
| CBraMod (FM) | 1.00 | **1.00** | **1.00** | 0.64 | 0.86 |
| REVE (FM) | 1.00 | **1.00** | **1.00** | 0.67 | 0.82 |
| LaBraM (FM) | 1.00 | 1.00 | 1.00 | 0.54 | 0.52 |
| LUNA (FM) | 1.00 | 1.00 | 0.88 | 0.72 | 0.88 |

**No method family dominates — each scenario has a different winner:** `spectral`→band-power,
`wave`→**SINDy** (system-ID recovers the governing wave dynamics) *and* phase-conn, `evoked`→**HMM**
(state-space catches the evoked transition) *and* the phase-FMs, `burst`→**DySCo** (dynamic-FC
catches the reconfiguration) *and* BIOT, `naturalistic`→**BIOT** (cross-frequency). And the **FMs
themselves split**: CBraMod/REVE/LaBraM own phase/waveform structure while BIOT owns cross-frequency
— a split a single-number leaderboard hides. This is what a netsim-style battery is *for*.

`neuro-dynadojo` is also an **adversarial-development** project: the scenario suite is meant to keep
evolving harder, more-discriminating scenarios (via `bench.adversarial_search`, and an LLM-driven
[LLaMEA](https://github.com/XAI-liacs/LLaMEA) loop that mutates scenario code to maximize
cross-method disagreement — see [`examples/llamea_evolve_scenarios.py`](examples/llamea_evolve_scenarios.py)).
Two **real runs** (both saved verbatim, both reproduce):
- **sonnet, budget 12** ([`examples/evolved_scenario.py`](examples/evolved_scenario.py), fitness
  0.095): rediscovered — unprompted — a traveling-wave-**direction** scenario carried purely by
  cross-channel phase-lag with a *matched power spectrum* — band-power blind (AUC 0.52), system-ID
  (SINDy) reads it (0.66). A clean single-winner case.
- **Opus, budget 30, raw disagreement** ([`examples/evolved_scenario_opus.py`](examples/evolved_scenario_opus.py),
  fitness 0.258 — **beats the hand-tuned battery's ~0.21**): a theta–gamma PAC contrast. Opus
  climbs the objective far more effectively, but the win is *instructive*: its class-0 phase-scramble
  leaked a gamma-power confound, so band-power/DMD ace it (1.00) instead of the intended
  nonlinearity-only methods. Raw cross-method disagreement can be gamed by an unintended easy feature.
- **Opus, budget 30, confound-aware (`targeted`)** ([`examples/evolved_scenario_targeted.py`](examples/evolved_scenario_targeted.py),
  margin **+0.47**): the fix and the payoff. Fitness = margin of the best genuine-dynamics method
  (SINDy/DySCo/HMM) over the best spectral method (band-power/DMD), so a scenario scores *only* if it
  beats **every** spectral method. Re-run under this objective, the **same model** was forced to build
  a genuinely spectrum-matched 6→40 Hz phase-coupling contrast — high-passing the coupled HF so no
  envelope bleeds into the low band, renormalizing HF power, matching channel variance — so band-power
  **and** DMD sit at chance (~0.50) while only nonlinear system-ID (SINDy) recovers it (1.00), and it
  **generalizes to held-out seeds**. Fitness shaping turned a confound-gamed win into a legitimate
  ground-truth scenario — the exact adversarial-design loop this platform is built for.

That evolved scenario is now a **first-class battery row**, folded into `scenarios.py` as `cfc_pac`
(a 6th scenario alongside the five above) with its anti-leakage controls preserved. Through the
battery's physical lead field it keeps the bred-for profile — SINDy 0.97, band-power 0.53, DMD 0.58,
DySCo 0.51, HMM 0.60 (n=40, seed 0) — a benchmark case the platform *discovered about itself* rather
than one a human hand-designed. (FM columns fill in when `scenario_benchmark.py` is re-run in the
container.)

Each scenario is calibrated by real HBN data and maps to an HBN task (resting → spectral,
SurroundSupp → evoked, videos → naturalistic). See
[`src/neurodynadojo/scenarios.py`](src/neurodynadojo/scenarios.py) and
[`examples/scenario_benchmark.py`](examples/scenario_benchmark.py).

## Layout

```
neurodynadojo/
  generators/   Systems: hopf (Hopf / Kuramoto / RingWave), netsim (Smith confound battery + real connectome), waves (Budzinski)
  algorithms/   fc (correlation, partial, coherence, imag-coh, wPLI, PLV), directed (Granger, DMD)
  challenges/   recovery metrics (edge AUC, directed AUC, wavenumber consistency)
  probes/       FM-ready linear probing (bring-your-own embed_fn; braindecode adapter; spectral baseline)
  bench.py      run_benchmark (scaling axes) + adversarial_search (auto-find breaking confounds)
examples/       runnable scripts reproducing the tables and the figure
```

## Related work & credit

- **FSLNets / netsim** — Smith, Miller, Salimi-Khorshidi, Webster, Beckmann, Nichols, Ramsey &
  Woolrich (2011), *Network modelling methods for FMRI*, NeuroImage 54:875. The ground-truth-recovery template.
- **DynaDojo** — Bhamidipaty, Bruzzese, Tran, Mrad & Kanwal (2023), *NeurIPS Datasets & Benchmarks*.
  The `System × Algorithm × Challenge` + scaling framing.
- **braindecode** — the PyTorch EEG model zoo with HuggingFace-style `from_pretrained`; the FM loading layer this builds on.
- **fmscope** (*The Identity Trap in EEG Foundation Models*) and **Tang et al. 2026** — complementary
  *real-data* FM audits; neuro-dynadojo is the synthetic, ground-truth counterpart.
- **Budzinski & Muller** (Chaos 2022; Phys Rev Research 2023) — the structure→traveling-wave generator.

## License

MIT.
