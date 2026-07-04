# Reviewer's guide

This document orients an independent reviewer: what to read first, how each claim maps to code and
data, how to reproduce results at increasing cost, the exact software environment, and the
safeguards (negative controls) against evaluation artefacts. The scientific write-up is
[`docs/TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md); this is the operational companion.

## Start here

1. [`docs/TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) — motivation, methods, results, limitations.
2. `figures/landscape_heatmap.png` — the method × scenario matrix at a glance.
3. `figures/cfc_pac_raincloud.png` — the headline result (per-seed distributions).
4. `src/neurodynadojo/scenarios.py` — every scenario's generative mechanism, in ~140 readable lines.
5. `examples/llamea_evolve_scenarios.py` — the adversarial loop and its two fitness modes.

## Claim → evidence map

| # | Claim (from the report) | Code | Data / figure | Reproduce |
|---|---|---|---|---|
| 1 | Every scenario is decodable by a *different* method family | `src/.../scenarios.py`, `algorithms/` | `results/landscape_matrix.csv`, `figures/landscape_heatmap.png` | Tier 1 + 2 |
| 2 | FMs split into two camps (phase/waveform vs cross-frequency) | — | landscape rows BIOT vs CBraMod/REVE/LaBraM | Tier 2 |
| 3 | Implementation matters: osl **TDE-HMM** ≫ Gaussian **HMM** | `algorithms/osl_dynamics.py`, `algorithms/dynamics.py` | landscape rows `HMM` vs `osl-TDE-HMM` | Tier 1 (HMM) + Tier 2 (osl) |
| 4 | LLaMEA raw-disagreement fitness is **gamed** by a spectral confound | `examples/llamea_evolve_scenarios.py` (`NDD_MODE=disagree`) | `examples/evolved_scenario_opus.py` (header) | Tier 3 |
| 5 | Confound-aware **targeted** fitness repairs it; same model, real scenario | same, `NDD_MODE=targeted` | `examples/evolved_scenario_targeted.py` | Tier 3 |
| 6 | `cfc_pac` defeats all 5 FMs + spectral/state-space; only SINDy & CEBRA read it | `scenarios.cfc_pac`, `examples/cfc_pac_seeds.py` | `figures/cfc_pac_seeds.json`, `figures/cfc_pac_raincloud.png` | Tier 1 (SINDy/CEBRA) + Tier 2 (FMs) |
| 7 | The decode is real signal, not harness leakage | `tests/test_negative_control.py` | — | Tier 0 |
| 8 | `cfc_pac` FM blind spot survives a realistic (3-shell) forward model | `scenarios._leadfield`, `generators/hopf.leadfield_3shell` | `figures/cfc_pac_3shell_raincloud.png` | Tier 1 (venv) + Tier 2 (FMs) |
| 9 | Spatially-embedding the connectome does not make FC recovery harder — the trend is delay-synchrony, not a leakage confound | `generators/hopf.structural_leakage_collinearity`, `examples/wiring_geometry_study.py` | `results/wiring_geometry_study.csv` | Tier 1 |

## Reproduction tiers

Cost rises with tier; the load-bearing claims are reproducible at Tier 0–1 without special hardware.

- **Tier 0 — `pytest` (CPU, ~90 s, no GPU/containers/keys).** 26 tests. Includes the scenario
  contracts, the fitness discrimination, the CEBRA contender, and the **negative controls**
  (`test_negative_control.py`): label-shuffle collapses the SINDy/`cfc_pac` decode from 0.90 to
  ≈0.44, and a pure-1/f scenario is undecodable — i.e. no index/order/balance leakage.
  ```bash
  pip install -e '.[sysid,latent]' && pytest -q
  ```
- **Tier 1 — classical / system-ID / latent rows (CPU venv, minutes).** The non-FM columns of the
  landscape and the SINDy/CEBRA points of the `cfc_pac` sweep.
  ```bash
  python examples/scenario_benchmark.py                 # (skips FMs gracefully outside the container)
  NDD_JSON=out.json python examples/cfc_pac_seeds.py    # SINDy 0.99, CEBRA 0.94 medians over 12 seeds
  NDD_LEADFIELD=3shell NDD_JSON=out3.json python examples/cfc_pac_seeds.py   # claim 8 (3-shell)
  python examples/wiring_geometry_study.py              # claim 9 (distance-wiring vs leakage)
  ```
- **Tier 2 — foundation-model and osl-dynamics rows (containers, GPU).** The FM columns and the
  osl-dynamics TDE-HMM row. Requires the two container images below.
  ```bash
  bash examples/run_leaderboard_container.sh examples/scenario_benchmark.py   # FM zoo
  bash examples/run_osl_container.sh                                          # osl TDE-HMM
  NDD_FMS=1  NDD_JSON=/scratch/cfc_fm.json  bash examples/run_leaderboard_container.sh examples/cfc_pac_seeds.py
  ```
- **Tier 3 — adversarial evolution (LLM key).** The LLaMEA loop; needs `ANTHROPIC_API_KEY`
  (Gemini/OpenAI/Ollama backends also ship with LLaMEA).
  ```bash
  NDD_MODE=targeted NDD_MODEL=claude-opus-4-8 NDD_BUDGET=30 python examples/llamea_evolve_scenarios.py
  ```
  The three champions we obtained are committed verbatim (`examples/evolved_scenario*.py`), so
  claims 4–5 are inspectable without rerunning the search.

## Environment provenance

The results in the report were produced with:

- **CPU venv** (Tier 0/1): Python 3.12; numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0,
  pysindy 2.1.0, hmmlearn 0.3.3, cebra 0.6.1, torch 2.12.1, llamea 1.2.0, anthropic 0.116.0.
- **FM container** (Tier 2): `nvcr.io/nvidia/pytorch:26.06-py3` with braindecode 1.5.2 + the
  *emeg-fm* zoo loader + *fmscope* probe. (Stock braindecode 1.6.x breaks the interpolated FM
  loaders; the pinned 1.5.2 is required.)
- **osl container** (Tier 2): `neurojax/oracle-osl:latest` with osl-dynamics 3.2.2 + TensorFlow.
  The `fsl` dependency (parcellation/plotting only) is stubbed in-process so the sensor-space
  TDE-HMM path imports cleanly.
- **LLM backends** (Tier 3): `claude-opus-4-8` and `claude-sonnet-5` via a ~40-line
  `Anthropic_LLM` subclass of `llamea.LLM`.

## Metric note (important for fair reading)

Foundation-model and `phase-conn` rows of the landscape are scored with *fmscope*'s balanced-accuracy
linear probe (inside its container); classical / system-ID / state-space / latent rows use 5-fold
LogReg AUC. Both are chance-0.50 and the shared rows (band-power, DMD, DySCo) agree across the two
scorers within ≈0.05. The `cfc_pac` result (claim 6, the raincloud) is computed under **one**
metric (LogReg AUC) across all methods and 12 seeds, so its cross-family comparison is strict. The
`results/landscape_matrix.csv` records the `metric` and `n` used for every row.

## What we would most like scrutinised

- The realism of the forward/noise model (radial lead field + spatially-correlated 1/f) and whether
  its idealisations bias any family.
- The fairness of the FM probing protocol (frozen embedding + linear probe; default checkpoints,
  montage mapping, resampling, normalisation).
- The confound-aware targeted fitness formulation (§4.2 of the report) — is the
  best-dynamics-minus-best-spectral margin the right leak-proof objective, and does it over-constrain?
- Whether the `cfc_pac` blind spot warrants a systematic follow-up (sweeping coupling frequency,
  phase depth, and SNR to map the boundary of what the FMs miss).
