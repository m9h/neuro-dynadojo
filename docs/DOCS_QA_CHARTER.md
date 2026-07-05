# Documentation & QA agent charter

A working brief for an agent (or human) dedicated to keeping `neuro-dynadojo`'s documentation
accurate and its claims trustworthy, as distinct from the research/engineering work itself. Written
against the repo's actual state as of this charter's creation (37 tests, 6 `docs/` +
root-level review docs, 18 distinct `NDD_*` environment variables across `examples/`), so it names
concrete gaps rather than generic advice. Update the baselines below as the project moves; a stale
charter is exactly the failure mode it exists to prevent.

## Mandate

This agent does not do the science. It verifies that what's written matches what's true, catches
drift before a reviewer does, and keeps the growing pile of cross-referencing documents from
silently diverging. Run it after every substantive push, and periodically (e.g. weekly) even
when nothing obviously changed — staleness accumulates quietly.

## 1. Documentation maintenance

- **Numeric-claim sync.** Grep README.md, docs/TECHNICAL_REPORT.md, docs/REVIEW_GUIDE.md,
  docs/REVIEW_RESPONSE.md, REPLY_TO_REVIEW.md for test counts ("N/M passed"), seed counts, and AUC/
  score values; confirm each still matches `pytest --collect-only` and the underlying
  `results/*.csv` / `figures/*.json`. These docs repeat the same numbers in prose, tables, and
  commit messages — the classic place for silent drift.
- **Environment-variable reference.** No single doc lists all `NDD_*` variables; they're documented
  piecemeal in individual script docstrings. Maintain a table like the seed below, keeping it
  synced as scripts add/remove variables (`grep -rhoE "NDD_[A-Z_]+" examples/*.py examples/*.sh`).

  | var | used by | meaning | default |
  |---|---|---|---|
  | `NDD_LEADFIELD` | `scenarios.py`, generators | `radial`\|`3shell`\|`bem` forward model | `radial` |
  | `NDD_PROBE` | `cfc_pac_seeds.py` | `linear`\|`kernel`\|`mlp` probe (`probes.cv_auc`) | `linear` |
  | `NDD_MODE` | `llamea_evolve_scenarios.py` | `targeted`\|`disagree` fitness | `targeted` |
  | `NDD_MODEL` | `llamea_evolve_scenarios.py` | LLaMEA backend model | `claude-sonnet-5` |
  | `NDD_BUDGET` | `llamea_evolve_scenarios.py` | evolution budget (generations) | `12` |
  | `NDD_OUT` | `llamea_evolve_scenarios.py` | path to write the evolved champion | unset |
  | `NDD_FMS` | `cfc_pac_seeds.py` | run the FM-zoo pass (container) | unset |
  | `NDD_OSL` | `cfc_pac_seeds.py` | run the osl-TDE-HMM pass (container) | unset |
  | `NDD_SEEDS` | `cfc_pac_seeds.py` | seed count for the sweep | `12` |
  | `NDD_NPER` | `cfc_pac_seeds.py`, `map_cfc_pac_boundary.py` | recordings per class | `40` / `30` |
  | `NDD_SEEDS_FAST` | `map_cfc_pac_boundary.py` | seeds for cheap methods | `5` |
  | `NDD_SEEDS_CEBRA` | `map_cfc_pac_boundary.py` | seeds for CEBRA (slower) | `3` |
  | `NDD_JSON` | `cfc_pac_seeds.py` | output JSON path (accumulates across passes) | required |
  | `NDD_SCRATCH` | container launchers | host path mounted at `/scratch` | unset |
  | `NDD_DEV` | `hbn_surroundsupp_decode.py` | `cpu`\|`cuda` | `cpu` |
  | `NDD_NSUBJ` | `hbn_surroundsupp_decode.py` | subject count | `40` |
  | `NDD_SKIP_ZOO` | `hbn_surroundsupp_decode.py` | skip the FM zoo | unset |
  | `NDD_CONNECTOME` | `netsim_real_connectome.py` | path to a real SC matrix | `desikan68_SC.npy` |
  | `NDD_SCI_HEADMODEL_ZIP` | `generators/montage.py` (`sci128`/`sci256`) | path to the SCI Utah EEG head-model zip | machine-specific; **added after this charter caught it missing** — see §6 |

- **`CITATION.cff` currency.** Check it names every method/paper the report now cites (SINDy, CEBRA,
  osl-dynamics, LLaMEA, Berg–Scherg, MNE/fsaverage BEM, DySCo, etc.) whenever a new one is added to
  `docs/TECHNICAL_REPORT.md`.
- **Docstring sync.** `src/neurodynadojo/{scenarios,algorithms/*,generators/*,probes/*}` docstrings
  must reflect actual current defaults and behavior — check after every parameter/signature change.
- **Link-check.** Verify every relative markdown link across `docs/` and root `.md` files resolves;
  these docs cross-reference each other (`../figures/...`, `../review_draft.md`) in ways that break
  silently when files move.
- **Status-flag currency.** `REPLY_TO_REVIEW.md`/`REVIEW_RESPONSE.md` mark phases/critiques as
  open/done/superseded. Confirm a flag flips promptly when the underlying work actually lands —
  nothing currently enforces this automatically.

## 2. Result reproducibility QA — the highest-value recurring check

- Periodically **re-run the Tier 0–2 reproduction commands** in `docs/REVIEW_GUIDE.md` and diff
  output against the numbers written in prose/tables. This is the single most valuable check for a
  repo whose credibility rests on numbers a reviewer can re-derive.
- **Trace every embedded number to a committed CSV/JSON**, and confirm that file was produced by the
  script credited for it — no orphaned or stale result files standing in for current ones.
- **Force figure regeneration** whenever the underlying CSV/JSON changes; nothing currently
  guarantees a PNG isn't stale relative to its data.

## 3. Test-suite QA

- **Coverage gaps on new surfaces**: e.g. is there a direct test for `NDD_LEADFIELD=bem` beyond the
  dead-source warning? For `Anthropic_LLM.query()`'s message-formatting beyond manual smoke tests?
- **Hidden nondeterminism**: every CV/CEBRA/LLM-touching test needs a fixed seed; flag any that
  don't.
- **CI fidelity**: confirm `.github/workflows/ci.yml` runs the current full test file set, and that
  container/API-key-dependent tests are explicitly skipped there (`pytest.importorskip`, etc.), not
  silently erroring or silently absent.
- **Runtime creep**: suite is 37 tests / ~75s today (from 24 tests earlier this project); flag when
  it's time to split fast vs. slow tests rather than let pre-commit runs quietly get too slow to run.

## 4. Security / secrets hygiene

- **Grep tool outputs, logs, and scratch files for secret patterns** (`hf_...`, `sk-...`,
  `ANTHROPIC_API_KEY` values, other API-key shapes) before they land anywhere persistent.
  **Concrete incident**: an `HF_TOKEN` value appeared in plaintext in a `docker run` process-listing
  (`pgrep -af`) tool output during this project's own session — a real exposure, not a hypothetical.
  That token should be rotated, and this check exists specifically because it already happened once.

## 5. Scientific-review consistency (the project is under active external review)

- **Re-verify scoping claims against current code.** E.g. "this bug lives in `simulate_hopf`, not
  the scenario battery" is only true until a refactor quietly changes the call path — a reviewer is
  relying on that claim staying accurate.
- **Own `docs/REVIEW_GUIDE.md`'s claim table as the single source of truth** for open/closed/
  superseded critiques, rather than letting status also live loosely in prose across the other three
  review docs.

## 6. Repo housekeeping

- **Sync `pyproject.toml` extras** (`fm`, `mne`, `sysid`, `latent`, `llm`, `dev`) against actual
  imports in `src/`/`examples/` — note `osl-dynamics` has no extra today because it only runs inside
  the `oracle-osl` container; confirm that's still true and documented as such, not silently assumed.
- **Watch for orphaned files** — results/figures with no doc referencing them, or docs referencing
  files that no longer exist.
- **Detect mirror drift** between `neuro-dynadojo/examples/` and the `smni-cmi/scripts/` mirror,
  currently kept in sync by manual `cp` after each change with nothing catching silent divergence.
- **External-data paths must be overridable, not hardcoded.** Caught the day this charter was
  written: a contributed EGI 128/256-channel montage loader (`generators/montage.py`) hardcoded an
  absolute path to a local dataset with no env-var override, unlike the project's existing
  `NDD_CONNECTOME` convention for external data — meaning it silently only worked on the machine it
  was authored on. Fixed with `NDD_SCI_HEADMODEL_ZIP`. This is the check that should catch the next
  one before it ships: any new `os.path.exists("/home/...")`-style hardcoded path in `src/` or
  `examples/` should be flagged and given an env-var override in the same commit.

## Out of scope

Writing the science, designing new scenarios/methods, deciding what to build next, and replying to
reviewers on substance are not this agent's job — see `docs/TECHNICAL_REPORT.md` and
`REPLY_TO_REVIEW.md` for that layer. This charter is deliberately narrow: accuracy and hygiene, not
research direction.
