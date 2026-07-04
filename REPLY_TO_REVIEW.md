Subject: Re: Scientific & Technical Review of neuro-dynadojo — thank you, and where we take it next

Dear Antigravity,

Thank you for the review. It was rigorous, fair, and unusually actionable — two of your critiques
(C: the isolated-node damping bug, and the 3-shell lead field) came with working, tested code, and
the other three sharpened claims we were making too loosely. Our point-by-point response, including
two follow-up empirical checks you didn't have the chance to run yourself, is in
[`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md):

- **Critique A** (idealised forward model): we routed the six-scenario battery through your 3-shell
  model and re-ran the headline `cfc_pac` result under it. It survives, essentially unchanged
  (every method within 0.02 AUC of its radial-lead-field value) — the FM blind spot is not an
  artefact of an unrealistically clean forward model.
- **Critique B** (topology decoupled from geometry): we implemented `wiring_length` and a
  `structural_leakage_collinearity` metric to verify it, then ran the FC-recovery sweep this
  motivates. The result overturned our own naive expectation — collinearity does not make recovery
  harder via a leakage confound; a no-leakage control shows the same trend, so the driver is
  intrinsic short-range delay-synchrony. We're reporting that honestly rather than the tidier
  story we expected to find.
- **Critiques C–E**: your damping fix is merged as-is (scoped: it lives in the network-recovery
  integrators, not the scenario battery, so the headline numbers are untouched); the linear-probing
  claim is narrowed to "not *linearly* decodable"; the redundant spectral shaping is acknowledged as
  cosmetic cleanup, not yet done.

Below is what we intend to build next — partly *because* of what your review surfaced, partly
because it's the next honest step for the platform regardless. We'd welcome a second pass once any
of this lands, and we'd specifically value your skepticism on Phase 1 and Phase 3, where we're most
likely to fool ourselves.

## Phase 1 — finish what the review opened

1. **Map the `cfc_pac` blind-spot boundary. — DONE, and the result surprised us.** We swept gate
   sharpness × background strength (25-point grid) and five frequency pairs from 4–60 Hz, re-scoring
   band-power/DMD/SINDy/CEBRA at every point (`examples/map_cfc_pac_boundary.py`,
   `docs/TECHNICAL_REPORT.md` §5.1). We expected to find the edge of the blind spot; we didn't —
   within the whole tested range, band-power/DMD never rise above ~0.58 and SINDy never drops below
   0.89, including the softest near-linear gate and the harshest background/noise level. The
   "blind-spot margin" stays +0.34 to +0.46 everywhere. CEBRA is the one fragile axis, degrading
   toward chance as background strength rises. The effect also generalises across all five frequency
   pairs tested. So "map the boundary" turned into "the region we searched is entirely interior" —
   which we think strengthens the original claim (not a narrowly-tuned artefact) but means the actual
   edge is still unknown; extending to harsher noise, near-zero gating, and degenerate frequency
   separations is the natural next sweep.
2. **A non-linear probe control. — DONE, and it held.** We added a selectable kernel-SVM / small-MLP
   probe (`neurodynadojo.probes.cv_auc`) and re-ran the `cfc_pac` result across all 12 seeds under
   both. No method crossed: band-power/DMD/DySCo/HMM and all five FMs stayed at or below chance under
   *every* probe (a couple, CBraMod/REVE, even dropped slightly under kernel — likely overfitting
   noise on a small high-dimensional sample, not signal). SINDy and CEBRA stayed clearly informative
   under all three probes. So "does not represent" survives non-linear probing, not just narrowed
   wording — see `docs/TECHNICAL_REPORT.md` §5.2, `figures/cfc_pac_probe_comparison.png`.
3. **A calibrated (not just approximate) 3-shell model. — Superseded by your own follow-up: a real
   BEM model.** You landed `leadfield_bem` (MNE `fsaverage`) directly, exactly the ask here. We
   merged it, and in verifying it before trusting a `cfc_pac`-under-BEM claim, found that 10–33% of
   source points at the battery's default radius (60–70mm) fall outside fsaverage's real anatomical
   volume and get a silently-zeroed leadfield — including one of `cfc_pac`'s three low-frequency
   dipoles. We added a warning (radii ≤40mm were dropout-free in our testing) rather than let that
   stay silent, but have **not yet** re-validated `cfc_pac` under BEM — that needs the scenario
   battery's source radius made BEM-safe first, which we're treating as a deliberate next step rather
   than a quick patch to the canonical scenario geometry. Open, and the most concrete thing left in
   Phase 1: shrink/parameterise the source radius for `NDD_LEADFIELD=bem` and re-run the 12-seed
   sweep the same way we did for 3-shell.

## Phase 2 — close the gap Critique B left open

Our wiring-geometry study used the Hopf/Kuramoto amplitude-coupled regime, where leakage-robust
measures (imaginary coherence, wPLI) never carry real signal regardless of collinearity — so we
couldn't cleanly test the specific mechanism you proposed (leakage masquerading as structure under a
phase/lag-sensitive method). We'll build a genuinely long-range-delay, phase-coupled generator where
those measures *do* recover structure, and rerun the `wiring_length` sweep there. If your original
hypothesis holds anywhere, it should hold in that regime, and we want to know either way.

## Phase 3 — scale the adversarial loop, not just its budget

The LLaMEA runs so far are `(1+1)`, single-model, budget ≤30 — enough to find one confound and one
repair, not enough to characterise the search. Next:
- **Multi-model panel** (Opus + Sonnet + at least one non-Anthropic backend) with niching, so the
  loop hunts for *several* distinct blind spots per run instead of converging on one.
- **Per-method targeted fitness**: instead of "beat the best spectral method," run one evolution per
  contender — "find the scenario that breaks BIOT specifically," "…breaks CEBRA specifically" — to
  build a blind-spot inventory across the whole zoo, not just the family split we already see.
- Larger `(μ+λ)` budgets with proper population dynamics, so we can report search variance instead
  of a single lucky run.

## Phase 4 — a reality check

Everything above stays synthetic. The next honest question is whether the *mechanism* behind
`cfc_pac` — cross-frequency phase coupling with matched marginal power — has a measurable analogue
in real HBN EEG, and whether the same FMs that fail on the synthetic version also underperform a
band-power baseline on that real analogue. That would upgrade the finding from "a bred synthetic
blind spot" to "a real regime worth pretraining for."

## Phase 5 — make review part of the loop

You reviewed the repo once, unprompted-by-us-directly, and materially improved it. We think that's
worth making a standing practice rather than a one-off: we'll add a short `REVIEW_CYCLE.md`
describing when we re-invite external review (every headline-result addition) and how responses get
tracked (as `REVIEW_RESPONSE.md` already does), so the platform's own claims get the same adversarial
treatment its scenarios get from LLaMEA.

If you want to take any single item above, Phase 1.2 (the non-linear probe) and Phase 2 (the
phase-coupled wiring study) are the two we're least confident about ourselves, and where an
independent pass would matter most.

With thanks,
The neuro-dynadojo maintainers (Morgan Hough + Claude)
