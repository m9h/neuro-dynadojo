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
2. **A non-linear probe control.** Your Critique D is right that a linear probe only shows linear
   decodability. We'll add a kernel-SVM / small-MLP probe alongside the linear one for `cfc_pac` and
   the full landscape — if FMs still fail non-linearly, "does not represent" becomes a much stronger
   claim; if a non-linear probe *does* recover it, that's an important correction to the whole
   result, and we'd rather find it than have a reviewer find it for us.
3. **A calibrated (not just approximate) 3-shell model**, validated against a real BEM reference
   (e.g. MNE's sample head model) rather than generic Berg–Scherg constants, so Critique A's fix is
   load-bearing, not just illustrative.

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
