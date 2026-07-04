# Response to the independent technical review

This document responds, point by point, to the code-level review in [`review_draft.md`](../review_draft.md)
(reviewer: "Antigravity"). The review was thorough, correct on the substance, and constructive — it
also contributed a genuine bug fix. We adopt it in full where it identifies a defect, address the
conceptual critiques with concrete options and an empirical robustness check, and note the two places
where a claim needs narrowing.

**Summary of changes made in response:**
- Kept the reviewer's **isolated-node damping fix** and **3-shell lead field** (both merged, both tested).
- Added a **`wiring_length`** option that spatially embeds the connectome (Critique B).
- Routed the **scenario battery through a selectable forward model** and **re-ran the headline
  result under the 3-shell model** (Critique A) — the finding survives.
- Revised the report's wording on linear probing (Critique D) and its limitations section.

---

## Critique A — infinite-medium lead field under-represents volume conduction · **Accepted, addressed, tested**

Correct and important: the radial `1/d²` lead field ignores the skull's spatial low-pass smearing.
We adopt the reviewer's opt-in **3-shell Berg–Scherg model** (`leadfield="3shell"` on the
generators) and additionally routed the **scenario battery** through a selectable forward model
(`NDD_LEADFIELD=3shell`), which the reviewer's change did not cover.

The decisive question is whether the headline result — the `cfc_pac` foundation-model blind spot —
is an artefact of the idealised forward model. It is not. Re-running the 12-seed sweep under the
3-shell (smeared) model:

| method | radial (median AUC) | **3-shell (median AUC)** |
|---|---|---|
| SINDy (system-ID) | 0.99 | **1.00** |
| CEBRA (latent) | 0.94 | **0.93** |
| band-power | 0.52 | 0.54 |
| DMD | 0.48 | 0.49 |
| DySCo | 0.56 | 0.51 |
| HMM | 0.50 | 0.48 |
| BIOT | 0.51 | 0.50 |
| CBraMod | 0.57 | 0.57 |
| REVE | 0.56 | 0.55 |
| LUNA | 0.59 | 0.58 |
| LaBraM | 0.65 | 0.63 |

(12 seeds each, single LogReg-AUC metric; radial from §5, 3-shell from this robustness run.)

The separation is unchanged: only nonlinear system-ID (and CEBRA) read `cfc_pac`; every spectral
method and every FM stay at chance under the more realistic forward model. As the reviewer notes in
their own synthesis, FMs failing *even in the cleaner spatial setting* only strengthens the claim —
and we now confirm they also fail in the smeared one. Full distributions:
`figures/cfc_pac_3shell_raincloud.png`.

Caveat we would add to the reviewer's fix: the 3-shell parameters (`g`, `mu`) are the standard
Berg-style approximation, not calibrated to a specific set of conductivity ratios, so it should be
read as a *more realistic alternative* forward model, not a validated head model. It is opt-in;
the default remains radial for continuity.

## Critique B — network topology decoupled from spatial geometry · **Accepted, option added**

A subtle and valid point: real cortical wiring is distance-penalised, so structural coupling and
volume-conduction leakage are collinear, whereas our generators made them orthogonal. We added a
`wiring_length` parameter to the network generators; with `wiring_length > 0` the edge probability
is scaled by `exp(−d_ij / L)`, embedding the connectome spatially (verified:
`test_distance_dependent_wiring_shortens_connections`, mean connection length 94 → 56). We have not
yet run a full FC-recovery study across the topology–geometry collinearity axis — that is exactly the
kind of controlled sweep the platform is built for, and we flag it as the natural follow-up.

Scope note: this critique concerns the **network-FC-recovery** generators (Hopf/Kuramoto/Netsim). The
six-scenario battery that produces the headline results uses fixed source topographies, not a
generated connectome, so those results are unaffected by the coupling.

## Critique C — isolated-node self-damping bug · **Accepted, fixed, tested**

A real bug and a clean catch. With row-normalised coupling `k·(Cn·z − z)`, a degree-0 node received
`−k·z`, shifting its Stuart–Landau bifurcation parameter from `a` to `a−k`. The reviewer's fix
(degree-guarded normalisation and `has_conn · z`) is mathematically correct and is retained, covered
by `test_isolated_node_no_damping`. **Scope:** the bug lived in `simulate_hopf`/`simulate_netsim`
(the network integrators used for FC recovery); the scenario battery builds signals from explicit
sinusoidal/Gabor sources and never calls those integrators, so the §3–§5 results are unchanged. We
have made this scoping explicit in the report so the fix is not mistaken for a correction to the
headline numbers.

## Critique D — linear probe ≠ representational capacity · **Accepted, wording narrowed**

Fair. A linear probe measures *linear* decodability; a factor could be encoded non-linearly. We use
frozen-embedding linear probing because it is the standard, conservative FM-evaluation convention and
the only thing that puts classical methods and FMs on one footing — but the precise claim is "the FM
does not *linearly* expose `cfc_pac`," not "cannot represent it." We have narrowed the report's
language accordingly and added a non-linear-probe control (kernel / small MLP) to future work.

## Critique E — redundant spectral shaping in `_bg_field` · **Acknowledged, minor**

Correct: `_pink` already applies `1/√f` shaping, and `_bg_field` re-shapes in the frequency domain,
so there is a redundant FFT/iFFT round-trip. The resulting spectrum is correct (exponent ≈1.13); the
inefficiency is cosmetic. We will streamline it into a single shaping step but it does not affect any
result.

---

## What we did not change, and why

- **Default forward model stays radial.** The 3-shell model is opt-in so existing numbers remain
  reproducible; both are available and the headline result is reported under both.
- **`cfc_pac` remains a single evolved point.** We treat it as an existence proof, not a map. Mapping
  the boundary of the FM blind spot (coupling frequency × phase depth × SNR) is the recommended
  follow-up, now that the robustness-to-forward-model question is answered.

We thank the reviewer. The critiques materially improved the rigor of the claims, and the two code
contributions are merged with attribution in the commit history.
