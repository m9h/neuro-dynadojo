# Response to the independent technical review

This document responds, point by point, to the code-level review in [`review_draft.md`](../review_draft.md)
(reviewer: "Antigravity"). The review was thorough, correct on the substance, and constructive — it
also contributed a genuine bug fix. We adopt it in full where it identifies a defect, address the
conceptual critiques with concrete options and an empirical robustness check, and note the two places
where a claim needs narrowing.

**Summary of changes made in response:**
- Kept the reviewer's **isolated-node damping fix** and **3-shell lead field** (both merged, both tested).
- Added a **`wiring_length`** option that spatially embeds the connectome, a
  `structural_leakage_collinearity` verification metric, and ran the FC-recovery sweep the critique
  motivates — the naive "collinearity makes recovery harder" hypothesis does not hold; the trend is
  driven by delay-synchrony, not a leakage confound (Critique B, **closed out empirically**).
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

## Critique B — network topology decoupled from spatial geometry · **Accepted, implemented, and empirically closed out**

A subtle and valid point: real cortical wiring is distance-penalised, so structural coupling and
volume-conduction leakage should be *collinear*, whereas the generators made them *orthogonal* by
drawing the connectome independently of node position. We:

1. Added `wiring_length` to the network generators — with `wiring_length > 0`, edge probability is
   scaled by `exp(−d_ij / L)`, embedding the connectome spatially (`modular_adjacency` /
   `directed_modular_adjacency`; verified in `test_distance_dependent_wiring_shortens_connections`,
   mean connection length 94→56mm).
2. Added `structural_leakage_collinearity(C, M)` — an odds-ratio metric that directly quantifies
   whether structural edges and strong-leakage node pairs co-occur (≈1 = independent/netsim-default;
   ≫1 = collinear/real-cortex-like), so the manipulation can be *verified*, not just asserted (unit
   tested against constructed independent- and collinear-by-design contingency tables).
3. Ran the FC-recovery sweep the critique calls for (`examples/wiring_geometry_study.py`,
   `results/wiring_geometry_study.csv`): edge-recovery AUC for a leakage-**vulnerable** measure
   (zero-lag correlation) and two leakage-**robust** measures (imaginary coherence, wPLI) as
   `wiring_length` shrinks from 0 (orthogonal) to 25mm (collinearity odds ratio 1.1 → 4.7), on the
   Hopf (amplitude-coupled) generator with `leak=0.8`, plus a **no-leakage control**
   (`leak=0.0`) that isolates a pure wiring-geometry effect from a leakage-collinearity effect.

**Result — more interesting, and more honest, than the naive hypothesis.** We expected collinearity
to make recovery *harder* (leakage mimicking structure at the same pairs). Instead, zero-lag
correlation AUC **rises** as wiring_length shrinks, in *both* the leaked and the no-leakage-control
condition, by a similar amount (leaked: 0.737→0.876; no-leak control: 0.898→0.990, wiring_length
0→25). Since the effect appears with leakage switched off, it is **not** the leakage-collinearity
confound we were testing for — it is a simpler, intrinsic consequence of short-range wiring: shorter
connections carry shorter conduction delay, which strengthens zero-lag synchrony (and hence
correlation-based recovery) regardless of leakage. Leakage still costs a fairly stable AUC penalty at
every wiring_length (≈0.12–0.16), so it remains a genuine confound — it just does not *interact*
multiplicatively with spatial collinearity the way the critique's mechanism predicted. The same
qualitative pattern (both tables move together) replicates under the phase-coupled Kuramoto
generator (`results/wiring_geometry_study.csv`), so it is not an artefact of the amplitude-coupled
regime. The leakage-robust measures (imaginary coherence, wPLI) stay near chance throughout, in both
conditions — consistent with existing test-suite documentation that this regime's structure imprints
on *zero-lag* amplitude coupling, not phase, so those measures are not informative for this
particular check (a phase-coupled *and* connectivity-recovering regime would be a cleaner probe;
existing tests already flag this as a known generator limitation).

**Reading for the review:** the critique's premise (structure and leakage should be modelled as
collinear, not orthogonal) is correct and now implemented and *verifiable* via
`structural_leakage_collinearity`. The specific failure mode it anticipated (collinearity → leakage
masquerading as structure → harder recovery) does not manifest in this generator; the dominant
driver of the observed trend is delay-synchrony, not a leakage confound. We consider this closed for
now, with the phase-coupled/lag-robust regime flagged as the natural place to look next if a
collinearity-specific interaction exists.

Scope note (unchanged): this critique concerns the **network-FC-recovery** generators
(Hopf/Kuramoto/Netsim). The six-scenario battery that produces the headline `cfc_pac` result uses
fixed source topographies, not a generated connectome, so those results are unaffected.

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
contributions are merged with attribution in the commit history. Our reply, including a forward
plan for what we build next in response to (and independent of) this review, is in
[`../REPLY_TO_REVIEW.md`](../REPLY_TO_REVIEW.md).
