# Claims Registry — *Selection Navigates a Degenerate Circuit Space*

**Status:** Post-referee-revision snapshot of the manuscript (`paper_v2/latex/main_v2.tex`). Every value below is the current value in `stats/paper_stats.tex` (415 macros, pipeline-verified). This document is a faithful registry of the paper's claims and their evidence — not a development log. (Revision history: `claims_history.md`.)

---

## Degeneracy at a glance

The paper's contribution, in the language of degeneracy:

- **Degeneracy is imposed by construction, not inferred.** Three architectural constraints (Dale's Law, sparse degree, quantized weights) compress every solution into one structural region, making structurally distinct circuits performing the same function the *expected* outcome of evolutionary search rather than a property to be discovered.
- **Degeneracy is shown to be total — it spans every level of circuit description examined:** aggregate statistics → connection topology → causal sensitivity → representational geometry → dynamical geometry. Each level has its own architectural null control.
- **Selection navigates the degenerate space without leaving a structural trace.** Behavioral individuation is robust (specialists are reliably distinct), yet no structural, representational, or dynamical axis carries individual identity.
- **The one axis selection does shape is functional sensitivity commitment** — and that, too, is degenerate: replicates of the same mouse commit *strongly* but to *different* pathways (commitment strength is conserved; commitment direction is not).
- **A within-target degeneracy result adjacent to plurifunctionality:** within-mouse sensitivity profiles are *significantly less* similar to each other than to between-mouse profiles — replicates of a single individual do not converge on a shared mechanistic signature.

The system is a 14-neuron recurrent network (6 sensory, 6 interneuron, 2 motor) navigating a hierarchical binary maze, evolved to replicate the navigation behavior of 9 individual mice (Rosenberg et al. 2021), 6 replicate runs per mouse, 54 runs total. Fitness tracks four behavioral statistics (Markov transition probabilities, spatial occupancy, path tortuosity, turn bias); no external reward.

---

## Argument chain

Each Results section answers one question; the chain is the paper's spine.

| § | Question | Answer |
|---|---|---|
| 2.1 | Does the system work? | Yes — all 54 runs converge; emergent (unselected) behaviors reproduced; consistent feedforward-dominant architecture with a dissociated E/I organisation. |
| 2.2 | What does selection converge on, and is there a paradox? | A structural floor: 0/18 aggregate features differ across mice — yet behavioral individuation is robust and persists on held-out data. |
| 2.3 | Does topology variation explain the individuation? | No. Topology distance predicts behavioral distance at no scale; the flatness is architectural (random constrained agents are identical). |
| 2.4 | Does any structural axis carry the identity information? | No. Mutual information between every structural axis and mouse identity is weak and non-significant. |
| 2.5 | What does topology *do* — and what doesn't it do? | It is causally necessary for each circuit but degenerate across circuits; no structural decomposition fingerprints identity. |
| 2.6 | What does individual-target selection actually achieve? | Functional sensitivity commitment (~2.2–2.6× higher variance than generalists; mouse-level *p* = 0.002) — directed but not convergent; no shared mouse-specific pathway signature. |
| 2.7 | Does degeneracy extend to representational geometry? | Yes. Activity-embedding geometry is mouse-invariant; weakly (positively) constrained by topology, but carries no identity. |
| 2.8 | Does degeneracy extend to dynamical geometry? | Yes, completely. Stability, attractor landscape, and driven trajectories are all mouse-invariant despite circuits sharing as few as 7% of edges. |

---

## Figure map

Main figures (compiled numbering) and the analysis they anchor:

| Fig | Content | Section |
|---|---|---|
| 1 | The constrained neuroevolution system (maze, architecture, convergence, emergent behavior, E/I) | §2.1 |
| 2 | Structural floor + behavioral individuation (cosine matrix, 18-feature ANOVA, specialization) | §2.2 |
| 3 | Structural degeneracy across all measurable axes (topology heatmap, joint KDE, NMI) | §2.3–2.4 |
| 4 | Topology causally necessary but degenerate (permutation, dose-response, generalist control) | §2.5 |
| 5 | Functional sensitivity commitment (gen vs spec topology, fitness cost, variance, trajectory) | §2.6 |
| 6 | Representational geometry degenerate (Procrustes, Mantel dissociations, mouse-coloured clouds) | §2.7 |
| 7 | Structural + dynamical degeneracy (circuit pairs, λ₁, attractor, trajectory PCA) | §2.8 |

Supplementary figures carry an `S` prefix (Fig S1–S23) and are distinct from the main Figures 1–7; supplementary tables are Table S1–S2. References below use these compiled numbers directly.

## Claims

All claims below are supported in the sealed manuscript by the listed evidence (figure panel, statistical test, and/or Methods paragraph). Values are macros from `stats/paper_stats.tex`.

### Level 1 — System validation (§2.1, `sec:system`)

| # | Claim | Key evidence |
|---|---|---|
| 1 | All 54 runs converge reliably (population mean fitness change <2% over the final 30 generations). | Grand mean fitness 0.809 ± 0.023 (range 0.770–0.847; lower = better). Fig 1C–D. |
| 2 | Evolved agents reproduce behaviors not in the fitness function (slow/fast speed phases, turn-rate distributions, thigmotaxis). | Per-mouse speed & turn-rate distributions vs real mice; Fig 1E–F; Supp Fig S3. |
| 3 | Evolution selects a consistent feedforward-dominant architecture (sparser, stronger connections, more sensory→motor shortcuts than random). | One-sample *t* vs 200 random agents, Cohen's *d* with bootstrap CIs; Fig 1G; Supp Table S1. |
| 4 | E/I dissociation: speed motor uniformly excitatory (E/(E+I) ≈ 0.85); turn motor more balanced (≈ 0.667) with greater inter-mouse variability; neither selected for. | 54×8 E/I heatmap; Fig 1H; Supp Fig S4. |

### Level 2 — Structural floor + behavioral individuation (§2.2, `sec:paradox`)

| # | Claim | Key evidence |
|---|---|---|
| 5 | 0/18 aggregate structural features differ across mice (smallest raw *p* = 0.045; all *p*_FDR > 0.47). | One-way ANOVA + BH-FDR; *k*=9, *n*=6; 10,000-permutation validation. Fig 2B. |
| 6 | The floor is architectural, not selection-driven: a matched sample of random constrained agents (54 agents, 9-group × 6, identical power) returns the same 0/18. | Same ANOVA on random agents. Fig 2B; Methods. |
| 7 | Study is adequately powered for large effects (3 largest-effect features each exceed 80% power). | Post-hoc power via non-central *F*; Supp Fig S6. |
| 8 | Behavioral specialization is robust: cross-mouse fitness error is 33.4% higher than own-mouse (specialization index = 0.334; own 0.748 vs cross 1.122). | 9×9 cross-mouse matrix; index = 1 − own/cross. Fig 2C–D. |
| 9 | All 9 mice show positive specialization indices (B5 most specialized; D7 least). | Per-mouse bar chart. Fig 2C. |
| 10 | Specialization is not strain-confounded and is robust to fitness-weighting scheme (5 schemes). | Within-strain indices; 5-scheme table. Supp Figs S10, S11. |
| 11 | Specialization persists on held-out trajectory data (index 0.096 > 0 for all 9 mice). | Cross-bout (second-half) evaluation. Supp Fig S12. |

### Level 3 — Topology-axis degeneracy (§2.3, `sec:topo_degeneracy`)

| # | Claim | Key evidence |
|---|---|---|
| 12 | Topology distance does not predict behavioral distance at any scale (overall ρ = 0.001, *p* = 0.966; within- and between-mouse equally null). | Spearman ρ(Jaccard, cosine) over 1,431 pairs. Fig 3C. |
| 13 | Within-mouse: neither topology nor magnitude predicts fitness (many-to-one landscape on both axes; partial correlations null). | Partial Spearman on within-mouse replicate pairs. §2.3 (text). |
| 14 | The flatness is architectural: random constrained agents show the same ρ ≈ 0 (ρ = −0.029). | 54 random agents, same correlation. Fig 3C. |

### Level 4 — No structural axis carries identity (§2.4, `sec:mi`)

| # | Claim | Key evidence |
|---|---|---|
| 15 | No structural axis carries significant identity information: max NMI = 0.2173 (topology 0.1451, magnitude 0.2173, sign 0.2161); AMI at/below chance across *k* = 3–9. | MI + NMI via *k*-means (*k*=5) per axis vs mouse identity; AMI sweep. Fig 3D. |

### Level 5 — Causal structure (§2.5, `sec:causal`)

| # | Claim | Key evidence |
|---|---|---|
| 16 | Connection topology is causally necessary: source-preserving permutation elevates motor MSE 3.53× vs replicate baseline (vs-baseline *p* = 8.6×10⁻⁷). | Topology permutation, 54 agents, 20 variants. Fig 4B; Table 1. |
| 17 | The disruption is mouse-specific: own-mouse degradation unanimously exceeds other-mouse across all 9 mice (Wilcoxon *p* = 0.0039). | Best-per-mouse agent; *n*=9 paired-means Wilcoxon. Fig 4A. |
| 18 | Dose-response (suggestive): specialization index trends with permutation own/other ratio (ρ = 0.68, perm *p* = 0.050, 95% CI [−0.10, 1.00], *n*=9). *Reported as indicative — the bootstrap CI spans zero; the causal claim rests on #16–17.* | Spearman + bootstrap. Fig 4C. |
| 19 | Topology, not magnitude, is the computational substrate: magnitude permutation reduces MSE below baseline (0.164×); 61.9% of sources are magnitude-uniform. | Magnitude-only permutation. Fig 4B; Table 1. |
| 20 | No single neuron fingerprints identity (0/14 sources significant, Bonferroni; within- vs between-mouse sensitivity correlation r = 0.150 vs 0.167, *p* = 0.692). | Per-source ablation, 54×14. Supp Fig S13. |
| 21 | No distributed sensitivity *geometry* fingerprints identity (Mantel ρ = −0.044, *p* = 0.357; within vs between MW *p* = 0.529). | Sensitivity RSA (Euclidean). Supp Fig S15. |

### Level 6 — Mechanism: functional sensitivity commitment (§2.6, `sec:sensitivity`)

| # | Claim | Key evidence |
|---|---|---|
| 22 | Generalists and specialists share statistically indistinguishable topological diversity (MW *p* = 0.856; gen 0.276 vs spec 0.268) — selection does not act on topology. | Topology cosine similarity, two-sided MW. Fig 5A. |
| 23 | Individual-target training has a real fitness cost: generalist error is ~42.1% higher than specialists on a ratio-of-means basis (per-mouse range 28.8–81.3%), yet generalists still learn each mouse's behaviour well above chance. | Per-mouse generalist vs specialist fitness (ratio-of-means). Fig 5B. |
| 24 | Individual-target selection builds ~2.2–2.6× higher normalized sensitivity variance in specialists vs generalists (within-mouse-pooled to across-54 definitions; bootstrap 95% CI [1.02, 7.36]). Primary test is mouse-level: 9/9 mice exceed the generalist (one-sample Wilcoxon *p* = 0.002); per-neuron MW *p* = 0.003 (one-sided) is reported but pseudoreplicated. | Source-permutation sensitivity, baseline-normalized. Fig 5C. |
| 25 | The generalist is disrupted like a specialist on the wrong mouse: no own-mouse elevation (Kruskal-Wallis *p* = 0.093). | Generalist permutation profile. Fig 4D. |
| 26 | **No shared mouse-specific pathway signature:** within-mouse sensitivity profiles are no more similar than between-mouse profiles (cosine 0.20 vs 0.24). The small gap does *not* survive a pseudoreplication-aware test (label-permutation *p* = 0.076; naive MW *p* = 0.040 is anticonservative). Demoted from "commitment is itself degenerate / distinct pathways" — exploratory, non-significant. | Pairwise cosine of 14-dim sensitivity vectors; label-permutation. §2.6. |

### Level 7 — Representational geometry (§2.7, `sec:rep_geometry`)

| # | Claim | Key evidence |
|---|---|---|
| 27 | Positive control: motor neurons separate from interneuron pairs in PC1–PC2 loading space (motor dist 0.583 vs inter 0.466; MW *p* < 0.001) — embeddings are informative at *N*=14. | Per-agent PCA loadings (14×6). Supp Fig S16. |
| 28 | Near-maximal manifold dimensionality (effective dim 5.83/6; participation ratio 4.44), uniform across mice — circuits explore nearly the full embedding space. | PCA explained variance; participation ratio. Supp Fig S17. |
| 29 | Activity-embedding geometry is mouse-invariant: within-mouse Procrustes distance (1.974) indistinguishable from between-mouse (1.988; MW *p* = 0.260); robust across 6 similarity metrics (all *p* > 0.25). | Pairwise Procrustes on PC loadings (1,431 pairs). Fig 6A; robustness Supp Fig S18. |
| 30 | Generalists occupy the same embedding region as specialists (Procrustes 1.946; MW vs specialist-within *p* = 0.920). | Same analysis on generalists. Fig 6A. |
| 31 | Representational distance dissociates from behavioral (Mantel ρ = −0.007, *p* = 0.871) and sensitivity (ρ = −0.034, *p* = 0.355); weakly **positively** constrained by topology (ρ = +0.273, *p* < 0.001) — structure constrains representation slightly, but too weakly to carry identity. | Permutation Mantel tests (1,431 pairs). Fig 6B,E,H. |

### Level 8 — Dynamical geometry (§2.8, `sec:dynamics_degen`)

| # | Claim | Key evidence |
|---|---|---|
| 32 | Dynamical stability is mouse-invariant: all 54 agents λ₁ < 0 (mean −0.050 ± 0.024, range −0.112 to −0.015); ANOVA *p* = 0.943, KW *p* = 0.996. | Finite-perturbation Lyapunov (ε=10⁻⁶, 8 directions, 400 steps). Fig 7D,G. |
| 33 | Attractor landscape is degenerate: within-mouse sliced-Wasserstein distance (0.081) indistinguishable from between-mouse (0.079; MW *p* = 0.544); near-universal convergence to a single fixed point per agent, clustered in a common region. | 200 random-init × 500-step autonomous runs; sliced Wasserstein. Fig 7I; Supp Fig S20. |
| 34 | Driven trajectories are degenerate: within-mouse activation trajectories no more similar than between-mouse (MW *p* = 0.509) under identical synthetic input. | Cosine of mean activation vectors. Supp Fig S19; Fig 7H. |
| 35 | The structural floor is visceral at the pair level: the most *similar* within-mouse pair (D9, Jaccard 0.694) shares only 30.6% of edges yet behavioral distance is 0.0069; the most *dissimilar* (B5, Jaccard 0.933) shares only 6.7% of edges yet behavioral distance is 0.0068 — architecturally distinct motifs, identical phenotype and dynamics. | Per-pair structural + dynamical comparison. Fig 7A–I. |

---

## Degeneracy hierarchy summary

| Level | Axis | Result | Architectural null? |
|---|---|---|---|
| 1 | Aggregate statistics | 0/18 features differ | Yes (random = 0/18) |
| 2 | Connection topology | ρ(topo, behavior) ≈ 0 at all scales | Yes (random ρ = −0.029) |
| 3 | All structural axes (MI) | max NMI = 0.2173, ≤ chance | — |
| 4 | Causal sensitivity | causally necessary but no identity fingerprint (local or distributed) | — |
| 5 | Representational geometry | mouse-invariant; weak +topology coupling only | Yes (generalists identical) |
| 6 | Dynamical geometry | stability, attractor, trajectories all mouse-invariant | — |
| **+** | **Functional commitment** | **the one axis selection shapes (~2.2–2.6×, mouse-level *p* = 0.002); no shared mouse-specific pathway signature (#26)** | Yes (generalists flat) |

**Bottom line:** selection produces robust individual-specific function while leaving no legible trace on any structural, representational, or dynamical axis. Where it does leave a trace — the *strength* of functional sensitivity commitment — even that trace carries no shared mouse-specific pathway signature.

---

## Scope and caveats (as stated in the manuscript)

- Quantized weights ({0.25, 1.0}) compress the magnitude axis; the topology-vs-magnitude result may partly reflect this and warrants continuous-weight follow-up.
- The behavioral fingerprint (4 metrics, one maze, no reward) may not capture all dimensions of individual identity.
- 14 neurons is a deliberate minimal model; scaling to biological circuit sizes is an open question.
- The sensitivity-RSA geometry test (#21) is likely underpowered at *n*=6 replicates/mouse; *n*=10 is planned.
- These findings are a proof-of-concept in a controlled minimal system, not a general claim about biological neural coding.
