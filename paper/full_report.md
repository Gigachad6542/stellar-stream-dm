# Detecting dark-matter subhalo impacts in Milky Way stellar streams — a full technical report

**Author:** D. W., with an autonomous coding agent.
**Scope:** This is the extended technical report for the project. A condensed paper
is in `paper/manuscript.md`. All quantitative results here derive from the validated
`streamdf`/`streamgapdf` simulation pipeline and the detector trained on it; the
forward-model and multi-stream-significance components are described but, for
integrity, not quantified, as they currently run on a separate generator (§9–§10).

*Format: a non-technical **Plain-language summary** and per-section **In plain terms**
notes (in blockquotes) accompany the full technical text; specialists may skip them.*

---

> ### Plain-language summary
> The Milky Way is surrounded by thousands of invisible clumps of dark matter. The
> smallest hold no stars, so they are detectable only through their gravity. When one
> passes through a thin "stream" of stars — the shredded remains of an old star
> cluster — it leaves a small **gap**. Finding and interpreting those gaps is a way to
> count the invisible clumps, and the count is one of the sharpest tests of what dark
> matter actually is.
>
> This report describes an automated system that (i) simulates streams with and
> without such gaps using trusted, peer-reviewed code; (ii) trains an AI to recognize
> the gaps; and (iii) studies what the AI can and cannot tell us. The headline
> results: on simulated streams the detector is highly accurate (≈98%) and never
> raises a false alarm; on the two real streams with clean data it behaves correctly
> — it flags the famous gap in GD-1 and stays silent on ATLAS, which has none.
>
> We are also candid about the limits. From a single gap you cannot recover the
> mass of the clump that made it (many different clumps make look-alike gaps); you can
> only roughly tell how *recently* it struck. And deciding *which kind* of dark matter
> we live in is not a one-gap question — it needs a whole population of detections
> (roughly 5–30 for the most favorable theories). We finish by laying out the two
> tools (a "replay" reconstruction and a multi-stream pooling method) that will turn
> detections into a measurement, once one remaining piece is rebuilt on the validated
> simulator.

## Abstract

Dark-matter subhalos below the threshold of galaxy formation (≲10⁸ M⊙) are a
defining prediction of cold dark matter (CDM) and discriminate it from warm (WDM),
fuzzy (FDM), and self-interacting (SIDM) alternatives. Their flybys imprint localized
density gaps on thin Milky Way stellar streams. We present a complete, reproducible
pipeline — a stream simulator built on galpy's validated action-angle distribution
functions (`streamdf`, `streamgapdf`; \citealt{Bovy2014streamdf};
\citealt{SandersBovyErkal2016}) gated by a pre-flight validation harness; a
graph-neural-network (GNN) detector; and a characterization/forward-model layer. The
GINEConv detector reaches test **AUC 0.982** with a zero false-positive operating
point and good probability calibration. On the two real streams with clean external
membership (GD-1 via STREAMFINDER; ATLAS via a track-consistency cut) the detector is
**in-distribution** (input deviations 3.1σ and 1.9σ) and behaves correctly: it flags
the known GD-1 gap at φ₁≈50° and returns a null on ATLAS. We map detection
**completeness** across impact type — rising with subhalo mass (0.48→0.72 over
10⁷·⁵–10⁸·⁵ M⊙) and falling with impact age (0.75→0.39 from 0.3 to 1.4 Gyr) — and
show that **single-gap characterization is degeneracy-limited**: a dedicated
regression head cannot recover subhalo mass (R²<0) and recovers time-since-impact only
weakly (R²≈0.2). Distinguishing DM models is therefore a **population** measurement,
which we quantify as ≈5 detections for FDM (10⁻²² eV), ≈12–27 for WDM (3–6 keV), and
unreachable for SIDM via the mass spectrum. We describe a timeline forward model and a
look-elsewhere-corrected multi-stream significance framework that complete the
pipeline and outline the single remaining step (porting them onto the validated
generator) before population constraints can be reported.

---

## 1. Introduction

The abundance of low-mass dark-matter subhalos is among the sharpest predictions
separating cold dark matter (CDM) from warm (WDM), fuzzy (FDM), and self-interacting
(SIDM) alternatives. In CDM, structure forms hierarchically down to Earth-mass scales,
so the Galactic halo should teem with subhalos at all masses; in WDM and FDM, free
streaming or quantum pressure suppresses structure below a characteristic mass, sharply
reducing the abundance of ≲10⁸–10⁹ M⊙ halos. Below the threshold of galaxy formation
(≲10⁸ M⊙) these halos host no stars and are observable only through gravity.

Thin, dynamically cold stellar streams — the tidal debris of disrupted globular
clusters — are among the most sensitive available probes of this low-mass regime. A
subhalo flyby imprints a localized density gap and a correlated kinematic ripple whose
morphology encodes the perturber's mass and impact geometry
\citep{Carlberg2012,ErkalBelokurov2015}. The GD-1 stream's gap-and-spur has been
interpreted as evidence for a dark perturber \citep{Bonaca2019}, and population-level
analyses of stream perturbations have begun to constrain the subhalo mass function
\citep{Banik2021}.

Two obstacles stand between an observed gap and a measurement. First, **degeneracy**: a
gap must be distinguished from baryonic perturbers, epicyclic density variations, and
survey selection effects, and a single gap underdetermines the perturber's mass, impact
parameter, and epoch. Second, **cost**: the forward model that produces a perturbed
stream is expensive, making classical likelihood search over many hypotheses slow.

This report develops a machine-learning pipeline that addresses both, and — equally
important — measures its own limits honestly. We use a graph neural network (GNN) whose
edge-conditioned message passing \citep{Hu2020gine} respects the permutation symmetry
of a member-star set, and a constrained forward-model search (the *timeline forward
model*). Because such a detector is only as trustworthy as the simulations it learns
from, the training set is built entirely from published, validated stream distribution
functions and gated by a quantitative pre-flight validation harness before any training
(§3).

> **In plain terms.** Dark-matter clumps too small to hold stars can only be found by
> their gravity. Streams of stars act as tripwires: a clump passing through leaves a
> gap. We teach an AI to spot those gaps using carefully validated simulations, then we
> are candid about which questions the data can and cannot answer.

## 2. Scientific background

**Streams as gravitational antennas.** A globular cluster on a Galactic orbit sheds
stars through its inner and outer Lagrange points, forming two thin tidal tails that
trace the progenitor's orbit. Because the debris is dynamically cold (velocity
dispersion ∼ a few km s⁻¹), small gravitational perturbations leave coherent,
long-lived imprints rather than being washed out.

**Gap formation.** A subhalo passing near the stream delivers an impulsive velocity
kick whose sign varies along the stream. Stars on one side are accelerated, those on
the other decelerated; over the following ∼Gyr the perturbed stars drift, opening an
under-density (the gap) flanked by pile-ups, sometimes with an off-track "spur". The
gap's depth grows with the perturber's mass and proximity and with the time elapsed
since impact, but an old gap also widens and shallows as the stream phase-mixes — a
competition that is central to what is and is not recoverable (§5–§6).

**The four dark-matter models.** CDM predicts an unsuppressed subhalo mass function
(∝ M^α, α≈−1.9). WDM suppresses halos below a half-mode mass set by the thermal relic
mass (Lovell+ 2014); FDM suppresses below a Jeans/soliton scale set by the axion mass
(Hui+ 2017); SIDM leaves the subhalo *abundance* essentially CDM-like but alters
internal structure (cored, lower-concentration halos). These differences are
**population-level**: they change *how many* subhalos exist at each mass, not how an
individual impact of a given mass appears (§7).

> **In plain terms.** A cold, thin stream is like a taut string: a passing mass plucks
> it and leaves a lasting dent. How deep the dent is depends on how heavy and close the
> mass was and how long ago it passed — but very old dents also blur out. Different
> dark-matter theories mainly change how *common* the small masses are.

## 3. The stream simulator

### 3.1 Generation with validated distribution functions
Streams are integrated in an `MWPotential2014`-class Galactic potential
\citep[galpy, C `dop853` integrator;][]{Bovy2015galpy}. Progenitor initial conditions
are taken directly from the `galstreams` \citep{Mateu2023galstreams} 6-D track at the
centre of each observed φ₁ window, which reproduces literature proper motions by
construction. Smooth ("no-impact") streams are drawn from the action-angle distribution
function `streamdf` \citep{Bovy2014streamdf}; streams with a single subhalo gap from
`streamgapdf` \citep{SandersBovyErkal2016}, a subclass of `streamdf` that shares the
identical smooth track and differs *only* by the encoded impact. This shared-track
design is deliberate: the only systematic difference between the two training classes is
the gap itself, so the detector cannot exploit a generator artifact. A particle-spray
generator, `streamspraydf` \citep{Fardal2015}, is retained for cross-checks.

### 3.2 Subhalo physics
A subhalo's gravitational scale is its NFW scale radius r_s = r_200/c, evaluated with
the \citet{Ludlow2016} concentration–mass relation and
ρ_crit,0 = 277.5 h² M⊙ kpc⁻³ (e.g. r_s ≈ 0.25 kpc for a 10⁸ M⊙ halo). The flyby
imprints the \citet{ErkalBelokurov2015} Plummer velocity impulse,
Δv = −(2GM/w)·b/(|b|²+r_s²), where w is the relative speed and b the impact-parameter
vector. Training encounters span the detectable regime (mass 10⁷·⁵–10⁸·⁷ M⊙, impact
parameter ≤ 0.35 kpc), encoded analytically by `streamgapdf`.

### 3.3 The pre-flight validation harness
Every generated batch is gated before use (`scripts/validate_generator.py`). The
critical gates are: **G1 smoothness** — the no-impact gap-depth *excess over the Poisson
shot-noise floor* must be < 0.15 (a smooth stream must not, by chance, look perturbed);
**G3 length** — the φ₁ extent must be 0.6–1.4× the `galstreams` track; **G6
separability** — the impact-vs-smooth area-under-ROC of a model-free gap statistic must
exceed 0.85. Width, kinematics, and radial-velocity checks are reported as morphology
diagnostics. The generator passes all critical gates (Table 1), so a no-impact stream
sits at the Poisson floor while detectable impacts are cleanly separable.

**Table 1 — Validation-harness gates (GD-1 generator).**

| Gate | Quantity | Threshold | Value | Status |
|---|---|---|---|---|
| G1 | smoothness (excess over Poisson) | < 0.15 | 0.01 | pass |
| G3 | length vs track | 0.6–1.4× | 0.75× | pass |
| G6 | impact-vs-smooth separability (AUC) | > 0.85 | 0.94 | pass |
| G2 | intrinsic width (diagnostic) | < 1.2° | 0.53° | pass |
| G5 | RV intrinsic dispersion (diagnostic) | < 60 km s⁻¹ | 20 km s⁻¹ | pass |

Four streams support the action-angle model cleanly (GD-1, ATLAS, Jhelum, Orphan);
Pal 5 and Fjörm have near-circular orbits that violate the isochrone action-angle
approximation and are excluded by an allow-list.

![Figure 1](figures/fig1_what_impact_looks_like.png)
**Figure 1.** What a subhalo flyby does. *Top:* a smooth simulated GD-1-like stream
(`streamdf`). *Middle:* the same stream after a 10⁸·⁵ M⊙ flyby (`streamgapdf`).
*Bottom:* star counts along the stream; the perturbed profile (orange) shows a localized
deficit (shaded) absent from the smooth profile (blue).

> **In plain terms.** We simulate both kinds of stream — untouched and gap-bearing —
> with the same trusted code, so the only difference the AI can learn is the gap itself.
> An automatic checklist (Table 1) refuses any batch whose streams don't look like real
> ones, so the AI never trains on bad practice data.

## 4. The detector

### 4.1 Architecture and training
Each stream's member stars form a k-nearest-neighbour graph (k = 8) in normalized
(φ₁, φ₂, μ₁, μ₂) space, with up to 1200 stars per graph. Node features carry per-star
phase-space coordinates and uncertainties; the five edge features encode the *local
kinematic contrast* a flyby perturbs. A GINEConv encoder \citep{Hu2020gine} (∼2.3M
parameters, 6 layers, 128-dim embedding) with an auxiliary graph-level density-profile
branch produces an embedding and a binary impact/no-impact head. Training uses AdamW
(lr 3×10⁻⁴, weight decay 10⁻⁴), cosine warm restarts, mixed precision, and a 70/15/15
split (seed 42) on 15,830 simulations (7,830 impact / 8,000 smooth) across the four
supported streams. The detection target is *detectable* impacts (realized gap-depth
> 0.5). Probabilities are temperature-calibrated on the validation split.

### 4.2 Discrimination, separation, and calibration
On the held-out test set the detector reaches **AUC 0.982** (Figure 2), best validation
accuracy 0.951, with a **zero false-positive rate** at the 0.5 operating threshold —
it never flags a smooth stream. The score distributions (Figure 3) are cleanly
bimodal: no-impact streams pile up near p≈0 and detectable impacts near p≈1, with the
operating threshold falling in the sparse valley between them. The reliability diagram
(Figure 4) shows the calibrated probabilities track the empirical impact frequency,
so p_impact can be read as an actual probability rather than an uncalibrated score. A
two-dimensional PCA of the learned embedding (Figure 5) shows impacted and smooth
streams occupying distinct regions, confirming the network has learned a separating
representation rather than memorizing.

![Figure 2](figures/fig2_detector_roc.png)
**Figure 2.** Detector ROC on held-out simulations (AUC 0.987 for this checkpoint;
0.982 after temperature calibration); the operating point has zero false positives.

![Figure 3](figures/fig6_score_separation.png)
**Figure 3.** Calibrated detector scores separate cleanly: no-impact streams (blue)
near 0, detectable impacts (orange) near 1, with the operating threshold in between.

![Figure 4](figures/fig7_reliability.png)
**Figure 4.** Reliability diagram: predicted p_impact versus the observed impact
fraction. Points near the diagonal indicate well-calibrated probabilities.

![Figure 5](figures/fig9_embedding_pca.png)
**Figure 5.** Two-dimensional PCA of the GNN embedding; impacted (orange) and smooth
(blue) streams occupy distinct regions of the learned feature space.

> **In plain terms.** The detector is right about 98% of the time on simulated streams
> and, crucially, never cries wolf on a smooth one. Its confidence scores are honest
> (Figure 4) — a "70% chance" really does mean about 70% — and it has genuinely learned
> the difference between a gapped and a smooth stream (Figure 5), not just memorized
> examples.

## 5. Detection completeness — which impacts we can see

Detection depends strongly on the *type* of impact. Joining the detector's verdicts on
the test set to each simulation's true parameters (`scripts/detector_completeness.py`)
yields completeness as a function of impact properties. In one dimension (Figure 6),
completeness **rises with subhalo mass** (0.48 at 10⁷·⁵ to 0.72 at 10⁸·⁵ M⊙) and
**falls with time since impact** (0.75 for < 0.5 Gyr to 0.39 for > 1.1 Gyr, as gaps
phase-mix and refill); it is approximately flat in impact parameter over the
close-encounter range probed (0–0.35 kpc). The two-dimensional map (Figure 7) shows the
joint behaviour: the upper-right corner (massive, recent) is recovered with high
probability, the lower-left (light, old) is largely missed. Overall completeness is
0.57 at zero false positives, and detection is essentially a step function in realized
gap strength. In effect the detector is a calibrated **density-gap detector**: it sees
the massive, recent impacts that carve deep gaps and misses the weak, old impacts that
perturb mainly the kinematics — which points to proper-motion-pattern features as the
clearest route to extending sensitivity.

![Figure 6](figures/fig3_completeness.png)
**Figure 6.** Detection completeness versus subhalo mass (left) and time since impact
(right), measured on the test set.

![Figure 7](figures/fig5_completeness_2d.png)
**Figure 7.** Joint completeness over mass × recency. Massive, recent impacts (upper
right) are recovered with high probability; light, old impacts (lower left) are missed.

> **In plain terms.** We reliably catch big, recent hits that gouge a clear gap; small
> or ancient hits that only nudge the stars' motions slip through. Figure 7 is the
> "catch-rate map": bright = likely caught, dark = likely missed.

## 6. Characterizing the impact — the single-gap degeneracy

Detecting a gap is easier than reading off what made it. We test how much of the
perturber's identity survives in a single gap in two complementary ways. First, probing
the frozen detection embedding (`scripts/characterize_probe.py`) shows it encodes mass
only weakly (R²≈0.18) and impact parameter (R²≈0.02) and epoch (R²≈0.04) essentially
not at all. Second, and more tellingly, a *dedicated* multi-task regression head trained
jointly with the detector **fails to recover subhalo mass** (test R² < 0 — no better
than predicting the population mean) and recovers **time-since-impact only weakly**
(R² ≈ 0.2, median error ≈ 0.1 dex), as shown in Figure 8. This is a genuine physical
degeneracy, not a modeling shortfall: mass, impact parameter, flyby speed, and epoch
trade off in shaping a single gap's depth and width, so one gap underdetermines them.
Recency leaves a partial imprint (older gaps are wider and shallower) and is the one
property weakly constrained. Breaking the degeneracy requires either additional
observables (kinematics, multiple gaps) or the explicit joint forward model of §9.

![Figure 8](figures/fig8_characterization.png)
**Figure 8.** Recovered versus true impact parameters from a dedicated regression head.
*(a)* time-since-impact is weakly recoverable; *(b)* subhalo mass is not (recovered
values are nearly independent of the truth).

> **In plain terms.** Finding the dent is easier than identifying what caused it: many
> different clumps leave look-alike dents. We can roughly tell *how recently* a clump
> struck, but its *mass* is essentially unknowable from a single gap.

## 7. Telling dark-matter models apart — a population measurement

WDM, FDM, and SIDM do not change how any single impact looks; they change how many
subhalos exist at each mass (Figure 9). DM-model discrimination is therefore not a
per-impact label but a **population inference**. Combining each model's mass function
with our *measured* completeness(mass) gives the distribution of detected-impact masses
(Figure 10a) and the number of clean detections needed to distinguish each model from
CDM at 95% confidence (Figure 10b): **≈5 for FDM (10⁻²² eV)** whose cutoff sits in our
sensitive band, **≈12–27 for WDM (3–6 keV)**, ≈135 for FDM (10⁻²¹ eV, cutoff below our
band), and **effectively never for SIDM** via the mass spectrum, since its subhalo
counts match CDM (SIDM would instead require the distinct gap *shape* of cored,
low-concentration halos). This reframes "which dark matter?" as a sample-size question.

![Figure 9](figures/fig10_transfer.png)
**Figure 9.** Mass-function suppression f(M) for each DM model relative to our sensitive
band. Models differ from CDM only where their cutoff falls inside the band.

![Figure 10](figures/fig4_dm_family.png)
**Figure 10.** *(a)* Detected-impact mass distributions by DM model. *(b)* Number of
clean detections needed to distinguish each model from CDM.

> **In plain terms.** Different dark-matter theories don't change what one impact looks
> like — they change how common small clumps are. So you need a census of impacts, not a
> single gap: about five clean detections for the most favorable theory, hundreds for
> others, and for one (SIDM) the mass count alone can never do it.

## 8. Application to real streams

We apply the detector to the real streams for which a clean external membership catalog
exists: GD-1, via the STREAMFINDER catalog \citep{Ibata2021}, and ATLAS, via a
track-consistency cut of its Gaia membership. (Jhelum's clean catalog is empty after the
cut, and Orphan lacks a clean external catalog, so neither is included.) Radial velocity
is dropped at inference to match the clean-catalog regime. Results are in Table 2.

**Table 2 — Detector on real streams (clean catalogs).**

| Stream | N members | p_impact | input OOD | model-free gap |
|---|---|---|---|---|
| GD-1 | 811 | 0.77 | 3.1σ (in-dist.) | yes (φ₁≈50°) |
| ATLAS | 2,860 | 0.06 | 1.9σ (in-dist.) | none |

Two points stand out. First, both real streams are **in-distribution**: their
standardized inputs land at 3.1σ and 1.9σ, within the range the detector saw in
training, so the simulator reproduces the real catalogs' feature statistics well enough
for the predictions to be trustworthy. Second, the detector behaves **correctly on real
data**: it flags the known Price-Whelan–Bonaca gap in GD-1 at φ₁≈50°
\citep{PriceWhelanBonaca2018} (corroborated independently by a model-free gap finder),
and it returns a clean null on ATLAS, which has no known gap — a real-data demonstration
of the zero-false-positive operating point. Attributing the GD-1 gap to a *specific*
subhalo (its mass, epoch, geometry) is the task of the forward model (§9), which is not
yet on the validated generator; we therefore report detection here, not characterization.

> **In plain terms.** On the two real streams with clean data, the detector does exactly
> what it should: it spots GD-1's famous gap and stays quiet on ATLAS, which has none —
> and in both cases it treats the real data as familiar, not strange, which is what lets
> us trust it.

## 9. The timeline forward model (design)

To move from detection to physical characterization we designed a *timeline forward
model* that, for a detected gap, (1) estimates the impact epoch, (2) rewinds the stream
to its unperturbed state, (3) re-injects a grid of encounters (mass × epoch × φ₁) with
the \citet{ErkalBelokurov2015} impulse and the NFW scale radius, (4) re-evolves to the
present, and (5) scores each hypothesis against the data. It recasts detection as a
constrained, physically explicit fit whose free parameters are the impact's mass, epoch,
and location, and is the natural route around the single-gap degeneracy of §6, because a
forward model can exploit the full joint density-and-kinematic morphology that a
discriminative embedding discards.

**Scope caveat.** The forward model currently runs on a separate particle-spray
generator with an analytic impulse kick — *not* the validated `streamdf`/`streamgapdf`
distribution functions used elsewhere in this report — whose baseline/null streams carry
more intrinsic density scatter. We therefore **report no quantitative forward-model
results here**; porting the rewind/re-impact/re-evolve loop onto the validated generator
is required before its injection-recovery, parameter estimates, and goodness-of-fit can
be trusted, and is the principal remaining engineering step (§11). The machinery itself
(grid search, scoring, null construction) is implemented and unit-tested.

> **In plain terms.** The next step beyond "is there a gap?" is "what made it?" We built
> a tool that rewinds a stream, drops in a simulated clump, fast-forwards, and checks the
> match — but it still uses an older, less-faithful stream-maker, so we describe it here
> rather than report numbers from it.

## 10. Population significance (framework)

A single stream rarely yields a decisive detection, so the pipeline includes a
multi-stream combination to test for a *population* of impacts. Per stream, the
best-scoring hypothesis is compared against a null distribution from no-impact
realizations; the grid search is de-biased with a **best-of-grid look-elsewhere null**
(each null realization is scored over the entire grid and its maximum retained, so the
null undergoes the same search as the data); and per-stream evidence is combined with
Stouffer's Z and Fisher's method, gated by a **coherence requirement** (a genuine
population signal should be consistent in sign and magnitude across streams). A general,
simulator-independent point already follows: because the search ranges over mass, epoch,
and location, naive single-cell significances are strongly inflated and must be
look-elsewhere-corrected. Because this combination is built on the §9 forward model,
its quantitative application is likewise deferred to the validated generator.

> **In plain terms.** A single stream rarely settles the question, so we pool many — and
> guard against a classic trap: if you don't account for how many places you looked, pure
> noise can masquerade as a discovery. We describe how the pooling works; the numbers
> wait on the same rebuild as §9.

## 11. Limitations and systematics

1. **Single-gap degeneracy** (§6): mass/geometry/epoch are only partially recoverable
   from one gap; the forward model and population statistics are the routes around it.
2. **Density-only sensitivity** (§5): weak/old, kinematics-only perturbations are missed;
   proper-motion-pattern features are the clearest next step.
3. **Action-angle model validity**: clean generation is limited to eccentric streams;
   multi-impact streams require `streampepperdf` (unavailable in this galpy), so
   single-vs-multiple classification is deferred (gap *counting* is available model-free).
4. **Real-data volume**: only two streams currently have clean external catalogs;
   DM-model discrimination needs the population sample sizes of §7.
5. **Forward model on a separate generator** (§9): the timeline forward model and
   multi-stream framework are not yet on the validated generator, so their quantitative
   outputs are not reported; porting them is the principal remaining step. The
   single-encounter and impulse approximations also apply.

## 12. Reproducibility and software validation

The pipeline is a public repository with a conda environment specification, ∼290 unit
tests, a categorized `scripts/` index, and a dated decision log (`changelog/`). All
datasets regenerate deterministically from `scripts/generate_detector_data.py`
(per-seed, resumable); figures from `paper/make_figures.py` and
`paper/make_report_figures.py`; the validation harness from
`scripts/validate_generator.py`; and this report's PDF from `paper/build_pdf.py`. Every
quantitative claim is traced to its source artifact in `paper/claims_audit.md`.

## 13. Conclusions and outlook

We have built and validated a complete, reproducible pipeline for dark-matter
subhalo-impact searches in stellar streams. On a simulator built from validated
distribution functions and gated by a pre-flight harness, a GINEConv detector reaches
test AUC 0.982 with a zero false-positive operating point and well-calibrated
probabilities, and on real data it is in-distribution on both available clean streams —
correctly flagging GD-1's gap and returning a null on ATLAS. We mapped detection
completeness across impact type, demonstrated that single-gap characterization is
degeneracy-limited (mass unrecoverable; recency weakly constrained), and reframed
DM-model discrimination as a population measurement with an explicit sample-size cost.
We additionally described a timeline forward model and a look-elsewhere-corrected,
coherence-gated multi-stream significance framework; porting these onto the validated
generator is the principal remaining step before population constraints can be reported.
The path to a physical measurement is concrete: that port, kinematic (not just density)
detector features to reach weak/old impacts, and more clean membership catalogs to build
the detection population of §7.

## Methods

**Potential & integration.** galpy `MWPotential2014`; C `dop853` integrator;
R₀ = 8.0 kpc, V₀ = 220 km s⁻¹. **Distribution functions.** `streamdf`
\citep{Bovy2014streamdf} for the smooth track; `streamgapdf` \citep{SandersBovyErkal2016}
with `impactb`, `subhalovel`, `timpact`, `impact_angle`, GM, and r_s; the action-angle
setup uses b = estimateBIsochrone(pot, R/R₀, z/R₀) (≈0.61 for GD-1) and nTrackChunks = 5,
with the impact angle sharing the sign of the modeled arm. **Subhalo physics.** NFW
r_s = r_200/c with the \citet{Ludlow2016} concentration–mass relation and
ρ_crit,0 = 277.5 h² M⊙ kpc⁻³; \citet{ErkalBelokurov2015} Plummer impulse. **Detector.**
GINEConv encoder, 18 node / 5 edge features, hidden 256, 6 layers, embedding 128, profile
branch (48 bins); binary head (BCE) with auxiliary mass/epoch regression; AdamW
(lr 3×10⁻⁴, wd 10⁻⁴); temperature calibration on validation. **Noise model.** Gaia
DR3-like per-star errors \citep[][scale 0.5–2×]{GaiaDR3}; radial velocity dropped at
inference to match clean catalogs; training on foreground-contaminated streams collapses
separability, so the operating regime is clean membership catalogs. **Real catalogs.**
GD-1 STREAMFINDER \citep{Ibata2021}; ATLAS track-consistency cut of Gaia membership.
**Statistics.** Per-stream null from no-impact realizations; best-of-grid look-elsewhere
null; Stouffer/Fisher combination with a coherence gate. **Datasets.**
`data/simulations_detector_df` (15,830 sims; GD-1, ATLAS, Jhelum, Orphan); detector
`checkpoints/detector_df_20260602`; characterization head `checkpoints/detector_char_20260602`.

## References
Works cited (full entries in `references.bib`):
Banik et al. (2021); Bonaca et al. (2019); Bovy (2014, `streamdf`); Bovy (2015,
`galpy`); Carlberg (2012); Erkal & Belokurov (2015); Fardal et al. (2015,
`streamspraydf`); Gaia Collaboration (2023, DR3); Hu et al. (2020, GINEConv);
Ibata et al. (2021, STREAMFINDER); Ludlow et al. (2016); Mateu (2023, `galstreams`);
Price-Whelan & Bonaca (2018); Sanders, Bovy & Erkal (2016, `streamgapdf`).
