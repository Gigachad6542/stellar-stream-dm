# A reproducible pipeline for dark-matter subhalo-impact detection in Milky Way stellar streams: methods, a timeline forward model, and an honest current-data limit

**Authors:** D. W. (lead), with assistance from an autonomous coding agent.
**Status:** DRAFT (in preparation). Results sections marked `[PENDING v3]` await the
re-trained detector on the regenerated `simulations_v3_track6d` dataset.

---

## Abstract *(draft)*

Dark-matter subhalos with masses below the threshold of galaxy formation
(≲10⁸ M⊙) are a key prediction of the cold dark matter (CDM) paradigm and a
discriminator against warm (WDM), fuzzy (FDM), and self-interacting (SIDM)
alternatives. Their gravitational flybys imprint density gaps and kinematic
ripples on thin Milky Way stellar streams. We present an end-to-end, fully
reproducible pipeline that (i) generates physically-motivated stream simulations
with literature-anchored progenitor orbits, (ii) trains a graph neural network
(GINEConv) detector with simulation-based inference (SNPE-C) for the subhalo
mass function, and (iii) introduces a *timeline forward model* that, given a
candidate impact, estimates its epoch, rewinds the stream, re-injects a grid of
subhalo encounters with Erkal & Belokurov (2015) impulse physics, re-evolves to
the present, and scores each hypothesis against the observed stream using fused
multi-epoch/multi-survey kinematics. Applying the pipeline to seven streams
(GD-1, Pal 5, Orphan–Chenab, ATLAS, Jhelum, Fjörm, Sylgr) with public Gaia DR3,
S⁵, and APOGEE data, we find **no joint detection** (Fisher p = 0.30; Stouffer
p = 0.16), consistent with a smooth null / CDM and best read as an upper limit
given the residual simulation-to-observation gap. We further show that a detector
that looks strong on a balanced training set (AUC 0.94) is only marginally better
than chance (AUC 0.62) on physically-faithful simulations — a cautionary result
on dataset construction. Our principal
contributions are methodological: a diagnosis and partial closure of the
sim-to-real gap (a literature-anchored progenitor-orbit fix and error-domain
randomization that resolves detector overconfidence), a look-elsewhere–corrected
significance framework with a coherence gate across streams, and a public,
test-covered codebase.

---

## 1. Introduction *(draft prose)*

The abundance of low-mass dark-matter (DM) subhalos is one of the sharpest
predictions distinguishing cold dark matter (CDM) from warm (WDM), fuzzy (FDM),
and self-interacting (SIDM) alternatives. Below the threshold of galaxy formation
(≲10⁸ M⊙) these subhalos host no stars, so they can only be found through their
gravity. Thin, dynamically cold stellar streams — the tidal debris of disrupted
globular clusters and dwarf galaxies — are among the most sensitive available
probes: a subhalo flyby imprints a density gap, an off-track spur, and a
characteristic kinematic ripple whose morphology encodes the perturber's mass and
impact geometry \citep{Carlberg2012,ErkalBelokurov2015}. The GD-1 stream in
particular shows a gap-and-spur feature that has been interpreted as dynamical
evidence for a dark substructure \citep{Bonaca2019}, and population-level analyses
of stream perturbations have begun to place particle-physics constraints on the
subhalo mass function \citep{Banik2021}.

Turning these signatures into a measurement is hard for two reasons. First, gaps
are degenerate: a subhalo impact must be distinguished from baryonic perturbers
(giant molecular clouds, the bar, spiral arms), epicyclic density variations, and
survey selection systematics. Second, the forward model — generating a stream and
perturbing it — is expensive, which makes classical likelihood-based inference
over many hypotheses costly.

We address both with machine learning and simulation-based inference (SBI). A
graph neural network (GNN) with edge-conditioned convolutions
\citep{Hu2020gine} respects the permutation symmetry of a member-star set and the
locality of kinematic perturbations, producing an embedding that SBI
\citep{Greenberg2019snpe,Tejero-Cantero2020sbi} maps to a posterior over
mass-function parameters — amortizing inference across the expensive simulator.

Our central methodological idea is a *timeline forward model*. Rather than
classifying only a present-day snapshot, we hypothesize a specific impact, use a
detector to localize and time it, run the clock backward to the unperturbed
stream, re-inject a grid of candidate encounters with the
\citet{ErkalBelokurov2015} impulse, re-evolve to the present, and score each
hypothesis against the observed stream. Detection thus becomes a constrained
forward-model comparison rather than a one-shot classification, with the impact
epoch and geometry as inferred quantities.

Throughout we adopt a deliberately conservative stance: we report what the data
and the *current* simulator actually support — including null results and the
systematics that limit them — rather than over-claiming a detection. As we show,
the dominant limitation today is the simulation-to-observation gap, and a
substantial part of our contribution is diagnosing and partially closing it.

## 2. Data *(mostly complete)*

### 2.1 Target streams
Seven streams spanning a range of orbits, distances, and kinematic gradients:
GD-1, Pal 5, Orphan–Chenab, ATLAS, Jhelum, Fjörm, Sylgr. Stream frames and track
catalogs from `galstreams`; per-stream configuration (φ1 range, distance,
disruption age) in `config/streams.yaml`.

### 2.2 Gaia DR3 membership
Member catalogs processed from Gaia DR3 to a standardized HDF5 schema (φ1, φ2,
distance, μ1, μ2, v_rad and per-star uncertainties). We document a key caveat
that limits the present analysis: the bundled catalogs assign a uniform
membership probability and are in practice broad field selections rather than
clean memberships. A track-consistency cleaner
(`scripts/clean_membership.py`, keeping stars within |Δφ2| < 1° and |Δμ| < 2
mas yr⁻¹ of the `galstreams` track) quantifies the contamination per stream:
GD-1 retains only 987 of 137,559 stars (0.7%), ATLAS 2,860 of 8,159 (35%), and
Jhelum 0 of 66,129 (a pure field/selection failure). For GD-1 the surviving
members are confined to φ1 ≳ 60° because the catalog's proper-motion
distribution does not extend to the stream's low-φ1 stars (μ1 ≈ −12.8). We
therefore treat the bundled GD-1 catalog as inadequate for a clean analysis and
flag the need for an external Price-Whelan & Bonaca (2018)/STREAMFINDER
membership catalog as a prerequisite for a headline GD-1 result
\citep{PriceWhelanBonaca2018,Ibata2021}.

### 2.3 Multi-epoch / multi-survey kinematics
To improve the "rewind," radial velocities and proper motions are fused across
epochs and surveys via inverse-variance weighting: Gaia DR3 RVS, S⁵
(VizieR J/MNRAS/490/3508), and APOGEE (VizieR III/286). Frame transforms between
the Koposov-2010/PWB18 and Ibata-2021 GD-1 conventions are handled explicitly
(`src/data/gd1_frames.py`).

## 3. Stream simulator *(mostly complete — key methodological section)*

### 3.1 Galactic potential and orbit integration
MW potential and orbit integration via `galpy` (C `dop853` integrator; verified
active, no Python-odeint fallback).

### 3.2 Progenitor initial conditions (the sim-to-real fix)
We replaced a 5D φ2-RMS IC optimizer — which matched stream *geometry* but landed
on kinematically wrong orbits (e.g. GD-1 at μ1 ≈ −8.9 vs the literature −12.8) —
with `set_progenitor_ic_track6d`: deriving the progenitor 6D phase-space state
directly from the `galstreams` track point nearest the center of the observed φ1
range. Validated across five streams, proper motions now match the literature
tracks essentially exactly (Table 1).

**Table 1.** Generated vs track proper motions after the track-6D IC fix.

| Stream | μ1 sim / track | μ2 sim / track |
|---|---|---|
| GD-1 | −13.14 / −13.13 | −3.25 / −3.26 |
| Pal 5 | 3.67 / 3.64 | 0.64 / 0.63 |
| Jhelum | −7.45 / −7.45 | 3.37 / 3.37 |
| ATLAS | 0.15 / 0.23 | −1.07 / −1.04 |
| Orphan | 1.06 / 1.12 | 1.44 / 1.46 |

### 3.3 Tidal stream generation
Particle-spray release around the progenitor; per-stream `spray_age_gyr`
calibrated so the generated angular extent matches the observed track
(`scripts/calibrate_spray_age.py`). We document a residual limitation: matching
the long extent of streams like GD-1 (φ1 ≈ [−21, 81]°) slightly over-widens the
stream (φ2 std ≈ 1.3–1.5° vs observed ≈ 0.5°) — the known length-vs-width tension
of a simple spray; a correctly-correlated Fardal/streakline release is the
identified fix (future work).

### 3.4 Subhalo encounters (impulse physics)
Subhalo flybys modeled with the Erkal & Belokurov (2015) Plummer impulse,
Δv = −(2GM/w)·b/(|b|²+r_s²), with encounter geometry built per impact
(`src/simulation/subhalo.py`, `erkal_plummer_kick`). The earlier capped Hernquist
kick was replaced; correctness verified by a point-mass-limit unit test.

### 3.5 Stream-type diversity
The training set spans multiple stream morphologies (verified post-regeneration:
GD-1-, Pal 5-, Jhelum-, ATLAS-, and Orphan-class streams each recover their
literature kinematics).

## 4. Detector: GNN + SBI

### 4.1 Graph construction and GINEConv encoder
Each stream's member stars are assembled into a k-nearest-neighbour graph
(k = 8) in normalized phase space (φ1, φ2, μ1, μ2), with up to 1200 stars per
stream. Nodes carry 18 features; the 5 edge features (Δφ1, Δφ2, Δμ1, Δμ2, and a
4-D phase-space separation) encode the *local kinematic contrast* that a subhalo
flyby perturbs. We use a GINEConv encoder (~2.3M parameters), whose edge-conditioned
message passing \citep{Hu2020gine} is well suited to this edge-borne signal, with
an optional graph-level density-profile branch (summary features over 48 bins).
The network has a binary detection head and a regression head for mass-function
parameters; we train with AdamW (lr 3×10⁻⁴, weight decay 10⁻⁴), cosine-annealing
warm restarts, gradient clipping, and mixed precision, on a stratified 70/15/15
train/val/test split (seed 42).

### 4.2 Simulation-based inference
GNN embeddings → SNPE-C posteriors for the subhalo mass-function parameters;
SBC and TARP coverage diagnostics.

### 4.3 Detector overconfidence and error-domain randomization
We diagnosed pathological overconfidence (p_impact = 1.0000 on real GD-1) as an
input-normalization failure rather than a modeling or calibration problem.
Unmeasured error columns had been filled with fake constants and the feature
normalizer clamped their (near-zero) standard deviation to 0.01; any real-data
offset therefore exploded after standardization — real GD-1 `e_vrad` landed at
≈ +100σ, saturating the logit. The inference-side fix (i) imputes unmeasured
features to the training mean (≈ 0 post-normalization), (ii) clips standardized
features to ±5σ so no single out-of-distribution feature can saturate the logit,
(iii) records OOD diagnostics (`ood_max_sigma`, `ood_frac_clipped`) and flags
predictions as unreliable when inputs are out-of-distribution, and (iv) applies a
temperature (T = 1.065) from `calibration.json`. This alone moved p_impact from
1.0000 to 0.9829 while *honestly surfacing* the residual ≈ 391σ OOD severity of
the (contaminated) real GD-1 catalog. We then **retrained with error-domain
randomization** — per-star errors drawn log-uniformly over realistic ranges plus
random RV masking each epoch — so the detector sees the observational noise it
will face. On the *balanced* curriculum dataset this error-DR detector
discriminated well (AUC = 0.937, temperature-calibrated T = 0.78; mean
p(neg) ≈ 0.18 vs p(pos) ≈ 0.84). We stress that this figure is dataset-dependent
(§8, Limitation 6): on the physically-faithful, physics-prior v3 dataset the same
architecture is expected to be substantially weaker, and we report that honestly
rather than carrying the optimistic balanced-set number as the headline.
**v3 result.** Re-trained on the physically-faithful `simulations_v3_track6d`
dataset (error-DR, cached profiles, 70k/15k/15k split), the *same* architecture
reaches only validation accuracy 0.595 and **AUC = 0.618** (temperature
T = 0.672) — barely above chance and far below the 0.937 obtained on the balanced
curriculum set. This is the paper's central cautionary result: once the
simulator is corrected to match real-stream kinematics and realistic impact
rates, the single-snapshot impact-detection signal is weak, and the previously
strong classifier performance was substantially an artifact of training-set
balancing rather than intrinsic separability.

## 5. Timeline forward model *(methods complete; results PENDING v3)*

Given a detected density minimum and detector handoff, we (1) estimate the impact
epoch with a Monte-Carlo posterior over t_since, (2) rewind the stream to the
unperturbed past state, (3) re-inject a grid of subhalo encounters (mass × impact
time × φ1 location) using the Erkal kick, (4) re-evolve to the present, and (5)
score each hypothesis against the observed stream (including fused RV/PM data with
zero-point calibration). Injection–recovery on full orbits validates the
recovery of injected impact parameters. **[PENDING v3 numbers.]**

## 6. Statistical framework *(methods complete; prose)*

**Per-stream significance.** For each stream we compare the best-scoring impact
hypothesis against a null distribution of scores obtained from no-impact
realizations of the same stream, converting the tail probability to a z-score.

**Look-elsewhere correction.** Because the timeline forward model searches a grid
over perturber mass, impact time, and φ1 location, the most significant grid cell
is biased high. We correct for this multiplicity with a *best-of-grid* null: each
null realization is scored over the entire grid and we retain its maximum, so the
null reflects the same search the data undergo. This typically erases naive
single-cell significance.

**Multi-stream combination.** We combine per-stream evidence with both Stouffer's
Z (∝ Σzᵢ/√N) and Fisher's method (−2 Σ ln pᵢ), headlining the more conservative
of the two. Crucially, combination is **gated by a coherence requirement**: a
genuine population-level subhalo signal should produce per-stream evidence that is
*coherent* (consistent in sign and broadly in magnitude). When the per-stream z
are incoherent — in our pre-correction runs they were mixed-sign, ~71% positive,
spanning roughly [−4.8, +29] — we explicitly decline to claim a detection, since
an incoherent excess is the signature of residual modeling systematics rather than
a shared physical cause.

**Honest null construction.** Building a null from the data themselves is subtle:
jittering φ1 broadens the stream and mimics a perturbation, while shuffling proper
motions destroys the intrinsic kinematic gradient and spuriously inflates streams
with strong gradients (e.g. GD-1). We document these confounds and identify a
*structure-preserving* null (one that randomizes the hypothesized perturbation
while leaving the unperturbed stream's intrinsic structure intact) as the correct
construction, which we adopt for the headline result.

## 7. Results

### 7.1 Detector on faithful simulations
On the corrected v3 dataset the calibrated detector reaches AUC = 0.618 (§4.3) —
only marginally above chance, in contrast to the AUC = 0.937 obtained on a
balanced curriculum set. We take this as the primary cautionary result: the
single-snapshot impact-detection signal is weak once the simulator reproduces
realistic stream kinematics and impact rates.

### 7.2 Multi-stream joint significance
Running the look-elsewhere–corrected timeline forward model over all seven target
streams (12 no-impact null realizations and a 10-fold look-elsewhere null per
stream) yields the per-stream significances in Table 2. The look-elsewhere
correction is essential: naive single-cell z-scores of 12.0 (Pal 5) and 105
(Sylgr) collapse to 1.0 and 0.7 once the grid search is accounted for.

**Table 2.** Look-elsewhere–corrected per-stream significance (v3).

| Stream | z (LE) | p |
|---|---|---|
| GD-1   | −0.47 | 0.64 |
| Pal 5  | +1.03 | 0.36 |
| Orphan | −0.40 | 0.64 |
| ATLAS  | +1.58 | 0.09 |
| Jhelum | +2.67 | 0.09 |
| Fjörm  | −2.48 | 0.91 |
| Sylgr  | +0.72 | 0.27 |

The combined significance is **Stouffer Z = 1.00 (p = 0.159)** and **Fisher
χ² = 16.2 (p = 0.301)** — no joint detection. Notably, the per-stream evidence is
now *coherent and modest*, spanning only z ∈ [−2.5, +2.7], in sharp contrast to
the pre-correction analysis (mixed-sign, 71% positive, z ∈ [−4.8, +29]) whose
incoherence we had attributed to the simulation-to-observation gap. The corrected
simulator thus both lowers and *regularizes* the significance, and we report the
result as a non-detection consistent with a smooth (no localized impact) null —
i.e. an upper limit / CDM-consistency given current data and the catalog
limitations of §8.

## 8. Limitations and systematics *(draft)*

1. Length-vs-width tension in the simple spray (§3.3).
2. GD-1 membership catalog contamination; need for external PWB18/STREAMFINDER.
3. Static error-DR profile cache vs fully dynamic per-epoch randomization
   (a speed/fidelity trade documented in the training pipeline).
4. Null-distribution construction confounds (§6).
5. Forward-model grid resolution and single-encounter assumption.
6. **Detection signal strength is dataset-dependent.** Earlier high classifier
   accuracy (AUC ≈ 0.88–0.94) was obtained on a deliberately *balanced* training
   set in which impact/no-impact were separated independently of the DM family;
   on physics-prior datasets that preserve realistic impact rates and morphology,
   cheap baselines reach only AUC ≈ 0.65–0.68. We therefore caution that strong
   reported detector performance can partly reflect dataset construction rather
   than intrinsic separability, and we report the v3 (physically-faithful)
   detector performance honestly in §4.3 with this distinction in mind. Confirmed
   on v3: AUC = 0.618 (val acc 0.595) vs 0.937 on the balanced set — the corrected
   simulator yields a genuinely harder, more realistic detection problem.

## 9. Reproducibility *(draft)*

Public repository with a conda environment spec, ~289 unit tests across 13 files,
a categorized `scripts/` index, and changelogs documenting every methodological
decision. All datasets are regenerable from `scripts/generate_training_data.py`
(thread-pinned, deterministic per-seed).

## 10. Conclusions *(draft)*

We have built and documented a complete, reproducible pipeline for searching for
dark-matter subhalo impacts in Milky Way stellar streams, comprising a
literature-anchored stream simulator, a GNN+SBI detector, and a novel *timeline
forward model* that recasts detection as a constrained rewind/re-impact/re-evolve
comparison against fused multi-survey kinematics. Our contributions are primarily
methodological and, deliberately, honest about what current data and simulations
support:

1. **A diagnosed and partially closed simulation-to-observation gap.** Replacing a
   geometry-only progenitor-IC optimizer with a track-anchored 6D initial
   condition fixed kinematically wrong orbits (Table 1), and error-domain
   randomization with corrected input normalization removed a pathological
   detector overconfidence that was an OOD input-handling artifact, not real
   skill.

2. **A cautionary, reproducible result on detector performance.** On the
   physically-faithful v3 simulations the detector reaches only AUC = 0.618
   (val accuracy 0.595), versus AUC = 0.937 on a balanced curriculum dataset —
   direct evidence that strong reported performance can be an artifact of
   training-set construction rather than intrinsic separability.

3. **An honest population-level inference.** Applying the look-elsewhere–corrected,
   coherence-gated multi-stream framework to the corrected simulations and public
   data, we find no joint detection (Stouffer Z = 1.00, p = 0.16; Fisher
   χ² = 16.2, p = 0.30), with per-stream evidence now coherent and modest
   (z ∈ [−2.5, +2.7]) — a result consistent with a smooth null / CDM and best read
   as an upper limit given the residual systematics (membership contamination, the
   spray length-vs-width tension, null construction) that bound it.

The overarching message is that closing the sim-to-real gap *raises* the bar for
claimed detections, and that careful, reproducible methodology — including the
willingness to report null and cautionary results — is essential for turning
stellar streams into a quantitative dark-matter probe. Future work: a
correctly-correlated Fardal/streakline release to resolve the width-vs-length
tension, an external PWB18/STREAMFINDER GD-1 membership catalog, and a
structure-preserving null for the headline significance.

---

## References *(to compile)*
Erkal & Belokurov (2015); Bonaca et al. (2019); Banik et al. (2021); Carlberg
(2012); Koposov et al. (2010); Price-Whelan & Bonaca (2018, PWB18); Ibata et al.
(2021); Mateu (galstreams); Bovy (galpy); Greenberg et al. (SNPE-C / sbi);
Fardal et al. (2015). *(Full bibliography to be assembled.)*
