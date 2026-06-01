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
S⁵, and APOGEE data, we find **[PENDING v3: a joint significance consistent with
no detection / an upper limit on the impact rate]**, which we interpret honestly
in the context of the residual simulation-to-observation gap. Our principal
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
distance, μ1, μ2, v_rad and per-star uncertainties). We document a key caveat:
the bundled GD-1 catalog is a broad field selection (~96% contamination); a
track-consistency cleaner (`scripts/clean_membership.py`) recovers on-track
members but exposes that a proper external PWB18/STREAMFINDER membership catalog
is required for a clean GD-1 analysis. *(Quantified per-stream contamination in
the 2026-05-31 membership-cleaning work.)*

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

## 4. Detector: GNN + SBI *(methods complete; numbers PENDING v3)*

### 4.1 Graph construction and GINEConv encoder
Member stars → k-NN phase-space graph; edge features carry the kinematic
perturbation signal. GINEConv encoder (~2.3M parameters) with an optional
density-profile branch (summary features, 48 bins).

### 4.2 Simulation-based inference
GNN embeddings → SNPE-C posteriors for the subhalo mass-function parameters;
SBC and TARP coverage diagnostics.

### 4.3 Detector overconfidence and error-domain randomization
We diagnosed pathological overconfidence (p ≈ 1.0) as an input-normalization
failure: an unmeasured-feature standard-deviation clamp pushed `e_vrad` to
≈ +100σ out-of-distribution. The fix imputes unmeasured features to the training
mean, clips to ±5σ, flags OOD inputs, and applies temperature scaling; combined
with **error-domain randomization** (per-star error draws + RV masking during
training) this closes much of the sim-to-real gap. Result: the spurious
391σ→13σ collapse and p 0.98→0.16 on real GD-1, validation AUC ≈ 0.937.
**[PENDING v3: re-trained on `simulations_v3_track6d`; report AUC, calibration,
real-stream scores.]**

## 5. Timeline forward model *(methods complete; results PENDING v3)*

Given a detected density minimum and detector handoff, we (1) estimate the impact
epoch with a Monte-Carlo posterior over t_since, (2) rewind the stream to the
unperturbed past state, (3) re-inject a grid of subhalo encounters (mass × impact
time × φ1 location) using the Erkal kick, (4) re-evolve to the present, and (5)
score each hypothesis against the observed stream (including fused RV/PM data with
zero-point calibration). Injection–recovery on full orbits validates the
recovery of injected impact parameters. **[PENDING v3 numbers.]**

## 6. Statistical framework *(methods complete)*

- Per-stream significance against a null distribution; the look-elsewhere effect
  corrected by a best-of-grid null.
- Multi-stream combination via Stouffer-Z and Fisher's method, gated by a
  **coherence** requirement (incoherent per-stream signs ⇒ not a detection).
- Honest null construction: we document confounds in data-driven nulls
  (φ1-jitter broadening; pm-shuffle destroying kinematic gradients) and why a
  structure-preserving null is needed.

## 7. Results *(PENDING v3)*

`[PENDING v3]` Re-run on the regenerated dataset + re-trained detector:
- Per-stream significances and the joint result (coherence-gated).
- Either an honest non-detection consistent with CDM, or an upper limit on the
  subhalo impact rate / mass function in the probed regime.
- Comparison to the pre-fix result (incoherent per-stream z, 71% positive,
  range [−4.8, +29]) which we attributed to the sim-to-real gap.

## 8. Limitations and systematics *(draft)*

1. Length-vs-width tension in the simple spray (§3.3).
2. GD-1 membership catalog contamination; need for external PWB18/STREAMFINDER.
3. Static error-DR profile cache vs fully dynamic per-epoch randomization
   (a speed/fidelity trade documented in the training pipeline).
4. Null-distribution construction confounds (§6).
5. Forward-model grid resolution and single-encounter assumption.

## 9. Reproducibility *(draft)*

Public repository with a conda environment spec, ~289 unit tests across 13 files,
a categorized `scripts/` index, and changelogs documenting every methodological
decision. All datasets are regenerable from `scripts/generate_training_data.py`
(thread-pinned, deterministic per-seed).

## 10. Conclusions *(PENDING v3)*

---

## References *(to compile)*
Erkal & Belokurov (2015); Bonaca et al. (2019); Banik et al. (2021); Carlberg
(2012); Koposov et al. (2010); Price-Whelan & Bonaca (2018, PWB18); Ibata et al.
(2021); Mateu (galstreams); Bovy (galpy); Greenberg et al. (SNPE-C / sbi);
Fardal et al. (2015). *(Full bibliography to be assembled.)*
