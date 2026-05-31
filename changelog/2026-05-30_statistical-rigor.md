# 2026-05-30 Statistical rigor: null distribution + multi-seed (timeline deliverable 3)

## Why

The forward model ranks candidate encounters by a combined score, but a raw
score is not interpretable: "how different is the best fit from the data, and is
that meaningful versus no impact at all?" needs a reference, and single-seed
scores carry sampling noise that reshuffles near-degenerate candidates.

## What changed

### New: `src/forward_model/significance.py`
- `compute_significance(candidate_score, null_scores)` -> `SignificanceResult`
  with a **z-score** `(null_mean - score)/null_std` and a one-sided empirical
  **p-value** `P(null <= score)` (add-one smoothed).

### `src/forward_model/pipeline.py`
- `_generate_base_stream(seed=...)` and `evaluate_candidate(params, seed=...)`
  now accept a seed override (fast mode regenerates a base realization per seed).
- `_score_stream_vs_obs(stream)` helper (shared scoring path, incl. the RV term).
- `build_null_distribution(n_realizations, seed0)` — scores many *unperturbed*
  realizations vs the observations (the no-impact baseline distribution).
- `evaluate_candidate_multiseed(params, seeds)` — mean/std of a candidate's score
  over several seeds.

### `scripts/run_timeline_forward_model.py`
- `--significance N`: build an N-realization null distribution and report the best
  candidate's z-score + p-value; saved to `significance.json`.
- `--n-seeds K`: re-rank the top candidates by their K-seed-averaged score
  (removes sampling noise); stores mean/std per candidate.
- `--significance-seed0`: first seed for the null realizations.

### Tests: `tests/test_significance.py` (6)
Strong candidate significant; candidate within null not significant; one-sided
p-value fraction; zero-std handling; serialization; empty-null error.

## Verification (known cases via injection)

| observed stream | best | null mean±std | z | p |
|---|---|---|---|---|
| **injected impact M=10^8.5** | 0.332 | 0.483 ± 0.046 | **3.29** | 0.062* |
| **no impact (unperturbed)** | 0.389 | 0.441 ± 0.091 | **0.58** | 0.375 |

The null distribution cleanly separates a real impact (3.3σ) from no impact
(0.6σ). *The injected p hits the n=15 empirical floor (1/16=0.0625); more null
realizations sharpen it — the z-score is the cleaner statistic at small n.

Full non-slow suite: **260 passed, 6 deselected**.

## Caveat (documented in the module)

Picking the *best* of many candidates beats the null by chance more often than a
single candidate (look-elsewhere). The first-order test here compares the best
score to the unperturbed-null distribution; a fully calibrated p-value would use
the distribution of *best* scores under the null (re-running the grid per null
realization) — supported by passing a best-score null distribution to
`compute_significance`.
