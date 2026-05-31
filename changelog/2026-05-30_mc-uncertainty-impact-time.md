# 2026-05-30 Monte-Carlo uncertainty-aware impact time (task 9)

The final timeline deliverable: turn the impact-time point estimate into a
posterior whose width reflects the present-day measurement precision.

## Method
`pipeline.monte_carlo_impact_time(n_realizations, error_scale)` resamples the
observed kinematics (pm1, pm2, vrad, dist) within their per-star errors, re-runs
the candidate grid on each realization, and records the best-fit (t_since, mass,
phi1). The spread of best-fit times is the impact-time posterior. Because
backward orbit integration amplifies velocity errors, this directly links
measurement precision to dated-impact precision; ``error_scale`` lets one preview
the gain from tighter (multi-epoch / future-release) data.

- `significance.py`: `ImpactTimePosterior` + `summarize_impact_time`.
- `pipeline.py`: `monte_carlo_impact_time` (restores observed data afterward).
- `run_timeline_forward_model.py`: `--mc-time N` / `--mc-error-scale`, saves
  `impact_time_posterior.json`.
- `tests/test_significance.py`: slow test (zero error -> zero spread;
  more error -> >= spread; median within grid).

## Result (injected 1.5 Gyr impact)
| error scale | t_since median | std [Gyr] |
|---|---|---|
| 0.0 | 1.50 (= truth) | 0.000 |
| 1.0 | 1.50 | 0.109 |
| 2.0 | 1.50 | 0.354 |

The posterior is centred on the truth and its width scales with the error level,
vanishing in the zero-error limit -- so the real multi-epoch RVs (and future
tighter PMs) measurably sharpen the dated impact.

## Report
Regenerated `Stellar_Stream_DM_Full_Report.pdf` with new subsection 13.7 and
Figure 20 (posterior width vs error scale). All four accuracy improvements and
the timeline deliverables are now complete; suite 269 passed (9 slow deselected).
