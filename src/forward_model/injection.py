"""
Injection-recovery testing for the timeline forward model.

This is the scientific validation step: before trusting the pipeline on real
data, prove that when a subhalo impact with *known* parameters is injected into
a synthetic stream, the full timeline pipeline recovers those parameters.

Workflow:
    1. Generate a synthetic "truth" stream = base stream + a known encounter
       (mass, time-since-impact, phi1), using a *different* random seed than the
       candidate simulations so recovery is not a trivial identity match.
    2. Inject that stream as the observed data (`prepare(observed_override=...)`).
    3. Run the candidate grid and score every candidate against the truth.
    4. Report the recovered (best-fit) parameters and their error vs the truth.

A successful recovery (best-fit close to truth, truth among the top candidates,
best beating the null) demonstrates that the search + scoring machinery actually
constrains the encounter parameters from the observable morphology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..simulation.stream_gen import StreamParticles, generate_stream
from ..simulation.subhalo import (
    EncounterParams,
    apply_impulse_approximation,
    scale_radius_from_mass,
)
from .evolve import generate_perturbed_stream_evolved
from .pipeline import ForwardModelConfig, TimelineForwardModel

log = logging.getLogger(__name__)


@dataclass
class InjectionRecoveryResult:
    """Outcome of one injection-recovery experiment."""

    # Injected truth
    truth_log10_mass: float
    truth_t_since_gyr: float
    truth_phi1: float
    truth_seed: int

    # Recovered (best-fit)
    recovered_log10_mass: float = 0.0
    recovered_t_since_gyr: float = 0.0
    recovered_phi1: float = 0.0
    recovered_score: float = 0.0

    # Errors (recovered - truth)
    mass_error_dex: float = 0.0
    t_since_error_gyr: float = 0.0
    phi1_error_deg: float = 0.0

    # Context
    null_score: float = 0.0
    best_beats_null: bool = False
    truth_rank: int = -1            # rank of the grid point nearest the truth (0 = best)
    n_candidates: int = 0
    fast_mode: bool = True

    def to_dict(self) -> dict:
        return {
            "truth": {
                "log10_mass": self.truth_log10_mass,
                "t_since_gyr": self.truth_t_since_gyr,
                "phi1": self.truth_phi1,
                "seed": self.truth_seed,
            },
            "recovered": {
                "log10_mass": self.recovered_log10_mass,
                "t_since_gyr": self.recovered_t_since_gyr,
                "phi1": self.recovered_phi1,
                "combined_score": self.recovered_score,
            },
            "errors": {
                "mass_dex": self.mass_error_dex,
                "t_since_gyr": self.t_since_error_gyr,
                "phi1_deg": self.phi1_error_deg,
            },
            "null_score": self.null_score,
            "best_beats_null": self.best_beats_null,
            "truth_rank": self.truth_rank,
            "n_candidates": self.n_candidates,
            "fast_mode": self.fast_mode,
        }


def generate_injected_stream(
    cfg: ForwardModelConfig,
    log10_mass: float,
    t_since_gyr: float,
    phi1: float,
    seed: int = 123,
    mws=None,
) -> dict:
    """Generate a synthetic observed stream containing one known encounter.

    Returns an observed-particle dict (phi1, phi2, pm1, pm2, dist, vrad,
    membership_prob) suitable for ``TimelineForwardModel.prepare(observed_override=...)``.
    """
    from ..simulation.potentials import get_mw_potential

    potential = get_mw_potential(cfg.config_path)
    if mws is None:
        import galstreams
        mws = galstreams.MWStreams(verbose=False)

    mass = 10.0 ** log10_mass
    encounter = EncounterParams(
        mass_solar=mass,
        scale_radius_kpc=scale_radius_from_mass(mass),
        impact_param_kpc=cfg.impact_param_kpc,
        flyby_vel_kms=cfg.flyby_vel_kms,
        encounter_phi1=phi1,
        t_since_impact_gyr=t_since_gyr,
        is_valid=True,
        is_massive=(mass > 1e8),
    )

    if cfg.use_fast_mode:
        base = generate_stream(
            stream_name=cfg.stream_name,
            potential=potential,
            n_stars=cfg.n_stars_sim,
            seed=seed,
            config_path=cfg.config_path,
            mws=mws,
        )
        perturbed = apply_impulse_approximation(base, encounter)
    else:
        perturbed = generate_perturbed_stream_evolved(
            stream_name=cfg.stream_name,
            potential=potential,
            encounter=encounter,
            n_stars=cfg.n_stars_sim,
            seed=seed,
            config_path=cfg.config_path,
            mws=mws,
        )

    n = len(perturbed.phi1)
    return {
        "phi1": perturbed.phi1,
        "phi2": perturbed.phi2,
        "pm1": perturbed.pm1,
        "pm2": perturbed.pm2,
        "dist": perturbed.dist,
        "vrad": perturbed.vrad,
        "membership_prob": np.ones(n, dtype=np.float64),
    }


def run_injection_recovery(
    cfg: ForwardModelConfig,
    truth_log10_mass: float,
    truth_t_since_gyr: float,
    truth_phi1: float,
    truth_seed: int = 123,
) -> tuple[InjectionRecoveryResult, list]:
    """Run a full injection-recovery experiment.

    The candidate grid is taken from ``cfg``; make sure the truth lies inside it.
    The truth stream uses ``truth_seed`` (distinct from ``cfg.base_seed``) so the
    recovery is non-trivial.

    Returns:
        (InjectionRecoveryResult, sorted list of CandidateResult).
    """
    import galstreams
    mws = galstreams.MWStreams(verbose=False)

    log.info("Injection: M=10^%.2f, t=%.2f Gyr, phi1=%.1f deg (seed=%d, %s)",
             truth_log10_mass, truth_t_since_gyr, truth_phi1, truth_seed,
             "fast" if cfg.use_fast_mode else "full orbit")

    obs = generate_injected_stream(
        cfg, truth_log10_mass, truth_t_since_gyr, truth_phi1, seed=truth_seed, mws=mws,
    )

    model = TimelineForwardModel(cfg)
    model._mws = mws
    model.prepare(observed_override=obs)
    results = model.run_grid()

    best = results[0]
    null_score = model.null_score.combined if model.null_score else float("inf")

    # Rank of the grid point closest to the truth (so we can report whether the
    # truth region scores well even if a near-degenerate point edges it out).
    def _dist(r):
        return (
            abs(r.log10_mass - truth_log10_mass) / max(cfg.log10_mass_step, 1e-6)
            + abs(r.t_since_gyr - truth_t_since_gyr) / max(cfg.t_since_step, 1e-6)
            + abs(r.impact_phi1 - truth_phi1) / 2.0
        )
    nearest = min(results, key=_dist)
    truth_rank = results.index(nearest)

    result = InjectionRecoveryResult(
        truth_log10_mass=truth_log10_mass,
        truth_t_since_gyr=truth_t_since_gyr,
        truth_phi1=truth_phi1,
        truth_seed=truth_seed,
        recovered_log10_mass=best.log10_mass,
        recovered_t_since_gyr=best.t_since_gyr,
        recovered_phi1=best.impact_phi1,
        recovered_score=best.score.combined,
        mass_error_dex=best.log10_mass - truth_log10_mass,
        t_since_error_gyr=best.t_since_gyr - truth_t_since_gyr,
        phi1_error_deg=best.impact_phi1 - truth_phi1,
        null_score=null_score,
        best_beats_null=best.score.combined < null_score,
        truth_rank=truth_rank,
        n_candidates=len(results),
        fast_mode=cfg.use_fast_mode,
    )
    return result, results
