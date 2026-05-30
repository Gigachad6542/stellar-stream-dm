"""Simulation-based inference (SBI) pipeline.

Modules:
    sbi_pipeline       — Prior construction, SNPE-C training, embedding wrappers
    posteriors         — Posterior sampling, credible intervals, evidence estimation
    calibration        — Simulation-based calibration (Talts+2018)
    diagnostics        — Posterior predictive checks, model misspecification detection
    suppression_scale  — Model-independent half-mode mass inference (primary constraint)
    hierarchical       — Hierarchical Bayesian multi-stream M_hm model (NumPyro/emcee)
"""
