"""
Timeline forward-modelling pipeline for stellar stream dark matter detection.

Instead of classifying streams into DM families, this approach asks:
"Which specific subhalo encounter parameters reproduce the observed stream
morphology?" by forcing encounters with explicit parameters, simulating the
resulting stream, and scoring against real Gaia data.

Modules:
    scoring  -- metric functions comparing simulated vs observed streams
    pipeline -- TimelineForwardModel orchestrator (data prep, simulation, scoring)
"""
