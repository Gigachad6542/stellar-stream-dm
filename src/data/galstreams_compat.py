"""Compatibility wrapper for constructing galstreams on modern pandas."""
from __future__ import annotations


def make_mwstreams(verbose: bool = False):
    """Construct ``galstreams.MWStreams`` without pandas Arrow-string crashes.

    galstreams converts Astropy tables to pandas DataFrames during startup.
    On Windows, repeated construction with pandas 3's inferred Arrow strings can
    trigger a native access violation. The library only needs ordinary Python
    strings for its small metadata tables, so disable Arrow-backed inference
    during construction.
    """
    import galstreams
    import pandas as pd

    options = []
    for key, value in (
        ("future.infer_string", False),
        ("mode.string_storage", "python"),
    ):
        try:
            pd.get_option(key)
        except Exception:
            continue
        options.extend((key, value))

    with pd.option_context(*options):
        return galstreams.MWStreams(verbose=verbose)
