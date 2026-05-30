"""
Gaia DR3 query and download utilities.

Queries the ESA Gaia TAP+ service via astroquery, caches raw results as FITS,
and joins with external membership catalogs where available.

Usage:
    from src.data.gaia_query import query_stream_members, load_membership_catalog
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table, join
from astroquery.gaia import Gaia

log = logging.getLogger(__name__)

# Gaia archive async login (anonymous is fine for public DR3)
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1


def _load_stream_config(stream_name: str, config_path: str = "config/streams.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["streams"][stream_name]


def _load_global_cuts(config_path: str = "config/streams.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["global_quality_cuts"]


def query_stream_members(
    stream_name: str,
    output_path: str | Path,
    config_path: str = "config/streams.yaml",
    force: bool = False,
) -> Table:
    """Query Gaia DR3 for all sources in the bounding box of a stream.

    Downloads to output_path as a FITS file if not already cached.
    Applies global quality cuts from streams.yaml.

    Args:
        stream_name: Key in config/streams.yaml (e.g., "GD1").
        output_path: Path to write the raw FITS catalog.
        config_path: Path to streams.yaml.
        force: Re-download even if the cache file exists.

    Returns:
        Astropy Table with Gaia DR3 columns plus quality flags.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        log.info("Loading cached Gaia query: %s", output_path)
        return Table.read(output_path)

    sc = _load_stream_config(stream_name, config_path)
    cuts = _load_global_cuts(config_path)

    # Build bounding box with 5-degree padding
    pad = 5.0
    ra_c = sc["ra_center_deg"]
    dec_c = sc["dec_center_deg"]
    phi1_min, phi1_max = sc["phi1_range_deg"]
    # Use explicit overrides from streams.yaml if present; otherwise compute from phi1 range.
    # Long streams (phi1 span > 60 deg) can generate boxes of 100×50 deg which return
    # tens of millions of rows — always pair with an icrs_pm_precut in that case.
    if "icrs_ra_width_override_deg" in sc:
        ra_width = sc["icrs_ra_width_override_deg"]
        log.info("Using explicit RA box width %.1f deg for %s", ra_width, stream_name)
    else:
        ra_width = abs(phi1_max - phi1_min) + 2 * pad

    if "icrs_dec_width_override_deg" in sc:
        dec_width = sc["icrs_dec_width_override_deg"]
        log.info("Using explicit Dec box width %.1f deg for %s", dec_width, stream_name)
    else:
        dec_width = abs(phi1_max - phi1_min) / 3.0 + 2 * pad

    # Optional ICRS PM pre-cut: dramatically reduces row count before coordinate transform
    pm_precut_clauses = ""
    if "icrs_pm_precut" in sc:
        pmc = sc["icrs_pm_precut"]
        pm_precut_clauses = (
            f"\n    AND gs.pmra IS NOT NULL"
            f"\n    AND gs.pmra BETWEEN {pmc['pmra_min']} AND {pmc['pmra_max']}"
            f"\n    AND gs.pmdec IS NOT NULL"
            f"\n    AND gs.pmdec BETWEEN {pmc['pmdec_min']} AND {pmc['pmdec_max']}"
        )
        log.info("Applying ICRS PM pre-cut: pmra [%.1f, %.1f], pmdec [%.1f, %.1f]",
                 pmc["pmra_min"], pmc["pmra_max"], pmc["pmdec_min"], pmc["pmdec_max"])

    adql = f"""
    SELECT
        gs.source_id, gs.ra, gs.dec,
        gs.l, gs.b,
        gs.parallax, gs.parallax_error,
        gs.pmra, gs.pmra_error,
        gs.pmdec, gs.pmdec_error,
        gs.radial_velocity, gs.radial_velocity_error,
        gs.phot_g_mean_mag,
        gs.phot_bp_mean_mag,
        gs.phot_rp_mean_mag,
        gs.bp_rp,
        gs.ruwe,
        gs.astrometric_excess_noise,
        gs.phot_bp_rp_excess_factor
    FROM gaiadr3.gaia_source AS gs
    WHERE CONTAINS(
        POINT('ICRS', gs.ra, gs.dec),
        BOX('ICRS', {ra_c}, {dec_c}, {ra_width}, {dec_width})
    ) = 1
    AND gs.phot_g_mean_mag < {cuts['phot_g_mean_mag_max']}
    AND (gs.parallax IS NULL OR gs.parallax < 0.5)
    AND gs.ruwe < {cuts['ruwe_max']}
    AND gs.phot_bp_rp_excess_factor < {cuts['bp_rp_excess_factor_max']}{pm_precut_clauses}
    """

    log.info("Launching async Gaia query for %s...", stream_name)

    # Retry with exponential backoff — Gaia archive returns HTTP 500 during
    # DR4 migration windows ("may be unstable at times").
    max_retries = 8
    backoff = 60  # seconds; doubles each attempt up to 600s
    for attempt in range(1, max_retries + 1):
        try:
            job = Gaia.launch_job_async(adql)
            while not job.is_finished():
                time.sleep(5)
            result = job.get_results()
            break  # success
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = min(backoff * (2 ** (attempt - 1)), 600)
            log.warning("Gaia query attempt %d/%d failed (%s). Retrying in %ds...",
                        attempt, max_retries, exc, wait)
            time.sleep(wait)
    log.info("Query returned %d rows for %s", len(result), stream_name)

    result = _apply_quality_cuts(result, cuts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write(str(output_path), format="fits", overwrite=True)
    log.info("Saved raw query to %s", output_path)
    return result


def _apply_quality_cuts(table: Table, cuts: dict) -> Table:
    """Remove rows that fail global quality criteria."""
    mask = np.ones(len(table), dtype=bool)
    if "ruwe" in table.colnames:
        mask &= table["ruwe"] < cuts["ruwe_max"]
    if "parallax" in table.colnames and "parallax_error" in table.colnames:
        with np.errstate(invalid="ignore", divide="ignore"):
            plx_snr = np.abs(table["parallax"]) / table["parallax_error"]
        mask &= ~(np.isfinite(plx_snr) & (plx_snr > cuts["parallax_over_error_max"]))
    if "phot_bp_rp_excess_factor" in table.colnames:
        mask &= table["phot_bp_rp_excess_factor"] < cuts["bp_rp_excess_factor_max"]
    if "astrometric_excess_noise" in table.colnames:
        mask &= table["astrometric_excess_noise"] < cuts["astrometric_excess_noise_max"]
    return table[mask]


def apply_proper_motion_cut(table: Table, stream_name: str, config_path: str = "config/streams.yaml") -> Table:
    """Retain only stars within the stream's expected proper motion range."""
    sc = _load_stream_config(stream_name, config_path)
    pm1_min, pm1_max = sc["pm1_range_masyr"]
    pm2_min, pm2_max = sc["pm2_range_masyr"]
    mask = (
        (table["pmra"] >= pm1_min)
        & (table["pmra"] <= pm1_max)
        & (table["pmdec"] >= pm2_min)
        & (table["pmdec"] <= pm2_max)
    )
    return table[mask]


def apply_color_magnitude_selection(
    table: Table,
    isochrone_path: str | Path,
    color_col: str = "bp_rp",
    mag_col: str = "phot_g_mean_mag",
    color_width: float = 0.15,
    mag_faint_limit: float = 21.0,
) -> Table:
    """Retain stars within a CMD box around a PARSEC isochrone.

    Args:
        table: Input catalog with color and magnitude columns.
        isochrone_path: Path to a two-column ASCII file (color, G_mag).
        color_width: Half-width of the CMD selection box in magnitudes.
        mag_faint_limit: Faint magnitude cutoff.

    Returns:
        Filtered table.
    """
    iso = np.loadtxt(isochrone_path, usecols=(0, 1))
    iso_color, iso_mag = iso[:, 0], iso[:, 1]

    colors = np.asarray(table[color_col])
    mags = np.asarray(table[mag_col])

    # For each star, interpolate expected isochrone color at its magnitude
    valid_iso = (iso_mag >= mags.min() - 1) & (iso_mag <= mag_faint_limit + 1)
    if valid_iso.sum() < 2:
        log.warning("Isochrone covers insufficient magnitude range; skipping CMD cut.")
        return table

    expected_color = np.interp(mags, iso_mag[valid_iso], iso_color[valid_iso])
    on_iso = np.abs(colors - expected_color) < color_width
    bright_enough = mags < mag_faint_limit
    return table[on_iso & bright_enough]


def load_membership_catalog(
    stream_name: str,
    raw_dir: str | Path,
    config_path: str = "config/streams.yaml",
) -> Optional[Table]:
    """Load a pre-built membership catalog for a stream.

    Returns None if no catalog file is found in raw_dir; the caller should
    then fall back to query_stream_members + statistical membership.

    Expected filename pattern: {raw_dir}/{stream_name}_members.fits
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / f"{stream_name}_members.fits"
    if not path.exists():
        log.warning("No pre-built membership catalog found at %s", path)
        return None
    table = Table.read(str(path))
    log.info("Loaded membership catalog for %s: %d stars", stream_name, len(table))
    return table
