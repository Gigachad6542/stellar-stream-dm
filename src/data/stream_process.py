"""
Transform raw Gaia catalogs into the standardized HDF5 processed format.

Handles:
- Stream-frame coordinate transforms (ICRS -> phi1/phi2) via galstreams
- Proper motion Jacobian transforms (uses astropy, not manual implementation)
- Dust extinction corrections (dustmaps Bayestar 2019)
- Photometric distance estimation for distant streams
- HDF5 schema write with normalization statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import astropy.coordinates as coord
import astropy.units as u
import h5py
import numpy as np
import yaml
from astropy.table import Table

from .galstreams_compat import make_mwstreams

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_stream_config(stream_name: str, config_path: str = "config/streams.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["streams"][stream_name]


# ---------------------------------------------------------------------------
# Stream frame transforms
# ---------------------------------------------------------------------------

def _get_stream_frame(sc: dict, stream_name: str):
    """Return the coordinate frame for ``stream_name``.

    If ``custom_frame_pole_ra_deg`` / ``custom_frame_pole_dec_deg`` are set in the
    stream config, builds a :class:`gala.coordinates.GreatCircleICRSFrame` with that
    pole (e.g. Jhelum, whose galstreams Jhelum-b-B19 track is misaligned with the
    actual stream by ~30° in phi2).  Otherwise falls through to the galstreams frame.
    """
    if "custom_frame_pole_ra_deg" in sc:
        import gala.coordinates as gc  # noqa: PLC0415
        pole = coord.SkyCoord(
            ra=sc["custom_frame_pole_ra_deg"] * u.deg,
            dec=sc["custom_frame_pole_dec_deg"] * u.deg,
        )
        log.debug(
            "Using custom GreatCircleICRSFrame for %s (pole RA=%.2f°, Dec=%.2f°)",
            stream_name, sc["custom_frame_pole_ra_deg"], sc["custom_frame_pole_dec_deg"],
        )
        return gc.GreatCircleICRSFrame(pole=pole)

    mws = make_mwstreams(verbose=False)
    return mws[sc["galstreams_key"]].stream_frame


def to_stream_frame(table: Table, stream_name: str, config_path: str = "config/streams.yaml") -> Table:
    """Convert ICRS positions and proper motions to stream-aligned (phi1, phi2) frame.

    Uses galstreams MWStreams object to get the great-circle rotation for this stream,
    or a custom GreatCircleICRSFrame when ``custom_frame_pole_ra/dec_deg`` is set in
    streams.yaml (e.g. Jhelum, whose galstreams frame is misaligned).
    Proper motions are transformed via astropy's full Jacobian, not by hand.

    Args:
        table: Astropy Table with columns ra, dec, pmra, pmdec (+ errors).
        stream_name: Key in streams.yaml.

    Returns:
        Table with added columns: phi1, phi2, pm_phi1, pm_phi2,
        e_pm_phi1, e_pm_phi2 (all in degrees / mas/yr).
    """
    sc = _load_stream_config(stream_name, config_path)
    frame = _get_stream_frame(sc, stream_name)

    # Build SkyCoord in ICRS with proper motions
    has_pm = "pmra" in table.colnames and "pmdec" in table.colnames
    # Use np.asarray() to strip any pre-existing Column units (e.g. Gaia FITS files
    # encode ra/dec in deg already; multiplying Column[deg] * u.deg → deg², which
    # astropy rejects with UnitTypeError).
    if has_pm:
        icrs = coord.SkyCoord(
            ra=np.asarray(table["ra"]) * u.deg,
            dec=np.asarray(table["dec"]) * u.deg,
            pm_ra_cosdec=np.asarray(table["pmra"]) * u.mas / u.yr,
            pm_dec=np.asarray(table["pmdec"]) * u.mas / u.yr,
            frame="icrs",
        )
    else:
        icrs = coord.SkyCoord(
            ra=np.asarray(table["ra"]) * u.deg,
            dec=np.asarray(table["dec"]) * u.deg,
            frame="icrs",
        )

    stream_coords = icrs.transform_to(frame)

    table = table.copy()
    table["phi1"] = stream_coords.phi1.deg
    table["phi2"] = stream_coords.phi2.deg

    if has_pm:
        try:
            table["pm_phi1"] = stream_coords.pm_phi1_cosphi2.to(u.mas / u.yr).value
            table["pm_phi2"] = stream_coords.pm_phi2.to(u.mas / u.yr).value
            # Propagate uncertainties via Monte Carlo (50 samples)
            if "pmra_error" in table.colnames and "pmdec_error" in table.colnames:
                table = _propagate_pm_errors(table, frame, n_samples=50)
        except TypeError:
            # Some galstreams frames (e.g. NGC3201-P21, Sylgr-I21) do not define a
            # differential (velocity) frame, so astropy cannot transform PMs to stream
            # coordinates.  Fall back to using ICRS pmra/pmdec as proxy features —
            # they are real measurements and still informative for the GNN even though
            # they are not in the stream-aligned frame.
            log.warning(
                "Stream frame for %s has no differential transform — "
                "using ICRS pmra/pmdec as proxy for pm_phi1/pm_phi2.", stream_name
            )
            table["pm_phi1"] = np.asarray(table["pmra"]).astype(np.float32)
            table["pm_phi2"] = np.asarray(table["pmdec"]).astype(np.float32)
            if "pmra_error" in table.colnames:
                table["e_pm_phi1"] = np.asarray(table["pmra_error"]).astype(np.float32)
            if "pmdec_error" in table.colnames:
                table["e_pm_phi2"] = np.asarray(table["pmdec_error"]).astype(np.float32)

    return table


def _propagate_pm_errors(table: Table, frame, n_samples: int = 1000) -> Table:
    """Monte-Carlo propagation of PM uncertainties into stream frame.

    Skipped silently if the stream frame has no differential transform support
    (e.g. NGC3201-P21, Sylgr-I21); in that case the caller already set
    e_pm_phi1/e_pm_phi2 from the ICRS error columns.
    """
    rng = np.random.default_rng(0)
    ra = np.asarray(table["ra"])
    dec = np.asarray(table["dec"])
    pmra = np.asarray(table["pmra"])
    pmdec = np.asarray(table["pmdec"])
    e_pmra = np.asarray(table["pmra_error"])
    e_pmdec = np.asarray(table["pmdec_error"])

    pm1_samples = []
    pm2_samples = []
    try:
        for _ in range(n_samples):
            sc = coord.SkyCoord(
                ra=ra * u.deg,
                dec=dec * u.deg,
                pm_ra_cosdec=(pmra + rng.normal(0, e_pmra)) * u.mas / u.yr,
                pm_dec=(pmdec + rng.normal(0, e_pmdec)) * u.mas / u.yr,
                frame="icrs",
            ).transform_to(frame)
            pm1_samples.append(sc.pm_phi1_cosphi2.to(u.mas / u.yr).value)
            pm2_samples.append(sc.pm_phi2.to(u.mas / u.yr).value)
    except TypeError:
        log.warning("MC PM error propagation skipped — stream frame has no differential support.")
        return table

    pm1_arr = np.array(pm1_samples)
    pm2_arr = np.array(pm2_samples)
    table["e_pm_phi1"] = pm1_arr.std(axis=0)
    table["e_pm_phi2"] = pm2_arr.std(axis=0)
    return table


# ---------------------------------------------------------------------------
# Dust correction
# ---------------------------------------------------------------------------

def compute_dust_correction(table: Table) -> Table:
    """Add dust-corrected magnitudes using the Bayestar 2019 3D dust map.

    Requires dustmaps package and the Bayestar map to be downloaded
    (dustmaps.bayestar.fetch() on first use).

    Adds columns: A_G, A_BP, A_RP, g0, bp0, rp0, bp_rp_0.
    """
    try:
        from dustmaps.bayestar import BayestarQuery  # noqa: PLC0415
    except ImportError:
        log.warning("dustmaps not available; skipping dust correction.")
        return table

    try:
        bq = BayestarQuery(max_samples=1)
    except (FileNotFoundError, OSError) as exc:
        log.warning("Bayestar 2019 dust map data not downloaded (%s); skipping dust correction. "
                     "Run `python -c \"from dustmaps.bayestar import fetch; fetch()\"` to download.", exc)
        return table

    # astropy Table has no .get(); use explicit column-existence check
    dist_kpc = np.asarray(table["dist"] if "dist" in table.colnames else np.ones(len(table)) * 10.0)

    # Strip Column units before attaching astropy units (FITS columns carry units already)
    coords_3d = coord.SkyCoord(
        ra=np.asarray(table["ra"]) * u.deg,
        dec=np.asarray(table["dec"]) * u.deg,
        distance=dist_kpc * u.kpc,
        frame="icrs",
    )
    ebv = bq(coords_3d, mode="median")
    ebv = np.where(np.isfinite(ebv), ebv, 0.0)

    # Schlafly & Finkbeiner 2011 Gaia band coefficients
    A_G = 2.740 * ebv
    A_BP = 3.374 * ebv
    A_RP = 2.035 * ebv

    table = table.copy()
    table["A_G"] = A_G
    table["A_BP"] = A_BP
    table["A_RP"] = A_RP
    if "phot_g_mean_mag" in table.colnames:
        table["g0"] = table["phot_g_mean_mag"] - A_G
    if "phot_bp_mean_mag" in table.colnames and "phot_rp_mean_mag" in table.colnames:
        table["bp0"] = table["phot_bp_mean_mag"] - A_BP
        table["rp0"] = table["phot_rp_mean_mag"] - A_RP
        table["bp_rp_0"] = table["bp0"] - table["rp0"]
    return table


# ---------------------------------------------------------------------------
# Distance estimation
# ---------------------------------------------------------------------------

def estimate_distances(table: Table, stream_name: str, config_path: str = "config/streams.yaml") -> Table:
    """Assign heliocentric distances to stream member stars.

    Nearby streams (distance_method: 'parallax'): invert Gaia parallax.
    Distant streams (distance_method: 'photometric'): use a simple
    main-sequence photometric parallax assuming the isochrone absolute magnitude
    at the star's dereddened color.

    Sets column 'dist' (kpc) and 'e_dist' (kpc).
    """
    sc = _load_stream_config(stream_name, config_path)
    method = sc.get("distance_method", "photometric")
    dist_range = sc["distance_kpc"]

    table = table.copy()

    if method == "parallax" and "parallax" in table.colnames:
        plx = np.asarray(table["parallax"])
        plx_err = np.asarray(table["parallax_error"] if "parallax_error" in table.colnames else np.zeros(len(table)))
        # Only use stars with positive, significant parallax
        good = (plx > 0) & (plx / np.maximum(plx_err, 1e-9) > 3)
        dist = np.full(len(table), np.mean(dist_range))
        dist[good] = 1.0 / plx[good]  # kpc (plx in mas)
        e_dist = np.full(len(table), 0.5 * (dist_range[1] - dist_range[0]))
        e_dist[good] = plx_err[good] / (plx[good] ** 2)
        table["dist"] = np.clip(dist, dist_range[0], dist_range[1])
        table["e_dist"] = e_dist

    else:
        # Photometric: assign stream mean distance with stream-width uncertainty
        d_mean = np.mean(dist_range)
        d_err = 0.15 * d_mean  # ~15% relative distance uncertainty
        table["dist"] = np.full(len(table), d_mean)
        table["e_dist"] = np.full(len(table), d_err)

    return table


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def write_to_hdf5(
    stream_name: str,
    table: Table,
    output_path: str | Path,
    track_table: Optional[Table] = None,
    gaia_release: str = "DR3",
) -> None:
    """Write processed stream to the standardized HDF5 schema.

    Schema:
        /streams/{stream_name}/members/{phi1,phi2,dist,pm1,pm2,vrad,errors,source_id,membership_prob}
        /streams/{stream_name}/track/{phi1_track,phi2_track,width_track}
        /streams/{stream_name}/meta/normalization/{feature_mean,feature_std}

    Args:
        stream_name: Key used as HDF5 group name.
        table: Processed astropy Table with stream-frame columns.
        output_path: Path to the output HDF5 file.
        track_table: Optional galstreams reference track.
        gaia_release: String label stored as attribute.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = ["phi1", "phi2", "dist", "pm_phi1", "pm_phi2"]
    error_cols = ["e_dist", "e_pm_phi1", "e_pm_phi2"]

    with h5py.File(str(output_path), "a") as f:
        grp = f.require_group(f"streams/{stream_name}")
        grp.attrs["gaia_release"] = gaia_release
        grp.attrs["n_members"] = len(table)

        mem = grp.require_group("members")
        _write_col(mem, "phi1", table, "phi1")
        _write_col(mem, "phi2", table, "phi2")
        _write_col(mem, "dist", table, "dist")
        _write_col(mem, "pm1", table, "pm_phi1")
        _write_col(mem, "pm2", table, "pm_phi2")
        _write_col(mem, "vrad", table, "radial_velocity", fill=np.nan)
        _write_col(mem, "e_dist", table, "e_dist", fill=1.0)
        _write_col(mem, "e_pm1", table, "e_pm_phi1", fill=0.1)
        _write_col(mem, "e_pm2", table, "e_pm_phi2", fill=0.1)
        _write_col(mem, "e_vrad", table, "radial_velocity_error", fill=np.nan)
        _write_col(mem, "source_id", table, "source_id", dtype=np.int64)
        _write_col(mem, "membership_prob", table, "membership_prob", fill=1.0)

        if track_table is not None:
            trk = grp.require_group("track")
            _write_col(trk, "phi1_track", track_table, "phi1")
            _write_col(trk, "phi2_track", track_table, "phi2")
            _write_col(trk, "width_track", track_table, "width", fill=1.0)

        # Store normalization statistics for the 11-feature node vector
        node_features = _extract_node_features(table)
        meta = grp.require_group("meta/normalization")
        for key in ("feature_mean", "feature_std"):
            if key in meta:
                del meta[key]
        meta["feature_mean"] = node_features.mean(axis=0).astype(np.float32)
        meta["feature_std"] = node_features.std(axis=0).astype(np.float32)

    log.info("Wrote processed stream %s (%d stars) to %s", stream_name, len(table), output_path)


def _write_col(grp: h5py.Group, name: str, table: Table, col: str, fill=0.0, dtype=np.float32) -> None:
    if name in grp:
        del grp[name]
    if col in table.colnames:
        arr = np.asarray(table[col]).astype(dtype)
    else:
        arr = np.full(len(table), fill, dtype=dtype)
    grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)


def _extract_node_features(table: Table) -> np.ndarray:
    """Return (N, 11) array of the standard node feature set."""
    def _get(col, fill=0.0):
        if col in table.colnames:
            return np.asarray(table[col], dtype=np.float32)
        return np.full(len(table), fill, dtype=np.float32)

    return np.column_stack([
        _get("phi1"), _get("phi2"), _get("dist"),
        _get("pm_phi1"), _get("pm_phi2"), _get("radial_velocity", fill=0.0),
        _get("e_dist", fill=1.0), _get("e_pm_phi1", fill=0.1),
        _get("e_pm_phi2", fill=0.1), _get("radial_velocity_error", fill=1.0),
        _get("membership_prob", fill=1.0),
    ])


# ---------------------------------------------------------------------------
# Top-level pipeline helper
# ---------------------------------------------------------------------------

def process_stream(
    stream_name: str,
    raw_table: Table,
    output_path: str | Path,
    config_path: str = "config/streams.yaml",
    track_table: Optional[Table] = None,
) -> None:
    """Full processing pipeline: frame transform → dust → distances → HDF5.

    Applies a two-pass strategy for large raw catalogs (>50K rows):
      Pass 1: Lightweight transform (no MC error propagation) → membership pre-filter
      Pass 2: Full transform with MC PM error propagation on the reduced catalog
    """
    log.info("Processing stream: %s  (%d input rows)", stream_name, len(raw_table))
    sc = _load_stream_config(stream_name, config_path)

    if len(raw_table) > 50_000:
        log.info("Large catalog (%d rows): applying lightweight pre-filter first.", len(raw_table))
        # Pass 1: quick stream-frame transform (skip MC error propagation)
        table_light = _to_stream_frame_no_mc(raw_table, stream_name, config_path)
        # Apply loose phi1/phi2/pm membership pre-filter
        phi1_min, phi1_max = sc["phi1_range_deg"]
        phi2_hw = sc.get("phi2_selection_deg", 10.0) * 2.0  # generous: 2× half-width
        pm1_min, pm1_max = sc.get("pm1_range_masyr", [-100, 100])
        pm2_min, pm2_max = sc.get("pm2_range_masyr", [-100, 100])
        pm1_pad = (pm1_max - pm1_min) * 0.5  # 50% padding
        pm2_pad = (pm2_max - pm2_min) * 0.5

        phi1_arr = np.asarray(table_light["phi1"])
        phi2_arr = np.asarray(table_light["phi2"])

        # galstreams returns phi1 in [0, 360].  When phi1_min < 0 the stream straddles
        # the 0/360 wrap (e.g. Sylgr at phi1=[345,11]).  Use the wrapped equivalent.
        if phi1_min < 0:
            phi1_mask = (phi1_arr >= (360 + phi1_min)) | (phi1_arr <= phi1_max)
        else:
            phi1_mask = (phi1_arr >= phi1_min) & (phi1_arr <= phi1_max)

        mask = phi1_mask & (np.abs(phi2_arr) <= phi2_hw)

        # Only apply PM cut when stream-frame PM columns exist; falling back to zeros
        # (for frames without differential support) would incorrectly cut all stars.
        if "pm_phi1" in table_light.colnames and "pm_phi2" in table_light.colnames:
            pm1_arr = np.asarray(table_light["pm_phi1"])
            pm2_arr = np.asarray(table_light["pm_phi2"])
            mask = (
                mask
                & (pm1_arr >= pm1_min - pm1_pad) & (pm1_arr <= pm1_max + pm1_pad)
                & (pm2_arr >= pm2_min - pm2_pad) & (pm2_arr <= pm2_max + pm2_pad)
            )
        raw_table = raw_table[mask]
        log.info("Pre-filter reduced catalog to %d rows (%.1f%%)",
                 len(raw_table), 100 * len(raw_table) / (len(table_light) or 1))

    # Pass 2 (or only pass for small catalogs): full transform with MC error propagation
    table = to_stream_frame(raw_table, stream_name, config_path)
    table = estimate_distances(table, stream_name, config_path)
    table = compute_dust_correction(table)
    write_to_hdf5(stream_name, table, output_path, track_table=track_table)


def _to_stream_frame_no_mc(
    table: Table, stream_name: str, config_path: str = "config/streams.yaml"
) -> Table:
    """Lightweight stream-frame transform: no MC error propagation.

    Used as a fast first pass on large catalogs to apply membership pre-filtering
    before the expensive Monte Carlo PM uncertainty propagation.
    """
    sc = _load_stream_config(stream_name, config_path)
    frame = _get_stream_frame(sc, stream_name)

    has_pm = "pmra" in table.colnames and "pmdec" in table.colnames
    # Strip Column units before attaching astropy units (same fix as to_stream_frame).
    if has_pm:
        icrs = coord.SkyCoord(
            ra=np.asarray(table["ra"]) * u.deg,
            dec=np.asarray(table["dec"]) * u.deg,
            pm_ra_cosdec=np.asarray(table["pmra"]) * u.mas / u.yr,
            pm_dec=np.asarray(table["pmdec"]) * u.mas / u.yr,
            frame="icrs",
        )
    else:
        icrs = coord.SkyCoord(
            ra=np.asarray(table["ra"]) * u.deg,
            dec=np.asarray(table["dec"]) * u.deg,
            frame="icrs",
        )

    stream_coords = icrs.transform_to(frame)
    t = table.copy()
    t["phi1"] = stream_coords.phi1.deg
    t["phi2"] = stream_coords.phi2.deg
    if has_pm:
        try:
            t["pm_phi1"] = stream_coords.pm_phi1_cosphi2.to(u.mas / u.yr).value
            t["pm_phi2"] = stream_coords.pm_phi2.to(u.mas / u.yr).value
        except TypeError:
            # Stream frame has no differential support — omit PM columns so the
            # pre-filter skips the PM range cut rather than using wrong values.
            log.warning("_to_stream_frame_no_mc: stream frame has no differential "
                        "transform for %s — PM columns omitted from pre-filter table.", stream_name)
    return t
