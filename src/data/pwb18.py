"""Mask-aware ingestion for the Price-Whelan & Bonaca (2018) GD-1 catalog.

Zenodo record 1295543 contains a large *region catalog* with published masks,
not a compact table of posterior stream memberships.  This module applies those
masks explicitly and records their semantics in the processed HDF5 output.
"""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

from .galstreams_compat import make_mwstreams
from .stream_process import to_stream_frame, write_to_hdf5

log = logging.getLogger(__name__)

PWB18_URL = "https://zenodo.org/records/1295543/files/gd1-with-masks.fits?download=1"
PWB18_MD5 = "996b1c2d187effac25c685cb153e5700"
PWB18_EXPECTED_BYTES = 2_007_829_440

MASK_ALIASES = {
    "pm_mask": ("pm_mask", "proper_motion_mask"),
    "gi_cmd_mask": ("gi_cmd_mask", "g_i_cmd_mask", "cmd_mask", "isochrone_mask"),
    "stream_track_mask": ("stream_track_mask", "track_mask", "phi2_mask"),
}

COLUMN_ALIASES = {
    "source_id": ("source_id",),
    "ra": ("ra", "raj2000"),
    "dec": ("dec", "dej2000"),
    "pmra": ("pmra", "pm_ra_cosdec"),
    "pmdec": ("pmdec", "pm_dec"),
    "pmra_error": ("pmra_error", "pm_ra_cosdec_error"),
    "pmdec_error": ("pmdec_error", "pm_dec_error"),
    "radial_velocity": ("radial_velocity", "vrad", "hrv"),
    "radial_velocity_error": ("radial_velocity_error", "e_vrad", "e_hrv"),
    "phi1_pwb18": ("phi1", "phi_1", "phi1_pwb18"),
    "phi2_pwb18": ("phi2", "phi_2", "phi2_pwb18"),
    "pm_phi1_pwb18": ("pm_phi1_cosphi2", "pm_phi1_pwb18"),
    "pm_phi2_pwb18": ("pm_phi2", "pm_phi2_pwb18"),
    "pm_phi1_no_reflex_pwb18": (
        "pm_phi1_cosphi2_no_reflex",
        "pm_phi1_no_reflex_pwb18",
    ),
    "pm_phi2_no_reflex_pwb18": ("pm_phi2_no_reflex", "pm_phi2_no_reflex_pwb18"),
}

SELECTION_EXPRESSIONS = {
    "track": "pm_mask & gi_cmd_mask & stream_track_mask",
    "pmcmd": "pm_mask & gi_cmd_mask",
    "all": "all rows",
}


def verify_md5(path: str | Path, expected: str = PWB18_MD5, chunk_size: int = 16 << 20) -> bool:
    """Return whether a file matches the published Zenodo MD5."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected.lower()


def download_pwb18(path: str | Path, force: bool = False) -> Path:
    """Download the published masked-region FITS file.

    This simple fallback downloader is intended for reproducibility.  For the
    2 GB file, a resumable/ranged downloader is preferable when available.
    """
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(PWB18_URL, path)
    return path


def _name_map(names: Iterable[str]) -> dict[str, str]:
    return {str(name).lower(): str(name) for name in names}


def _resolve_column(
    names: dict[str, str],
    aliases: Iterable[str],
    *,
    required: bool = False,
) -> Optional[str]:
    for alias in aliases:
        if alias.lower() in names:
            return names[alias.lower()]
    if required:
        raise KeyError(f"Required FITS column missing; tried {tuple(aliases)}")
    return None


def _mask_array(data, names: dict[str, str], key: str) -> np.ndarray:
    name = _resolve_column(names, MASK_ALIASES[key], required=True)
    values = np.asarray(data[name])
    if values.dtype.kind in {"S", "U"}:
        normalized = np.char.upper(values.astype("U"))
        return np.isin(normalized, ("T", "TRUE", "1", "Y", "YES"))
    return values.astype(bool, copy=False)


def load_pwb18_selection(
    path: str | Path,
    selection: str = "track",
    max_rows: Optional[int] = None,
) -> Table:
    """Load one published PWB18 mask selection without materializing all columns.

    Args:
        path: ``gd1-with-masks.fits``.
        selection: ``track`` (PM+CMD+stream-track), ``pmcmd`` (PM+CMD), or
            ``all``.
        max_rows: Optional deterministic truncation for smoke tests only.
    """
    if selection not in SELECTION_EXPRESSIONS:
        raise ValueError(f"Unknown PWB18 selection {selection!r}")

    path = Path(path)
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        data = hdul[1].data
        names = _name_map(data.names)

        if selection == "all":
            mask = np.ones(len(data), dtype=bool)
        else:
            mask = _mask_array(data, names, "pm_mask") & _mask_array(data, names, "gi_cmd_mask")
            if selection == "track":
                mask &= _mask_array(data, names, "stream_track_mask")

        indices = np.flatnonzero(mask)
        if max_rows is not None:
            indices = indices[: int(max_rows)]

        table = Table()
        for canonical, aliases in COLUMN_ALIASES.items():
            source = _resolve_column(
                names,
                aliases,
                required=canonical in {"ra", "dec", "pmra", "pmdec"},
            )
            if source is not None:
                table[canonical] = np.asarray(data[source][indices])
        for mask_name in MASK_ALIASES:
            source = _resolve_column(names, MASK_ALIASES[mask_name], required=True)
            values = np.asarray(data[source][indices])
            if values.dtype.kind in {"S", "U"}:
                normalized = np.char.upper(values.astype("U"))
                values = np.isin(normalized, ("T", "TRUE", "1", "Y", "YES"))
            table[mask_name] = values.astype(bool, copy=False)

    table["selection_weight"] = np.ones(len(table), dtype=np.float32)
    table["membership_prob"] = np.ones(len(table), dtype=np.float32)
    table.meta["source"] = "Price-Whelan & Bonaca 2018; Zenodo 1295543"
    table.meta["selection"] = selection
    table.meta["selection_expression"] = SELECTION_EXPRESSIONS[selection]
    table.meta["membership_semantics"] = (
        "Published boolean selection masks; membership_prob=1 is a pipeline "
        "compatibility weight, not a posterior membership probability."
    )
    table.meta["catalog_role"] = (
        "PWB18 masked region catalog. PM+CMD selects the wider probable-member "
        "candidate sample, including off-track morphology and contamination; "
        "stream_track_mask isolates the main-track subset."
    )
    return table


def compute_pwb18_selection_profile(
    path: str | Path,
    bin_width_deg: float = 1.0,
    smooth_sigma_bins: float = 2.0,
) -> dict[str, np.ndarray | float | str]:
    """Estimate the main-track excess density and contamination fraction.

    The PM+CMD-selected stars outside ``stream_track_mask`` provide a local
    contamination control. The full region catalog supplies a per-longitude
    track/off-track occupancy ratio that approximately corrects the unequal
    mask footprints. Because the off-track sample can contain the GD-1 spur or
    cocoon, this is conservative for main-track density and must not be used as
    the 2D off-track morphology likelihood. It is not a calibrated membership
    posterior or final density likelihood.
    """
    from scipy.ndimage import gaussian_filter1d

    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        data = hdul[1].data
        names = _name_map(data.names)
        phi1_name = _resolve_column(names, COLUMN_ALIASES["phi1_pwb18"], required=True)
        phi1 = np.asarray(data[phi1_name], dtype=np.float64)
        pmcmd = _mask_array(data, names, "pm_mask") & _mask_array(data, names, "gi_cmd_mask")
        track = _mask_array(data, names, "stream_track_mask")

        finite = np.isfinite(phi1)
        phi1 = phi1[finite]
        pmcmd = pmcmd[finite]
        track = track[finite]

    lo = np.floor(np.nanmin(phi1) / bin_width_deg) * bin_width_deg
    hi = np.ceil(np.nanmax(phi1) / bin_width_deg) * bin_width_deg
    edges = np.arange(lo, hi + bin_width_deg * 1.01, bin_width_deg)
    centers = 0.5 * (edges[:-1] + edges[1:])

    all_track = np.histogram(phi1[track], bins=edges)[0].astype(np.float64)
    all_offtrack = np.histogram(phi1[~track], bins=edges)[0].astype(np.float64)
    pmcmd_track = np.histogram(phi1[pmcmd & track], bins=edges)[0].astype(np.float64)
    pmcmd_offtrack = np.histogram(phi1[pmcmd & ~track], bins=edges)[0].astype(np.float64)

    def smooth(values: np.ndarray) -> np.ndarray:
        if smooth_sigma_bins <= 0:
            return values.copy()
        return gaussian_filter1d(values, smooth_sigma_bins, mode="nearest")

    smooth_all_track = smooth(all_track)
    smooth_all_offtrack = smooth(all_offtrack)
    smooth_pmcmd_offtrack = smooth(pmcmd_offtrack)
    footprint_ratio = smooth_all_track / np.maximum(smooth_all_offtrack, 1.0)
    expected_background_track = smooth_pmcmd_offtrack * footprint_ratio
    excess = pmcmd_track - expected_background_track
    variance = np.maximum(pmcmd_track, 1.0) + (
        footprint_ratio**2 * np.maximum(pmcmd_offtrack, 1.0)
    )
    excess_significance = excess / np.sqrt(variance)
    signal_fraction = np.clip(
        np.maximum(excess, 0.0) / np.maximum(pmcmd_track, 1.0),
        0.0,
        1.0,
    )

    return {
        "method": (
            "Diagnostic PM+CMD on-track excess using off-track contamination "
            "and full-catalog track/off-track occupancy correction"
        ),
        "bin_width_deg": float(bin_width_deg),
        "smooth_sigma_bins": float(smooth_sigma_bins),
        "bin_edges_pwb18": edges,
        "bin_centers_pwb18": centers,
        "all_track_counts": all_track,
        "all_offtrack_counts": all_offtrack,
        "pmcmd_track_counts": pmcmd_track,
        "pmcmd_offtrack_counts": pmcmd_offtrack,
        "footprint_ratio": footprint_ratio,
        "expected_background_track": expected_background_track,
        "main_track_excess": excess,
        "main_track_excess_significance": excess_significance,
        "main_track_signal_fraction": signal_fraction,
    }


def _track_selection_weights(table: Table, selection_profile: dict) -> np.ndarray:
    """Map binned PWB18 main-track signal fractions onto selected stars."""
    edges = np.asarray(selection_profile["bin_edges_pwb18"], dtype=np.float64)
    fractions = np.asarray(
        selection_profile["main_track_signal_fraction"], dtype=np.float64
    )
    indices = np.searchsorted(edges, np.asarray(table["phi1_pwb18"]), side="right") - 1
    valid = (indices >= 0) & (indices < len(fractions))
    weights = np.zeros(len(table), dtype=np.float32)
    weights[valid] = fractions[indices[valid]].astype(np.float32)
    return weights


def _gd1_i21_track():
    mws = make_mwstreams(verbose=False)
    track = mws["GD-1-I21"]
    frame = track.stream_frame
    track_i21 = track.track.transform_to(frame)
    phi1 = (np.asarray(track_i21.phi1.deg) + 180.0) % 360.0 - 180.0
    order = np.argsort(phi1)
    table = Table()
    table["phi1"] = phi1[order]
    table["phi2"] = np.asarray(track_i21.phi2.deg)[order]
    table["width"] = np.full(len(table), 0.5)
    return track, frame, table


def transform_pwb18_to_i21(table: Table, config_path: str = "config/streams.yaml") -> Table:
    """Transform a selected PWB18 table into the pipeline's GD-1-I21 frame."""
    transformed = to_stream_frame(table, "GD1", config_path=config_path)
    transformed["phi1"] = (
        (np.asarray(transformed["phi1"], dtype=float) + 180.0) % 360.0
    ) - 180.0
    track, frame, _ = _gd1_i21_track()
    track_i21 = track.track.transform_to(frame)
    phi1_track = (np.asarray(track_i21.phi1.deg) + 180.0) % 360.0 - 180.0
    order = np.argsort(phi1_track)
    distance = np.asarray(track_i21.distance.to_value(u.kpc))[order]
    interp = interp1d(
        phi1_track[order],
        distance,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    transformed["dist"] = interp(np.asarray(transformed["phi1"], dtype=float))
    transformed["e_dist"] = np.full(len(transformed), 0.5)
    return transformed


def write_pwb18_selection(
    raw_path: str | Path,
    output_path: str | Path,
    selection: str = "track",
    config_path: str = "config/streams.yaml",
    max_rows: Optional[int] = None,
    selection_profile: Optional[dict] = None,
) -> dict:
    """Transform and write one PWB18 selection to a dedicated HDF5 catalog."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    table = load_pwb18_selection(raw_path, selection=selection, max_rows=max_rows)
    if selection == "track" and selection_profile is not None:
        table["selection_weight"] = _track_selection_weights(table, selection_profile)
    transformed = transform_pwb18_to_i21(table, config_path=config_path)
    _, _, track_table = _gd1_i21_track()
    write_to_hdf5("GD1", transformed, output_path, track_table=track_table, gaia_release="DR2")

    with h5py.File(output_path, "a") as handle:
        group = handle["streams/GD1"]
        group.attrs["source"] = table.meta["source"]
        group.attrs["selection"] = selection
        group.attrs["selection_expression"] = SELECTION_EXPRESSIONS[selection]
        group.attrs["membership_semantics"] = table.meta["membership_semantics"]
        group.attrs["catalog_role"] = table.meta["catalog_role"]
        group.attrs["raw_path"] = str(raw_path)
        members = group["members"]
        for name in (
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "phi1_pwb18",
            "phi2_pwb18",
            "pm_phi1_pwb18",
            "pm_phi2_pwb18",
            "pm_phi1_no_reflex_pwb18",
            "pm_phi2_no_reflex_pwb18",
            "selection_weight",
            "pm_mask",
            "gi_cmd_mask",
            "stream_track_mask",
        ):
            if name not in transformed.colnames:
                continue
            if name in members:
                del members[name]
            dtype = np.bool_ if name in MASK_ALIASES else np.float32
            members.create_dataset(
                name,
                data=np.asarray(transformed[name], dtype=dtype),
                compression="gzip",
                compression_opts=4,
            )
        if selection_profile is not None:
            profile_group = group.require_group("selection_profile")
            profile_group.attrs["method"] = selection_profile["method"]
            profile_group.attrs["scientific_status"] = (
                "Diagnostic background/footprint correction; not a calibrated "
                "membership posterior or final density likelihood. Off-track "
                "signal may include the GD-1 spur/cocoon."
            )
            profile_group.attrs["bin_width_deg"] = selection_profile["bin_width_deg"]
            profile_group.attrs["smooth_sigma_bins"] = selection_profile["smooth_sigma_bins"]
            for name, values in selection_profile.items():
                if name in {"method", "bin_width_deg", "smooth_sigma_bins"}:
                    continue
                if name in profile_group:
                    del profile_group[name]
                profile_group.create_dataset(
                    name,
                    data=np.asarray(values, dtype=np.float64),
                    compression="gzip",
                    compression_opts=4,
                )

    return {
        "selection": selection,
        "selection_expression": SELECTION_EXPRESSIONS[selection],
        "n_rows": len(transformed),
        "phi1_i21_range": [
            float(np.nanmin(transformed["phi1"])),
            float(np.nanmax(transformed["phi1"])),
        ],
        "n_stream_track_mask": int(np.sum(transformed["stream_track_mask"])),
        "selection_weight_sum": float(np.sum(transformed["selection_weight"])),
        "catalog_role": table.meta["catalog_role"],
        "output_path": str(output_path),
    }
