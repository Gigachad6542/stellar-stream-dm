"""Ingestion helpers for the Jarvis et al. DESI DR2 GD-1 v3 catalog."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

from .galstreams_compat import make_mwstreams
from .stream_process import to_stream_frame, write_to_hdf5


DESI_GD1_RECORD_ID = 19889980
DESI_GD1_VERSION = "3"
DESI_GD1_ARCHIVE_BYTES = 40_942_870
DESI_GD1_ARCHIVE_MD5 = "29a2e81ba5423f56e5869b5661e9cb5a"
DESI_GD1_ARCHIVE_URL = (
    "https://zenodo.org/api/records/19889980/files/"
    "Jarvis_GD1_DESI_DR2_Zenodo.zip/content"
)

EXTRA_COLUMNS = (
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "phi1_desi",
    "phi2_desi",
    "vgsr",
    "feh",
    "feh_error",
    "p_thin",
    "p_cocoon",
    "p_background",
    "selection_weight",
    "distmod",
    "delta_phi2",
    "delta_pm_phi1",
    "delta_pm_phi2",
    "delta_vgsr",
)


def verify_md5(path: str | Path, expected: str, chunk_size: int = 16 << 20) -> bool:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected.lower()


def _distance_from_modulus(distmod: np.ndarray) -> np.ndarray:
    return 10.0 ** ((distmod - 10.0) / 5.0)


def _distance_track(table1_path: str | Path):
    track = Table.read(table1_path)
    phi1 = np.asarray(track["phi1"], dtype=float)
    distance = np.asarray(track["distance"], dtype=float)
    order = np.argsort(phi1)
    return interp1d(
        phi1[order],
        distance[order],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )


def load_desi_gd1_table(
    table7_path: str | Path,
    table1_path: str | Path,
    selection: str = "thin",
    distance_error_kpc: float = 0.5,
) -> Table:
    """Standardize the DESI member table for the project HDF5 schema."""
    if selection not in {"thin", "member"}:
        raise ValueError(f"Unknown DESI GD-1 selection {selection!r}")

    raw = Table.read(table7_path)
    result = Table()
    mapping = {
        "source_id": "SOURCE_ID",
        "ra": "RA",
        "dec": "Dec",
        "pmra": "PM_RA",
        "pmra_error": "PM_RA_ERR",
        "pmdec": "PM_DEC",
        "pmdec_error": "PM_DEC_ERR",
        "radial_velocity": "V_LOS",
        "radial_velocity_error": "V_ERR",
        "phi1_desi": "phi1",
        "phi2_desi": "phi2",
        "vgsr": "VGSR",
        "feh": "FEH",
        "feh_error": "FEH_ERR",
        "p_thin": "P_THIN",
        "p_cocoon": "P_COCOON",
        "distmod": "DISTMOD",
        "delta_phi2": "DELTA_PHI2",
        "delta_pm_phi1": "DELTA_PM_PHI1",
        "delta_pm_phi2": "DELTA_PM_PHI2",
        "delta_vgsr": "DELTA_VGSR",
    }
    for target, source in mapping.items():
        result[target] = np.asarray(raw[source])

    p_thin = np.clip(np.asarray(result["p_thin"], dtype=float), 0.0, 1.0)
    p_cocoon = np.clip(np.asarray(result["p_cocoon"], dtype=float), 0.0, 1.0)
    p_member = np.clip(p_thin + p_cocoon, 0.0, 1.0)
    result["p_background"] = np.clip(1.0 - p_member, 0.0, 1.0)
    result["membership_prob"] = p_thin if selection == "thin" else p_member
    result["selection_weight"] = p_thin

    distmod = np.asarray(result["distmod"], dtype=float)
    distance = _distance_from_modulus(distmod)
    missing = ~np.isfinite(distance)
    if np.any(missing):
        distance[missing] = _distance_track(table1_path)(
            np.asarray(result["phi1_desi"], dtype=float)[missing]
        )
    result["dist"] = distance
    result["e_dist"] = np.full(len(result), float(distance_error_kpc))

    result.meta["selection"] = selection
    result.meta["source"] = (
        "Jarvis et al., Characterizing the GD-1 Stream with DESI DR2 Data, "
        f"Zenodo {DESI_GD1_RECORD_ID} v{DESI_GD1_VERSION}"
    )
    result.meta["membership_semantics"] = (
        "thin: membership_prob=P_THIN; member: "
        "membership_prob=clip(P_THIN+P_COCOON, 0, 1)."
    )
    result.meta["selection_weight_semantics"] = (
        "selection_weight=P_THIN for main-track density-profile scoring."
    )
    result.meta["distance_uncertainty_semantics"] = (
        f"e_dist={distance_error_kpc:g} kpc placeholder; Table7 provides DISTMOD "
        "but no per-star distance uncertainty. Missing DISTMOD values use Table1 track."
    )
    return result


def _gd1_i21_track_table() -> Table:
    mws = make_mwstreams(verbose=False)
    track = mws["GD-1-I21"]
    coords = track.track.transform_to(track.stream_frame)
    phi1 = (np.asarray(coords.phi1.deg) + 180.0) % 360.0 - 180.0
    order = np.argsort(phi1)
    result = Table()
    result["phi1"] = phi1[order]
    result["phi2"] = np.asarray(coords.phi2.deg)[order]
    result["width"] = np.full(len(result), 0.5)
    return result


def write_desi_gd1_selection(
    table7_path: str | Path,
    table1_path: str | Path,
    output_path: str | Path,
    selection: str = "thin",
    config_path: str = "config/streams.yaml",
) -> dict:
    """Transform and write one DESI v3 GD-1 selection."""
    table = load_desi_gd1_table(table7_path, table1_path, selection=selection)
    transformed = to_stream_frame(table, "GD1", config_path=config_path)
    transformed["phi1"] = (
        (np.asarray(transformed["phi1"], dtype=float) + 180.0) % 360.0
    ) - 180.0
    write_to_hdf5(
        "GD1",
        transformed,
        output_path,
        track_table=_gd1_i21_track_table(),
        gaia_release="DR3+DESI-DR2",
    )

    with h5py.File(output_path, "a") as handle:
        group = handle["streams/GD1"]
        for key in (
            "source",
            "selection",
            "membership_semantics",
            "selection_weight_semantics",
            "distance_uncertainty_semantics",
        ):
            group.attrs[key] = table.meta[key]
        group.attrs["zenodo_record_id"] = DESI_GD1_RECORD_ID
        group.attrs["zenodo_version"] = DESI_GD1_VERSION
        group.attrs["table7_path"] = str(table7_path)
        group.attrs["table1_path"] = str(table1_path)
        members = group["members"]
        for name in EXTRA_COLUMNS:
            if name in members:
                del members[name]
            members.create_dataset(
                name,
                data=np.asarray(transformed[name], dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )

    membership = np.asarray(transformed["membership_prob"], dtype=float)
    return {
        "selection": selection,
        "n_rows": len(transformed),
        "n_membership_ge_0_5": int(np.sum(membership >= 0.5)),
        "membership_weight_sum": float(np.sum(membership)),
        "thin_weight_sum": float(np.sum(transformed["p_thin"])),
        "cocoon_weight_sum": float(np.sum(transformed["p_cocoon"])),
        "n_finite_rv": int(np.sum(np.isfinite(transformed["radial_velocity"]))),
        "n_finite_distmod": int(np.sum(np.isfinite(transformed["distmod"]))),
        "phi1_i21_range": [
            float(np.nanmin(transformed["phi1"])),
            float(np.nanmax(transformed["phi1"])),
        ],
        "output_path": str(output_path),
    }
