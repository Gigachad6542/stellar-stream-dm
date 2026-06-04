"""Fuse PWB18 spatial selection with DESI v3 spectroscopy by sky position."""
from __future__ import annotations

import shutil
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import SkyCoord


DESI_EXTRA_MAPPING = {
    "desi_source_id": "source_id",
    "desi_p_thin": "p_thin",
    "desi_p_cocoon": "p_cocoon",
    "desi_p_background": "p_background",
    "desi_membership_prob": "membership_prob",
    "desi_delta_phi2": "delta_phi2",
    "desi_delta_pm_phi1": "delta_pm_phi1",
    "desi_delta_pm_phi2": "delta_pm_phi2",
    "desi_delta_vgsr": "delta_vgsr",
    "desi_dist": "dist",
    "desi_e_dist": "e_dist",
}


def fuse_pwb18_desi(
    pwb18_path: str | Path,
    desi_path: str | Path,
    output_path: str | Path,
    match_radius_arcsec: float = 1.0,
) -> dict:
    """Copy a PWB18 track catalog and attach matched DESI spectroscopy."""
    pwb18_path = Path(pwb18_path)
    desi_path = Path(desi_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pwb18_path, output_path)

    with h5py.File(output_path, "a") as output, h5py.File(desi_path, "r") as desi:
        output_group = output["streams/GD1"]
        output_members = output_group["members"]
        desi_members = desi["streams/GD1/members"]
        for name in ("ra", "dec"):
            if name not in output_members or name not in desi_members:
                raise KeyError(f"Fusion requires preserved ICRS {name} in both catalogs")

        pwb_coord = SkyCoord(
            output_members["ra"][:] * u.deg,
            output_members["dec"][:] * u.deg,
        )
        desi_coord = SkyCoord(
            desi_members["ra"][:] * u.deg,
            desi_members["dec"][:] * u.deg,
        )
        nearest, separation, _ = pwb_coord.match_to_catalog_sky(desi_coord)
        matched = separation.arcsec <= float(match_radius_arcsec)
        matched_desi = nearest[matched]
        if len(np.unique(matched_desi)) != len(matched_desi):
            raise RuntimeError("DESI sky cross-match is not one-to-one")

        for target in ("vrad", "e_vrad"):
            values = output_members[target][:]
            values[matched] = desi_members[target][:][matched_desi]
            output_members[target][:] = values

        if "desi_match" in output_members:
            del output_members["desi_match"]
        output_members.create_dataset(
            "desi_match",
            data=matched,
            compression="gzip",
            compression_opts=4,
        )

        for target, source in DESI_EXTRA_MAPPING.items():
            if target in output_members:
                del output_members[target]
            source_values = desi_members[source][:]
            if target == "desi_source_id":
                values = np.full(len(matched), -1, dtype=np.int64)
            else:
                values = np.full(len(matched), np.nan, dtype=np.float32)
            values[matched] = source_values[matched_desi]
            output_members.create_dataset(
                target,
                data=values,
                compression="gzip",
                compression_opts=4,
            )

        output_group.attrs["fusion_source"] = (
            "PWB18 PM+CMD+track spatial selection with Jarvis et al. DESI DR2 "
            "v3 spectroscopy attached by ICRS sky-position cross-match."
        )
        output_group.attrs["fusion_match_radius_arcsec"] = float(match_radius_arcsec)
        output_group.attrs["fusion_n_desi_matches"] = int(np.sum(matched))
        output_group.attrs["fusion_max_match_separation_arcsec"] = float(
            np.max(separation.arcsec[matched]) if np.any(matched) else np.nan
        )
        output_group.attrs["fusion_pwb18_path"] = str(pwb18_path)
        output_group.attrs["fusion_desi_path"] = str(desi_path)

        phi1_pwb18 = output_members["phi1_pwb18"][:]
        canonical = matched & (np.abs(phi1_pwb18 - (-40.0)) <= 3.0)
        spur = matched & (np.abs(phi1_pwb18 - (-20.0)) <= 3.0)

    return {
        "output_path": str(output_path),
        "n_pwb18_track": int(len(matched)),
        "n_desi_matches": int(np.sum(matched)),
        "match_radius_arcsec": float(match_radius_arcsec),
        "max_match_separation_arcsec": float(
            np.max(separation.arcsec[matched]) if np.any(matched) else np.nan
        ),
        "n_rv_canonical_gap_window": int(np.sum(canonical)),
        "n_rv_spur_window": int(np.sum(spur)),
    }
