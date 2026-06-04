"""Tests for PWB18 + DESI GD-1 catalog fusion."""
from __future__ import annotations

import h5py
import numpy as np

from src.data.gd1_fusion import fuse_pwb18_desi


def _write_catalog(path, ra, dec, source_ids, vrad, extra=None):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("streams/GD1/members")
        values = {
            "ra": ra,
            "dec": dec,
            "source_id": source_ids,
            "vrad": vrad,
            "e_vrad": np.full(len(ra), np.nan),
            "phi1_pwb18": np.array([-40.0, -20.0, 0.0])[: len(ra)],
        }
        values.update(extra or {})
        for name, data in values.items():
            group.create_dataset(name, data=np.asarray(data))


def test_fusion_attaches_one_to_one_spectroscopy(tmp_path):
    pwb = tmp_path / "pwb.h5"
    desi = tmp_path / "desi.h5"
    out = tmp_path / "fused.h5"
    _write_catalog(
        pwb,
        [10.0, 20.0, 30.0],
        [0.0, 0.0, 0.0],
        [1, 2, 3],
        [np.nan, np.nan, np.nan],
    )
    extra = {
        "p_thin": [0.9, 0.8],
        "p_cocoon": [0.05, 0.1],
        "p_background": [0.05, 0.1],
        "membership_prob": [0.95, 0.9],
        "delta_phi2": [0.0, 0.1],
        "delta_pm_phi1": [0.0, 0.1],
        "delta_pm_phi2": [0.0, 0.1],
        "delta_vgsr": [0.0, 1.0],
        "dist": [8.0, 9.0],
        "e_dist": [0.5, 0.5],
    }
    _write_catalog(
        desi,
        [10.0, 20.0],
        [0.0, 0.0],
        [101, 102],
        [-100.0, -50.0],
        extra=extra,
    )
    with h5py.File(desi, "a") as handle:
        handle["streams/GD1/members/e_vrad"][:] = [1.0, 2.0]

    result = fuse_pwb18_desi(pwb, desi, out)

    assert result["n_desi_matches"] == 2
    with h5py.File(out, "r") as handle:
        members = handle["streams/GD1/members"]
        assert members["desi_match"][:].tolist() == [True, True, False]
        assert np.allclose(members["vrad"][:2], [-100.0, -50.0])
        assert np.isnan(members["vrad"][2])
