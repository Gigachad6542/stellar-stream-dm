"""Tests for the DESI DR2 GD-1 v3 ingestion helpers."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from src.data.desi_gd1 import load_desi_gd1_table


@pytest.fixture
def desi_tables(tmp_path):
    table7 = Table()
    table7["SOURCE_ID"] = [1, 2, 3]
    table7["RA"] = [140.0, 141.0, 142.0]
    table7["Dec"] = [30.0, 31.0, 32.0]
    table7["phi1"] = [-40.0, -20.0, 0.0]
    table7["phi2"] = [0.0, 1.0, -1.0]
    table7["VGSR"] = [-100.0, -50.0, 0.0]
    table7["V_LOS"] = [-120.0, -70.0, -20.0]
    table7["V_ERR"] = [1.0, 2.0, 3.0]
    table7["FEH"] = [-2.0, -2.1, -1.9]
    table7["FEH_ERR"] = [0.1, 0.1, 0.2]
    table7["P_THIN"] = [0.8, 0.1, 0.2]
    table7["P_COCOON"] = [0.1, 0.7, 0.1]
    table7["PM_RA"] = [-8.0, -7.0, -6.0]
    table7["PM_RA_ERR"] = [0.1, 0.1, 0.2]
    table7["PM_DEC"] = [-2.0, -1.0, 0.0]
    table7["PM_DEC_ERR"] = [0.1, 0.1, 0.2]
    table7["DISTMOD"] = [14.5, np.nan, 15.0]
    table7["PM_PHI1"] = [-8.0, -7.0, -6.0]
    table7["PM_PHI1_ERR"] = [0.1, 0.1, 0.2]
    table7["PM_PHI2"] = [-2.0, -1.0, 0.0]
    table7["PM_PHI2_ERR"] = [0.1, 0.1, 0.2]
    table7["GMAG0"] = [18.0, 18.5, 19.0]
    table7["RMAG0"] = [17.5, 18.0, 18.5]
    table7["DELTA_PHI2"] = [0.0, 1.0, -1.0]
    table7["DELTA_PM_PHI1"] = [0.0, 0.1, -0.1]
    table7["DELTA_PM_PHI2"] = [0.0, 0.1, -0.1]
    table7["DELTA_VGSR"] = [0.0, 1.0, -1.0]
    table7_path = tmp_path / "Table7.fits"
    table7.write(table7_path)

    table1 = Table({"phi1": [-40, -20, 0], "distance": [8.0, 9.0, 10.0]})
    table1_path = tmp_path / "Table1.fits"
    table1.write(table1_path)
    return table7_path, table1_path


def test_thin_selection_uses_thin_probability(desi_tables):
    table = load_desi_gd1_table(*desi_tables, selection="thin")

    assert np.allclose(table["membership_prob"], [0.8, 0.1, 0.2])
    assert np.allclose(table["selection_weight"], table["p_thin"])
    assert np.allclose(table["radial_velocity"], [-120.0, -70.0, -20.0])


def test_member_selection_combines_thin_and_cocoon(desi_tables):
    table = load_desi_gd1_table(*desi_tables, selection="member")

    assert np.allclose(table["membership_prob"], [0.9, 0.8, 0.3])
    assert np.allclose(table["p_background"], [0.1, 0.2, 0.7])


def test_missing_distance_modulus_uses_track(desi_tables):
    table = load_desi_gd1_table(*desi_tables, selection="thin")

    assert table["dist"][1] == pytest.approx(9.0)
    assert np.isfinite(table["dist"]).all()
