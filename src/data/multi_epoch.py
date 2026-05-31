"""
Multi-epoch / multi-survey data fusion for the timeline "rewind".

The accuracy with which we can rewind a stream (reconstruct its un-impacted past
state and infer *when* a subhalo struck) is limited by the precision of the
present-day 6D phase space, because backward orbit integration diverges as
``dx(t) ~ dv * t`` — velocity errors dominate over Gyr timescales.

This module pulls **real, public** data from multiple epochs/surveys to tighten
that present-day state:

* Radial velocity (the missing 6th dimension — ``vrad`` is empty in the base
  Gaia DR3 membership tables):
    - Gaia DR3 RVS (`gaiadr3.gaia_source.radial_velocity`) for bright members.
    - S5, the Southern Stellar Stream Spectroscopic Survey DR1 (Li et al. 2022,
      ApJ 928, 30; VizieR ``J/ApJ/928/30``) for the southern streams.
* Proper motion second epoch:
    - Gaia DR2 astrometry via the ``gaiadr3.dr2_neighbourhood`` cross-match, for
      inverse-variance fusion with DR3 and for projecting the gains from future
      releases (DR4/DR5) whose longer baselines shrink PM errors as
      ``sigma_mu ~ baseline^-1.5``.

All measurements are combined by inverse-variance weighting, which both reduces
the uncertainty and cross-checks systematics. Catalogs are cached under
``data/raw/multi_epoch`` so repeated runs do not re-hit the archives.

References
----------
Gaia Collaboration (DR2 2018, DR3 2022); Lindegren et al. 2021 (astrometry).
Li et al. 2022, ApJ 928, 30 (S5 DR1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .radial_velocity import RadialVelocityData

log = logging.getLogger(__name__)


# Approximate astrometric baselines of Gaia data releases [years].
# PM random error scales ~ baseline^-1.5 (longer baseline -> tighter PM).
GAIA_DR_BASELINES_YR = {
    "DR2": 1.83,    # 22 months
    "EDR3": 2.83,   # 34 months
    "DR3": 2.83,    # same astrometry as EDR3
    "DR4": 5.5,     # ~66 months (nominal mission)
    "DR5": 10.0,    # ~10 years (extended mission)
}


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------

def inverse_variance_combine(
    values: list[np.ndarray],
    errors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance combine aligned measurements from several catalogs.

    Each ``values[k]`` / ``errors[k]`` is a length-N array (one per source),
    with NaN where catalog ``k`` has no measurement for that source. Returns the
    combined value and combined 1-sigma error per source (NaN where no catalog
    measured it).

    fused = sum_k w_k x_k / sum_k w_k ,  sigma = 1/sqrt(sum_k w_k),
    with w_k = 1/sigma_k^2 (0 where the value or error is missing/non-positive).
    """
    if not values:
        raise ValueError("need at least one catalog to combine")
    n = len(values[0])
    wsum = np.zeros(n)
    wxsum = np.zeros(n)
    for x, e in zip(values, errors):
        x = np.asarray(x, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(e) & (e > 0)
        w = np.zeros(n)
        w[good] = 1.0 / e[good] ** 2
        wsum += w
        wxsum += w * np.where(good, x, 0.0)
    fused = np.full(n, np.nan)
    err = np.full(n, np.nan)
    nz = wsum > 0
    fused[nz] = wxsum[nz] / wsum[nz]
    err[nz] = 1.0 / np.sqrt(wsum[nz])
    return fused, err


def pm_error_scaling_factor(target_release: str, reference_release: str = "DR3") -> float:
    """Predicted PM-error ratio between two Gaia releases (target / reference).

    Uses sigma_mu ~ baseline^-1.5. A value < 1 means the target release has
    *smaller* PM errors than the reference. E.g. DR4 vs DR3 ~ 0.37 (≈2.7x tighter).
    """
    b_t = GAIA_DR_BASELINES_YR[target_release]
    b_r = GAIA_DR_BASELINES_YR[reference_release]
    return float((b_r / b_t) ** 1.5)


# ---------------------------------------------------------------------------
# Gaia fetchers (real TAP queries)
# ---------------------------------------------------------------------------

def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def fetch_gaia_dr3_rvs(
    source_ids: np.ndarray,
    cache_path: Optional[str | Path] = None,
    chunk_size: int = 5000,
) -> RadialVelocityData:
    """Fetch real Gaia DR3 RVS radial velocities for the given DR3 source_ids.

    Only bright members (G <~ 14) have RVS, so coverage is sparse for faint
    streams. Returns a RadialVelocityData with only the matched stars.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        d = np.load(cache_path)
        log.info("Loaded cached Gaia RVS (%d stars) from %s", len(d["source_id"]), cache_path)
        return RadialVelocityData(
            source_ids=d["source_id"], rv_km_s=d["rv"], rv_error_km_s=d["e_rv"],
            survey="Gaia_RVS", snr=np.full(len(d["source_id"]), np.nan),
        )

    from astroquery.gaia import Gaia

    ids = [int(s) for s in np.asarray(source_ids).ravel()]
    out_id, out_rv, out_erv = [], [], []
    for j, chunk in enumerate(_chunks(ids, chunk_size)):
        id_list = ",".join(str(s) for s in chunk)
        adql = (
            "SELECT source_id, radial_velocity, radial_velocity_error "
            "FROM gaiadr3.gaia_source "
            f"WHERE source_id IN ({id_list}) AND radial_velocity IS NOT NULL"
        )
        log.info("Gaia RVS chunk %d (%d ids)...", j + 1, len(chunk))
        try:
            r = Gaia.launch_job_async(adql).get_results()
        except Exception as e:
            log.warning("  Gaia RVS chunk %d failed: %s", j + 1, e)
            continue
        out_id.extend(int(x) for x in r["source_id"])
        out_rv.extend(float(x) for x in r["radial_velocity"])
        out_erv.extend(float(x) for x in r["radial_velocity_error"])

    sid = np.array(out_id, dtype=np.int64)
    rv = np.array(out_rv, dtype=np.float64)
    erv = np.array(out_erv, dtype=np.float64)
    log.info("Gaia RVS: %d / %d members have radial velocities", len(sid), len(ids))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, source_id=sid, rv=rv, e_rv=erv)
    return RadialVelocityData(
        source_ids=sid, rv_km_s=rv, rv_error_km_s=erv,
        survey="Gaia_RVS", snr=np.full(len(sid), np.nan),
    )


def fetch_gaia_dr2_proper_motions(
    dr3_source_ids: np.ndarray,
    cache_path: Optional[str | Path] = None,
    chunk_size: int = 5000,
) -> dict:
    """Fetch Gaia DR2 PMs as a second epoch, matched to DR3 source_ids.

    Uses ``gaiadr3.dr2_neighbourhood`` to map DR3 -> DR2 and keeps the best
    neighbour (smallest magnitude difference, then angular distance) per DR3 id.

    Returns a dict of arrays aligned by the unique matched DR3 source_id:
    ``{dr3_source_id, pmra, pmdec, pmra_error, pmdec_error}`` (ICRS).
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        d = np.load(cache_path)
        log.info("Loaded cached Gaia DR2 PMs (%d stars) from %s", len(d["dr3_source_id"]), cache_path)
        return {k: d[k] for k in d.files}

    from astroquery.gaia import Gaia

    ids = [int(s) for s in np.asarray(dr3_source_ids).ravel()]
    best: dict[int, tuple] = {}  # dr3 id -> (mag_diff, ang_dist, pmra, pmdec, e_pmra, e_pmdec)
    for j, chunk in enumerate(_chunks(ids, chunk_size)):
        id_list = ",".join(str(s) for s in chunk)
        adql = (
            "SELECT n.dr3_source_id, n.angular_distance, n.magnitude_difference, "
            "d2.pmra, d2.pmdec, d2.pmra_error, d2.pmdec_error "
            "FROM gaiadr3.dr2_neighbourhood AS n "
            "JOIN gaiadr2.gaia_source AS d2 ON d2.source_id = n.dr2_source_id "
            f"WHERE n.dr3_source_id IN ({id_list}) AND d2.pmra IS NOT NULL"
        )
        log.info("Gaia DR2 PM chunk %d (%d ids)...", j + 1, len(chunk))
        try:
            r = Gaia.launch_job_async(adql).get_results()
        except Exception as e:
            log.warning("  Gaia DR2 chunk %d failed: %s", j + 1, e)
            continue
        for row in r:
            did = int(row["dr3_source_id"])
            md = abs(float(row["magnitude_difference"])) if row["magnitude_difference"] is not None else 99.0
            ad = float(row["angular_distance"]) if row["angular_distance"] is not None else 99.0
            key = (md, ad)
            if did not in best or key < best[did][0]:
                best[did] = (key, float(row["pmra"]), float(row["pmdec"]),
                             float(row["pmra_error"]), float(row["pmdec_error"]))

    sid = np.array(sorted(best.keys()), dtype=np.int64)
    out = {
        "dr3_source_id": sid,
        "pmra": np.array([best[s][1] for s in sid]),
        "pmdec": np.array([best[s][2] for s in sid]),
        "pmra_error": np.array([best[s][3] for s in sid]),
        "pmdec_error": np.array([best[s][4] for s in sid]),
    }
    log.info("Gaia DR2: matched %d / %d members to a second-epoch PM", len(sid), len(ids))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **out)
    return out


# ---------------------------------------------------------------------------
# S5 spectroscopic survey (radial velocities for southern streams)
# ---------------------------------------------------------------------------

S5_VIZIER_CATALOG = "J/MNRAS/490/3508"   # Li et al. 2019, S5 survey RV catalog
S5_VIZIER_TABLE = "s5dr1"


def fetch_s5_rvs(
    cache_path: Optional[str | Path] = None,
    good_star_only: bool = True,
) -> Optional[RadialVelocityData]:
    """Fetch S5 radial velocities (VizieR ``J/MNRAS/490/3508``, Li et al. 2019).

    Uses the ``s5dr1`` table: Gaia source id (``Gaia``), calibrated line-of-sight
    velocity (``velcalib``, falling back to the posterior median ``vel50``) and
    its error (``velcalib_std`` / ``vel_std``). Optionally keeps only stars
    flagged ``good_star`` == 1. Returns None if the catalog cannot be retrieved.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        log.info("Loaded cached S5 RVs (%d stars) from %s", len(d["source_id"]), cache_path)
        return RadialVelocityData(
            source_ids=d["source_id"], rv_km_s=d["rv"], rv_error_km_s=d["e_rv"],
            survey="S5", snr=np.full(len(d["source_id"]), np.nan),
        )

    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["**"], row_limit=-1)
        cats = v.get_catalogs(S5_VIZIER_CATALOG)
    except Exception as e:
        log.warning("S5 fetch failed (VizieR): %s", e)
        return None

    tab = None
    for t in cats:
        name = t.meta.get("name", "")
        if name.endswith(S5_VIZIER_TABLE) or "Gaia" in t.colnames and "vel50" in t.colnames:
            tab = t
            break
    if tab is None:
        log.warning("S5: table %s not found in %s", S5_VIZIER_TABLE, S5_VIZIER_CATALOG)
        return None

    cols = tab.colnames
    rv_col = "velcalib" if "velcalib" in cols else "vel50"
    err_col = "velcalib_std" if "velcalib_std" in cols else ("vel_std" if "vel_std" in cols else None)

    gaia = np.array(tab["Gaia"]).astype("int64", copy=False)
    rv = np.array(tab[rv_col], dtype=np.float64)
    if err_col:
        erv = np.array(tab[err_col], dtype=np.float64)
    elif "vel84" in cols and "vel16" in cols:
        erv = 0.5 * (np.array(tab["vel84"], float) - np.array(tab["vel16"], float))
    else:
        erv = np.full(len(gaia), 2.0)

    # Drop S5 sentinel/failed-fit values (|v| ~ 1999) and implausible speeds.
    keep = np.isfinite(rv) & (gaia > 0) & (np.abs(rv) < 800.0)
    keep &= np.isfinite(erv) & (erv > 0) & (erv < 50.0)
    if good_star_only and "good_star" in cols:
        keep &= np.array(tab["good_star"]).astype(bool)
    gaia, rv, erv = gaia[keep], rv[keep], erv[keep]

    log.info("S5 (%s/%s): %d stars (rv=%s err=%s, good_star_only=%s)",
             S5_VIZIER_CATALOG, S5_VIZIER_TABLE, len(gaia), rv_col, err_col, good_star_only)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, source_id=gaia, rv=rv, e_rv=erv)
    return RadialVelocityData(
        source_ids=gaia, rv_km_s=rv, rv_error_km_s=erv,
        survey="S5", snr=np.full(len(gaia), np.nan),
    )


APOGEE_VIZIER_CATALOG = "III/286/catalog"   # APOGEE-2 DR17 allStar (Abdurro'uf+ 2022)


def fetch_apogee_rvs(
    source_ids: np.ndarray,
    ra_range: tuple[float, float],
    dec_range: tuple[float, float],
    cache_path: Optional[str | Path] = None,
) -> Optional[RadialVelocityData]:
    """Fetch APOGEE DR17 heliocentric RVs for members in a sky box, by Gaia id.

    APOGEE (VizieR ``III/286``) provides ``GaiaEDR3`` ids and heliocentric RV
    (``HRV``, error ``e_HRV``). Northern streams such as GD-1 fall outside the
    S5 footprint but are partially covered by APOGEE. The all-sky catalog is
    queried within the stream's RA/Dec bounding box, then matched to the member
    ``source_ids`` (Gaia EDR3 ids equal DR3 ids). Returns None on failure.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        d = np.load(cache_path)
        log.info("Loaded cached APOGEE RVs (%d stars) from %s", len(d["source_id"]), cache_path)
        return RadialVelocityData(
            source_ids=d["source_id"], rv_km_s=d["rv"], rv_error_km_s=d["e_rv"],
            survey="APOGEE", snr=np.full(len(d["source_id"]), np.nan))

    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["GaiaEDR3", "HRV", "e_HRV"], row_limit=-1)
        v.column_filters = {
            "RAJ2000": f"{ra_range[0]:.3f}..{ra_range[1]:.3f}",
            "DEJ2000": f"{dec_range[0]:.3f}..{dec_range[1]:.3f}",
            "HRV": "!=",
        }
        cats = v.get_catalogs(APOGEE_VIZIER_CATALOG)
    except Exception as e:
        log.warning("APOGEE fetch failed (VizieR): %s", e)
        return None
    if not len(cats):
        return None
    tab = cats[0]

    member = set(int(s) for s in np.asarray(source_ids).ravel())
    out_id, out_rv, out_e = [], [], []
    for row in tab:
        gid = str(row["GaiaEDR3"]).strip()
        if not gid or gid in ("--", "0"):
            continue
        gid = int(gid)
        if gid not in member:
            continue
        rv = float(row["HRV"])
        if not np.isfinite(rv) or abs(rv) > 600.0:
            continue
        e = float(row["e_HRV"]) if ("e_HRV" in tab.colnames and np.isfinite(row["e_HRV"])) else 1.0
        out_id.append(gid); out_rv.append(rv); out_e.append(max(e, 0.1))

    sid = np.array(out_id, dtype=np.int64)
    rv = np.array(out_rv); e = np.array(out_e)
    log.info("APOGEE DR17: %d / %d members matched with RV", len(sid), len(member))
    if cache_path and len(sid):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, source_id=sid, rv=rv, e_rv=e)
    return RadialVelocityData(source_ids=sid, rv_km_s=rv, rv_error_km_s=e,
                              survey="APOGEE", snr=np.full(len(sid), np.nan))


# ---------------------------------------------------------------------------
# Fusion across surveys, aligned to a stream's members
# ---------------------------------------------------------------------------

@dataclass
class FusionReport:
    """Per-field provenance / coverage summary for a fused catalog."""
    n_members: int = 0
    rv_n_measured: int = 0
    rv_sources: dict = field(default_factory=dict)        # survey -> n matched
    rv_median_error: float = float("nan")
    pm2epoch_n_matched: int = 0

    def to_dict(self) -> dict:
        return {
            "n_members": self.n_members,
            "rv_n_measured": self.rv_n_measured,
            "rv_sources": self.rv_sources,
            "rv_median_error_km_s": self.rv_median_error,
            "pm_second_epoch_n_matched": self.pm2epoch_n_matched,
        }


def fuse_radial_velocities(
    member_source_ids: np.ndarray,
    rv_catalogs: list[RadialVelocityData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Align several RV catalogs to a member list and inverse-variance combine.

    Returns ``(rv, e_rv, mask, per_survey_counts)`` aligned to
    ``member_source_ids`` (NaN where no survey measured the star).
    """
    member_source_ids = np.asarray(member_source_ids, dtype=np.int64)
    n = len(member_source_ids)
    index = {int(s): i for i, s in enumerate(member_source_ids)}

    values, errors = [], []
    per_survey = {}
    for cat in rv_catalogs:
        if cat is None or len(cat.source_ids) == 0:
            continue
        v = np.full(n, np.nan)
        e = np.full(n, np.nan)
        matched = 0
        for sid, rv, erv in zip(cat.source_ids, cat.rv_km_s, cat.rv_error_km_s):
            i = index.get(int(sid))
            if i is not None and np.isfinite(rv):
                v[i] = rv
                e[i] = erv if (np.isfinite(erv) and erv > 0) else 2.0
                matched += 1
        if matched:
            values.append(v)
            errors.append(e)
            per_survey[cat.survey] = matched

    if not values:
        return np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), {}

    rv, e_rv = inverse_variance_combine(values, errors)
    mask = np.isfinite(rv).astype(np.float32)
    return rv, e_rv, mask, per_survey


def fuse_proper_motions_icrs(
    member_source_ids: np.ndarray,
    dr3_pmra: np.ndarray, dr3_pmdec: np.ndarray,
    dr3_pmra_err: np.ndarray, dr3_pmdec_err: np.ndarray,
    dr2: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Inverse-variance fuse DR3 + DR2 proper motions in ICRS.

    Returns fused ``(pmra, pmdec, pmra_err, pmdec_err, n_matched_dr2)``. Note the
    main multi-epoch PM gain is *future* (DR4/DR5); DR2 fusion mainly cross-checks
    DR3 and yields a modest ~10% error reduction.
    """
    member_source_ids = np.asarray(member_source_ids, dtype=np.int64)
    n = len(member_source_ids)
    index = {int(s): i for i, s in enumerate(member_source_ids)}

    dr2_pmra = np.full(n, np.nan); dr2_pmdec = np.full(n, np.nan)
    dr2_e_pmra = np.full(n, np.nan); dr2_e_pmdec = np.full(n, np.nan)
    n_matched = 0
    for k, sid in enumerate(dr2.get("dr3_source_id", [])):
        i = index.get(int(sid))
        if i is not None:
            dr2_pmra[i] = dr2["pmra"][k]; dr2_pmdec[i] = dr2["pmdec"][k]
            dr2_e_pmra[i] = dr2["pmra_error"][k]; dr2_e_pmdec[i] = dr2["pmdec_error"][k]
            n_matched += 1

    pmra, e_pmra = inverse_variance_combine([dr3_pmra, dr2_pmra], [dr3_pmra_err, dr2_e_pmra])
    pmdec, e_pmdec = inverse_variance_combine([dr3_pmdec, dr2_pmdec], [dr3_pmdec_err, dr2_e_pmdec])
    return pmra, pmdec, e_pmra, e_pmdec, n_matched
