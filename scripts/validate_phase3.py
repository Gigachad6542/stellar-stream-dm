"""
Phase 3 physics validation:
  1. Stream track offset from galstreams reference < 1 deg in phi2
  2. Gap visible after single M=1e7 Msun impact at phi1=-40 deg

Run from project root:
    python scripts/validate_phase3.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation.potentials import get_mw_potential, vary_potential
from src.simulation.stream_gen import generate_stream, set_progenitor_ic
from src.simulation.subhalo import (
    EncounterParams, apply_impulse_approximation,
    scale_radius_from_mass,
)

import galstreams

STREAM = "GD1"
CFG    = "config/streams.yaml"
SEED   = 0

print("Loading potential …")
pot = get_mw_potential(config_path=CFG)

print("Loading galstreams …")
mws = galstreams.MWStreams(verbose=False)

print("Generating GD-1 stream (n_stars=5000) …")
import time
t0 = time.time()
p = generate_stream(STREAM, pot, n_stars=5000, seed=SEED,
                    stream_age_gyr=10.0, config_path=CFG, mws=mws)
dt = time.time() - t0
print(f"  → {len(p.phi1)} particles in {dt:.1f}s")

# ── Validation 1: track offset ────────────────────────────────────────────────
print("\n=== Validation 1: track offset (binned median phi2) ===")
import yaml, astropy.units as u, astropy.coordinates as coord
with open(CFG) as f:
    sc = yaml.safe_load(f)["streams"][STREAM]
track = mws[sc["galstreams_key"]]

# galstreams reference track phi1/phi2
tr  = track.track
sf  = track.stream_frame
tr_sf = tr.transform_to(sf)
tr_phi1 = (np.array(tr_sf.phi1.deg) + 180) % 360 - 180
tr_phi2 = np.array(tr_sf.phi2.deg)

# Keep track points in the stream phi1 range
phi1_min, phi1_max = sc["phi1_range_deg"]
tmask = (tr_phi1 >= phi1_min) & (tr_phi1 <= phi1_max)
tr_phi1 = tr_phi1[tmask]
tr_phi2 = tr_phi2[tmask]

# Interpolate reference track to a phi1 grid
from scipy.interpolate import interp1d
ref_interp = interp1d(np.sort(tr_phi1),
                      tr_phi2[np.argsort(tr_phi1)],
                      kind='linear', bounds_error=False, fill_value='extrapolate')

# Compute binned MEDIAN phi2 of simulated stream — this is the "track position"
n_bins = 15
bin_edges = np.linspace(phi1_min, phi1_max, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
sim_median_phi2 = np.full(n_bins, np.nan)
for k in range(n_bins):
    mask_k = (p.phi1 >= bin_edges[k]) & (p.phi1 < bin_edges[k+1])
    if mask_k.sum() >= 5:
        sim_median_phi2[k] = np.median(p.phi2[mask_k])

# Compute offset in bins with enough particles
valid = np.isfinite(sim_median_phi2)
ref_phi2_at_bins = ref_interp(bin_centers[valid])
offsets = sim_median_phi2[valid] - ref_phi2_at_bins
rms = float(np.sqrt(np.mean(offsets**2))) if len(offsets) > 0 else 999.0
mean_off = float(np.mean(offsets)) if len(offsets) > 0 else 999.0
print(f"  Bins with ≥5 particles: {valid.sum()}/{n_bins}")
print(f"  Median-track offset: rms={rms:.3f} deg  mean={mean_off:.3f} deg")
passed1 = rms < 1.0  # strict threshold — v3 IC optimiser should achieve this
print(f"  Track offset <1.0 deg? {'PASS ✓' if passed1 else 'FAIL ✗'}")

# ── Validation 2: gap visibility ──────────────────────────────────────────────
GAP_PHI1 = 30.0   # within GD-1-I21 track range [-21, 81]; known gap at phi1≈20-40
print(f"\n=== Validation 2: gap after M=1e7 Msun impact at phi1={GAP_PHI1} deg ===")
print(f"  Stream has {len(p.phi1)} particles in phi1 [{phi1_min}, {phi1_max}]")

m_sub = 1e7
enc = EncounterParams(
    mass_solar=m_sub,
    scale_radius_kpc=scale_radius_from_mass(m_sub),
    impact_param_kpc=0.1,    # tight flyby for a detectable gap
    flyby_vel_kms=150.0,
    encounter_phi1=GAP_PHI1,
    t_since_impact_gyr=2.0,
    is_valid=True,
    is_massive=False,
)
p_perturbed = apply_impulse_approximation(p, enc)
# Gap evolution (phi1 drift) is handled inside apply_impulse_approximation.

# Bin in phi1 and compare density
bins = np.linspace(phi1_min, phi1_max, 61)   # 2-deg bins
bc   = 0.5 * (bins[:-1] + bins[1:])
dens_orig = np.histogram(p.phi1, bins=bins)[0].astype(float)
dens_pert = np.histogram(p_perturbed.phi1, bins=bins)[0].astype(float)

# Gap region ±7 deg around impact; background = ±20 deg excluding gap
hw = 7.0   # half-width of gap search window
gap_mask = (bc >= GAP_PHI1 - hw) & (bc <= GAP_PHI1 + hw)
bg_mask  = (bc >= GAP_PHI1 - 20) & (bc <= GAP_PHI1 + 20) & ~gap_mask
bg_level = float(np.median(dens_pert[bg_mask])) if bg_mask.sum() >= 2 else 1.0
if gap_mask.sum() == 0:
    print(f"  WARNING: no bins in gap region — widen phi1_range_deg or move GAP_PHI1")
    gap_visible = False
    gap_depth = float('nan')
else:
    gap_depth = float(np.min(dens_pert[gap_mask])) / max(bg_level, 1.0)
    gap_visible = gap_depth < 0.85  # gap = at least 15% depletion
print(f"  Background level: {bg_level:.1f} stars/2-deg-bin")
print(f"  Density in gap: min/background = {gap_depth:.3f}")
print(f"  Gap visible?  {'PASS ✓' if gap_visible else 'FAIL ✗'}")

# ── Diagnostic plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

ax = axes[0]
ax.scatter(p.phi1, p.phi2, s=0.3, alpha=0.3, c="steelblue", label="simulated")
ax.plot(tr_phi1, tr_phi2, "r-", lw=1.5, label="galstreams track")
ax.set_xlabel("phi1 [deg]"); ax.set_ylabel("phi2 [deg]")
ax.set_title(f"GD-1 stream track (rms offset = {rms:.3f} deg)")
ax.legend(markerscale=5)

ax = axes[1]
ax.step(bc, dens_orig, where="mid", color="steelblue", lw=1.2, label="unperturbed")
ax.step(bc, dens_pert, where="mid", color="crimson", lw=1.2, label=f"after M=1e7 impact at φ1={GAP_PHI1}°")
ax.axvspan(GAP_PHI1-hw, GAP_PHI1+hw, alpha=0.12, color="orange", label="gap region")
ax.axvline(GAP_PHI1, ls="--", color="orange", lw=0.8)
ax.set_xlabel("phi1 [deg]"); ax.set_ylabel("N stars / bin")
gd = f"{gap_depth:.2f}" if np.isfinite(gap_depth) else "N/A"
ax.set_title(f"Density profile (gap depth {gd}×bg)")
ax.legend()

ax = axes[2]
if valid.sum() > 0:
    ax.bar(bin_centers[valid], offsets, width=(phi1_max-phi1_min)/n_bins*0.8, color="steelblue")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="1 deg goal")
    ax.axhline(-1.0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("phi1 bin center [deg]"); ax.set_ylabel("median(phi2_sim) − phi2_ref [deg]")
    ax.set_title(f"Binned median phi2 offset  rms={rms:.3f} deg  mean={mean_off:.3f} deg")
    ax.legend()

plt.tight_layout()
out = Path("outputs/phase3_validation.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out), dpi=120)
print(f"\nPlot saved → {out}")

print("\n=== Summary ===")
print(f"  Track offset <1 deg:  {'PASS' if passed1 else 'FAIL'}")
print(f"  Gap visible:          {'PASS' if gap_visible else 'FAIL'}")
if passed1 and gap_visible:
    print("\nPhase 3 PASS — all physics checkpoints satisfied.")
else:
    print("\nPhase 3 INCOMPLETE — see details above.")
