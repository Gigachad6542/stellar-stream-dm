# 2026-05-30 Full-orbit injection-recovery validation (improvement D)

Validated the physically-correct full orbit-integrated forward-model path
end-to-end, not just the fast impulse approximation.

## What was validated

1. **Full-orbit inject + recover.** A known 10^9 Msun impact (t=1.5 Gyr,
   phi1=20 deg) is injected by spraying particles, integrating them to the
   impact epoch, applying the Erkal+2015 kick, and integrating forward to the
   present; the full grid is then evaluated the same way. On a coarse 12-point
   grid the impact is recovered within one grid step on each axis (mass 8.5 vs
   9.0, phi1 10 vs 20, time exact) and beats the unperturbed null. Runtime ~18 s
   for the grid (1200-star streams).

2. **Cross-mode (full-orbit truth, fast recovery).** Injecting with full orbit
   integration and recovering with the fast impulse mode still recovers the mass
   exactly (10^9), but the best fit only marginally beats the null (1.38 vs
   1.42) -- the impulse approximation is a usable but degraded stand-in for the
   full non-linear evolution, as expected. This quantifies the cost of the fast
   shortcut.

## Change
- `tests/test_injection_recovery.py`: added `test_injection_recovery_full_orbit`
  (slow) asserting the full-orbit path beats null and recovers mass + phi1 within
  one grid step.

This closes the four "as close to perfect as possible" accuracy improvements:
A (Erkal+2015 kick), B (look-elsewhere significance), C (GD-1 APOGEE RVs), and
D (full-orbit injection-recovery validation).
