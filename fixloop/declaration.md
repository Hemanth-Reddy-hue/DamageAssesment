# Fix Loop Declaration  [marker: CEILING-CLAMP-DECLARATION]

Committed BEFORE the fix. The earlier G6 declaration (commit 42d0cdb) was written
before any implementation existed and has no real before/after run behind it; it is
archived in fixloop/archive/ and is not claimed as a fix loop.

## Failing gate
G2 (ceiling height) and G8 (calibration): the pipeline reports a ceiling that is not
an observation with an interval that claims centimetre precision.

## Evidence (reproduce: python bench/fixloop_run.py --label before)
- out/lidar/plan.json: ceiling_height = 2.4 m, [2.3808, 2.4192], method "conformal".
  The value is exactly the constant used by np.clip(..., 2.4, 3.2) in the LiDAR branch of
  src/areamap/geometry/planes.py; out/HOUSE1/plan.json (photo) shows the same 2.4.
- When no ceiling plane is observed, planes.py sets method="prior" but the interval is
  still calculate_interval(...) = about +/-1.9 cm at LiDAR tier.
- Photo-tier clouds are synthesized at ceiling_height_prior, so a "measured" photo
  ceiling is circular.

## Root cause
planes.py (ceiling block): out-of-range or unobserved ceilings are replaced by clamp
constants / priors, then given a tier-default tight interval. Provenance is lost.

## Fix
1. Ceiling is "measured" only when a floor AND a ceiling plane were found and the
   height is within 2.0 to 4.5 m. Otherwise method="prior", value 2.60 m (or the raw
   photo value), interval value +/- 0.40 m, plus a pipeline warning.
2. Photo tier: always "prior".
3. No np.clip on the ceiling.

## Predictions (also in fixloop/prediction.json, written now)
- Before (hypothesis to confirm): both LiDAR scans report 2.4 / "conformal".
- After, single_scan_floor_only: method "prior", interval width >= 0.79 m, warning present.
- After, single_scan_with_ceiling: method "measured", interval width <= 0.06 m.
  Risk: if the raw ceiling plane is outside 2.0-4.5 m the result becomes "prior";
  the post-mortem will report whichever happens.

## What this fix does NOT claim
It does not improve geometric accuracy. It makes the reported ceiling honest and the
interval truthful. Accuracy against tape remains NOT MEASURED.
