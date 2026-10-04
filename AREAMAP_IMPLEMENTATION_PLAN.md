# AreaMap: End-to-End Implementation & Verification Plan (6 hours)

Built from: README.md, the audit file, and README_CODEBASE_REFERENCE.md (source, outputs, git log).
Machine: Windows, CPU only. No tape, no consumer-app export, no LiDAR iPhone.

---

## 0. Ground rules

1. **No number without a source.** Every gate result is computed from `plan.json` files and a reference CSV, and carries one evidence label:

| Label | Meaning |
|---|---|
| `MEASURED` | Reference is tape or laser |
| `INDICATIVE` | Reference is a tile count, a LiDAR-derived value or a rough estimate |
| `SYNTHETIC` | Reference is a synthetic fixture (tests only) |
| `NOT MEASURED` | No reference data exists |

2. **Status vocabulary** is `PASS`, `FAIL`, `INCONCLUSIVE`, `NOT MEASURED`. The harness only says `PASS` when the reference is precise enough to resolve the gate. It says `FAIL` only when the error exceeds tolerance even after allowing for reference error. Otherwise it says `INCONCLUSIVE`. This is what makes rough estimates usable without lying.
3. **Git order is evidence.** Declaration commit, then fix commit, then after-run commit. Nothing is rewritten or squashed.
4. **Delete every hardcoded figure** (78.6%, 100%, 64.2%, 87.84%, 64.50 m2, 11.2% to 5.1%, 1.2 cm, 0.9 cm, 89.4%) from scripts, README, report and device matrix.
5. **Branch first:** `git checkout -b hardening/honest-gates`.

---

## 1. What the code shows (corrections and additions to the audit)

These change the plan, so read them first.

| # | Finding | Where | Consequence |
|---|---|---|---|
| F1 | The ceiling is clamped in the **LiDAR and photo branches** of `planes.py` and still labeled `measured` (it becomes `conformal` in the interval). `calibrate.py` does not touch methods. It only copies intervals. | `planes.py`, ceiling block | The fix goes in `planes.py` (+ a warning in `geometry.py`), not in calibrate |
| F2 | When the ceiling is unobserved, the code sets `method="prior"` but still emits a tight interval (about +/-1.9 cm at LiDAR) | `planes.py`, `calculate_interval(...)` | A prior must carry a wide band |
| F3 | **Photo-tier ceiling is circular.** `recover_metric_scale_and_points` and the depth engine place points at `ceiling_height_prior` (2.50), then RANSAC "finds" a ceiling there | `tiers/photo.py` | Photo ceiling is always a prior |
| F4 | `method="conformal"` is a label on a fixed percentage (0.8% / 2.5% / 6.5%). Nothing is conformal | `uncertainty.py` | Relabel to `tier_prior` (WP-F) |
| F5 | `extent_metric` for damage comes from a VLM guess (`estimated_extent_m2`) and gets a +/-0.8% LiDAR interval. It also silently defaults to 0.75 m2 | `damage.py` | Wide `vlm_estimate` band; no invented default |
| F6 | The damage prompt is **inline in `damage.py`**; `prompts/damage.md` is never read | `damage.py` | Fix the inline prompt, keep the .md in sync |
| F7 | `out/lidar/plan.json` contains damage notes "Offline heuristic detection". That fallback was deleted from `client.py`. The outputs are **stale and partly fabricated** | `out/*` | Regenerate every output before using it as a baseline |
| F8 | `qa_critic.py` lists `check_polygon_closure` in `checks_run` but never runs it | `qa_critic.py` | Implement it or stop claiming it (WP-A3) |
| F9 | `plan.json` contains `timings` and `output_dir`, so two runs never hash identically | export | Determinism test must strip them. `main.py` has **no `--seed` flag**; seeds are already fixed to 42 in code |
| F10 | `scripts/run_capture.py` defaults `--tier lidar` and **forces** it on any input | `run_capture.py` | Default to auto (None) |
| F11 | `stitch_node` hardcodes `enable_drift_correction=True` and `drift_correction_applied=True` | `stitch.py` | Drift ablation currently cannot be run. Add a config flag |
| F12 | Node failures crash the whole run (`ingest` raises on missing/failed input). README M15 requires a partial result with warnings | `main.py` | Wrap each node (WP-A4) |
| F13 | Changing the VLM prompt changes the cache key, so cached VLM answers (e.g. HOUSE1 cracks) stop replaying. With `OFFLINE=1` damage becomes "unavailable" until the cache is re-warmed | `cache.py`, `client.py` | Re-warm once with network and a key (Phase 2), commit the cache for the demo captures |
| F14 | `fixloop/declaration.md` for G6 was committed in `42d0cdb` before any implementation, with no real before run | git log | Write a **new** declaration (ceiling), archive the old one, say so openly |

---

## 2. Timeline (6:00, clock times assume a 10:25 start)

| Clock | Phase | Output | Commit |
|---|---|---|---|
| 10:25-10:55 | **P0** Branch, regenerate baseline, write declaration + prediction | `fixloop/before/`, `declaration.md`, `prediction.json` | C1 |
| 10:55-11:55 | **P1** Ceiling fix (WP-B) + tests, after-run, `diff.patch` | `planes.py`, `geometry.py` | C2, C3 |
| 11:55-12:50 | **P2** Damage fix, QA closure, drift flag, `run_capture`, crash-safe pipeline, interval relabel | WP-A1..A4, WP-E, WP-F | C4-C7 |
| 12:50-13:10 | Buffer / meal | | |
| 13:10-14:40 | **P3** Real harness, ablation, reference data | `bench/*`, `Data/ground_truth/*` | C8, C9 |
| 14:40-15:25 | **P4** Verification suite, log results | `reports/verification_log.md` | C10 |
| 15:25-16:00 | **P5** Docs: compliance matrix, report, README, device matrix | | C11 |
| 16:00-16:25 | Final: full `pytest -q`, `bench/harness.py` twice, `handoff.zip` | | C12 |

Parallel work: start the 3-minute `pytest -q` and long video runs in a second terminal, then write the next block while they run. Reference data (tile counts, estimates) can be collected by a second person at home during P1 to P3.

---

## 3. P0: Baseline and declaration (10:25-10:55)

### 3.1 Commands (PowerShell, repo root)

```powershell
git checkout -b hardening/honest-gates
git status                         # must be clean
python -m pytest -q                # note the pass count (expect 116)
```

Add `bench/fixloop_run.py` (section 8.4 below), then:

```powershell
python bench\fixloop_run.py --label before
# writes fixloop\before\summary.json and fixloop\before\<case>\plan.json
```

Do **not** edit any source file before the declaration commit.

### 3.2 Archive the old declaration, write the new one

```powershell
mkdir fixloop\archive
git mv fixloop\declaration.md fixloop\archive\declaration_g6_placeholder.md
```

`fixloop/declaration.md` (new):

```markdown
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
```

`fixloop/prediction.json`:

```json
{
  "before_hypothesis": {
    "floor_only":   {"value": 2.4, "method": "conformal"},
    "with_ceiling": {"value": 2.4, "method": "conformal"}
  },
  "after": {
    "floor_only":   {"method": "prior",    "min_width_m": 0.79},
    "with_ceiling": {"method": "measured", "max_width_m": 0.06}
  }
}
```

### 3.3 Commit C1

```powershell
git add fixloop bench\fixloop_run.py
git commit -m "fixloop: declare ceiling-clamp defect with predictions (before any fix)"
git log --oneline -1               # record this hash: DECL
```

**Acceptance P0:** `fixloop/before/summary.json` exists; the declaration commit contains no `src/` change.

---

## 4. P1: Ceiling fix (WP-B), 10:55-11:55

### 4.1 `src/areamap/geometry/planes.py`

At the top, add:

```python
import logging
logger = logging.getLogger(__name__)

CEILING_PRIOR_M = 2.60
CEILING_PRIOR_HALF_WIDTH_M = 0.40      # plausibility band for residential ceilings, not a statistical interval
CEILING_MIN_M, CEILING_MAX_M = 2.0, 4.5
```

Replace everything from `ceiling_observed = True` down to (not including) `# 2. Extract Vertical Wall Planes` with:

```python
    # Ceiling is "measured" only if a floor AND a ceiling plane were observed and plausible.
    ceiling_method = "measured"
    if floor is not None and ceiling is not None:
        floor_z = floor["z_mean"]
        ceiling_z = ceiling["z_mean"]
        ceiling_height_val = float(ceiling_z - floor_z)
        if not (CEILING_MIN_M <= ceiling_height_val <= CEILING_MAX_M):
            logger.warning(
                "Ceiling candidate %.2f m outside [%.1f, %.1f]; using prior", 
                ceiling_height_val, CEILING_MIN_M, CEILING_MAX_M,
            )
            ceiling_method = "prior"
            ceiling_height_val = CEILING_PRIOR_M
    elif floor is not None:
        floor_z = floor["z_mean"]
        z_high = float(np.percentile(points[:, 2], 95))
        ceiling_z = z_high                       # used only to bound wall extraction
        ceiling_height_val = CEILING_PRIOR_M
        ceiling_method = "prior"
    else:
        z_min, z_max = np.percentile(points[:, 2], [5, 95])
        floor_z, ceiling_z = float(z_min), float(z_max)
        ceiling_height_val = CEILING_PRIOR_M
        ceiling_method = "prior"

    if tier == "photo":
        # Photo clouds are synthesized at the prior ceiling height: not an observation.
        ceiling_method = "prior"
```

Replace the ceiling interval block (starts at `ceil_interval = calculate_interval(ceiling_height_val, "ceiling", tier)` and ends after the `c_margin` lines) with:

```python
    if ceiling_method == "prior":
        ceil_interval = Interval(
            value=round(ceiling_height_val, 4),
            lo=round(ceiling_height_val - CEILING_PRIOR_HALF_WIDTH_M, 4),
            hi=round(ceiling_height_val + CEILING_PRIOR_HALF_WIDTH_M, 4),
            confidence_level=0.90,
            method="prior",
            tier=tier,
        )
    else:
        ceil_interval = calculate_interval(ceiling_height_val, "ceiling", tier)
        if scale_relative_uncertainty > 0:
            c_margin = ceiling_height_val * scale_relative_uncertainty
            ceil_interval.lo = max(0.5, round(ceil_interval.lo - c_margin, 3))
            ceil_interval.hi = round(ceil_interval.hi + c_margin, 3)
```

`residual_ceiling` is no longer used; delete any leftover references. The photo bbox fallback still reads `ceiling_height_val`; that is fine.

### 4.2 `src/areamap/nodes/geometry.py`

After the loop that builds `room_geometry`, add:

```python
    for rid, geom in room_geometry.items():
        if geom.ceiling_height.method == "prior":
            msg = (f"Room {rid}: ceiling height is a prior ({geom.ceiling_height.value:.2f} m, "
                   f"band +/-{(geom.ceiling_height.hi - geom.ceiling_height.lo)/2:.2f} m), not a measurement.")
            if msg not in warnings:
                warnings.append(msg)
```

### 4.3 Test: `tests/unit/test_ceiling_provenance.py`

```python
import numpy as np
from areamap.geometry.planes import fit_room_planes
from areamap.tiers.lidar import _generate_synthetic_box


def _noisy(box, seed=0, sigma=0.004):
    rng = np.random.default_rng(seed)
    return box + rng.normal(0, sigma, box.shape)


def test_full_box_lidar_is_measured_and_tight():
    pts = _noisy(_generate_synthetic_box(4.0, 3.0, 2.6, n_points=8000))
    g = fit_room_planes(pts, tier="lidar")
    assert g.ceiling_height.method == "measured"
    assert abs(g.ceiling_height.value - 2.6) < 0.05
    assert (g.ceiling_height.hi - g.ceiling_height.lo) <= 0.06


def test_floor_only_is_prior_with_wide_band():
    box = _generate_synthetic_box(4.0, 3.0, 2.6, n_points=8000)
    pts = _noisy(box[box[:, 2] <= 1.4])
    g = fit_room_planes(pts, tier="lidar")
    assert g.ceiling_height.method == "prior"
    assert (g.ceiling_height.hi - g.ceiling_height.lo) >= 0.79


def test_implausible_ceiling_is_not_clamped_to_2_4():
    pts = _noisy(_generate_synthetic_box(4.0, 3.0, 1.5, n_points=8000))
    g = fit_room_planes(pts, tier="lidar")
    assert g.ceiling_height.method == "prior"
    assert g.ceiling_height.value != 2.4


def test_photo_tier_ceiling_is_always_prior():
    pts = _noisy(_generate_synthetic_box(4.0, 3.2, 2.5, n_points=4000), sigma=0.03)
    g = fit_room_planes(pts, tier="photo")
    assert g.ceiling_height.method == "prior"
```

If a fixture assumption is off (for example the generator's point density), adjust `n_points` or the `z <= 1.4` cut, not the assertions.

### 4.4 Run, after-run, diff, commits

```powershell
python -m pytest tests\unit\test_ceiling_provenance.py -q
python -m pytest -q                       # fix any older test that asserted the 2.4 clamp
git add src tests
git commit -m "fix: ceiling is measured only when observed; priors carry wide band and warning"   # C2 -> FIX
python bench\fixloop_run.py --label after
git diff DECL..FIX -- src > fixloop\diff.patch
python bench\verify_fixloop.py            # writes fixloop\verify_report.json
```

Write `fixloop/postmortem.md` (10 lines): predicted vs actual for each of the four predictions, honest either way, and a note that accuracy against tape is not measured.

```powershell
git add fixloop
git commit -m "fixloop: after-run, real diff.patch, post-mortem"       # C3
```

**Acceptance P1:** `verify_report.json` shows `order_ok: true`; prediction rows are filled in with actual values; full `pytest -q` is green.

---

## 5. P2: Remaining fixes (11:55-12:50)

### WP-A1: `src/areamap/nodes/damage.py`

Add helpers above `damage_node`:

```python
def _resolve_surface_id(surface: str, state: CaptureState) -> tuple[str | None, str]:
    """Return (surface_id, note). Wall damage cannot be tied to a specific wall from one image."""
    s = (surface or "").lower()
    if "ceil" in s:
        return "ceiling", ""
    if "floor" in s:
        return "floor", ""
    geom = None
    for rid in state.rooms:
        geom = state.room_geometry.get(rid)
        if geom:
            break
    if geom is None and state.room_geometry:
        geom = next(iter(state.room_geometry.values()))
    if geom and geom.walls:
        return geom.walls[0].wall_id, (
            f"wall not localized from a single image; assigned to {geom.walls[0].wall_id}")
    return None, "no wall geometry available to bind damage to"


def _clean_bbox(bb) -> tuple[list[float] | None, bool, str]:
    """Return (bbox, whole_image_fallback, note). bbox must be [ymin,xmin,ymax,xmax] in [0,1]."""
    try:
        v = [float(x) for x in bb]
    except (TypeError, ValueError):
        return None, False, "bbox missing"
    if len(v) != 4 or any(x < 0.0 or x > 1.0 for x in v):
        return None, False, "bbox invalid (not normalized [0,1])"
    ymin, xmin, ymax, xmax = v
    if ymax <= ymin or xmax <= xmin:
        return None, False, "bbox degenerate"
    if (ymax - ymin) * (xmax - xmin) > 0.95:
        return v, True, "whole_image_fallback: bbox covers >95% of image"
    return v, False, ""
```

Replace the VLM prompt and schema so boxes are constrained:

```python
"bounding_box_2d": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 4, "maxItems": 4,
                    "description": "[ymin, xmin, ymax, xmax] normalized to 0..1"},
```

```python
    prompt = (
        "You are a professional property damage assessor. "
        "Inspect the image for visible surface damage: water stains, mold, cracks, spalling, efflorescence. "
        "If there is no damage, return an empty damage_findings array. Do NOT fabricate findings. "
        "For each real finding give: damage_class, severity (minor|moderate|severe), "
        "surface (ceiling|wall|floor), and bounding_box_2d as [ymin, xmin, ymax, xmax] with every value "
        "normalized to the range 0..1 relative to the image (never pixels, never the whole image unless "
        "the damage truly fills it). Also give estimated_extent_m2 only if you can justify it; otherwise omit it."
    )
```

Replace the loop body that builds `DamageRegion` (from `extent_m2 = ...` to the `damage_regions.append(...)`) with:

```python
    for i, item in enumerate(findings):
        d_class = item.get("damage_class", "water_stain")
        if d_class == "none":
            continue
        severity = item.get("severity")
        if severity not in ("minor", "moderate", "severe"):
            severity = "minor"

        surface = item.get("surface") or ("ceiling" if d_class == "water_stain" else "wall")
        surface_id, sid_note = _resolve_surface_id(surface, state)
        if surface_id is None:
            state.warnings.append(f"Damage {d_class} #{i+1} dropped: {sid_note}")
            continue

        bbox, whole, bbox_note = _clean_bbox(item.get("bounding_box_2d"))

        # The extent is a VLM estimate, not a measurement: wide band, labeled as such.
        raw_extent = item.get("estimated_extent_m2")
        notes = [n for n in (item.get("notes", ""), sid_note, bbox_note) if n]
        confidence = 0.6
        if isinstance(raw_extent, (int, float)) and raw_extent > 0:
            v = float(raw_extent)
            half = 0.5 if not whole else 1.0
            lo, hi = max(0.0, v * (1 - min(half, 0.9))), v * (1 + half)
        else:
            v, lo, hi = 0.5, 0.1, 2.0
            confidence = 0.3
            notes.append("extent not provided by VLM; wide placeholder band, not a measurement")
        if whole:
            confidence = min(confidence, 0.3)

        damage_regions.append(DamageRegion(
            damage_id=f"dmg_{d_class}_{i+1:02d}",
            surface_id=surface_id,
            damage_class=d_class,
            severity=severity,
            extent_metric=Interval(value=round(v, 4), lo=round(lo, 4), hi=round(hi, 4),
                                   confidence_level=0.90, method="vlm_estimate", tier=state.tier or "lidar"),
            bounding_box_2d=bbox,
            confidence=confidence,
            notes="; ".join(notes),
        ))
```

Add `Interval` to the import from `areamap.state`. Update `src/areamap/llm/prompts/damage.md` to match (add the `surface` and "never pixels" lines) so the two files do not disagree.

### WP-A1 test: `tests/unit/test_damage_surface_binding.py`

```python
import pytest
from areamap.state import CaptureState, RoomGeometry, WallSegment
from areamap.geometry.uncertainty import calculate_interval
from areamap.llm.client import LLMResult
import areamap.nodes.damage as dmg
from areamap.nodes.scope import scope_node
from areamap.nodes.qa_critic import qa_critic_node


def _room(rid):
    iv = lambda v: calculate_interval(v, "wall", "lidar")
    pts = [[0, 0], [4, 0], [4, 3], [0, 3]]
    walls = [WallSegment(wall_id=f"{rid}_w{i+1}", start=pts[i], end=pts[(i+1) % 4], length=iv(3.0)) for i in range(4)]
    return RoomGeometry(room_id=rid, room_name=rid, ceiling_height=calculate_interval(2.6, "ceiling", "lidar"),
                        floor_area=calculate_interval(12.0, "area", "lidar"), walls=walls, floor_polygon=pts)


class _Fake:
    def __init__(self, items): self.items = items
    def generate_structured(self, **kw):
        return LLMResult(ok=True, data={"damage_findings": self.items}, status="ok")


@pytest.mark.parametrize("rid", ["room_00", "room_A", "hallway"])
def test_wall_damage_binds_to_existing_wall(monkeypatch, rid):
    items = [{"damage_class": "crack", "severity": "moderate", "surface": "wall",
              "bounding_box_2d": [0.2, 0.2, 0.5, 0.6], "estimated_extent_m2": 0.5}]
    monkeypatch.setattr(dmg, "get_llm_client", lambda *a, **k: _Fake(items))
    monkeypatch.setattr(dmg, "_find_representative_image", lambda p: None)
    st = CaptureState(capture_path="x", tier="lidar", rooms=[rid], room_geometry={rid: _room(rid)})
    st.damage = dmg.damage_node(st)["damage"]
    wall_ids = {w.wall_id for w in st.room_geometry[rid].walls}
    assert st.damage and st.damage[0].surface_id in wall_ids
    st.scope_items = scope_node(st)["scope_items"]
    qa = qa_critic_node(st)["qa_report"]
    assert not [f for f in qa.failed_checks if "non-existent surface" in f]


def test_whole_image_bbox_flagged_and_extent_is_wide(monkeypatch):
    items = [{"damage_class": "crack", "severity": "minor", "surface": "wall",
              "bounding_box_2d": [0.0, 0.0, 1.0, 1.0], "estimated_extent_m2": 1.0}]
    monkeypatch.setattr(dmg, "get_llm_client", lambda *a, **k: _Fake(items))
    monkeypatch.setattr(dmg, "_find_representative_image", lambda p: None)
    st = CaptureState(capture_path="x", tier="lidar", rooms=["room_00"], room_geometry={"room_00": _room("room_00")})
    d = dmg.damage_node(st)["damage"][0]
    assert "whole_image_fallback" in d.notes
    assert d.extent_metric.method == "vlm_estimate"
    assert (d.extent_metric.hi - d.extent_metric.lo) / d.extent_metric.value >= 0.9


def test_pixel_bbox_is_rejected(monkeypatch):
    items = [{"damage_class": "crack", "severity": "minor", "surface": "ceiling",
              "bounding_box_2d": [0, 0, 1000, 800], "estimated_extent_m2": 0.4}]
    monkeypatch.setattr(dmg, "get_llm_client", lambda *a, **k: _Fake(items))
    monkeypatch.setattr(dmg, "_find_representative_image", lambda p: None)
    st = CaptureState(capture_path="x", tier="lidar", rooms=["room_00"], room_geometry={"room_00": _room("room_00")})
    assert dmg.damage_node(st)["damage"][0].bounding_box_2d is None
```

### WP-A2: re-warm the VLM cache (F13)

```powershell
# needs one provider key in .env (e.g. MISTRAL_API_KEY) and internet, once
$env:OFFLINE="0"
python main.py Data\HOUSE1 --no-local-models
$env:OFFLINE="1"
python main.py Data\HOUSE1 --no-local-models       # must now replay from data\cache
git add data\cache
```

Check that `out/HOUSE1/plan.json` has `qa_report.passed == true`, no whole-image box, and no failed `surface_reference_integrity`. For an unseen room at the walk-in with no internet, damage will correctly report "unavailable". Say this in the report.

### WP-A3: `src/areamap/nodes/qa_critic.py`: implement the closure check you already list

Add to `checks_run`: `"check_ceiling_provenance"` (informational) and implement closure inside the room loop:

```python
    # polygon closure + area consistency
    for r_id, room in state.room_geometry.items():
        n = len(room.walls)
        for i, w in enumerate(room.walls):
            nxt = room.walls[(i + 1) % n]
            gap = ((w.end[0] - nxt.start[0]) ** 2 + (w.end[1] - nxt.start[1]) ** 2) ** 0.5
            if gap > 0.02:
                failed_checks.append(f"{r_id}: wall {w.wall_id} does not close to next wall (gap {gap:.3f} m)")
        if len(room.floor_polygon) >= 3:
            pts = room.floor_polygon
            a = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                        for i in range(len(pts)))) / 2.0
            if room.floor_area.value > 0 and abs(a - room.floor_area.value) / room.floor_area.value > 0.05:
                failed_checks.append(f"{r_id}: polygon area {a:.2f} vs floor_area {room.floor_area.value:.2f} differ >5%")
        if room.ceiling_height.method == "prior":
            adjustments_made.append(f"{r_id}: ceiling is a prior (informational, not a failure)")
```

Test (add to a new `tests/unit/test_qa_critic_checks.py`): corrupt one wall's `end` by 0.3 m and assert the failure is reported; inject a 20% wall error into `floor_area` and assert it is flagged.

### WP-A4: `main.py`: crash-safe pipeline (F12)

Add near the top-level helpers:

```python
def _run_node(label, node_fn, state, failures):
    try:
        updates = node_fn(state)
    except Exception as exc:  # partial-result policy (M15)
        msg = f"NODE_FAILED {label}: {type(exc).__name__}: {str(exc)[:200]}"
        state.warnings.append(msg)
        failures.append(msg)
        return False
    if updates and isinstance(updates, dict):
        for k, v in updates.items():
            setattr(state, k, v)
    return True


def _write_emergency_plan(state, out_dir):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "plan.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
```

In `run_pipeline`, replace the `steps` loop, the separate `scope_node` / `qa_critic_node` calls and the final `export_node(...)` with:

```python
    steps = [
        ("[1/9] Ingestion & Tier Router", ingest_node),
        ("[2/9] RANSAC Room Geometry & Planes", geometry_node),
        ("[3/9] Opening Cutouts & Phantom Suppression", openings_node),
        ("[4/9] Multi-Room Alignment & Stitching", stitch_node),
        ("[5/9] Interval Calibration", calibrate_node),
        ("[6/9] Damage Proposals", damage_node),
        ("[7/9] Forensic Rules", concealed_node),
        ("[8/9] Repair Scope", scope_node),
        ("[9/9] QA Critic", qa_critic_node),
    ]
    failures: list[str] = []
    for label, node_fn in steps:
        print(f"  -> {label}...", end="", flush=True)
        ok = _run_node(label, node_fn, state, failures)
        print(" [DONE]" if ok else " [FAILED]")

    if failures:
        from areamap.state import QAReport
        prior = state.qa_report
        state.qa_report = QAReport(
            passed=False,
            checks_run=prior.checks_run if prior else [],
            failed_checks=failures + (prior.failed_checks if prior else []),
            overall_confidence=0.0,
        )

    def _export(s):
        return export_node(s, output_dir=resolved_output_dir)
    if not _run_node("export", _export, state, failures):
        _write_emergency_plan(state, resolved_output_dir)

    try:
        _print_summary(state, resolved_output_dir)
    except Exception as exc:
        print(f"[AreaMap] summary skipped: {exc}")
    return state.model_dump()
```

In `main()`, after `run_pipeline(...)` capture the result and `sys.exit(2)` if any warning starts with `NODE_FAILED` (a clear partial-result exit code, no stack trace).

Test `tests/unit/test_partial_result.py`:

```python
from pathlib import Path
import main as m


def test_missing_input_yields_partial_plan_not_crash(tmp_path):
    out = tmp_path / "o"
    res = m.run_pipeline(str(tmp_path / "does_not_exist"), output_dir=str(out), no_llm=True, no_local_models=True)
    assert any(w.startswith("NODE_FAILED") for w in res["warnings"])
    assert res["qa_report"]["passed"] is False
    assert (out / "plan.json").exists()
```

### WP-C1: drift flag (F11)

`config.py`, inside `AreaMapConfig`:

```python
    drift_correction: bool = Field(default_factory=lambda: os.getenv("DRIFT_CORRECTION", "1").lower() in ("1", "true", "yes"))
```

`nodes/stitch.py`: `from areamap.config import settings`, then:
- single-room return: `drift_correction_applied=False` (nothing to correct)
- `enable_drift_correction=settings.drift_correction`
- final `StitchedPlan(... drift_correction_applied=settings.drift_correction ...)`

If an old test asserts `drift_correction_applied is True` for a single room, update it: that value was a hardcoded constant.

### WP-D: `scripts/run_capture.py`

Change `--tier` to `choices=["auto","photo","video","lidar"], default="auto"` and pass `tier=None if args.tier == "auto" else args.tier` into `CaptureState`.

### WP-E: relabel intervals (F4), timebox 20 minutes

```powershell
Select-String -Path src\*,tests\*,bench\* -Pattern "conformal" -Recurse
```

In `uncertainty.py` change the default `method="conformal"` to `method="tier_prior"`. Update any test that asserts `"conformal"`. Doc line for the report: "Intervals are tier-dependent relative bands plus sensor noise floors. They become empirical (conformal) only if `calibration_report.py` is fitted on enough reference data; with the data available they are not."

If more than about five tests break, stop and do only the report wording change.

### Commits

```
C4  fix: damage surface-id binding, normalized bbox, vlm_estimate extent + tests
C5  fix: qa_critic closure/area checks + tests
C6  fix: crash-safe pipeline (partial result) + drift flag + run_capture auto tier
C7  chore: relabel interval method tier_prior (if done)
```

Run `python -m pytest -q` before each commit.

---

## 6. P3: Real harness and reference data (13:10-14:40)

### 6.1 `bench/gates.py` (thresholds only, no results)

Keep `GATES_CONFIG` as it is. **Delete `EVALUATED_GATES`.** Add:

```python
THRESHOLDS = {
    "G1": {"abs_m": 0.02, "min_frac": 0.85},
    "G2": {"abs_m": 0.015},
    "G3": {"abs_m": 0.01, "rel": 0.005},
    "G5": {"rel": 0.08},
    "G6": {"rel": 0.08},
    "G7": {"rel": 0.03},
    "G8": {"lo": 0.85, "hi": 0.95, "min_n": 10},
}
```

Run `Select-String -Path . -Pattern "EVALUATED_GATES" -Recurse` and fix any importer.

### 6.2 `bench/harness.py`

```python
"""M2 benchmark harness. Every number is computed from plan.json files and reference CSVs.
Nothing is stored here: a gate with no reference data prints NOT MEASURED."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.gates import GATES_CONFIG, THRESHOLDS  # noqa: E402

REF_SIGMA_REL = {"tape": 0.002, "laser": 0.001, "tile": 0.015, "lidar_ref": 0.01,
                 "estimate": 0.05, "synthetic": 0.0}
TRUSTED = {"tape", "laser"}


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plan(path, root):
    return json.loads((Path(root) / path).read_text(encoding="utf-8"))


def classify(err, tol, sigma):
    """hit: provably within tolerance; miss: outside even allowing for reference error; else unclear."""
    if err > tol + sigma:
        return "miss"
    if err <= tol and sigma <= tol / 2:
        return "hit"
    return "unclear"


def _row(case, tier, rtype, feat, gt, iv, source):
    return dict(case=case, tier=tier, type=rtype, feature=feat, gt=gt, pred=iv["value"],
                lo=iv["lo"], hi=iv["hi"], err=abs(iv["value"] - gt),
                sigma=REF_SIGMA_REL.get(source, 0.05) * gt,
                covered=iv["lo"] <= gt <= iv["hi"], method=iv.get("method"), source=source)


def evaluate_case(case, root):
    plan = _plan(case["plan_json"], root)
    tier, name, room = plan.get("tier"), case["case_id"], case["room_id"]
    gt_all = [r for r in _csv(Path(root) / case["gt_csv"]) if (r.get("dimension_m") or "").strip()]
    src = lambda r: (r.get("source") or "synthetic").strip()
    rows, ev = [], {"missed": 0, "phantom": 0, "overlaps": 0, "room_found": True}
    geom = plan.get("room_geometry", {}).get(room)
    gt = [r for r in gt_all if r["room_id"] == room]
    if geom is None:
        ev["room_found"] = False
    else:
        gw = sorted((float(r["dimension_m"]), src(r)) for r in gt if r["type"] == "wall_length")
        pw = sorted(geom["walls"], key=lambda w: w["length"]["value"])
        for i, ((g, s), w) in enumerate(zip(gw, pw)):
            rows.append(_row(name, tier, "wall_length", f"wall_rank_{i+1}", g, w["length"], s))
        for r in gt:
            if r["type"] == "ceiling_height":
                rows.append(_row(name, tier, "ceiling_height", "ceiling", float(r["dimension_m"]), geom["ceiling_height"], src(r)))
            elif r["type"] == "floor_area":
                rows.append(_row(name, tier, "floor_area", "floor_area", float(r["dimension_m"]), geom["floor_area"], src(r)))
        pool = list(geom.get("openings", []))
        for r in (x for x in gt if x["type"] == "opening_width"):
            kind = r["feature"].split("_")[0]
            cands = [o for o in pool if o["type"] == kind]
            if not cands:
                ev["missed"] += 1
                continue
            best = min(cands, key=lambda o: abs(o["width"]["value"] - float(r["dimension_m"])))
            pool.remove(best)
            rows.append(_row(name, tier, "opening_width", r["feature"], float(r["dimension_m"]), best["width"], src(r)))
        if str(case.get("openings_complete", "0")) == "1":
            ev["phantom"] += sum(1 for o in pool if o["type"] in ("door", "window"))
    sp = plan.get("stitched_plan")
    for r in gt_all:
        if r["type"] == "footprint_area" and sp:
            rows.append(_row(name, tier, "footprint_area", "footprint", float(r["dimension_m"]), sp["total_footprint_area"], src(r)))
    ev["overlaps"] = sum("overlap" in w.lower() for w in plan.get("warnings", []))
    return rows, ev


def evaluate_repeat(pair, root):
    a, b = _plan(pair["plan_a"], root), _plan(pair["plan_b"], root)
    ga, gb = a["room_geometry"].get(pair["room_a"]), b["room_geometry"].get(pair["room_b"])
    items = []
    if ga and gb:
        wa = sorted(w["length"]["value"] for w in ga["walls"])
        wb = sorted(w["length"]["value"] for w in gb["walls"])
        for x, y in zip(wa, wb):
            tol = max(THRESHOLDS["G3"]["abs_m"], THRESHOLDS["G3"]["rel"] * x)
            items.append("hit" if abs(x - y) <= tol else "miss")
        ca, cb = ga["ceiling_height"], gb["ceiling_height"]
        if ca["method"] == "measured" and cb["method"] == "measured":
            items.append("hit" if abs(ca["value"] - cb["value"]) <= 0.01 else "miss")
    return items


def _evidence(rows):
    s = {r["source"] for r in rows}
    if not s:
        return "-"
    if s <= TRUSTED:
        return "MEASURED"
    if s <= {"synthetic"}:
        return "SYNTHETIC"
    return "INDICATIVE"


def _gate(gid, items, req, evidence, extra=""):
    d = GATES_CONFIG[gid]["name"]
    n = len(items)
    if n == 0:
        return dict(gate=gid, name=d, status="NOT MEASURED", evidence="-", detail="no reference data", n=0)
    hit, unc, miss = items.count("hit"), items.count("unclear"), items.count("miss")
    if hit / n >= req:
        st = "PASS"
    elif (hit + unc) / n < req:
        st = "FAIL"
    else:
        st = "INCONCLUSIVE"
    return dict(gate=gid, name=d, status=st, evidence=evidence,
                detail=f"{hit} hit / {unc} unclear / {miss} miss of {n}{extra}", n=n)


def _g8(rows):
    per, widths = {}, {}
    for t in ("lidar", "video", "photo"):
        rs = [r for r in rows if r["tier"] == t and r["sigma"] <= ((r["hi"] - r["lo"]) / 2) / 2]
        per[t] = (len(rs), (sum(r["covered"] for r in rs) / len(rs)) if rs else None)
        ws = [(r["hi"] - r["lo"]) / r["pred"] for r in rows if r["tier"] == t and r["type"] == "wall_length" and r["pred"]]
        widths[t] = sum(ws) / len(ws) if ws else None
    ok_tiers = {t: v for t, v in per.items() if v[0] >= THRESHOLDS["G8"]["min_n"]}
    det = "; ".join(f"{t}: n={n} cov={'-' if c is None else f'{c:.0%}'}" for t, (n, c) in per.items())
    if not ok_tiers:
        return dict(gate="G8", name=GATES_CONFIG["G8"]["name"], status="NOT MEASURED", evidence="-",
                    detail=f"too few precise reference points ({det})", n=0)
    inside = all(THRESHOLDS["G8"]["lo"] <= c <= THRESHOLDS["G8"]["hi"] for _, c in ok_tiers.values())
    return dict(gate="G8", name=GATES_CONFIG["G8"]["name"], status="PASS" if inside else "FAIL",
                evidence=_evidence([r for r in rows if r["tier"] in ok_tiers]), detail=det,
                n=sum(n for n, _ in ok_tiers.values()))


def _from_json(gid, path, root, label):
    p = Path(root) / path
    if not p.exists():
        return dict(gate=gid, name=GATES_CONFIG[gid]["name"], status="NOT MEASURED", evidence="-",
                    detail=f"{path} not found", n=0)
    d = json.loads(p.read_text(encoding="utf-8"))
    return dict(gate=gid, name=GATES_CONFIG[gid]["name"], status=d["status"], evidence=label,
                detail=d["detail"], n=d.get("n", 1))


def run_harness(manifest, repeat=None, root=ROOT):
    rows, events = [], {"missed": 0, "phantom": 0, "overlaps": 0}
    mp = Path(root) / manifest
    for case in (_csv(mp) if mp.exists() else []):
        r, ev = evaluate_case(case, root)
        rows += r
        for k in events:
            events[k] += ev[k]
    rep_items = []
    rp = Path(root) / repeat if repeat else None
    for pair in (_csv(rp) if rp and rp.exists() else []):
        rep_items += evaluate_repeat(pair, root)

    T = THRESHOLDS
    sel = lambda rt, tier: [r for r in rows if r["type"] == rt and r["tier"] == tier]
    g1r = sel("opening_width", "lidar")
    g1 = [classify(r["err"], T["G1"]["abs_m"], r["sigma"]) for r in g1r] + ["miss"] * (events["missed"] + events["phantom"])
    g2r = sel("ceiling_height", "lidar")
    g2 = ["miss" if r["method"] != "measured" else classify(r["err"], T["G2"]["abs_m"], r["sigma"]) for r in g2r]
    g5r = sel("footprint_area", "photo")
    g5 = [classify(r["err"], T["G5"]["rel"] * r["gt"], r["sigma"]) for r in g5r]
    if g5r and events["overlaps"]:
        g5.append("miss")
    g6r, g7r = sel("wall_length", "photo"), sel("wall_length", "video")
    g6 = [classify(r["err"], T["G6"]["rel"] * r["gt"], r["sigma"]) for r in g6r]
    g7 = [classify(r["err"], T["G7"]["rel"] * r["gt"], r["sigma"]) for r in g7r]

    gates = [
        _gate("G1", g1, T["G1"]["min_frac"], _evidence(g1r), f" (missed={events['missed']}, phantom={events['phantom']})"),
        _gate("G2", g2, 1.0, _evidence(g2r), " (a prior counts as a miss)"),
        _gate("G3", rep_items, 1.0, "SELF-CONSISTENCY"),
        _from_json("G4", "out/ablation_drift.json", root, "MEASURED (internal, no truth)"),
        _gate("G5", g5, 1.0, _evidence(g5r), f" (overlap warnings={events['overlaps']})"),
        _gate("G6", g6, 1.0, _evidence(g6r)),
        _gate("G7", g7, 1.0, _evidence(g7r)),
        _g8(rows),
        dict(gate="G9", name=GATES_CONFIG["G9"]["name"], status="NOT MEASURED", evidence="-",
             detail="no consumer-app export available", n=0),
        _from_json("G10", "fixloop/verify_report.json", root, "MEASURED (git + runs)"),
    ]
    return dict(gates=gates, rows=rows, events=events)


def _print(res):
    print("=" * 100)
    print(f"{'Gate':<5}| {'Name':<28}| {'Status':<13}| {'Evidence':<24}| Detail")
    print("-" * 100)
    for g in res["gates"]:
        print(f"{g['gate']:<5}| {g['name'][:27]:<28}| {g['status']:<13}| {g['evidence']:<24}| {g['detail']}")
    print("=" * 100)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Data/ground_truth/manifest.csv")
    ap.add_argument("--repeat", default="Data/ground_truth/repeat_pairs.csv")
    ap.add_argument("--out", default="out/bench_results.json")
    a = ap.parse_args(argv)
    res = run_harness(a.manifest, a.repeat)
    _print(res)
    op = ROOT / a.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    return 1 if any(g["status"] == "FAIL" for g in res["gates"]) else 0


if __name__ == "__main__":
    sys.exit(main())
```

`Makefile` target `fixloop` should call `bench/verify_fixloop.py`, not `harness.py --fixloop`.

Test `tests/unit/test_harness_synthetic.py`:

```python
import csv, json
from pathlib import Path
from bench.harness import run_harness


def _iv(v, tier="photo"):
    return {"value": v, "lo": v * 0.95, "hi": v * 1.05, "confidence_level": 0.9, "method": "tier_prior", "tier": tier}


def test_exact_synthetic_room_gives_pass_and_unmeasured(tmp_path):
    walls = [{"wall_id": f"r_w{i}", "length": _iv(v)} for i, v in enumerate([4.0, 3.0, 4.0, 3.0])]
    plan = {"tier": "photo", "warnings": [], "room_geometry": {"r": {
        "walls": walls, "ceiling_height": _iv(2.6), "floor_area": _iv(12.0), "openings": []}}}
    (tmp_path / "p.json").write_text(json.dumps(plan))
    with open(tmp_path / "gt.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "feature", "dimension_m", "type", "source"])
        for i, v in enumerate([4.0, 3.0, 4.0, 3.0]):
            w.writerow(["r", f"wall_{i+1}", v, "wall_length", "synthetic"])
    with open(tmp_path / "m.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "plan_json", "gt_csv", "room_id", "openings_complete"])
        w.writerow(["c", str(tmp_path / "p.json"), str(tmp_path / "gt.csv"), "r", "0"])
    res = run_harness(str(tmp_path / "m.csv"), None, root=tmp_path)
    g = {x["gate"]: x for x in res["gates"]}
    assert g["G6"]["status"] == "PASS" and g["G6"]["evidence"] == "SYNTHETIC"
    assert g["G1"]["status"] == "NOT MEASURED"
    assert g["G9"]["status"] == "NOT MEASURED"
```

A second test with a 20% wall error and `source=synthetic` must give `G6 == FAIL`.

### 6.3 `bench/ablation_drift.py` (real)

```python
"""G4: run the same capture with drift correction ON and OFF and report what changes.
No ground truth is assumed: the script reports differences, not accuracy, unless --gt-footprint is given."""
import argparse, json, subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(capture, out, enabled, tier):
    env = dict(os.environ, DRIFT_CORRECTION="1" if enabled else "0")
    cmd = [sys.executable, "main.py", capture, "--out", out, "--no-llm", "--no-local-models"]
    if tier != "auto":
        cmd += ["--tier", tier]
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    return json.loads((ROOT / out / "plan.json").read_text(encoding="utf-8"))


def _centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--tier", default="auto")
    ap.add_argument("--gt-footprint", type=float, default=None)
    a = ap.parse_args()
    on = _run(a.capture, "out/ablation_on", True, a.tier)
    off = _run(a.capture, "out/ablation_off", False, a.tier)

    rooms = list(on["room_geometry"])
    fp_on = on["stitched_plan"]["total_footprint_area"]["value"]
    fp_off = off["stitched_plan"]["total_footprint_area"]["value"]
    shifts = []
    for r in rooms:
        if r in off["room_geometry"]:
            c1, c2 = _centroid(on["room_geometry"][r]["floor_polygon"]), _centroid(off["room_geometry"][r]["floor_polygon"])
            shifts.append(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5)
    ov = lambda p: sum("overlap" in w.lower() for w in p.get("warnings", []))
    max_shift = max(shifts) if shifts else 0.0
    changed = max_shift > 0.01 or abs(fp_on - fp_off) > 0.01

    detail = (f"rooms={len(rooms)}, footprint ON={fp_on:.2f} OFF={fp_off:.2f} m2, "
              f"max room shift={max_shift:.2f} m, overlaps ON={ov(on)} OFF={ov(off)}")
    if len(rooms) <= 1:
        detail += "; single room: nothing to correct"
    elif not changed:
        detail += "; correction had no effect (shared SfM frame or identity constraints)"
    if a.gt_footprint:
        detail += (f"; error vs reference ON={abs(fp_on-a.gt_footprint)/a.gt_footprint:.1%} "
                   f"OFF={abs(fp_off-a.gt_footprint)/a.gt_footprint:.1%} (reference is an estimate)")
    else:
        detail += "; no ground truth: differences reported, accuracy not claimed"
    out = dict(status="PASS" if len(rooms) > 1 else "INCONCLUSIVE", detail=detail, n=1)
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/ablation_drift.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(detail)


if __name__ == "__main__":
    main()
```

Run: `python bench\ablation_drift.py Data\RealHouse --tier photo` and, if it has several rooms, on `Data\1BHKRoom\1bhKRoom.mp4`. If the video run uses a shared SfM frame, the report will say the correction had no effect. Keep that sentence. It is the honest answer and the report must say so.

### 6.4 `bench/headtohead.py`

Replace the body with a function that prints `G9 NOT MEASURED: no consumer-app export in Data/app_exports/` and exits 0. Keep a docstring describing the expected CSV (`case,feature,type,app_value`) so it can be wired in if an export ever appears. Remove `SHARED_DIMENSIONS`.

### 6.5 `bench/calibration_report.py`

Delete the hardcoded dict. Make it load `out/bench_results.json` and print per-tier `n`, coverage and mean relative wall width, plus the line "calibration is not statistically meaningful below n=10 precise reference points per tier". It can reuse `_g8` from the harness.

### 6.6 Reference data (30-45 min, can be done in parallel)

**File layout**

```
Data/ground_truth/manifest.csv        # which plan.json is compared with which csv
Data/ground_truth/repeat_pairs.csv    # G3 pairs (two runs of one room)
Data/ground_truth/<case>.csv          # reference values
```

CSV columns: `room_id,feature,dimension_m,type,source`. Allowed types: `wall_length`, `ceiling_height`, `floor_area`, `opening_width`, `footprint_area`. Allowed sources: `tape`, `laser`, `tile`, `lidar_ref`, `estimate`, `synthetic`. Rows with an empty `dimension_m` are skipped. Wall order does not matter (the harness matches sorted lengths).

`manifest.csv`:

```csv
case_id,plan_json,gt_csv,room_id,openings_complete
house_own_photo,out/OwnHouse/plan.json,Data/ground_truth/own_house.csv,room_00,0
house1_photo,out/HOUSE1/plan.json,Data/ground_truth/house1.csv,room_00,0
realhouse_photo,out/RealHouse/plan.json,Data/ground_truth/realhouse.csv,room_00,0
bhk_video,out/1BHKRoom/plan.json,Data/ground_truth/bhk.csv,room_00,0
singleroom_video_vs_lidar,out/SingleRoom_video/plan.json,Data/ground_truth/singleroom_lidar_ref.csv,room_00,0
```

`repeat_pairs.csv` (only if the two LiDAR scans are the same room; check that the floor areas are within about 10% before using it):

```csv
plan_a,plan_b,room_a,room_b
out/single_scan_floor_only/plan.json,out/single_scan_with_ceiling/plan.json,room_00,room_00
```

**How to get reference values without a tape (best first)**

1. **Floor tiles:** count full tiles along a wall plus a fraction for the cut tile, times the tile size. Use `source=tile`. Standard sizes are 600 mm, 600x1200 mm and 2 ft, but read the real size off the box or a spare tile.
2. **LiDAR as reference for the video tier:** `bench/make_lidar_ref.py` (below) turns `out/lidar/plan.json` into `singleroom_lidar_ref.csv` with `source=lidar_ref`. Then run the video tier on the same room (`python main.py Data\SingleRoom\rgb.mp4 --tier video --out out\SingleRoom_video`). Both runs come from our own pipeline, so call the result "consistency between tiers", not accuracy.
3. **Rough estimates** (house, 1BHK, RealHouse): `source=estimate`. The harness will treat these as 5% uncertain, so G7 (3%) and G1/G2 will come out `INCONCLUSIVE`. That is correct and is what you report.

`bench/make_lidar_ref.py`:

```python
import csv, json, sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
rows = []
for rid, g in plan["room_geometry"].items():
    for i, w in enumerate(sorted(w["length"]["value"] for w in g["walls"])):
        rows.append([rid, f"wall_{i+1}", round(w, 3), "wall_length", "lidar_ref"])
    rows.append([rid, "floor_area", round(g["floor_area"]["value"], 3), "floor_area", "lidar_ref"])
    if g["ceiling_height"]["method"] == "measured":
        rows.append([rid, "ceiling_height", round(g["ceiling_height"]["value"], 3), "ceiling_height", "lidar_ref"])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["room_id", "feature", "dimension_m", "type", "source"])
    w.writerows(rows)
```

Run: `python bench\make_lidar_ref.py out\SingleRoom\plan.json Data\ground_truth\singleroom_lidar_ref.csv` (regenerate `out\SingleRoom` first with the fixed code).

**Walk-in tip (use it in the protocol):** for the video and photo tiers, ask for one known dimension in the unseen space (door height or ceiling height) and pass `--reference-height <m>`. `main.py` already supports it.

### 6.7 `bench/fixloop_run.py` and `bench/verify_fixloop.py` (also needed for P0/P1)

`bench/fixloop_run.py`:

```python
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = {"floor_only": "Data/single_scan_floor_only", "with_ceiling": "Data/single_scan_with_ceiling"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["before", "after"])
    a = ap.parse_args()
    summary = {}
    for name, cap in CASES.items():
        out = f"fixloop/{a.label}/{name}"
        subprocess.run([sys.executable, "main.py", cap, "--tier", "lidar", "--out", out,
                        "--no-llm", "--no-local-models"], cwd=ROOT, check=False)
        plan = json.loads((ROOT / out / "plan.json").read_text(encoding="utf-8"))
        g = next(iter(plan["room_geometry"].values()))
        c = g["ceiling_height"]
        summary[name] = dict(value=c["value"], lo=c["lo"], hi=c["hi"], width=round(c["hi"] - c["lo"], 4),
                             method=c["method"], warned=any("ceiling" in w.lower() and "prior" in w.lower()
                                                            for w in plan.get("warnings", [])))
    p = ROOT / f"fixloop/{a.label}/summary.json"
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

`bench/verify_fixloop.py`:

```python
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def first(marker, path=None):
    args = ["log", "--reverse", "--format=%H", f"-S{marker}"] + (["--", path] if path else [])
    out = git(*args).splitlines()
    return out[0] if out else None


decl = first("CEILING-CLAMP-DECLARATION", "fixloop/declaration.md")
fix = first("CEILING_PRIOR_M", "src/areamap/geometry/planes.py")
order_ok = False
if decl and fix:
    order_ok = subprocess.run(["git", "merge-base", "--is-ancestor", decl, fix], cwd=ROOT).returncode == 0 and decl != fix

pred = json.loads((ROOT / "fixloop/prediction.json").read_text())
before = json.loads((ROOT / "fixloop/before/summary.json").read_text())
after = json.loads((ROOT / "fixloop/after/summary.json").read_text())
checks = []
fo, wc = after["floor_only"], after["with_ceiling"]
checks.append(("floor_only prior", fo["method"] == pred["after"]["floor_only"]["method"]))
checks.append(("floor_only width", fo["width"] >= pred["after"]["floor_only"]["min_width_m"]))
checks.append(("with_ceiling method", wc["method"] == pred["after"]["with_ceiling"]["method"]))
checks.append(("with_ceiling width", wc["width"] <= pred["after"]["with_ceiling"]["max_width_m"]))
checks.append(("before had conformal 2.4", all(v["method"] == "conformal" and abs(v["value"] - 2.4) < 1e-6 for v in before.values())))
diff_nonempty = (ROOT / "fixloop/diff.patch").exists() and (ROOT / "fixloop/diff.patch").stat().st_size > 0
ok = order_ok and diff_nonempty
detail = (f"declaration {str(decl)[:7]} precedes fix {str(fix)[:7]}: {order_ok}; diff.patch non-empty: {diff_nonempty}; "
          + ", ".join(f"{n}={'ok' if v else 'MISSED'}" for n, v in checks))
rep = dict(status="PASS" if ok else "FAIL", detail=detail, n=len(checks) + 2, order_ok=order_ok, checks=dict(checks))
(ROOT / "fixloop/verify_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
print(detail)
```

Missed predictions do not fail G10 (the gate is about the process); they are printed and must be explained in `postmortem.md`.

### 6.8 Commits

```
C8  bench: real harness, thresholds-only gates, drift ablation, no hardcoded results
C9  data: reference CSVs with source column, manifest, repeat pairs
```

**Acceptance P3:** `python bench\harness.py` run twice gives byte-identical `out\bench_results.json`; no gate prints PASS with `evidence=-`; `G9` prints `NOT MEASURED`.

---

## 7. P4: Verification suite (14:40-15:25)

Record every result (including failures) in `reports/verification_log.md`: command, date, result, anything surprising.

| # | Check | Command | Pass criterion |
|---|---|---|---|
| V1 | Unit + new tests | `python -m pytest -q` | All green (116 + new tests) |
| V2 | Regenerate every output | `python main.py <capture> ...` for SingleRoom, single_scan_floor_only, single_scan_with_ceiling, 1BHKRoom video, HOUSE1, RealHouse | Each writes `plan.json` and `plan.svg`; none contains "Offline heuristic detection" |
| V3 | Determinism | `python scripts\verify_determinism.py Data\SingleRoom` (and photo, video) | Canonical hashes equal. If video/COLMAP or the depth model differs between runs, log it as "not deterministic" with the cause; do not claim otherwise |
| V4 | Offline | `python scripts\run_no_network.py Data\HOUSE1 --no-local-models` and the video | Exits 0, no network exception in output |
| V5 | Failure injection | `python scripts\make_corrupt_fixtures.py` then run each fixture | Exit code 0 or 2, `plan.json` written, `NODE_FAILED` or ingest warnings present, no traceback |
| V6 | QA regression | `pytest tests\unit\test_qa_critic_checks.py` | Bad surface id, 0.3 m wall gap, 20% area error all flagged |
| V7 | Gate run | `python bench\harness.py` twice, `fc`/`Get-FileHash` the JSON | Identical; honest statuses |
| V8 | Fix loop | `python bench\verify_fixloop.py` | `order_ok: true` |
| V9 | Cold run | Fresh clone, new venv, `pip install -r requirements.txt`, one command per tier | Time each step; targets: setup < 15 min, LiDAR ~30 s, video ~90 s, photo ~45 s on CPU (record actuals, do not claim targets) |

### Scripts

`scripts/verify_determinism.py`:

```python
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def canon(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    for k in ("timings", "output_dir"):
        d.pop(k, None)
    return json.dumps(d, sort_keys=True, indent=1)


def main():
    capture = sys.argv[1]
    extra = sys.argv[2:]
    hs = []
    for i in (1, 2):
        out = f"out/det_run{i}"
        subprocess.run([sys.executable, "main.py", capture, "--out", out, "--no-llm", "--no-local-models", *extra],
                       cwd=ROOT, check=False)
        hs.append(hashlib.sha256(canon(ROOT / out / "plan.json").encode()).hexdigest())
    print(hs)
    print("DETERMINISTIC" if hs[0] == hs[1] else "NOT DETERMINISTIC (see diff of out/det_run1 vs out/det_run2)")
    sys.exit(0 if hs[0] == hs[1] else 1)


if __name__ == "__main__":
    main()
```

`scripts/run_no_network.py`:

```python
import os, runpy, socket, sys

os.environ["OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def _blocked(*a, **k):
    raise OSError("NETWORK BLOCKED by run_no_network.py")


socket.socket.connect = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
```

`scripts/make_corrupt_fixtures.py`:

```python
import csv, shutil
from pathlib import Path
import cv2
import numpy as np

OUT = Path("Data/test_corrupt")
OUT.mkdir(parents=True, exist_ok=True)

# 1) one blurred photo
src = next(Path("Data/RealHouse").glob("*.png"))
img = cv2.imread(str(src))
(OUT / "photo_one_blurred").mkdir(exist_ok=True)
cv2.imwrite(str(OUT / "photo_one_blurred/a.jpg"), cv2.GaussianBlur(img, (51, 51), 0))

# 2) all-black video (3 s, 30 fps)
vw = cv2.VideoWriter(str(OUT / "black.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
for _ in range(90):
    vw.write(np.zeros((480, 640, 3), np.uint8))
vw.release()

# 3) LiDAR bundle (first 5 frames only) with NaN odometry rows
srcdir, dst = Path("Data/single_scan_floor_only"), OUT / "lidar_nan"
dst.mkdir(exist_ok=True)
for f in srcdir.iterdir():
    if f.is_file():
        shutil.copy(f, dst / f.name)
for sub in ("depth", "confidence"):
    if (srcdir / sub).is_dir():
        (dst / sub).mkdir(exist_ok=True)
        for f in sorted((srcdir / sub).iterdir())[:5]:
            shutil.copy(f, dst / sub / f.name)
odo = dst / "odometry.csv"
rows = list(csv.reader(open(odo, newline="")))
for r in rows[2:5]:
    for j in range(len(r)):
        try:
            float(r[j])
            r[j] = "nan"
        except ValueError:
            pass
csv.writer(open(odo, "w", newline="")).writerows(rows)
print("fixtures written to", OUT)
```

Expected behaviours: the blurred single photo gives a result with warnings and wide intervals (or a clean `NODE_FAILED ingest` partial plan); the black video gives `NODE_FAILED` with "Reconstruction failed", exit 2, and a `plan.json`; the NaN odometry gives either dropped poses with a warning or a `NODE_FAILED` partial plan. If any of these produces a traceback, that is a bug to fix before the defense.

---

## 8. P5: Documents (15:25-16:00)

### 8.1 `compliance_matrix.md`

Replace all "Scaffolded", "In Progress" and "Complete" with these statuses only: `Verified (tests)`, `Implemented, not measured`, `Indicative`, `Not measured`, `Stub / removed`. Fill every row's artifact column with a path that exists.

| Row | Status to write |
|---|---|
| M0 | Verified (tests) |
| M1, M3, M4, M5, M11, M12, M13, M14, M15 | Verified (tests) (M13 only after WP-A3; M15 partial-result only after WP-A4) |
| M2 | Verified (tests) via `test_harness_synthetic.py`; results depend on references |
| M6, M7, M8, M9 | Implemented, not measured (no tape) |
| M10 | Implemented, not measured: extent is a VLM estimate (`vlm_estimate`), no SAM masks, damage needs network or a warmed cache |
| M16 | Stub / removed: no consumer-app export |
| M17 | Verified (git + runs) via `verify_fixloop.py` |
| M18 | Implemented, not measured (no outside tester) |
| M19 | Verified once the report is regenerated from `bench_results.json` |
| G1, G2 | Not measured (no tape/laser; LiDAR cannot be its own reference) |
| G3 | Indicative (self-consistency of two LiDAR scans) |
| G4 | Measured (internal, no truth) |
| G5, G6, G7, G8 | Indicative or Inconclusive, copied from the harness output |
| G9 | Not measured |
| G10 | Measured (git + runs) |

### 8.2 `reports/technical_report.md`: edits

- Section 4: delete the ablation table (64.50 / 68.20 / 64.95). Paste the single line from `out/ablation_drift.json` and the sentence about whether correction changed anything.
- Section 5: delete the 91.2/88.7/87.4 coverages. State that intervals are tier-based bands plus noise floors (`tier_prior`) and that empirical coverage is `NOT MEASURED` unless `calibration_report.py` shows n >= 10 precise references.
- Sections 6, 7, 8: replace the gate table with `reports/gate_table.md`, generated by `scripts/make_report_tables.py` from `out/bench_results.json` (code below). Delete section 7 (head-to-head) and replace it with "G9 not measured: no consumer-app export was available." Rewrite section 8 as the real fix loop (ceiling) using `fixloop/postmortem.md`.
- Section 9 (disclosure table): update to what is actually in the code (Depth Anything via the depth engine or the geometric fallback, CLIP room classifier, Mistral/Groq/Ollama/OpenRouter/Gemini VLM providers, COLMAP/pycolmap, LangGraph). Qwen2-VL appears in the old table but the config does not use it; remove it unless you really use it.
- Add a "Limits of evidence" section: no tape or laser, references are tile counts/LiDAR-derived/estimates, which gates are `NOT MEASURED` and why.
- Section 10: add "ceiling is a prior when unobserved and always for photo tier", "wall damage is not localized to a specific wall from a single image", "damage needs a VLM provider or warmed cache".

`scripts/make_report_tables.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "out/bench_results.json").read_text(encoding="utf-8"))
lines = ["| Gate | Name | Status | Evidence | Detail |", "|---|---|---|---|---|"]
for g in res["gates"]:
    lines.append(f"| {g['gate']} | {g['name']} | {g['status']} | {g['evidence']} | {g['detail']} |")
(ROOT / "reports/gate_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote reports/gate_table.md")
```

### 8.3 Other documents

- `reports/device_matrix.md`: replace the "Measured Benchmark Performance" column with "See reports/gate_table.md (reference quality noted)". Remove 0.9 cm, 1.2 cm, 92.3%, 1.8%, 5.1%, 4.8%.
- `protocol/capture_protocol.md`: add a "Walk-in" box: ask for one known dimension and run with `--reference-height`; one sentence that damage inspection needs internet or a warmed cache. Remove the claim that an outside person has tested it unless that happens.
- `README.md`: add a short "Current status" section that points to `reports/gate_table.md`. Do not leave the 19-module "Fully completed" summary from the first audit file anywhere in the repo.
- `fixloop/before/README.md` and `after/README.md` placeholders: replace with the generated summaries.

### 8.4 Commit

```
C11 docs: evidence-labelled compliance matrix, report, device matrix, protocol, README status
```

---

## 9. Final hour (16:00-16:25)

```powershell
python -m pytest -q
python bench\harness.py ; python bench\harness.py       # compare out\bench_results.json hashes
python scripts\make_report_tables.py
python bench\verify_fixloop.py
git status                                              # clean
git log --oneline | Select-Object -First 15
Compress-Archive -Path . -DestinationPath handoff.zip   # exclude venv, Data\*\depth, out\det_run*, .env
```

Before zipping, make sure `.env` and API keys are not included. `git log` should show: C1 declaration, then C2 fix, then C3 after-run, then the rest.

---

## 10. Definition of done

- [ ] `git log` shows the declaration commit before the ceiling fix commit; `fixloop/verify_report.json` has `order_ok: true`; `diff.patch` is a real diff
- [ ] `floor_only` reports a ceiling `prior` with a >= 0.79 m band and a warning; `with_ceiling` reports `measured` (or the post-mortem explains why not)
- [ ] `out/HOUSE1/plan.json`: `qa_report.passed == true`, no `room_01_w1`, no whole-image box
- [ ] No output file contains "Offline heuristic detection"
- [ ] `bench/` contains no hardcoded results; `EVALUATED_GATES` is gone; `G9` is `NOT MEASURED`
- [ ] `python bench\harness.py` twice gives identical JSON; every PASS/FAIL line has an evidence label
- [ ] `pytest -q` is green and includes the new ceiling, damage, QA, partial-result and harness tests
- [ ] A missing/corrupt input produces a `plan.json` with `NODE_FAILED` warnings and exit code 2, never a bare traceback
- [ ] `reports/verification_log.md` lists V1-V9 with real results, including any that failed
- [ ] `compliance_matrix.md` uses only the five statuses; the report, README and device matrix contain none of the old hardcoded figures
- [ ] Offline run (V4) completes with the network blocked

---

## 11. What this plan cannot give you

- **G1 (opening widths) and G2 (ceiling +/-1.5 cm):** these need tape or laser. With the data you have they stay `NOT MEASURED`. Do not turn them into PASS.
- **G9:** no consumer-app export, so `NOT MEASURED`.
- **G7 (+/-3%) and G1/G2 from rough estimates:** the harness will say `INCONCLUSIVE` because a 5% estimate cannot resolve a 3% gate. That is the correct outcome, and the report should explain why.
- **Accuracy:** nothing here makes the geometry more accurate. It fixes provenance, calibration labels, the damage and QA defects, and the evidence trail. If a computed number misses a gate, report it as it is.

## 12. Risks and the cut order if something slips

| Risk | Response |
|---|---|
| Old tests assert `"conformal"` or the 2.4 clamp | Update the tests; if WP-E breaks more than about five, do only the wording change in the report |
| A COLMAP/depth run is nondeterministic | Log V3 as "not deterministic", name the cause, do not hide it |
| No VLM key or no internet for cache warming | Damage will be `unavailable`; say so in the report; keep the damage code fixes and tests (they use a fake client) |
| Tile counts not collected in time | Use `estimate` sources; the harness already handles that |
| Time runs out in P5 | Do compliance matrix and the gate table first, then report section 8 (fix loop), then the rest |

Never cut: the declaration commit, the ceiling fix, the real harness, deleting hardcoded numbers, the crash-safe pipeline, and the verification log.
