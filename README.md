# AreaMap: iPhone Capture to Dimensioned Floor Plan, Damage Findings and Repair Scope

Working name. Pipeline that turns an iPhone capture of a property (photos, video or LiDAR) into a stitched, dimensioned whole-property floor plan, per-surface damage regions, concealed-damage flags, scope line items, and a confidence interval on every measurement.

> **Design rule:** anything that produces a number is deterministic Python. LLMs/VLMs only do semantics (what is the damage, what is the repair item) and QA review, are run at temperature 0, and are cached by input hash.

---

## 0. Assumptions (we cannot confirm with the organizers)

| # | Assumption | Consequence |
|---|---|---|
| A1 | Walk-in machine may be **CPU-only** | Every model needs a CPU path; benchmark timing on CPU |
| A2 | Internet may be **unavailable** at the defense | All weights pre-downloaded; LLM nodes have a local fallback and a replay cache |
| A3 | The "published schema" from Round 1 is **not available to us** | We define `schema/capture_v1.json` ourselves and document it as our interpretation |
| A4 | Capture route = **Route 2 (stock app + one-page protocol)** | No Swift dependency; LiDAR ingest depends on the chosen app's export format (verify on Day 1) |
| A5 | Exactly one Pro iPhone with LiDAR is needed for the LiDAR tier | If none is available, see Risks (section 12) |
| A6 | Walk-in test gives us an iPhone 15 or newer, tier chosen on the day | All three tiers must run cold with one command |

---

## 1. Problem in one paragraph

Per capture, produce: dimensioned per-room plan (walls, ceiling height, floor area, openings), stitched multi-room plan with correct adjacency, per-surface damage regions (class + metric extent), concealed-damage flags with the rule that fired, scope line items keyed to surfaces, a confidence interval on every measurement, JSON to a schema, and a rendered plan. Three input tiers, same output contract, intervals widen as data thins.

| Tier | Input | Device | Accuracy gate |
|---|---|---|---|
| Photos | 2 to 8 stills per room, one folder per room, no depth, no poses | iPhone 15+ | wall lengths and footprint within +/-8%, calibrated intervals |
| Video | One handheld walkthrough clip | iPhone 15+ | +/-3% |
| LiDAR | Depth, poses, intrinsics | Pro-class iPhone | opening widths <= 2 cm on >= 85%, ceiling <= 1.5 cm |

### Official gates (copied from the case study)

| Gate | Threshold |
|---|---|
| G1 Opening widths | <= 2 cm on >= 85% of openings; a missed opening and a phantom opening each count as a miss |
| G2 Ceiling height | <= 1.5 cm per room; spread across repeated captures <= 1 cm; report must say whether we are "repeatable but biased" or "unrepeatable" |
| G3 Repeatability | Two captures of same room, same tier, agree within 1 cm or 0.5% per wall |
| G4 Drift accountability | Report states drift handling; ablation shows stitched footprint with it on and off; "poses used as-is" is an automatic fail |
| G5 Photo whole-property stitch | Per-room photo folders produce one stitched plan, correct adjacency, no room overlaps, footprint within +/-8% with calibrated intervals |
| G6 Photo wall lengths | within +/-8% with calibrated intervals |
| G7 Video wall lengths | within +/-3% |
| G8 Calibration | Scored at every tier; confident garbage on thin input caps total score |
| G9 Head-to-head | Beat or tie a consumer app on >= 70% of shared dimensions (2 benchmark rooms, LiDAR tier) |
| G10 Fix loop | Worst gate, root cause, shipped fix, regenerable before/after run, readable diff (25% of score) |

Score weights: walk-in test 30, fix loop 25, verified benchmark 15, compliance matrix 10, head-to-head 10, capture route quality 5, process evidence 5.

---

## 2. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Pin all versions in `pyproject.toml` on Day 1 |
| Orchestration | **LangGraph** (graph of nodes, typed shared state) | Deterministic nodes and LLM nodes in one graph |
| LLM plumbing | **LangChain** (model wrappers, structured output) | Only used in LLM nodes |
| Tool boundary | **MCP** (local stdio server exposing geometry and render tools) + in-process fallback | Fallback is mandatory: if MCP breaks live, we flip one flag |
| State/validation | pydantic v2 | State object and JSON schema generated from models |
| Geometry | numpy, scipy, Open3D, trimesh, shapely | Plane RANSAC, point clouds, polygons for floor plans |
| Pose graph | GTSAM or g2o (via python bindings), fallback scipy least squares | Evaluate install difficulty on Day 1 |
| Video SfM | COLMAP (primary, CPU), VGGT or MASt3R (optional, if CPU time acceptable) | Verify on a real clip by hour 4 |
| Depth (photo/video) | Depth Pro, MoGe-2, Depth Anything V2 metric | Pick winner on our own benchmark photos |
| VLM (damage) | Qwen3-VL 4B/8B (local, Apache 2.0), Gemma 3 4B fallback | Run via transformers or Ollama/llama.cpp |
| Segmentation | SAM 3 (or SAM 2 fallback) | Mask + depth gives metric area |
| Detection (openings, optional) | SAM 3 text prompts, Grounding DINO | Geometry check always has the last word |
| Image/video | OpenCV, Pillow, ffmpeg, exifread | EXIF focal length, frame sampling |
| Rendering | matplotlib or svgwrite to SVG/PNG | Dimension lines, adjacency, intervals |
| Testing | pytest, hypothesis (optional) | Bench harness is its own CLI |
| Packaging | Makefile, `uv` or `pip`, optional Docker | Clean-machine setup in < 15 min |

---

## 3. System architecture

```
 capture folder ──▶ [M1 Ingest/Router]  deterministic
                          │ tier = photo | video | lidar
        ┌─────────────────┼─────────────────┐
   [M8 Photo tier]   [M7 Video tier]   [M3 LiDAR tier]
   depth+layout+     frames+SfM+scale   depth+poses to
   scale priors                          point cloud
        └─────────────────┼─────────────────┘
                   [M4 Room geometry]    planes, walls, ceiling, area
                   [M5 Openings]         doors, windows
                   [M6 Stitcher]         pose graph, loop closure, adjacency
                   [M9 Calibrator]       tier-aware intervals
              ┌───────────┴───────────┐
        [M10 Damage]            [M11 Concealed rules]
        VLM + SAM + depth       deterministic rule engine
              └───────────┬───────────┘
                   [M12 Scope]           LLM, structured output keyed to surfaces
                   [M13 QA critic]       rules first, optional LLM review
                          │ fail: widen intervals / rerun node / flag
                   [M14 Export/Render]   JSON + SVG/PNG
```

| Node | Type | Why |
|---|---|---|
| Ingest, Geometry, Openings, Stitcher, Calibrator, Concealed rules, Export | Deterministic | Numbers, repeatability gate |
| Damage, Scope | LLM/VLM, temp 0, cached | Semantics only |
| QA critic | Rules (LLM review optional) | Must be reproducible |

### Shared state (pydantic)

`CaptureState`: `capture_path`, `tier`, `device_meta`, `rooms[]`, `point_clouds{}`, `room_geometry{}`, `openings{}`, `stitched_plan`, `intervals{}`, `damage[]`, `concealed_flags[]`, `scope_items[]`, `qa_report`, `timings{}`, `warnings[]`.

---

## 4. File structure

```
areamap/
├── README.md
├── compliance_matrix.md          # requirement -> file path -> artifact -> status
├── Makefile                      # make setup | run CAPTURE=... | bench | ablation | fixloop | test
├── pyproject.toml
├── .env.example
├── schema/
│   └── capture_v1.json           # generated from pydantic; our documented interpretation
├── scripts/
│   ├── fetch_models.sh           # downloads all weights into models/
│   ├── run_capture.py            # THE one command per capture
│   └── make_report_tables.py     # regenerate every number in the report
├── models/                       # gitignored; filled by fetch_models.sh
├── src/areamap/
│   ├── graph.py                  # LangGraph wiring
│   ├── state.py                  # typed shared state
│   ├── config.py                 # flags: USE_MCP, USE_LLM, OFFLINE, DEVICE
│   ├── nodes/
│   │   ├── ingest.py
│   │   ├── geometry.py
│   │   ├── openings.py
│   │   ├── stitch.py
│   │   ├── calibrate.py
│   │   ├── damage.py
│   │   ├── concealed.py
│   │   ├── scope.py
│   │   ├── qa_critic.py
│   │   └── export.py
│   ├── tiers/
│   │   ├── lidar.py              # depth + poses + intrinsics to point cloud
│   │   ├── video.py              # frame select, SfM, scale recovery
│   │   └── photo.py              # depth, layout, scale priors
│   ├── geometry/
│   │   ├── planes.py             # RANSAC, wall/floor/ceiling
│   │   ├── openings.py           # cutout analysis
│   │   ├── posegraph.py          # loop closure, drift correction
│   │   ├── adjacency.py          # door-based room connection
│   │   └── uncertainty.py        # error propagation
│   ├── mcp_server/
│   │   ├── server.py             # geometry + render tools over stdio
│   │   └── client.py             # with in-process fallback
│   ├── llm/
│   │   ├── client.py             # provider switch: local | api
│   │   ├── cache.py              # hash-keyed deterministic replay
│   │   └── prompts/              # damage.md, scope.md
│   ├── rules/concealed_rules.yaml  # rule id, trigger, flag text
│   ├── catalog/scope_items.yaml    # damage class -> line items
│   └── render/plan_svg.py
├── bench/
│   ├── harness.py                # computes all gates, prints pass/fail table
│   ├── gates.py                  # thresholds from section 1
│   ├── headtohead.py             # our LiDAR vs consumer app, per dimension
│   ├── ablation_drift.py         # footprint with drift correction on/off
│   ├── calibration_report.py     # interval coverage per tier
│   └── timing.py
├── data/
│   ├── raw/<room>/<tier>/<run>/  # sensor logs, photos, video
│   ├── ground_truth/             # laser/tape CSV per room
│   ├── app_exports/              # competitor exports
│   └── cache/                    # LLM replay cache (committed, small)
├── fixloop/
│   ├── declaration.md            # one page: gate, root cause, fix, predicted number
│   ├── before/                   # regenerable run output
│   ├── after/
│   └── diff.patch
├── protocol/capture_protocol.md  # one page for a non-engineer
├── reports/
│   ├── technical_report.md       # max 6 pages
│   └── device_matrix.md
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/                 # tiny synthetic rooms with known geometry
```

---

## 5. APIs, keys, accounts and downloads we need

The case study allows any pretrained model, dataset or API **with disclosure**, but everything must run without calling our own infrastructure. So: no required cloud dependency; APIs are optional accelerators with local fallbacks.

### Required (free)

| Item | Purpose | Notes |
|---|---|---|
| **Hugging Face account + access token** | Download weights (Qwen3-VL, Depth Anything, MoGe-2, SAM, etc.) | Some models are gated (accept license on model page). Set `HF_TOKEN` |
| **Model weights on disk** | Offline operation | `scripts/fetch_models.sh`; verify checksums; total size budget decided Day 1 |
| **Stock capture app(s)** | Capture route | Candidates: Record3D, 3D Scanner App, Polycam, SiteScape. **Verify which exports depth + poses + intrinsics on our device** |
| **Consumer scanning app (free tier)** | Head-to-head | Polycam or magicplan; record app name and version; export must be saved to `data/app_exports/` |
| **Laser measurer / tape** | Ground truth | Not an API, but a hard requirement |
| **A LiDAR iPhone (Pro)** | LiDAR tier | See A5 |
| **GitHub repo** | Process evidence | Commit as we work |

### Optional (each has a local fallback)

| Item | Used by | Fallback if unavailable |
|---|---|---|
| Hosted LLM/VLM API key (e.g. Anthropic, Google Gemini free tier, OpenAI) | Damage, Scope nodes (better quality) | Local Qwen3-VL via transformers/Ollama; replay cache |
| Ollama or llama.cpp runtime | Local VLM/LLM serving | transformers directly |
| LangSmith account | Tracing/debugging graph runs | Python logging + JSON run logs |
| Apple Developer account | Only if Route 1 (own iOS app) | Not needed for Route 2 |
| Roboflow/other hosted inference | Not needed | n/a |

### Environment variables (`.env.example`)

```
HF_TOKEN=
LLM_PROVIDER=local        # local | anthropic | gemini | openai
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
OPENAI_API_KEY=
OFFLINE=1                 # 1 = never call external APIs, use cache + local models
USE_MCP=1                 # 0 = in-process tool calls
DEVICE=auto               # auto | cpu | cuda | mps
LLM_CACHE_DIR=data/cache
```

### Disclosure table (goes in the technical report)

Every model/API used: name, version, license, purpose, local or hosted, which tier uses it.

---

## 6. Modules, tasks and acceptance criteria

Legend: **Owner** A = capture/benchmark data, B = geometry/stitching, C = photo/video tiers, D = orchestration/LLM/export. (If fewer than four people, merge roles; see section 11.)

### M0 Contracts and schema (Owner: all, Day 1 hours 0 to 1)

Tasks
- [ ] Define pydantic models for state and output; generate `schema/capture_v1.json`
- [ ] Fix coordinate conventions (meters, Y-up or Z-up, right-handed), room-local vs plan frame
- [ ] Define interval format: `{value, lo, hi, confidence_level, method, tier}`
- [ ] Repo skeleton and first commits

Acceptance
- [ ] A hand-written sample output validates against the schema
- [ ] Every measurement field requires an interval (schema rejects missing intervals)
- [ ] Conventions documented in `docs` section of this README

### M1 Ingest and router (Owner: D)

Tasks
- [ ] Detect tier from folder layout (photo folders / video file / depth+poses bundle)
- [ ] Validate files, read EXIF (focal length, 35mm equivalent, orientation), ARKit intrinsics
- [ ] Reject or warn on bad input (blur, too dark, too few photos)

Acceptance
- [ ] Correct tier detected on 100% of benchmark captures plus 3 malformed fixtures
- [ ] Malformed input produces a clear error or warning, never a crash with a stack trace
- [ ] Runs in < 5 s on any capture (excluding frame extraction)

### M2 Benchmark harness (Owner: A, Day 1 hours 3 to 6)

Tasks
- [ ] Ground-truth CSV format per room (wall lengths, ceiling height, openings, areas, adjacency)
- [ ] `bench/harness.py` computes G1 to G9 and prints a pass/fail table
- [ ] Matching logic for openings (missed + phantom counted as misses)

Acceptance
- [ ] Harness run on a synthetic fixture with known answers reproduces the expected numbers exactly
- [ ] One command regenerates the full table from raw data
- [ ] Output is deterministic (two runs byte-identical)

### M3 LiDAR tier ingest (Owner: A/B)

Tasks
- [ ] Parse chosen app export (depth, RGB, poses, intrinsics) into a unified point cloud
- [ ] Depth confidence filtering, outlier removal, voxel downsample
- [ ] Handle app-specific quirks (units, axis flips, timestamps)

Acceptance
- [ ] Point cloud of a known box-shaped room: fitted plane normals within 1 degree of orthogonal
- [ ] Reprojection check: back-projected depth lines up with RGB edges (visual check saved to disk)
- [ ] Processes a 2-minute scan in < 3 min on CPU

### M4 Room geometry (Owner: B)

Tasks
- [ ] RANSAC plane fitting; classify floor, ceiling, walls; enforce Manhattan assumption as a soft prior with an off-switch for non-rectilinear rooms
- [ ] Ceiling height, wall lengths, floor polygon and area
- [ ] Uncertainty propagation (plane fit residual, pose noise)

Acceptance
- [ ] LiDAR tier ceiling height error <= 1.5 cm on all benchmark rooms (G2)
- [ ] Wall length error <= 1.5 cm on LiDAR rooms (internal target, stricter than gate)
- [ ] Floor area within 1% on LiDAR rooms
- [ ] Non-rectangular room fixture (L-shape) handled without crashing and flagged
- [ ] Mirror/glass fixture: either correct geometry or an explicit low-confidence flag with widened intervals

### M5 Openings (Owner: B)

Tasks
- [ ] Detect doors/windows from plane cutouts and RGB (SAM 3 / open-vocab prompts) cross-checked with geometry
- [ ] Width, height, sill height; type classification
- [ ] Phantom suppression (mirrors, dark furniture, doorway-shaped alcoves)

Acceptance
- [ ] G1: width error <= 2 cm on >= 85% of openings on the LiDAR benchmark, with missed and phantom each counted as a miss
- [ ] Zero phantom openings on the mirror fixture, or flagged with low confidence
- [ ] Detection precision and recall reported separately in the benchmark report

### M6 Stitcher with drift handling (Owner: B)

Tasks
- [ ] Place rooms in a common frame using connector/door constraints and pose graph
- [ ] Loop closure or plane-anchored correction; adjacency matrix
- [ ] No-overlap check; footprint polygon
- [ ] **Ablation script:** footprint with drift correction on vs off

Acceptance
- [ ] G4: report states the method; `bench/ablation_drift.py` outputs a table and image for on/off
- [ ] Correct adjacency on 100% of benchmark connections
- [ ] Zero room overlaps in all benchmark outputs
- [ ] Footprint area within 2% (LiDAR) of ground truth
- [ ] Correction measurably reduces footprint error vs "as-is" on the multi-room capture, or the report explains honestly why not

### M7 Video tier (Owner: C)

Tasks
- [ ] Frame selection (sharpness, coverage, baseline), motion-blur rejection
- [ ] SfM/reconstruction (COLMAP primary), metric scale recovery (depth model + door/ceiling priors + iPhone intrinsics)
- [ ] Hand the scaled cloud to M4/M5/M6, so downstream code is shared with LiDAR

Acceptance
- [ ] G7: wall lengths within +/-3% on benchmark rooms
- [ ] Reconstruction failure is detected and reported (no output with fake confidence); intervals widen accordingly
- [ ] 3-minute clip processes in a time budget agreed on Day 1 (target < 10 min on CPU, record actual)
- [ ] Same clip run twice gives identical output (fixed seeds)

### M8 Photo tier (Owner: C)

Tasks
- [ ] Per-room: metric depth (Depth Pro / MoGe-2), layout/plane estimation, scale priors (EXIF focal length, door height, depth model)
- [ ] Cross-room: adjacency from door/threshold visibility across folders, then stitch
- [ ] Wide, honest intervals; refuse or flag when < 2 usable photos in a room

Acceptance
- [ ] G6: wall lengths within +/-8% on benchmark rooms
- [ ] G5: multi-room photo folders produce **one stitched plan**, correct adjacency, no overlaps, footprint within +/-8%
- [ ] Single-room-only behavior is a failure; test with the multi-room fixture
- [ ] Works with 2 photos per room and with 8 photos per room
- [ ] Interval coverage meets M9 criteria

### M9 Calibrator (Owner: B + A)

Tasks
- [ ] Per-tier, per-measurement-type interval models fitted on benchmark errors (e.g. conformal-style residual quantiles)
- [ ] Input-quality inflation (low texture, low light, few photos) widens intervals
- [ ] Coverage report per tier

Acceptance
- [ ] Nominal 90% intervals cover ground truth in 85% to 95% of cases per tier on held-out benchmark data (we must hold some captures out of fitting)
- [ ] Interval width ordering: photo > video > LiDAR on every measurement type
- [ ] Degraded-input fixtures (dark, blurred, 2 photos) produce visibly wider intervals than clean input
- [ ] No tier ever emits an interval narrower than the sensor's physical floor (e.g. 0.5 cm)

### M10 Damage detection (Owner: D)

Tasks
- [ ] VLM proposes damage class and location per surface image (2 classes minimum, e.g. water stain, crack; add mold/impact if time)
- [ ] SAM mask on flagged region; mask + depth + plane gives metric area/length on the surface
- [ ] Output keyed to surface IDs from M4

Acceptance
- [ ] Both staged damage classes detected in the furnished benchmark room
- [ ] Metric extent within 15% of tape-measured damage (looser than geometry; state honestly)
- [ ] No damage reported on a clean-room control capture (false positive rate reported)
- [ ] Cached replay gives identical output
- [ ] Every region carries an interval

### M11 Concealed-damage rules (Owner: D)

Tasks
- [ ] Rule engine in `concealed_rules.yaml` (e.g. ceiling stain under a wet-area room flags possible leak; baseboard swelling plus wall stain flags possible moisture behind wall)
- [ ] Output includes rule id and the evidence that triggered it

Acceptance
- [ ] Each rule has a unit test with a triggering and a non-triggering fixture
- [ ] Every flag lists the rule id that fired
- [ ] Deterministic; no LLM involvement

### M12 Scope generation (Owner: D)

Tasks
- [ ] Map damage class + extent to line items from `catalog/scope_items.yaml` (item, unit, quantity, surface id)
- [ ] LLM only for phrasing and for edge cases, with structured output validated against the schema

Acceptance
- [ ] Every scope item references an existing surface id; schema validation passes
- [ ] Quantities derived from measured extent, not invented by the LLM (unit test)
- [ ] Items have intervals propagated from the damage measurement

### M13 QA critic (Owner: D)

Tasks
- [ ] Rule checks: angles near 90 degrees where Manhattan applies, wall sum vs polygon closure, area consistency, overlap, adjacency symmetry, interval sanity
- [ ] On failure: widen intervals, rerun a node with fallback settings, or flag in `warnings`

Acceptance
- [ ] Each check has a failing fixture and the critic catches it
- [ ] Critic never silently edits a measurement; all changes are logged
- [ ] Deliberately corrupted output (inject a 20% wall error) is flagged

### M14 Export and render (Owner: D)

Tasks
- [ ] JSON to `schema/capture_v1.json`
- [ ] SVG/PNG plan: walls, openings, dimensions, adjacency, damage overlay, interval annotation

Acceptance
- [ ] JSON validates against schema on all benchmark outputs
- [ ] Plan is recognizable as a floor plan to a non-engineer (peer review by someone outside the team)
- [ ] Render is deterministic

### M15 Orchestration: LangGraph + MCP (Owner: D)

Tasks
- [ ] Graph wiring with typed state; conditional edges for tier routing and QA retry
- [ ] MCP server exposing geometry and render tools; client with in-process fallback
- [ ] `scripts/run_capture.py <path>`: one command per capture, structured logs, timings

Acceptance
- [ ] One command, any tier, produces JSON + plan with no manual steps
- [ ] `USE_MCP=0` gives byte-identical output to `USE_MCP=1`
- [ ] `OFFLINE=1` run completes with no network access (test with network disabled)
- [ ] Node failure produces a partial result with warnings, never a crash with no output

### M16 Head-to-head (Owner: A)

Tasks
- [ ] Same 2 benchmark rooms scanned in a consumer app (free tier); export saved
- [ ] `bench/headtohead.py` outputs one table: dimension by dimension, our error vs theirs

Acceptance
- [ ] G9: beat or tie on >= 70% of shared dimensions, or the report states honestly where we lose
- [ ] App name and version recorded; export file committed

### M17 Fix loop (Owner: whoever owns the worst gate)

Tasks
- [ ] Freeze benchmark run ("before"), identify the single worst gate and failing number
- [ ] Write `fixloop/declaration.md` (gate, root-cause hypothesis with evidence, fix, predicted number) **before** shipping the fix
- [ ] Ship fix, regenerate "after", produce `diff.patch`

Acceptance
- [ ] Before and after both regenerate from one command each
- [ ] Declaration committed before the fix commit (git history proves order)
- [ ] After-run number compared with prediction in a short post-mortem, honest either way

### M18 Capture route and protocol (Owner: A)

Tasks
- [ ] `protocol/capture_protocol.md`: one page covering what to install, how to walk, how long, what to avoid (mirrors, glare, motion), how to hand files over
- [ ] Separate sections for photo, video and LiDAR tiers
- [ ] `reports/device_matrix.md`: tier vs hardware vs honest accuracy

Acceptance
- [ ] A person outside the team follows the page literally and produces a usable capture on the first try
- [ ] Install-to-first-capture time recorded
- [ ] Unambiguous: every instruction is a concrete action ("walk along walls at about 0.5 m/s"), not "scan slowly"

### M19 Documentation and compliance (Owner: D + A)

Tasks
- [ ] `compliance_matrix.md`: requirement, file path, artifact, status
- [ ] Technical report (max 6 pages): architecture, tier design, device matrix, drift handling, error budget, calibration analysis, fix loop story, known failure modes (include mirrors, glass, wet-look surfaces, low light)
- [ ] README tested on a clean machine

Acceptance
- [ ] Fresh clone to first result in < 15 minutes on a clean machine (timed by someone who did not write the README)
- [ ] Every number in the report regenerates from `scripts/make_report_tables.py`
- [ ] Report <= 6 pages

---

## 7. Benchmark set we must build (composition is mandated)

| Item | Requirement |
|---|---|
| B1 | One multi-room capture: 3 or more rooms plus a connector (hallway/doorway) |
| B2 | One furnished room with staged damage spanning **two** damage classes |
| B3 | The same rooms captured at **all three tiers**, multi-room set included; photo tier delivered as per-room folders |
| B4 | At least one room captured **twice at the same tier** (repeatability) |
| B5 | Laser or tape ground truth on everything; raw sensor data and measurements committed |
| B6 | Hard cases: at least one mirror/glass scene and one low-light scene, reported honestly |
| B7 | Hold-out: reserve some captures that are never used for calibration fitting |

Ground-truth checklist per room: every wall length, ceiling height at 3 points, every opening width/height/sill, floor diagonals (to check squareness), damage extents, door-to-door adjacency.

---

## 8. Testing strategy

| Level | What | Where |
|---|---|---|
| Unit | Plane fitting on synthetic boxes, interval math, rule engine, schema validation | `tests/unit` |
| Integration | Full graph on synthetic fixtures with known geometry, all three tiers | `tests/integration` |
| Benchmark | Real captures vs ground truth, all gates | `bench/harness.py` |
| Determinism | Run twice, diff outputs byte-for-byte (seeds fixed, LLM cache on) | `make test` |
| Offline | Run with network disabled | `OFFLINE=1` test |
| Cold run rehearsal | Fresh space, fresh capture, strict protocol, all three tiers, timed | Day 2 last hour |

Synthetic fixtures (build early, they unblock everyone): generated box rooms with doors/windows and known dimensions, rendered to depth maps + poses, used to test M3 to M6 before real captures exist.

---

## 9. Two-day plan

### Day 1: foundation and LiDAR end-to-end

| Time | Task | Owner |
|---|---|---|
| 0:00-1:00 | M0: schema, state, conventions, repo skeleton; first commits | All |
| 1:00-3:00 | **Capture session 1**: verify capture app export, scan multi-room + furnished room at all tiers, repeat one room, ground truth | A (+C) |
| 1:00-5:00 | M3, M4 on synthetic fixtures then real LiDAR data | B |
| 1:00-5:00 | M15 skeleton: graph with stub nodes, M14 export | D |
| 1:00-4:00 | M7 video: frame select, COLMAP, scale recovery attempt | C |
| 3:00-6:00 | M2 harness | A |
| 4:00 | **Model timing check on target hardware** (depth, VLM, SfM) | C |
| 5:00-8:00 | M5 openings, M6 stitcher v1 | B |
| 4:00-8:00 | M8 photo tier v1 (single room) | C |
| 5:00-8:00 | M10 damage (cached), M11 rules | D |
| End of day | **Checkpoint:** LiDAR tier through the full graph; harness prints gates | All |

### Day 2: tiers, calibration, fix loop, delivery

| Time | Task | Owner |
|---|---|---|
| 0:00-3:00 | M8 photo stitching across rooms; M7 hardening | C |
| 0:00-2:00 | M6 ablation (drift on/off) | B |
| 0:00-3:00 | M9 calibrator + coverage report | B + A |
| 0:00-3:00 | M12 scope, M13 critic, M15 one-command runner | D |
| 3:00-4:00 | M16 head-to-head | A |
| 4:00-4:30 | Full benchmark; choose worst gate; write fix declaration | All |
| 4:30-7:00 | Ship the fix; regenerate before/after (M17) | Gate owner |
| 6:00-8:00 | M18, M19: protocol, report, compliance matrix, clean-machine README test | D + A |
| Last hour | Walk-in rehearsal on an unseen space, all three tiers | All |

Commit at every working step; the history is scored.

---

## 10. Definition of done (checklist)

- [ ] Compliance matrix complete, every row has a file path and status
- [ ] Capture route (protocol page + device matrix) delivered
- [ ] Repo runs on a clean machine in < 15 min, one command per capture
- [ ] Reproduction bundle regenerates every reported number from raw inputs; live path also runs
- [ ] Benchmark report: gates at three tiers, repeatability table, head-to-head table, timing
- [ ] Fix loop bundle: declaration, before, after, diff
- [ ] Technical report <= 6 pages, including known failure modes (mirrors, glass, wet-look surfaces, low light)
- [ ] Raw benchmark data: sensor logs, ground truth, app exports
- [ ] All three tiers pass a cold run on an unseen space
- [ ] Model and API disclosure table included

---

## 11. Cut list (if time or people are short)

In this order:
1. Keep the QA critic rule-based only (no LLM)
2. Damage: two classes only (e.g. water stain, crack)
3. Photo tier: simple depth + priors + very honest wide intervals before anything clever
4. Skip optional layout models and optional tracing
5. Merge roles: B owns M3 to M6 and M9; C owns M7 and M8; D owns everything else; A owns data, harness, protocol

Never cut: harness, drift ablation, photo multi-room stitch, calibration, fix loop, commits.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No Pro iPhone for LiDAR tier | Borrow a device for capture sessions; capture once and keep raw data. The walk-in test only needs the organizers' phone |
| Capture app does not export depth/poses in a usable form | Test two apps in the first hour; write the ingest adapter per app |
| Video SfM fails on texture-poor walls | Detect failure, fall back to depth-prior scale with wide intervals |
| Photo tier error exceeds 8% | Report honestly; widen intervals; calibration is scored too |
| Mirrors/glass produce phantom geometry | Mirror detection flag, QA critic, wider intervals, documented in failure modes |
| CPU too slow in the demo | Choose smaller models, quantized weights, cache; record timings |
| LLM unavailable offline | Local VLM + replay cache; Scope mostly template-driven |
| MCP transport fails live | `USE_MCP=0` in-process path |
| Defense: cannot explain a design choice | Keep designs simple; each module owner writes a 5-line rationale in the report |

---

## 13. Quick start (to be finalized by Day 2)

```bash
git clone <repo> && cd areamap
make setup                       # env + deps
bash scripts/fetch_models.sh     # weights (needs HF_TOKEN, once)
python scripts/run_capture.py data/raw/<capture_folder> --out out/
make bench                       # all gates + tables
make ablation                    # drift on/off
make fixloop                     # regenerate before/after
```

Outputs per capture: `out/plan.json` (validates against `schema/capture_v1.json`), `out/plan.svg`, `out/run_log.json`.
