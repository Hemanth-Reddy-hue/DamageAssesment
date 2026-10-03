# AreaMap Multi-Room Architecture Audit & Technical Fix Roadmap

## 1. Project Context & Objectives
Home damage assessment requires reconstructing a metric, dimensioned multi-room floor plan from ordinary perspective iPhone still images (and walkthrough video) without requiring users to manually pre-sort images into room folders.

This document serves as the persistent audit and implementation specification for resolving current multi-room and global floor-plan reconstruction failures.

---

## 2. Audited Project Architecture

| File | Purpose | Key Functions / Classes |
|---|---|---|
| [`src/areamap/state.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/state.py) | Pydantic data contracts and schemas | `Interval`, `WallSegment`, `Opening`, `RoomGeometry`, `StitchedPlan`, `CaptureState` |
| [`src/areamap/graph.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/graph.py) | Graph pipeline runner | `build_areamap_graph()`, `SimpleGraphRunner` |
| [`src/areamap/nodes/ingest.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/nodes/ingest.py) | Tier routing & multi-room folder parsing | `detect_tier()`, `ingest_node()` |
| [`src/areamap/tiers/photo.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/tiers/photo.py) | Photo EXIF, vanishing pitch, unprojection | `extract_exif_intrinsics()`, `detect_vertical_vanishing_pitch()`, `recover_metric_scale_and_points()`, `ingest_photo_capture()` |
| [`src/areamap/nodes/geometry.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/nodes/geometry.py) | Per-room plane fitting driver | `geometry_node()` |
| [`src/areamap/geometry/planes.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/geometry/planes.py) | RANSAC floor/ceiling & wall planes | `fit_plane_ransac()`, `extract_horizontal_planes()`, `extract_vertical_wall_planes()`, `intersect_2d_lines()`, `fit_room_planes()` |
| [`src/areamap/nodes/openings.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/nodes/openings.py) | Aperture detection driver | `openings_node()` |
| [`src/areamap/geometry/openings.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/geometry/openings.py) | Wall unrolling & density cutouts | `unroll_wall_points()`, `detect_cutouts_on_wall()`, `detect_openings_from_cutouts()` |
| [`src/areamap/nodes/stitch.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/nodes/stitch.py) | Frame alignment, collision shift, footprint | `stitch_node()` |
| [`src/areamap/geometry/adjacency.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/geometry/adjacency.py) | Door matching, SE(2) door snap, polygon clip | `infer_room_adjacency()`, `align_room_pair_se2()`, `apply_se2_transform_to_room()`, `_clip_polygon()`, `check_room_overlaps()` |
| [`src/areamap/geometry/posegraph.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/geometry/posegraph.py) | SE(2) matrix math & Manhattan angle snap | `make_se2_matrix()`, `extract_se2_params()`, `snap_to_manhattan_angle()`, `optimize_pose_graph()` |
| [`src/areamap/render/plan_svg.py`](file:///c:/Users/Chetana/Downloads/DamageAssesment/src/areamap/render/plan_svg.py) | 2D dimensioned floor-plan SVG visualizer | `render_plan_svg()` |

---

## 3. End-to-End Pipeline Trace

```
iPhone Images in directory: capture_path
  │
  ▼
[src/areamap/graph.py: SimpleGraphRunner.invoke()]
  │
  ▼
1. INGESTION & CLUSTERING:
   src/areamap/nodes/ingest.py: ingest_node(state)
   ├── detect_tier(capture_path) -> "photo"
   ├── subdirs = [d for d in capture_path.iterdir() if d.is_dir()]
   └── for each subfolder:
         src/areamap/tiers/photo.py: ingest_photo_capture(photo_dir)
         ├── for each photo_path in photo_files:
         │     ├── extract_exif_intrinsics(photo_path) -> (fx, fy, cx, cy)
         │     ├── detect_vertical_vanishing_pitch(photo_path, intrinsics) -> pitch_rad
         │     └── recover_metric_scale_and_points(photo_path, ...) -> points_3d
         ├── np.vstack(accumulated_clouds)
         └── voxel grid downsampling (5 cm) -> data/cache/cloud_{room_id}.npy
  │
  ▼
2. PER-ROOM 3D PLANE & BOUNDARY RECONSTRUCTION:
   src/areamap/nodes/geometry.py: geometry_node(state)
   └── for each room_id:
         src/areamap/geometry/planes.py: fit_room_planes(pts, room_id, ...)
         ├── extract_horizontal_planes(pts) -> RANSAC floor (z=0) & ceiling (z=H)
         ├── extract_vertical_wall_planes(pts) -> RANSAC 4 vertical planes
         ├── intersect_2d_lines(n_i, d_i, n_j, d_j) -> 2D corner vertices
         ├── calculate_polygon_area(vertices) -> floor_area
         └── calculate_interval() -> conformal bounds on height, area, walls
  │
  ▼
3. WALL CUTOUT & APERTURE DETECTION:
   src/areamap/nodes/openings.py: openings_node(state)
   └── for each room_id:
         src/areamap/geometry/openings.py: detect_openings_from_cutouts(walls, pts, ...)
         ├── for each wall:
         │     ├── unroll_wall_points(wall, pts) -> 2D (u, v) wall plane coordinates
         │     ├── detect_cutouts_on_wall(wall, pts) -> 2 cm column histogram void detection
         │     └── classify opening: door (sill <= 0.35m) vs window (sill > 0.35m)
         └── geom.openings = detected openings
  │
  ▼
4. MULTI-ROOM ADJACENCY & GLOBAL FRAME STITCHING:
   src/areamap/nodes/stitch.py: stitch_node(state)
   ├── src/areamap/geometry/adjacency.py: infer_room_adjacency(rooms)
   │     └── matches doors across rooms if abs(d1.width - d2.width) <= 0.20m
   ├── for each connection:
   │     src/areamap/geometry/adjacency.py: align_room_pair_se2(from_room, to_room, door1, door2)
   │     ├── rotation: R(theta) * norm_to = -norm_from
   │     └── translation: t = p_from - R * p_to
   ├── src/areamap/geometry/posegraph.py: optimize_pose_graph(relative_poses, loop_closures)
   │     ├── dead-reckoning forward composition: T_v = T_u @ rel_T
   │     └── snap_to_manhattan_angle(theta, tol=8 deg)
   ├── Collision Resolution:
   │     for each placed room:
   │       _clip_polygon(curr_geom, placed_geom)
   │       if overlap > 0.05 m²: shift curr_geom along +X axis
   └── StitchedPlan: aggregate footprint area & bounding envelope polygon
  │
  ▼
5. EXPORT & VISUALIZATION:
   src/areamap/nodes/export.py: export_node(state) -> plan.json & plan.svg
```

---

## 4. Current Multi-Room Logic Audit (10 Core Questions)

1. **Same Room Discrimination:** None exists at image level. System depends 100% on filesystem directory separation.
2. **Different Room Discrimination:** Strictly determined by whether files reside in different folder paths.
3. **Physical Connectivity:** Compares door opening widths (`abs(d1.width - d2.width) <= 0.20m`).
4. **Reference Image Selection:** None. Points from all images in a room are stacked at origin $(0, 0, 0)$. First folder is root.
5. **Relative Image Pose Estimation:** None. Assumes all photos were taken from $(0, 0, 1.45)$ looking horizontally.
6. **Room Transformation:** Solves 2D $SE(2)$ rigid matrix matching door centers and opposing outward wall normals.
7. **Drift Prevention:** Multiplies transforms along tree; snaps yaw angle to multiples of $90^\circ$ if within $\pm 8^\circ$.
8. **Duplicate Room Prevention:** Creates exactly one room per directory name. No visual deduplication.
9. **Room Merging:** Rooms remain separate polygons in a shared coordinate system; no topological fusion.
10. **Disconnected Room Handling:** Chains disconnected rooms blindly to the right along $+X$ axis.

---

## 5. Coordinate Systems Specification

```
[Pixel (u, v)] ──Pinhole──▶ [Camera (X_c, Y_c, Z_c)] ──Pitch Rect──▶ [Room 3D (X, Y, Z)]
                                                                           │
                                                                       RANSAC
                                                                           ▼
[SVG Canvas (x_svg, y_svg)] ◀──Normalize── [Global 2D (X_plan, Y_plan)] ◀──SE(2)── [Room 2D (x, y)]
```

* **Pixel Coordinates:** Origin $(0, 0)$ top-left, $+u$ right, $+v$ down.
* **Camera Coordinates:** $+X_c$ right, $+Y_c$ down, $+Z_c$ forward. Pinhole: $f_x = (f_{35} / 36.0) \cdot W$, $f_y = f_x$.
* **Room-Local 3D Euclidean:** Right-handed metric: $+X$ right, $+Y$ forward, $+Z$ up (floor $Z=0$).
* **Room-Local 2D Plane:** Horizontal slice at $Z=0$ bounded by RANSAC wall lines $n_{2d} \cdot p + d = 0$.
* **Global Plan Frame:** 2D metric coordinates anchored to Room 1; rooms positioned via $3 \times 3$ $SE(2)$ matrices:
  $$T = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}$$
* **SVG Canvas:** Normalized $[1200 \times 900]$ space with dynamic scale $s = \min(950 / \text{span}_x, 680 / \text{span}_y, 85.0)$.

---

## 6. Technical Failure Modes Ranking

| Severity | Failure Mode | Technical Mechanism |
|---|---|---|
| **CRITICAL** | **No Image Matching / Epipolar Tracking** | Photos in a room are not registered relative to each other; stacking assuming $(0,0,0)$ causes ghost walls. |
| **CRITICAL** | **Directory-Dependent Room Clustering** | Unorganized photos in a single folder cannot be partitioned into rooms; lumped into `"room_01"`. |
| **CRITICAL** | **Door Pairing by Width Alone** | All interior doors are $0.80\text{–}0.90\text{ m}$; leads to random false room-to-room topology. |
| **CRITICAL** | **Ad-Hoc X-Axis Collision Shift** | Pushing overlapping rooms along $+X$ breaks door connectivity and turns 2D floor plans into 1D trains. |
| **HIGH** | **Fragile Height/Baseboard Scale Prior** | Assuming $1.45\text{ m}$ camera height and baseboard at $55\text{–}85\%$ image height causes $>50\%$ error on non-level shots. |
| **HIGH** | **Disconnected Room Snapping** | Snapping unmatched rooms along $+X$ creates false adjacencies. |
| **HIGH** | **Absence of Visual Loop Closure** | Multi-room loops accumulate drift without automatic feature-based loop detection. |
| **HIGH** | **Vanishing Line Pitch Fragility** | Edge-based pitch estimation fails on plain walls, curtains, or wallpaper. |
| **MEDIUM** | **Manhattan Grid Over-Constraint** | $90^\circ$ snapping breaks diagonal walls and angled extensions. |
| **MEDIUM** | **Greedy Spanning Tree Order** | Topological layout changes depending on alphabetical folder sorting. |
| **MEDIUM** | **Bounding Box Footprint** | Outer envelope computed via rectangular bounding box instead of boolean polygon union. |
| **LOW** | **Multi-Floor Handling** | All rooms projected onto single $Z=0$ plane; no vertical floor separation. |

---

## 7. ZInD Comparison: Transferable vs. Non-Transferable

* **Transferable Concepts:**
  1. *Primary vs. Secondary View Hierarchy:* Pick one primary anchor image per room; register other photos to it.
  2. *Portal/Doorway Association:* Treat doors as explicit graph nodes linking two adjacent room coordinate frames.
  3. *Global Pose Graph Optimization:* Jointly optimize relative room positions using doorway coincidence, wall parallelism, and non-penetration penalties.
  4. *Structured 2D Polygon Layout:* Represent each room as an ordered 2D polygon with metric wall line segments.
* **Non-Transferable Concepts:**
  1. *Equirectangular $360^\circ$ Geometry:* Perspective photos have narrow FOV ($\approx 65^\circ$) and cannot see all walls in one shot.
  2. *Single-Shot 4-Corner Layout Networks:* Models like HorizonNet assume all 4 ceiling/floor corners are visible in one panorama.

---

## 8. Proposed 6-Stage Architecture

```
Unorganized iPhone Stills (arbitrary order)
  │
  ▼
Stage 1: Per-Image Feature Extraction & Metric Depth
  ├── EXIF focal length extraction
  ├── Monocular metric depth estimation (Depth Anything V2 Metric)
  ├── Local keypoints & descriptors (SuperPoint)
  └── Global image embedding (DINOv2)
  │
  ▼
Stage 2: Visual Place Recognition & Room Clustering
  ├── Pairwise cosine similarity matrix across global embeddings
  ├── Community detection (Leiden / Louvain graph clustering)
  └── Automatic room assignment & Primary Keyframe selection
  │
  ▼
Stage 3: Per-Room Multi-View Reconstruction
  ├── Intra-cluster feature matching (LightGlue) & relative poses (PnP)
  ├── Multi-view metric point cloud fusion
  ├── RANSAC wall plane extraction & 2D polygon fitting
  └── Door and window opening detection
  │
  ▼
Stage 4: Visual Portal Verification
  ├── Feature matching across doorway apertures
  └── Topological Room Adjacency Graph construction
  │
  ▼
Stage 5: Global Floor-Plan Graph Optimization
  ├── Joint SE(2) non-linear least squares optimization (Scipy)
  ├── Doorway alignment constraints + Non-penetration barrier penalties
  └── Loop-closure residual distribution
  │
  ▼
Stage 6: Multi-Room Blueprint & Damage Scoping
  ├── True boolean polygon union of room boundaries
  ├── VLM damage surface projection
  └── Dimensioned SVG blueprint & JSON schema export
```

---

## 9. Component Action Matrix

| Component | Status | Action |
|---|---|---|
| Pydantic State & Contracts (`state.py`) | Production Ready | **KEEP** |
| SVG Floor Plan Visualizer (`plan_svg.py`) | Production Ready | **KEEP** |
| Conformal Uncertainty Engine (`uncertainty.py`) | Production Ready | **KEEP** |
| Damage & Concealed Rules (`damage.py`, `concealed.py`) | Production Ready | **KEEP** |
| RANSAC Plane Extraction (`planes.py`) | Functional | **MODIFY** for non-rectangular polygons |
| Wall Cutout Detector (`openings.py`) | Functional | **MODIFY** to fuse with 2D object detection |
| Pose Graph Optimizer (`posegraph.py`) | Minimal | **MODIFY** to use Scipy least-squares |
| Door Adjacency Inference (`adjacency.py`) | Flawed | **REPLACE** width matching with visual portal matching |
| Single-Image Depth & Scale (`photo.py`) | Flawed | **REPLACE** height prior with metric depth model |
| Collision Push along $+X$ (`stitch.py`) | Flawed | **REPLACE** with constrained non-overlap optimization |
| Image Clustering Engine | Missing | **ADD** DINOv2 room clustering module |
| Intra-Room Camera Tracker | Missing | **ADD** SuperPoint + LightGlue feature matching & PnP |
| Visual Portal Verifier | Missing | **ADD** cross-door visual co-visibility verification |

---

## 10. The 5 Priority Implementation Steps

1. **Automatic Room Clustering (`src/areamap/clustering/room_cluster.py`):**
   Extract DINOv2 global embeddings and cluster images via graph community detection so unorganized photo folders are automatically partitioned into rooms.
2. **Visual Portal Doorway Matching (`src/areamap/geometry/portal_matcher.py`):**
   Replace width-only door pairing in `adjacency.py` with visual feature correspondences across doorways.
3. **Metric Monocular Depth Integration (`src/areamap/tiers/photo.py`):**
   Replace hardcoded $1.45\text{ m}$ camera height prior and baseboard line assumption with metric monocular depth estimation.
4. **Intra-Room Camera Pose Tracking (`src/areamap/sfm/`):**
   Use SuperPoint + LightGlue feature matching between images in the same room cluster to compute relative poses $(R_k, t_k)$ before fusing point clouds.
5. **Constrained SE(2) Pose Graph Optimization (`src/areamap/nodes/stitch.py`):**
   Replace the $+X$ collision shift with joint non-linear least-squares optimization (`scipy.optimize.least_squares`) minimizing doorway residuals while penalizing polygon overlaps.
