# AreaMap — Video Tier Fix: Implementation Spec

Audience: the coding assistant (Copilot) that will edit the repo.
Scope: everything that touches the **video tier** — LLM client, keyframes, SfM, scale, room segmentation, failure policy.
Out of scope for now: openings, damage, scope, concealed, calibration, stitching design (separate pass later).

Environment: Windows, PowerShell, Python venv. Package root `src/areamap/`.

---

## 0. Read this first

### 0.1 Files you must read before editing (not all were reviewed when this spec was written)

| File | Why |
|---|---|
| `src/areamap/nodes/ingest.py` (≈ lines 120–160) | Consumes `ingest_video_capture()`. Writes `rooms_dict["room_01"]` in several places. Must be adapted to the new return contract (WP7). |
| `src/areamap/tiers/photo.py` | Owns `_get_depth_engine()`, `DepthEngine.predict()`, `recover_metric_scale_and_points()`. Check whether `predict()` is conditioned on the camera-height / ceiling **priors** (suspected; see F9). |
| `src/areamap/geometry/registration.py` | `estimate_relative_pose_essential`, `icp_align` (odometry fallback path). |
| `src/areamap/config.py` | `settings` object (`llm_provider`, `offline`, `gemini_*`, keys). Add new settings here (WP2). |
| `src/areamap/llm/cache.py` | `LLMCache` key format. Keep it compatible. |
| `src/areamap/state.py` | `CaptureState`, `RoomGeometry` — add provenance / warning fields if missing. |
| `src/areamap/nodes/damage.py` (line ≈ 111) | Also calls `generate_structured`; it will receive the new "unavailable" result instead of fabricated damage (WP1). |
| `src/areamap/geometry/room_discovery.py`, `posegraph.py` | Use `room_{idx:02d}` ids. Video tier must use the same id scheme (F13). |

### 0.2 Code-vs-log mismatch — verify before trusting the log

The log from the failed run says `SfM registration low: 105/153 frames (<80%)`.
The uploaded `video_sfm.py` caps keyframes at **30** (line 30) and its warning text is `SfM registration partial` with a 60 % threshold (line ~92).
So the run that produced the log used a **different version** of the code than the uploaded file (or there is a second copy of the module).

**Task 0:** run

```powershell
Get-ChildItem -Path . -Recurse -Filter video_sfm.py | Select-Object FullName, LastWriteTime
Get-ChildItem -Path . -Recurse -Include *.py | Select-String -Pattern "SfM registration low"
```

Find which file prints that message and make sure there is exactly **one** `video_sfm.py` and that it is the one being imported. Do not proceed until this is resolved, otherwise fixes will be applied to a file that is not executed.

---

## 1. Findings (root causes)

Evidence is from the uploaded files and the run log. Line numbers are approximate.

### A. LLM client — `llm/client.py`

| # | Finding | Evidence | Impact |
|---|---|---|---|
| F1 | **Retrying a quota error.** Daily-quota 429s are retried 3× (5 s, 10 s backoff) plus a `time.sleep(4)` per call. | L48–64, L114; log shows `retryDelay: 77553s` | ≈ 27 s wasted per frame; 153 frames ≈ 1 h stall. Retrying can never succeed when the reset is ~21 h away. |
| F2 | **No circuit breaker, no call budget.** Every keyframe triggers a new API call even after the provider is known dead. | `video_sfm.py` L237 loop over all keyframes | One failure repeats N times. |
| F3 | **Parse failures are retried as API calls.** `_parse_text_response` raises `ValueError`, which goes through the same retry loop. | L51–60 | On a 20-requests/day quota, a bad JSON reply burns up to 3 requests. |
| F4 | **Fabricated results.** After retries fail, `_offline_fallback` returns fake data: room type `"living_room"` for every frame, fake `water_stain` damage with extent 0.85 m², fake scope item. | L160–200 | Fake output looks real downstream. This is the most dangerous bug in the file. Every room is labelled `living_room`, which collapses to one room. |
| F5 | **Anthropic/OpenAI paths ignore the image and the schema.** They call `llm.invoke(prompt)` only, with hard-coded old model names. | L86–87, L130–131 | Vision tasks silently run text-only. |
| F6 | **Full-resolution frames are sent.** No downscaling before base64. | `_call_api` | Slow, token-heavy, hits tokens-per-minute limits. |
| F7 | **Only paid/limited providers supported.** No Mistral / Groq / Ollama (local) path. | `_call_api` | Cannot use free tiers or run offline. |

### B. Keyframes, SfM, scale — `tiers/video.py`, `tiers/video_sfm.py`

| # | Finding | Evidence | Impact |
|---|---|---|---|
| F8 | **Plane/scale math is done on an un-scaled SfM cloud using metric thresholds.** `extract_horizontal_planes` uses 0.04 m RANSAC threshold and a 0.30 m "distinct plane" gap, but COLMAP units are arbitrary. Also it assumes the floor normal is near +Z in a frame that is just the first camera's frame. | `video_sfm.py` L102–122; `planes.py` L85, L113 | Floor/ceiling often not found, or found as the same plane → `Scaled ceiling 0.02 m` in the log. |
| F9 | **Scale recovery is circular.** `engine.predict(..., camera_height=1.45, seam_v=0.7*H, ceiling_height=2.6)` feeds the priors into the depth model, then the result is cross-checked against the same 2.6 m prior, and on mismatch the code overwrites the scale with `2.6 / sfm_height`. | L148, L196–206 | Reported ceiling height is whatever the prior is; measured scale is meaningless; the failure is hidden. (Confirm in `photo.py`.) |
| F10 | **Gravity is not estimated from the cameras.** Floor RANSAC is used to find "up". | L108–122 | Fails in dim, textureless rooms. Camera up-vectors are a much stronger cue for handheld video. |
| F11 | **Registration count is probably wrong.** `largest_model.num_images()` may count unregistered images; later loops iterate `largest_model.images` including unregistered ones. | L80–85, L130 | Wrong registration ratio; possible exceptions on images without a pose. Use `num_reg_images()` / registered-only iteration (check installed pycolmap version). |
| F12 | **Fragile matching/mapping setup.** Sequential matching with small overlap, `loop_detection=True` (needs a vocab tree, may need network), `multiple_models=False`, `min_model_size=3`, no focal prior (COLMAP defaults to ≈ 1.2×max(w,h), far from an iPhone's ≈ 65° HFOV), no feature caps, keyframes chosen by time not by parallax. | L30–33, L52–69 | Fragmented models (105/153), Cholesky/BA failures in the log, slow CPU SIFT (4.8 min). With `multiple_models=False` the other fragments are discarded — which for a 2-room clip is typically the second room. |
| F13 | **Room ids are LLM labels.** Segments are keyed by labels such as `living_room`; non-contiguous repeats merge; ids don't match `room_{idx:02d}` used elsewhere. Last segment end is `timestamp + 10.0` (magic number). | L237–260 | Wrong room splitting, id mismatch downstream. |
| F14 | **Points↔array alignment by dict-order index.** `valid_idx` walks `points3D.items()` in lock-step with a filtered array. | L285–293 | Works only if iteration order and filter are identical; fragile. Keep `(point_id, xyz)` together. |
| F15 | **Keyframe extractor is slow and memory-heavy.** Seeks with `cap.set(POS_FRAMES)` per sample (inexact on H.264/HEVC) and keeps full-resolution frames for all candidates. | `video.py` L86–103 | Wrong timestamps, high RAM on 4K video. |

### C. Silent fallbacks to fake geometry

| # | Finding | Evidence | Impact |
|---|---|---|---|
| F16 | Several paths silently substitute a synthetic 4 × 3 × 2.6 m box: missing video file, SfM failure → odometry fallback with empty result, `geometry.py` when no cloud, `planes._create_fallback_room`. | `video.py` L329–333, L361–368; `geometry.py` L19; `planes.py` L286, L429 | This is the likely cause of "2-room video comes out as one 4 × 3 room". Output looks real but is invented. |
| F17 | **Dataset-specific hard-coding:** `if "singleroom" in parent_name: room_prior = (5.35, 5.90)`. | `video.py` L413 | Overfits to sample data; not acceptable in a real pipeline. |
| F18 | **Ceiling hidden for video tier:** if ceiling plane is missing or outside 2.0–2.7 m, it is silently set to 2.40 m. | `planes.py` L319–321 | Placeholder reported as a measurement. |
| F19 | `ingest_video_capture` returns `scaled_pts` that is actually a **dict of rooms** (from SfM) but is typed/handled as an ndarray; `intrinsics` is `{}` on the SfM path; no registration/scale info is returned. | `video.py` L383–390 | Metadata is empty; downstream cannot judge quality. |
| F20 | RANSAC uses unseeded `np.random.choice`. | `planes.py` L29 | Non-deterministic results between runs. |

---

## 2. Design principles for the fix

1. **Never fabricate.** If a stage cannot produce a result, it returns an explicit failure/unavailable status with a reason. Nothing downstream may treat a prior as a measurement.
2. **Fail fast on quota.** A daily-quota error disables that provider for the rest of the run immediately.
3. **LLM is optional.** The video pipeline must produce correct geometry and room splits with **no** LLM. The LLM only adds optional room-type names.
4. **Priors are priors.** Priors can seed or sanity-check, but a result produced by a prior is tagged `method="prior"` with low confidence and wide intervals.
5. **Everything observable.** Each stage logs counts, timings, and a `quality` summary into the result so QA and users can see what happened.
6. **Deterministic.** Seed all RNGs.

---

## 3. Target data flow

```
video file
  │
  ├─ WP3  Keyframe extraction (sequential read, downscale, parallax-aware, blur/glare filtered)
  │
  ├─ WP4  SfM (exhaustive matching for ≤120 frames, tuned options, keep ALL models)
  │         └─ fallback chain if registration low
  │
  ├─ WP5  Gravity from camera up-vectors → metric scale from fused cues → plane sanity
  │
  ├─ WP6  Room segmentation from covisibility graph (no LLM) → ids room_00, room_01, …
  │         └─ optional: 1 LLM call per room to name it (cached, budgeted)
  │
  └─ WP7  VideoReconstruction result (typed) → ingest.py
            status = ok | degraded | failed   (never a synthetic box)
```

Typed result (new, e.g. `src/areamap/tiers/video_types.py`):

```python
@dataclass
class ScaleInfo:
    factor: float | None            # SfM units -> metres
    method: str                     # "depth_fused" | "reference" | "door" | "prior" | "none"
    confidence: float               # 0..1
    cues: dict[str, dict]           # per-cue estimate, n_samples, spread
    relative_uncertainty: float     # e.g. 0.08 for ±8 %

@dataclass
class RegistrationInfo:
    n_frames: int
    n_registered: int
    ratio: float
    n_models: int
    model_sizes: list[int]
    mean_reproj_error_px: float
    strategy_used: str              # "exhaustive_default" | "exhaustive_relaxed" | ...

@dataclass
class VideoRoom:
    room_id: str                    # "room_00", "room_01", ...
    points: np.ndarray              # (N,3) metres, z-up, shared world frame
    camera_positions: np.ndarray    # (M,3)
    time_range_s: tuple[float, float]
    room_type: str | None           # optional name, from LLM if available
    room_type_source: str           # "llm" | "none"

@dataclass
class VideoReconstruction:
    status: str                     # "ok" | "degraded" | "failed"
    rooms: list[VideoRoom]
    transitions: list[dict]         # doorway candidates between rooms (frame ids, positions)
    scale: ScaleInfo
    registration: RegistrationInfo
    intrinsics: dict
    video_meta: dict
    warnings: list[str]
    timings: dict[str, float]
    llm: dict                       # {"provider": ..., "calls": n, "disabled_reason": ... | None}
```

---

## 4. Work packages

Implement in this order. Each WP has acceptance criteria; do not move on until they pass.

---

### WP1 — Rewrite `llm/client.py`

**Goal:** no stalls on quota errors, no fabricated output, free-provider support, image-aware for all providers.

**1.1 Single OpenAI-compatible implementation.**
Mistral, Groq, Ollama (local), OpenRouter, and Gemini (OpenAI-compatible endpoint) all accept the OpenAI chat-completions format. Implement **one** `_call_openai_compat(base_url, api_key, model, messages, ...)` using the `openai` SDK (or `httpx`) and drive providers from a config table:

| provider key | base_url | key env var | notes |
|---|---|---|---|
| `mistral` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` | free "Experiment" tier; 1 req/s, high monthly token cap (verify in console) |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | needs a vision-capable model; watch tokens/min |
| `ollama` | `http://localhost:11434/v1` | none | local, no quota; e.g. a small Qwen-VL / LLaVA / Moondream model |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `:free` models, small daily cap |
| `gemini` | Google OpenAI-compat endpoint | `GEMINI_API_KEY` | keep as optional |
| `anthropic`, `openai` | native | existing keys | keep, but **must send the image** (fix F5) and use configurable model names |

Model names must be **settings**, not literals (e.g. `settings.llm_model_mistral`). Do not hard-code model ids; they change. Verify each model supports image input before relying on it.

**1.2 Provider chain.**
`settings.llm_provider_chain = ["mistral", "groq", "ollama"]` (comma-separated env var). Try providers in order; skip any provider that is missing a key, unreachable, or has its breaker open. If every provider fails → return unavailable (1.5).

**1.3 Error classification** (this is the core fix for F1/F3):

| Error | Action |
|---|---|
| 429 / `RESOURCE_EXHAUSTED` with a long reset (`retryDelay` > 60 s, or text containing "quota", "per day", "free_tier") | **Non-retryable.** Open that provider's breaker until the reset time (or until process end). Move to the next provider. No sleep. |
| 429 with short reset (≤ 10 s) | One retry after `min(retryDelay, 10)` s. |
| 5xx / timeout / connection error | Up to 2 retries, 1 s then 3 s, jittered. |
| 400 / 401 / 403 / 404 (bad request, auth, model not found) | Non-retryable. Open breaker for that provider for the run. |
| JSON parse / schema error | **No extra API call retry.** Try local JSON repair (strip fences, extract the first `{...}` block); if still invalid, return a parse-failure result. |

Extract `retryDelay` from the Google error payload (`RetryInfo.retryDelay`, e.g. `"77553s"`) and from `Retry-After` headers.

**1.3b** Remove the unconditional `time.sleep(4)`. Replace with a per-provider token-bucket limiter configured from settings (`rpm` per provider); sleep only when the bucket is empty.

**1.4 Budget and breaker.**
- `settings.llm_max_calls_per_run` (default 12). When exceeded, further calls return unavailable immediately.
- Global per-run kill switch `settings.llm_enabled` and CLI flag `--no-llm`.
- Wall-clock cap per call (`timeout=30 s`) and per run (`llm_max_total_seconds`, default 120 s).

**1.5 Return type.** Replace the dict-or-fabricated-dict with:

```python
@dataclass
class LLMResult:
    ok: bool
    data: dict | None
    status: str          # "ok" | "unavailable" | "quota" | "parse_error" | "disabled" | "budget"
    provider: str | None
    model: str | None
    cached: bool
    detail: str | None   # short reason, no secrets
```

`generate_structured(...) -> LLMResult`. **Delete** `_offline_fallback` and every fabricated payload. Existing callers (`video_sfm.py`, `nodes/damage.py`) must handle `ok=False` by skipping the LLM-derived output and adding a warning to state — never by inventing data.

**1.6 Image handling.**
- Decode with OpenCV/Pillow, downscale so the long side is ≤ 768 px (configurable), JPEG quality ≈ 80, then base64 data URL.
- For a multi-frame request, build a **single collage** (e.g. 2×2 grid of frames) so one request covers several frames.

**1.7 Caching.**
- Key = hash(prompt + image bytes after downscale + provider + model).
- Read the cache on every run (online and offline). Cache only `ok=True` results.
- Never cache failures.

**1.8 Logging/telemetry.**
- One concise log line per call: provider, model, ms, tokens (if given), status.
- Maintain `client.stats = {"calls": n, "ok": n, "cached": n, "failed": n, "disabled_providers": {...}}`; expose it so the video result can carry `llm` info.
- Never log API keys or full payloads. Truncate error text to ~300 characters in console output (the current code prints entire 429 JSON bodies).

**Acceptance (WP1):**
- With an exhausted Gemini key as the only provider, a 153-frame loop finishes the LLM stage in **< 5 s total** and returns `status="quota"`; no `time.sleep` > 1 s occurs after the first failure.
- With a mock server returning 429 `retryDelay: 77553s`, exactly **1** request is made per provider per run.
- `grep -rn "_offline_fallback\|living_room" src/` returns nothing in `llm/`.
- A parse error produces ≤ 1 API request.
- Unit tests cover each row of the error table (mock HTTP).

---

### WP2 — Settings (`config.py`) and CLI

Add (with env-var names, defaults, and docstrings):

```
LLM_ENABLED=true
LLM_PROVIDER_CHAIN=mistral,groq,ollama
LLM_MAX_CALLS_PER_RUN=12
LLM_MAX_TOTAL_SECONDS=120
LLM_TIMEOUT_SECONDS=30
LLM_IMAGE_MAX_SIDE=768
LLM_MODEL_MISTRAL=...      # set by user; verify vision support
LLM_MODEL_GROQ=...
LLM_MODEL_OLLAMA=...
LLM_RPM_MISTRAL=50 (example; verify)  LLM_RPM_GROQ=...   # per-provider limiter
OLLAMA_BASE_URL=http://localhost:11434/v1

VIDEO_MAX_FRAMES=90
VIDEO_MAX_IMAGE_SIDE=1600
VIDEO_MATCHER=auto            # auto | exhaustive | sequential
VIDEO_MIN_REG_RATIO=0.60
VIDEO_STAGE_TIMEOUT_S=600
ALLOW_SYNTHETIC=false         # only tests/demos may set true
KNOWN_REFERENCE_M=            # optional user-supplied length, see WP5
KNOWN_REFERENCE_KIND=         # "ceiling_height" | "door_height" | "wall_length"
```

CLI (`main.py`): `--no-llm`, `--reference-height 2.7`, `--allow-synthetic` (demo only), `--rooms-json path`.
Provide a `.env.example` listing variable **names only** (no keys). Make sure `.env` is in `.gitignore`.

**Acceptance:** `python main.py <video> --no-llm` runs the whole video tier without importing any LLM SDK.

---

### WP3 — Keyframe extraction (`tiers/video.py`)

Rewrite `extract_sharp_keyframes` (keep the name and a compatible return shape; add fields).

1. **Sequential decode.** Read with `cap.grab()` / `cap.retrieve()` in order; compute the timestamp from `CAP_PROP_POS_MSEC` (fallback: `idx/fps`). No `CAP_PROP_POS_FRAMES` seeking. Check `CAP_PROP_ORIENTATION_AUTO`/rotation metadata so portrait iPhone videos are not sideways (log the final frame size).
2. **Analysis at low resolution.** Do blur / flow computations on a ≈ 480 px-wide copy. Keep full-res only for chosen keyframes, then downscale those to `VIDEO_MAX_IMAGE_SIDE` (default 1600) before writing JPEGs. Never hold more than a few frames in memory.
3. **Candidate sampling:** every ~4th–6th frame (≈ 5–8 fps for 30 fps video).
4. **Quality filters per candidate:**
   - sharpness: variance of Laplacian, compared with a **rolling local median** (reject if < 0.6× local median) rather than a global percentile only;
   - exposure: reject frames with > 25 % of pixels saturated (> 245) or mean intensity too low (log how many dark frames);
   - texture: reject if detected corner/feature count (e.g. FAST/ORB at low-res) is below a floor (≈ 150).
5. **Parallax-aware selection.** Walk the candidates in time order. Accept a new keyframe when the median feature displacement vs. the last keyframe (pyramidal LK flow on tracked corners at low-res) is ≥ ~5–8 % of image width, **or** the time gap exceeds ~2.0 s. This avoids near-duplicate frames when the user pauses, and large gaps when moving fast. Cap at `VIDEO_MAX_FRAMES` (default 90); if over the cap, drop the lowest-quality frames evenly.
6. **Pure-rotation detection.** From the same flow, estimate the ratio of translation vs. rotation (e.g. homography-explained fraction). If > 60 % of the clip is consistent with a pure pan, add warning `"video_mostly_rotation: SfM scale/depth unreliable; ask user to walk instead of pivoting"`.
7. **Metadata returned:** total/sampled/kept counts, rejection counts per reason, final frame size, fps, duration, `quality` summary and the warnings above.
8. Remove the `target_keyframes=8` odometry-path default; pass the configured count.

**Acceptance:** 60 s 1080p/4K video → between 40 and 90 keyframes, peak RAM < ~1 GB, extraction < 30 s on CPU, timestamps monotonic, no frame larger than `VIDEO_MAX_IMAGE_SIDE`.

---

### WP4 — SfM reconstruction (`tiers/video_sfm.py`)

Refactor `reconstruct_video_sfm` into small functions; each returns data and logs timings.

**4.1 Intrinsics prior.**
Set an initial focal length from the iPhone HFOV (≈ 65°, `fx = (w/2)/tan(hfov/2)`; for portrait video use the matching axis — verify) via `ImageReaderOptions.camera_params` and keep `SIMPLE_RADIAL`, single shared camera. Allow focal-length refinement; keep principal point and extra params fixed (as now).

**4.2 Feature extraction.**
Use SIFT options: `max_image_size` = 1600, `max_num_features` ≈ 8000, RootSIFT normalization, `estimate_affine_shape=False`, `domain_size_pooling=False`. If the installed pycolmap has CUDA, enable GPU; otherwise CPU. Confirm option names with `help(pycolmap.SiftExtractionOptions)` on the installed version (`pycolmap.__version__`) — names differ across releases.

**4.3 Matching.**
- If `n_frames ≤ 120` (default case): **exhaustive matching**. It is cheap at this size and removes the dependence on loop-closure / vocab tree.
- Otherwise sequential matching with `overlap ≥ 15` and **no** `loop_detection` unless a vocab-tree file is present locally (do not trigger a network download implicitly).
- Enable guided matching if available.

**4.4 Mapping.**
- `multiple_models = True`, `max_num_models` ≈ 6; `min_model_size` ≈ 6–8 (not 3).
- Keep `ba_refine_principal_point=False`, `ba_refine_extra_params=False`.
- Raise `init_min_tri_angle` to ≈ 4–6° (parallax requirement) — confirm the property path (`opts.mapper.init_min_tri_angle`).
- Candidate knobs if registration is low: `mapper.min_num_matches` lower (≈ 12–15), `mapper.abs_pose_min_num_inliers` lower, more local BA iterations. Verify names exist in the installed version.

**4.5 Choose and report models.**
- Use **registered** image counts: `num_reg_images()` (or iterate registered ids only). Fix F11 everywhere `num_images()` / `.images` is used.
- Sort models by registered count. Keep **all** models with ≥ `min_model_size`; the largest is "primary".
- Compute registration ratio = registered / n_frames and the mean reprojection error. Fill `RegistrationInfo`.

**4.6 Fallback chain** (stop at the first that reaches `VIDEO_MIN_REG_RATIO` for the **union** of models, or at least ≥ 2 rooms' worth):
1. exhaustive + default options;
2. exhaustive + relaxed mapper options (4.4 knobs) and more features (≈ 12000);
3. *(optional upgrade)* learned features/matcher (e.g. SuperPoint + LightGlue via `hloc` or `lightglue` if installable) — much better on plain plaster and dim light; torch is already a dependency through the depth model;
4. if still below threshold: return `status="degraded"` with the best model(s) and warnings **or** `status="failed"` if < 3 registered frames. **Do not** switch to the synthetic box.

**4.7 Point extraction (fix F14).**
Build `points = [(point3D_id, xyz, track_len, error)]`; keep only `error < 1.5 px`, `track_len ≥ 3`; then remove statistical outliers (e.g. remove points beyond ~3 MAD from the median in each axis, or Open3D-style radius outlier removal). Keep the ids with the points all the way through scaling and room assignment; no positional index tricks.

**4.8 Temp directory.**
`ingest_video_capture` currently uses `tempfile.mkdtemp` and never cleans up. Use `out/<name>/sfm/` (stable, inspectable), reuse it if a cache hash of (video mtime + options) matches, and clean images older than N runs.

**4.9 Timing.**
Log per-step time (extract, match, map, scale, segment). Enforce `VIDEO_STAGE_TIMEOUT_S`; on timeout return `failed` with the reason instead of hanging.

**Acceptance (WP4):**
- Log shows `registered / frames` computed from registered images only.
- On the sample 60 s video: the registration ratio is reported and ≥ 0.6, **or** the result is `degraded` with a clear warning; never silent.
- Both rooms of a 2-room clip survive as models/segments (no discarded fragments).
- Total SfM time for ≤ 90 frames on CPU ≲ 6 min (target; report actual).

---

### WP5 — Gravity alignment and metric scale

Do this **before** any metric-threshold plane fitting.

**5.1 Gravity from cameras (fixes F10).**
- For every registered image get the camera-to-world rotation; the camera "up" direction in the world frame is the world-space image of the camera's −y axis (COLMAP/OpenCV: +y is down).
- `up = normalize(median/mean of the unit up-vectors)` (use a robust mean: weighted by inlier count, drop outliers > 30° from the median).
- Rotate the whole reconstruction (points + camera centers) so `up → +Z`. Handheld phone video held roughly upright makes this reliable.
- **Refine** with the floor/ceiling normal later (WP5.4) only if it agrees within ~10°.

**5.2 Scale-free plane sanity (optional but recommended).**
Normalize the cloud to unit scale (divide by median camera-to-point distance) before RANSAC, so thresholds are relative (e.g. 1.5 % of that distance). Alternatively run RANSAC **after** scale (5.3). Either way, **never** apply metric thresholds to un-scaled data.

**5.3 Metric scale from fused cues (fixes F9).**
Compute independent scale estimates; each carries a value, sample count, spread (MAD), and a confidence:

| Cue | How | Notes |
|---|---|---|
| **A. Metric depth model (primary)** | For ~10 frames **evenly spread over the clip**, run the depth model **without** conditioning it on camera-height/ceiling priors. Per frame, `ratio = metric_depth(px) / sfm_depth(px)` over SfM-tracked pixels; take the median per frame, then the median across frames; spread across frames = reliability. | If `DepthEngine.predict` mandatory args force priors, add a `priors=None` path or a raw-depth method. Confirm in `photo.py`. Use only pixels with SfM depth > 0.1 and metric depth in a sane range. |
| **B. Camera height** | Camera centers above the floor plane ≈ 1.2–1.6 m for handheld. | Weak (±20 %). Use only as a sanity bound. |
| **C. Door height** | If a door-sized opening with floor-to-head extent is found in the SfM points/frames, door ≈ 2.0–2.1 m. | Medium; needs opening detection (later). Leave a hook. |
| **D. User reference (best)** | `--reference-height` / `KNOWN_REFERENCE_M` (ceiling height, door height, or a wall length). | If supplied, this wins: `method="reference"`, confidence 0.9+. |

**Fusion rule:**
1. If D exists → use it; log others as cross-checks.
2. Else use A. Accept A when ≥ 5 frames contribute and the across-frame spread < 15 %.
3. Else use B/C as `method="prior"`, confidence ≤ 0.3, and set `relative_uncertainty ≥ 0.25`.
4. If two cues disagree by > 20 %: **do not overwrite the primary with the prior.** Keep the higher-confidence cue, widen `relative_uncertainty`, and add warning `"scale_cues_disagree"` with both values.
5. Remove the code at L196–206 that replaces the scale with `2.6/sfm_height`.

The result (`ScaleInfo`) must be propagated so calibration (later) can widen intervals according to `relative_uncertainty` instead of using a fixed ±6.5 %.

**5.4 Floor/ceiling detection (after scaling).**
In `planes.py` (video tier only):
- Floor/ceiling candidates must satisfy camera-based constraints: floor plane is **below all camera centers** (median camera height above it in 0.9–2.0 m); ceiling is **above all camera centers**.
- Require a minimum inlier fraction and horizontal extent (> ~2 m in both axes) so a table top or a stray cluster is not accepted.
- If the ceiling is not found, return `ceiling_height = None` with `provenance="not_observed"`. **Delete** the "set to 2.40" clamp at `planes.py` L319–321 for video; the geometry node must then use a prior only with the `method="prior"` tag and a widened interval.
- Seed RANSAC: use `rng = np.random.default_rng(seed)` instead of `np.random.choice` (F20).

**5.5 Sanity checks to emit as warnings (not silent corrections):**
- scaled ceiling outside 2.0–4.0 m,
- camera height above floor outside 0.8–2.2 m,
- floor area per room < 3 m² or > 60 m²,
- scale factor changes > 15 % between the first and second half of the clip (drift).

**Acceptance (WP5):**
- No code path divides by `sfm_height` to force 2.6 m.
- With `--reference-height`, reported ceiling is within ~3 % of the supplied value on a clip where the ceiling plane is found.
- Without reference, `ScaleInfo.method`, `confidence`, and `relative_uncertainty` are always populated.
- The "Scaled ceiling 0.02" situation produces `scale_cues_disagree` / `scale_unreliable` with values, not a silent prior.

---

### WP6 — Room segmentation (no LLM required)

Replaces Phase 3 in `video_sfm.py` (the per-keyframe LLM loop at L230–260).

**6.1 Covisibility graph.**
Nodes = registered images. Edge weight = number of 3D points observed by both images (from tracks). Keep edges only between images that are temporally within a window **or** share many points.

**6.2 Community detection.**
Run weighted community detection (Louvain / `networkx.algorithms.community.louvain_communities`, or greedy modularity) on the covisibility graph. Rooms appear as dense clusters joined by thin "doorway" bridges.

**6.3 Temporal regularization.**
- Order frames by time; assign each frame its community.
- Smooth: remove segments shorter than ~4 s (merge into the neighbour with higher covisibility), and median-filter isolated label flips.
- Re-merge communities that are non-contiguous in time **only** if their camera positions overlap strongly in XY (revisiting the same room); otherwise keep separate.
- Replace the magic `+10.0` end time with `video duration`.

**6.4 Geometric validation per segment.**
Project the segment's points to XY. Reject (merge) segments whose observed floor/wall footprint is < ~3 m² or which have fewer than ~300 points; flag segments whose footprint is huge relative to the camera path.

**6.5 Doorway transitions.**
For each pair of consecutive segments, record a `transition` = {frame ids around the boundary, camera positions, time}. Because all rooms come from one SfM model, **they already share one world frame**: output per-room points **without recentring**, so the later stitch stage can use identity transforms. If rooms come from **different** SfM models (fragmentation), mark `transition["aligned"]=False` so the stitching stage knows to align them via the doorway frames.

**6.6 Room ids and names.**
- Ids are always `room_00`, `room_01`, … in time order (consistent with `room_discovery.py`).
- `room_type` is optional metadata, never an id.
- Optional naming step (only if LLM is enabled and available): for each segment, pick 4 well-spread frames, make a 2×2 collage, make **one** call with a prompt like: *"These four images are from the same room of a home. Classify the room type. Choose one of: living_room, bedroom, kitchen, bathroom, hallway, dining_room, balcony, storage, other. Respond with JSON only."* Cache it. If the call returns `ok=False`, set `room_type=None` and continue. Total LLM calls = number of rooms (≈ 1–5), well under any free quota.
- Honor `rooms.json` override (`segments` with `room_id`, `start_time`, `end_time`) if present next to the video or passed via `--rooms-json`; validate its schema and ranges.

**6.7 Point assignment.**
Use track visibility as now (≥ 70 % of observations from one room → that room). Points with no ≥ 60 % majority (seen from both sides of a doorway) go to `transitions[i]["points"]` instead of being forced into the "max" room.

**Acceptance (WP6):**
- On the 2-room sample video (`1bhKRoom.mp4`), output has **≥ 2** `VideoRoom`s with ids `room_00`, `room_01`; each has ≥ 300 points and a plausible footprint.
- Segmentation produces the same result with `--no-llm`.
- Unit test with a synthetic two-room scene (two point-box clusters joined by a doorway, camera path passing through) yields exactly two segments.

---

### WP7 — Failure policy, `ingest_video_capture`, and contract

**7.1 New return contract.**
`ingest_video_capture(path) -> VideoReconstruction` (WP3 types). Update `nodes/ingest.py` (lines ≈ 130–160) to: write each `VideoRoom.points` to `data/cache/cloud_<room_id>.npy`, set `state.rooms`, `state.point_clouds`, `state.device_meta` (include `scale`, `registration`, `llm`, `warnings`), and append `VideoReconstruction.warnings` to `state.warnings`.

**7.2 Remove synthetic substitution (F16, F17, F18).**
- Delete the synthetic box in the "file not found" branch and in `recover_walkthrough_point_cloud` fallbacks. Missing file → raise `FileNotFoundError` with a clear message.
- `status="failed"` → stop the pipeline for that capture with an explicit error and the reasons; do not continue with fake geometry. Allow synthetic data only when `ALLOW_SYNTHETIC=true`, and in that case tag the output `provenance="synthetic"`, add warning `"SYNTHETIC GEOMETRY — NOT A MEASUREMENT"`, and make the QA critic fail the run.
- `nodes/geometry.py` L19 (`_generate_synthetic_box` when no cloud) and `planes._create_fallback_room`: same policy — no silent fallback outside `ALLOW_SYNTHETIC`.
- Remove the `"singleroom" in parent_name` block and its (5.35, 5.90) priors. Any room-size prior must be generic and tagged as a prior.

**7.3 Monocular odometry path.**
Keep `recover_walkthrough_point_cloud` only as a **second-tier** fallback when SfM is `failed`, and mark every result from it `status="degraded"`, `scale.method="prior"` unless a reference is supplied. Its input must come from the new keyframe extractor. It must never fall through to a synthetic box.

**7.4 Metadata.**
Always return populated `intrinsics` (the SfM camera or the estimated one), `video_meta`, `registration`, `scale`, `timings`, and `llm` stats. Remove the placeholder `scale_recovery` string and `"gate_g7_target"` literal; those claim an accuracy that is not computed.

**7.5 QA hooks (data only; QA node itself later).**
Expose booleans/flags that the QA critic can read: `status`, `registration.ratio`, `scale.method`, `scale.confidence`, `llm.disabled_reason`, `warnings`, `provenance`.

**Acceptance (WP7):**
- `grep -rn "_generate_synthetic_box\|singleroom\|synthetic_fallback" src/` shows matches only behind `ALLOW_SYNTHETIC`.
- A deliberately corrupt/blank video yields `status="failed"` and a clear error; **no** plan is produced from invented data.
- `ingest.py` no longer contains hard-coded `rooms_dict["room_01"]` for the video path.

---

### WP8 — `geometry/planes.py` (video-relevant parts only)

Keep wall/polygon logic for the later design pass. Change only:
1. Seed RANSAC (F20) with a deterministic `Generator`; accept `seed` argument.
2. Video-tier ceiling handling per WP5.4 (no silent 2.40).
3. Accept optional `camera_positions` and `up` hints in `fit_room_planes(...)`; use them for floor/ceiling selection.
4. Thresholds (0.035 / 0.04 / 0.30) stay in metres, which is valid because the cloud is now metric.
5. Return provenance such as `"measured" | "bbox" | "prior" | "not_observed"` per quantity and propagate `scale.relative_uncertainty` into the interval width when scale method is not `"reference"`.

**Acceptance:** two runs on the same cloud give identical output; ceiling height is `None`/`prior` (tagged) instead of 2.40 when no ceiling plane is found.

---

### WP9 — Observability

- One structured `run_log` entry for the video stage: timings per step, counts, `status`, warnings, LLM stats.
- Console progress lines at INFO level for each step (`[video] keyframes 62 kept / 480 sampled`, `[video] SfM exhaustive: 58/62 registered, 1 model`, `[video] scale 1.83 via depth_fused conf 0.72 ±9 %`, `[video] rooms: 2`).
- Replace the many `print(...)` calls with `logging`.
- Truncate third-party error text in logs.

---

## 5. Tests

Create `tests/video/`:

| Test | What it checks |
|---|---|
| `test_llm_errors.py` | Mock HTTP server: long-reset 429 → 1 request, no sleep; short 429 → 1 retry; 5xx → ≤ 2 retries; 401 → no retry; parse error → no extra request; provider chain falls through to the next provider; budget exceeded → `status="budget"`. |
| `test_llm_no_fabrication.py` | With no provider available, `generate_structured` returns `ok=False`; no key named `damage_findings`, `room_type`, etc. is ever produced by the client itself. |
| `test_keyframes.py` | Synthetic video with a static section, a fast-pan section, and a dark section → static duplicates dropped, blurred/dark frames rejected, count within limits, timestamps monotonic. |
| `test_scale_fusion.py` | Feed fake cue sets: agreeing cues → fused value; disagreeing cues → warning + wider uncertainty, **never** forced to a prior; reference overrides. |
| `test_gravity.py` | Synthetic cameras with known tilt → recovered up vector within 2°. |
| `test_room_segmentation.py` | Synthetic two-room scene → exactly two segments, ids `room_00/room_01`, contiguous in time. |
| `test_failure_policy.py` | Blank video → `status="failed"`; `ALLOW_SYNTHETIC=false` never yields synthetic provenance. |
| `test_determinism.py` | Same cloud/seed → identical planes. |

Smoke runs (PowerShell):

```powershell
python main.py Data\1BHKRoom\1bhKRoom.mp4 --no-llm
python main.py Data\1BHKRoom\1bhKRoom.mp4 --reference-height 2.7
```

Check `out\<name>\plan.json` and `run_log.json` for: `rooms ≥ 2`, `scale.method`, `registration.ratio`, `llm.calls == 0` for `--no-llm`, no `synthetic` provenance.

---

## 6. Order of work and definition of done

1. Task 0 (find the real `video_sfm.py`; one copy only).
2. WP1 + WP2 (client, settings) — unblocks every run; ship first.
3. WP3 (keyframes).
4. WP4 (SfM).
5. WP5 (gravity + scale).
6. WP6 (rooms).
7. WP7 + WP8 (contract, failure policy, planes tweaks).
8. WP9 + tests.

**Done when**
- A 60 s video completes end-to-end on a quota-exhausted machine with no LLM stall.
- Every output carries `scale.method`, `scale.confidence`, `registration.ratio`, and `warnings`.
- No fabricated or synthetic data can reach `plan.json` unless explicitly enabled and tagged.
- The 2-room sample produces ≥ 2 rooms.
- Tests above pass.

---

## 7. Notes and risks

- **pycolmap API differs by version.** Run `python -c "import pycolmap; print(pycolmap.__version__)"` and inspect signatures (`help(pycolmap.extract_features)`, `help(pycolmap.match_exhaustive)`, `help(pycolmap.incremental_mapping)`) before writing option names. Names above are the common ones, not guaranteed.
- **Dim, plain-plaster rooms are hard for SIFT.** If registration stays low after WP4.6 step 2, the learned-matcher step is the best upgrade; guidance to the user on capture (walk, don't pivot; light on; avoid glare from bulbs/windows) should be shown when `video_mostly_rotation` or low registration warnings fire.
- **Free-tier limits change.** Provider names, limits, and model ids in WP1 must be verified in each provider's console; none are hard-coded.
- **Depth-model scale accuracy** is typically within ~10–15 % indoors and can be worse on unusual scenes; that is why `--reference-height` exists and why uncertainty is reported.
- **Do not paste API keys** into logs, tests, or the repo.
