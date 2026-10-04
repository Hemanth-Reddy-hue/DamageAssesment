# AreaMap

Turn an iPhone capture of a property (photos, a walkthrough video, or LiDAR) into a dimensioned floor plan, a damage assessment, and a repair scope — with an uncertainty interval on every number.

- **Runs fully locally.** No cloud LLM, no API keys, no request quotas. Small models come from Hugging Face and are downloaded automatically the first time you run `main.py`.
- **Never invents geometry.** If a capture cannot be reconstructed, the run fails with a clear reason or is flagged as degraded. Synthetic placeholder rooms are disabled unless you explicitly enable them for demos.
- **Honest numbers.** Every output records how the metric scale was obtained, how many frames registered, and which warnings fired.

---

## Status of this document

This README describes the target behaviour after the video-tier fixes in `VIDEO_TIER_FIX_SPEC.md` are applied. Items marked **[in progress]** are specified but may not be in the code yet. Check `run_log.json` after a run to see what actually executed.

| Area | State |
|---|---|
| Photo / LiDAR tiers, plane fitting, openings, damage, scope, QA, export | Existing (known issues are tracked separately) |
| Video tier: COLMAP reconstruction | Existing; being hardened **[in progress]** |
| Video tier: keyframe selection, scale fusion, room segmentation, failure policy | **[in progress]** |
| Local Hugging Face models loaded at startup (replaces the cloud LLM client) | **[in progress]** |

---

## 1. Quick start (Windows, PowerShell)

Requirements: Python 3.10 or newer, about 3 GB of free disk for model weights, and 8 GB RAM or more. A GPU is optional.

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (CPU build of PyTorch shown; see section 3 for GPU)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. (Optional) copy the environment template and edit it
Copy-Item .env.example .env

# 4. Run on a video
python main.py Data\1BHKRoom\1bhKRoom.mp4
```

On the **first run**, `main.py` downloads the local models into the Hugging Face cache and loads them before processing starts. You will see progress bars. Later runs start from the cache and work offline.

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window.

---

## 2. What happens when you run `main.py`

```
python main.py <capture>
  │
  ├─ 1. Load settings (.env)
  ├─ 2. Model startup  ← local Hugging Face models are downloaded if missing, then loaded
  │        • depth model (metric depth for scale recovery)
  │        • room classifier (CLIP / SigLIP, names rooms)
  │        • optional local vision-language model (off by default)
  ├─ 3. Detect tier (photo / video / lidar)
  └─ 4. Run the pipeline and write out/<FolderName>/
```

Model startup is **soft-failing for optional models**: if the room classifier cannot be loaded (no internet on the first run, disk full), rooms are named `Room 1`, `Room 2`, … and the run continues with a warning. The depth model is needed for scale recovery on the video tier; if it is unavailable the run falls back to a clearly flagged prior or a user-supplied reference height (see section 6).

Turn the model startup off for a quick geometry-only run:

```powershell
python main.py Data\1BHKRoom\1bhKRoom.mp4 --no-local-models
```

---

## 3. Dependencies and local models

### 3.1 `requirements.txt` (relevant lines)

Libraries go in `requirements.txt`. Model **weights** are not pip packages: they are fetched from the Hugging Face Hub by `transformers` the first time they are needed and cached on disk.

```text
numpy
opencv-python
pycolmap
scipy
networkx
pyyaml
pillow
tqdm

# local models
torch
transformers
huggingface_hub
accelerate
safetensors
sentencepiece

# pipeline / agents (existing)
langgraph
langchain-core
```

Notes:

- `torch` is listed so a plain `pip install -r requirements.txt` works, but on Windows it is better to install it first with the CPU or CUDA index URL (see the quick start) so you get the right build.
- GPU (NVIDIA): replace the torch line in the quick start with the command from the PyTorch "Get Started" page for your CUDA version. Set `MODELS_DEVICE=cuda`.
- Remove the cloud client packages you no longer use (`langchain-google-genai`, `langchain-anthropic`, `langchain-openai`). The pipeline does not call any hosted model.

### 3.2 Which models are used

| Role | Default model (Hub id) | Size (approx.) | Required? |
|---|---|---|---|
| Room naming | `openai/clip-vit-base-patch32` | ≈ 0.6 GB | Optional (rooms get numbered names without it) |
| Metric depth | the depth model configured in `tiers/photo.py` | varies | Needed for scale recovery unless `--reference-height` is given |
| Local vision-language model | disabled by default; e.g. `HuggingFaceTB/SmolVLM-500M-Instruct` | ≈ 1–2 GB | Optional, for future damage descriptions |

Model ids are settings, not code. Confirm an id exists on huggingface.co before changing it, and prefer small variants on CPU.

### 3.3 Hugging Face token (optional)

Public models download without a token. A token is only needed for **gated** models (for example some Gemma releases, where you must accept the licence on the model page first) and gives higher download rate limits.

```env
HF_TOKEN=your_token_here
```

Create it at huggingface.co → Settings → Access Tokens (a read-only token is enough). Never commit it.

### 3.4 Cache location and offline use

- Default cache: `C:\Users\<you>\.cache\huggingface`. Change it with `HF_HOME`.
- After the first successful download you can run with no internet: set `HF_HUB_OFFLINE=1`.
- To download everything in advance (for example before travelling or a demo):

```powershell
python -m areamap.models.prefetch
```

- Windows may print a warning about symlinks in the cache. It is harmless. Silence it with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

---

## 4. Configuration (`.env`)

Copy `.env.example` to `.env`. Every value has a default; set only what you want to change.

```env
# --- Local models ---
MODELS_PRELOAD=true                       # load models at startup
MODELS_DEVICE=auto                        # auto | cpu | cuda
ROOM_CLASSIFIER_MODEL=openai/clip-vit-base-patch32
ROOM_CLASSIFIER_MIN_SCORE=0.30            # below this, the room is named "Room N"
LOCAL_VLM_MODEL=                          # empty = disabled
HF_TOKEN=                                 # optional, gated models only
HF_HOME=                                  # optional cache folder
HF_HUB_OFFLINE=0                          # 1 = never touch the network

# --- Video tier ---
VIDEO_MAX_FRAMES=90
VIDEO_MAX_IMAGE_SIDE=1600
VIDEO_MATCHER=auto                        # auto | exhaustive | sequential
VIDEO_MIN_REG_RATIO=0.60
VIDEO_STAGE_TIMEOUT_S=600

# --- Scale ---
KNOWN_REFERENCE_M=                        # optional, e.g. 2.7
KNOWN_REFERENCE_KIND=                     # ceiling_height | door_height | wall_length

# --- Safety ---
ALLOW_SYNTHETIC=false                     # demos/tests only; tags output as synthetic
```

`.env` must be listed in `.gitignore`. Commit only `.env.example`.

---

## 5. Usage

```powershell
# Video walkthrough
python main.py Data\1BHKRoom\1bhKRoom.mp4

# A folder (the tier is detected from its contents)
python main.py Data\HOUSE1

# Give the pipeline a real measurement (recommended for accurate metres)
python main.py Data\1BHKRoom\1bhKRoom.mp4 --reference-height 2.7

# Provide room boundaries yourself
python main.py Data\1BHKRoom\1bhKRoom.mp4 --rooms-json Data\1BHKRoom\rooms.json

# Geometry only, no local models
python main.py Data\1BHKRoom\1bhKRoom.mp4 --no-local-models
```

Running `python main.py` with no arguments opens the interactive prompt.

| Flag | Meaning |
|---|---|
| `--reference-height M` | Known ceiling height in metres. Overrides estimated scale (best accuracy). |
| `--rooms-json PATH` | Manual room segments (see below). |
| `--no-local-models` | Skip loading Hugging Face models. Rooms get numbered names; scale uses reference or a flagged prior. |
| `--allow-synthetic` | Demo only. Allows placeholder geometry, tagged `provenance: synthetic`, and QA fails the run. |

`rooms.json` format:

```json
{
  "segments": [
    {"room_id": "room_00", "start_time": 0.0,  "end_time": 28.5},
    {"room_id": "room_01", "start_time": 28.5, "end_time": 60.0}
  ]
}
```

---

## 6. How the video tier works

1. **Keyframes.** The clip is read sequentially. Blurred, over-exposed, dark, or featureless frames are rejected. New keyframes are chosen when the camera has moved enough (parallax), not at fixed times. Frames are downscaled before reconstruction.
2. **Structure from Motion (COLMAP).** Features are extracted with a focal-length prior for an iPhone-class camera. Matching is exhaustive for up to about 120 frames. All reconstructed fragments are kept, not just the largest. If too few frames register, a relaxed second attempt runs. If that also fails, the run reports `degraded` or `failed`.
3. **Gravity.** "Up" is estimated from the camera orientations, then the model is rotated so up is +Z.
4. **Metric scale.** COLMAP has arbitrary units. Scale is obtained from, in order of trust: a user reference (`--reference-height`), the metric depth model sampled across the clip, then weak priors. The chosen method and its uncertainty are recorded. Disagreeing cues produce a warning, never a silent override.
5. **Rooms.** Rooms are found from the covisibility graph of frames (frames that see the same points group together; doorways are the thin links between groups). Segments are cleaned up in time, and ids are always `room_00`, `room_01`, … The local classifier then names each room from four sample frames. No model is needed to split rooms.
6. **Hand-off.** Points for every room stay in one shared world frame, so later stitching does not need to guess alignment.

### Capture tips (these matter more than any setting)

- **Walk through the space; do not stand still and pivot.** Pure rotation gives no depth information and breaks reconstruction.
- Move slowly and steadily. Keep the phone upright.
- Turn the lights on. Avoid pointing straight at bulbs and bright windows.
- Include textured surfaces (furniture, door frames, wall edges). Plain plaster alone is hard to track.
- Pass through doorways slowly, and look back at the room you just left for a second.
- Say or note one real measurement (ceiling height or a door width) and pass it with `--reference-height`.

---

## 7. Outputs

Each run writes `out\<FolderName>\`:

| File | Content |
|---|---|
| `plan.json` | Full machine-readable plan (rooms, walls, openings, damage, scope, intervals). Schema: `schema/capture_v1.json`. |
| `plan.svg` | Dimensioned 2D floor plan with openings and damage markers. |
| `run_log.json` | Stage timings, tier, quality summary, warnings, QA results. |

Check these fields before trusting any number:

| Field | What it tells you |
|---|---|
| `status` | `ok`, `degraded`, or `failed`. |
| `registration.ratio` | Share of keyframes that were reconstructed. Below about 0.6 means a weak model. |
| `scale.method` / `scale.confidence` / `scale.relative_uncertainty` | How metres were obtained. `prior` means a guess; supply `--reference-height`. |
| `provenance` | `measured`, `bbox`, `prior`, `not_observed`, or `synthetic`. Anything other than `measured` needs a human look. |
| `warnings` | Everything the pipeline is unsure about, in plain language. |
| `qa_report` | Automatic consistency checks (closed polygons, surface references, ceiling range). |

---

## 8. Project layout

```
main.py                      CLI entry
src/areamap/
  config.py                  settings (.env)
  state.py                   CaptureState and typed models
  models/                    local model manager and prefetch   [in progress]
  nodes/                     ingest, geometry, openings, stitch, calibrate,
                             damage, concealed, scope, qa_critic, export
  tiers/                     photo.py, video.py, video_sfm.py, lidar.py
  geometry/                  planes, scene_geometry, registration, posegraph,
                             room_discovery, drift, calibration, uncertainty
  rules/ , catalog/          forensic rules, scope item catalog
  render/                    SVG plan renderer
  mcp_server/                MCP server exposing the pipeline
schema/capture_v1.json       output schema
Data/                        sample captures
out/                         run outputs
data/cache/                  point-cloud and model caches
tests/                       unit and smoke tests
```

---

## 9. Testing

```powershell
pytest -q
pytest tests\video -q
```

Smoke run on the two-room sample without any model downloads:

```powershell
python main.py Data\1BHKRoom\1bhKRoom.mp4 --no-local-models --reference-height 2.7
```

Expected: at least two rooms (`room_00`, `room_01`), `scale.method = reference`, no `synthetic` provenance in `plan.json`.

---

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| First run is slow and prints download bars | Normal. Models are being downloaded once. Use `python -m areamap.models.prefetch` to do it ahead of time. |
| `OSError` / connection error while loading a model | No internet and the model is not cached. Connect once, or run with `--no-local-models`. |
| "gated repo" / 401 from the Hub | The model needs licence acceptance. Accept it on its model page and set `HF_TOKEN`. |
| Out-of-memory when loading | Use a smaller model, set `MODELS_DEVICE=cpu`, and leave `LOCAL_VLM_MODEL` empty. |
| Rooms are named "Room 1", "Room 2" | Classifier disabled, not loaded, or below `ROOM_CLASSIFIER_MIN_SCORE`. Geometry is unaffected. |
| `status: degraded`, low `registration.ratio` | Weak footage. Re-capture by walking, with lights on; check `warnings`. |
| COLMAP log shows "Linear solver failure ... Cholesky" | Poorly conditioned frames (little parallax, plain walls). Re-capture, or lower `VIDEO_MAX_FRAMES` and raise image quality; see capture tips. |
| `scale.method = prior` | No reference and the depth cues disagreed. Pass `--reference-height`. |
| Only one room found in a multi-room video | Check `registration` (rooms may have registered as separate fragments) or provide `--rooms-json`. |
| Run takes very long | Check the timings in `run_log.json`. Reduce `VIDEO_MAX_FRAMES` or `VIDEO_MAX_IMAGE_SIDE`. |
| PowerShell: `grep` not found | Use `Select-String`, for example `Get-ChildItem src -Recurse -Filter *.py \| Select-String "pattern"`. |

---

## 11. Security and privacy

- Everything runs on your machine. Photos and videos are not uploaded anywhere by the pipeline.
- Model downloads contact huggingface.co only. After caching, set `HF_HUB_OFFLINE=1` to guarantee no network use.
- Keep `.env` and any token out of version control and out of screenshots and chats. If a token leaks, revoke it in your Hugging Face settings and create a new one.

---

## 12. Known limitations

- Metric accuracy depends on scale. Without a reference measurement, depth-based scale is typically within about 10–15 % indoors and can be worse; the plan reports its own uncertainty.
- Dim, featureless rooms can fragment the reconstruction. The pipeline reports this instead of hiding it.
- Opening detection, damage detection, and scope generation are being reviewed separately; treat their outputs as drafts until that pass is complete.

---

## 13. License

Add your license here.
