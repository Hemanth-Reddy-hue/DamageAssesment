"""Metric monocular depth estimation engine (closes the Gate-G6 metric gap).

Provides metric depth maps from single RGB images using a cascade of backends:

  Backend 1 – Depth Anything V2 (HuggingFace transformers + torch)
    Model: depth-anything/Depth-Anything-V2-Small-hf
    Output: affine-invariant relative depth → rescaled to metric via architectural
            anchors (floor=0, seam=camera_height, ceiling=ceiling_height_prior).

  Backend 2 – MiDaS via OpenCV DNN (no torch, uses MiDaS ONNX / caffe model)
    Requires models/midas_v21_small_256.onnx (downloaded on first use).
    Falls back gracefully to Backend 3 if model not cached.

  Backend 3 – Geometric triangulation (always available, CPU-only)
    Uses known camera intrinsics + detected floor seam + camera height from
    scene_geometry.py to build a per-pixel depth estimate purely from ray
    geometry.  This is a prior-based estimate but is grounded in the actual
    image geometry rather than a hardcoded box.

Usage:
    from areamap.geometry.depth_engine import DepthEngine

    engine = DepthEngine()   # auto-selects best available backend
    depth_m = engine.predict(
        image_bgr,          # (H, W, 3) uint8 BGR numpy array
        intrinsics,         # {fx, fy, cx, cy, width, height}
        camera_height=1.45, # metres – from scene_geometry.estimate_camera_height
        seam_v=420.0,       # y-pixel of floor-wall seam
        ceiling_height=2.50 # metres – architectural prior
    )
    # depth_m: (H, W) float32, metric metres (0 = invalid / too close)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _is_offline() -> bool:
    """Check the offline flag from settings (loaded from .env).
    
    Uses the config system so .env is always respected.
    Falls back to the OFFLINE env var if settings can't be imported.
    """
    try:
        from areamap.config import settings
        return settings.offline
    except Exception:
        # Fallback: check env var. Default to FALSE so models load if not explicitly blocked.
        return os.environ.get("OFFLINE", "0").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEPTH_ANYTHING_V2_HF_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_MIDAS_ONNX_URL = (
    "https://github.com/isl-org/MiDaS/releases/download/v3_1/"
    "dpt_swin2_tiny_256.onnx"
)
_MIDAS_ONNX_FILENAME = "dpt_swin2_tiny_256.onnx"

# Maximum raw model output value treated as "valid" (normalisation bound)
_DEPTH_PERCENTILE_LO = 2
_DEPTH_PERCENTILE_HI = 98

# Metric range for clipping (realistic indoor residential bounds)
METRIC_MIN_M = 0.20
METRIC_MAX_M = 10.00
METRIC_INDOOR_MAX_M = 10.00


# ---------------------------------------------------------------------------
# Backend 1: Depth Anything V2 via transformers + torch
# ---------------------------------------------------------------------------

def _try_load_depth_anything_v2(device: str = "cpu"):
    """Attempt to load Depth-Anything-V2-Metric-Indoor from HuggingFace cache.

    This is a METRIC model: its output is already in meters, no rescaling needed.
    It loads from local cache if available (no internet required once downloaded).

    Returns (pipeline_fn, 'depth_anything_v2') or (None, None) on failure.
    """
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        # Use settings.offline (reads from .env) — NOT a hardcoded default.
        # This ensures the model is used when it is cached locally,
        # even when OFFLINE=1 was intended only to block LLM/API calls.
        offline = _is_offline()
        log.debug("[DepthEngine] offline=%s, attempting to load: %s", offline, _DEPTH_ANYTHING_V2_HF_ID)

        # If offline, use local_files_only to load from cache without network.
        # If not offline, allow download on first use.
        pipe = hf_pipeline(
            task="depth-estimation",
            model=_DEPTH_ANYTHING_V2_HF_ID,
            device=device,
            local_files_only=offline,  # Use cache only when offline
        )
        log.info("[DepthEngine] Depth-Anything-V2-Metric-Indoor loaded on %s (offline=%s)", device, offline)
        return pipe, "depth_anything_v2"
    except Exception as exc:
        log.warning("[DepthEngine] Could not load Depth-Anything-V2-Metric-Indoor: %s", exc)
        return None, None


def _predict_depth_anything(pipe, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Run Depth-Anything-V2-Metric-Indoor and return a (H, W) float32 METRIC depth map.

    The Metric Indoor variant outputs depth directly in meters — no affine rescaling needed.
    The 'predicted_depth' key contains the raw metric tensor.
    """
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = pipe(pil_img)

        # 'predicted_depth' is the raw model output tensor (metric metres for Metric-Indoor)
        if "predicted_depth" in result:
            depth_tensor = result["predicted_depth"]
            if hasattr(depth_tensor, "cpu"):
                depth_tensor = depth_tensor.cpu().numpy()
            elif hasattr(depth_tensor, "numpy"):
                depth_tensor = depth_tensor.numpy()
            else:
                depth_tensor = np.array(depth_tensor)
            depth = np.squeeze(depth_tensor).astype(np.float32)
        else:
            # Fallback: convert PIL image to float array
            depth = np.array(result["depth"], dtype=np.float32)

        # Resize to original image resolution if needed
        if depth.shape[:2] != image_bgr.shape[:2]:
            depth = cv2.resize(
                depth, (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Sanity check: metric indoor depth should be in [0.1, 20] metres
        valid_ratio = float(np.mean((depth > 0.1) & (depth < 20.0)))
        if valid_ratio < 0.1:
            log.warning("[DepthEngine] Suspiciously few valid depth pixels (%.1f%%). "
                        "Model output range: [%.3f, %.3f]. Skipping frame.",
                        valid_ratio * 100, float(depth.min()), float(depth.max()))
            return None

        log.debug("[DepthEngine] Depth: min=%.3f median=%.3f max=%.3f m (valid=%.1f%%)",
                  float(depth.min()), float(np.median(depth)), float(depth.max()), valid_ratio * 100)
        return depth
    except Exception as exc:
        log.warning("[DepthEngine] Depth-Anything inference failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Backend 2: MiDaS via OpenCV DNN (ONNX)
# ---------------------------------------------------------------------------

def _get_midas_model_path(models_dir: Path) -> Optional[Path]:
    """Return path to MiDaS ONNX file, downloading if necessary."""
    model_path = models_dir / _MIDAS_ONNX_FILENAME
    if model_path.exists():
        return model_path

    offline = os.environ.get("OFFLINE", "1").lower() in ("1", "true", "yes")
    if offline:
        log.debug("OFFLINE=1 – skipping MiDaS download")
        return None

    try:
        import urllib.request
        log.info("Downloading MiDaS ONNX to %s …", model_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MIDAS_ONNX_URL, model_path)
        log.info("MiDaS ONNX downloaded successfully")
        return model_path
    except Exception as exc:
        log.warning("Could not download MiDaS: %s", exc)
        return None


def _try_load_midas(models_dir: Path):
    """Attempt to load MiDaS via OpenCV DNN.

    Returns (net, input_size, 'midas') or (None, None, None) on failure.
    """
    model_path = _get_midas_model_path(models_dir)
    if model_path is None:
        return None, None, None
    try:
        net = cv2.dnn.readNet(str(model_path))
        log.info("MiDaS ONNX loaded from %s", model_path)
        return net, 256, "midas"
    except Exception as exc:
        log.debug("Could not load MiDaS: %s", exc)
        return None, None, None


def _predict_midas(net, input_size: int, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Run MiDaS DNN and return a (H, W) float32 relative depth map."""
    try:
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(input_size, input_size),
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        output = net.forward()  # (1, 1, H_out, W_out) or (1, H_out, W_out)
        if output.ndim == 4:
            depth = output[0, 0]
        elif output.ndim == 3:
            depth = output[0]
        else:
            depth = output

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)
    except Exception as exc:
        log.debug("MiDaS inference failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Backend 3: Geometric triangulation (always available)
# ---------------------------------------------------------------------------

def _geometric_depth(
    image_bgr: np.ndarray,
    intrinsics: Dict[str, float],
    camera_height: float,
    seam_v: float,
    ceiling_height: float,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Generate a per-pixel metric depth estimate using pure ray geometry.

    Model:
      - Camera at height h_cam above the floor (z=0 plane).
      - Floor plane: z = 0.
      - Ceiling plane: z = ceiling_height.
      - For each pixel (u, v):
          * elevation angle φ = atan2((v - cy), fy) + pitch_rad
          * if φ > φ_seam  → pixel is on the floor / below seam
              depth = h_cam / tan(φ)   (intersection with z=0 plane)
          * if φ < φ_ceiling  → pixel is near ceiling
              depth = (ceiling_height - h_cam) / tan(-φ)
          * else  → wall region: linear interpolation between floor and ceiling depths

    Returns:
        depth_m: (H, W) float32, metric metres.  0 = invalid.
    """
    H, W = image_bgr.shape[:2]
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Elevation angle at the seam and at the top of the image (including pitch)
    phi_seam = float(np.arctan2(seam_v - cy, fy) + pitch_rad)
    phi_top = float(np.arctan2(0.0 - cy, fy) + pitch_rad)

    # Build pixel coordinate grids
    u_grid = np.arange(W, dtype=np.float32)
    v_grid = np.arange(H, dtype=np.float32)
    UU, VV = np.meshgrid(u_grid, v_grid)

    phi = np.arctan2(VV - cy, fy) + pitch_rad  # (H, W) elevation per pixel

    # Region masks
    floor_mask = phi >= phi_seam                        # below seam → floor
    ceiling_mask = phi <= min(phi_top * 0.3, -0.10)     # near top → ceiling
    wall_mask = ~floor_mask & ~ceiling_mask

    depth_m = np.zeros((H, W), dtype=np.float32)

    # Clamp minimum elevation to 0.05 rad (~2.9 deg) to prevent runaway depths near horizon
    tan_phi_floor = np.tan(np.clip(phi[floor_mask], 0.05, np.pi / 2 - 0.01))
    depth_m[floor_mask] = np.clip(camera_height / tan_phi_floor, METRIC_MIN_M, METRIC_INDOOR_MAX_M)

    # Ceiling region
    tan_phi_ceil = np.tan(np.clip(-phi[ceiling_mask], 0.15, np.pi / 2 - 0.01))
    depth_m[ceiling_mask] = np.clip(
        (ceiling_height - camera_height) / (tan_phi_ceil + 1e-6),
        METRIC_MIN_M, METRIC_INDOOR_MAX_M,
    )

    # Wall region: linear interpolation between typical floor depth at seam
    # and a reasonable far-wall distance
    if np.any(wall_mask):
        phi_s_clamp = max(phi_seam, 0.22)
        d_seam = float(np.clip(camera_height / np.tan(phi_s_clamp), 0.8, METRIC_INDOOR_MAX_M))
        d_wall_far = float(np.clip(d_seam * 1.15, 1.0, METRIC_INDOOR_MAX_M))

        v_in_wall = VV[wall_mask]
        v_wall_min = float(np.where(ceiling_mask)[0].max()) if np.any(ceiling_mask) else 0.0
        v_wall_max = float(np.where(floor_mask)[0].min()) if np.any(floor_mask) else float(H)
        v_span = max(v_wall_max - v_wall_min, 1.0)
        frac = np.clip((v_in_wall - v_wall_min) / v_span, 0.0, 1.0)
        depth_m[wall_mask] = (d_wall_far + frac * (d_seam - d_wall_far)).astype(np.float32)

    return depth_m


# ---------------------------------------------------------------------------
# Affine rescaling: relative depth → metric depth
# ---------------------------------------------------------------------------

def _affine_rescale_to_metric(
    rel_depth: np.ndarray,
    camera_height: float,
    seam_v: float,
    ceiling_height: float,
    intrinsics: Dict[str, float],
    invert: bool = True,
) -> np.ndarray:
    """Rescale a relative (affine-invariant) depth map to metric metres.

    Depth Anything / MiDaS outputs are proportional to 1/Z (disparity-like).
    We recover metric scale using two anchor depths:

      Anchor A: floor at the seam row  → depth_A = camera_height / tan(phi_seam)
      Anchor B: ceiling row (top 10%)  → depth_B = (ceiling_height - camera_height) / tan(phi_ceiling)

    We solve for scale s and shift t such that:
        s * rel(y_floor) + t = depth_A
        s * rel(y_ceiling) + t = depth_B

    Args:
        rel_depth  : (H, W) relative depth map (may be inverse-depth / disparity)
        camera_height: metres
        seam_v     : seam y-pixel
        ceiling_height: metres
        intrinsics : camera dict
        invert     : if True, invert before fitting (DA-V2 / MiDaS output is disparity)

    Returns:
        metric_depth : (H, W) float32 in metres, clipped to [METRIC_MIN_M, METRIC_MAX_M]
    """
    fy = intrinsics["fy"]
    cy = intrinsics["cy"]
    H, W = rel_depth.shape

    if invert:
        # Convert disparity → relative depth
        d_inv = rel_depth.copy()
        d_inv = (d_inv - d_inv.min()) / (d_inv.max() - d_inv.min() + 1e-8)
        rel = 1.0 / (d_inv + 0.01)  # avoid divide-by-zero
    else:
        rel = rel_depth.copy().astype(np.float64)

    # Normalize rel to [0, 1]
    r_lo = float(np.percentile(rel, _DEPTH_PERCENTILE_LO))
    r_hi = float(np.percentile(rel, _DEPTH_PERCENTILE_HI))
    if r_hi - r_lo < 1e-6:
        # Degenerate map – return flat prior
        return _geometric_depth(
            np.zeros((H, W, 3), np.uint8), intrinsics,
            camera_height, seam_v, ceiling_height,
        )
    rel_norm = np.clip((rel - r_lo) / (r_hi - r_lo), 0.0, 1.0)

    # Anchor A: seam row (floor boundary)
    seam_row = int(np.clip(seam_v, 0, H - 1))
    phi_seam = np.arctan2(seam_v - cy, fy)
    phi_seam_clamped = max(float(phi_seam), 0.05)
    depth_A = float(np.clip(camera_height / np.tan(phi_seam_clamped), 0.5, 8.0))
    val_A = float(np.median(rel_norm[seam_row, :]))

    # Anchor B: ceiling row (top 8% of image)
    ceil_row = max(0, int(H * 0.08))
    phi_ceil = np.arctan2(float(ceil_row) - cy, fy)
    phi_ceil_clamped = max(float(-phi_ceil), 0.05)
    depth_B = float(np.clip(
        (ceiling_height - camera_height) / np.tan(phi_ceil_clamped),
        0.5, 8.0,
    ))
    val_B = float(np.median(rel_norm[ceil_row, :]))

    # Solve 2×2 linear system: [val_A, 1; val_B, 1] [s; t] = [depth_A; depth_B]
    A_mat = np.array([[val_A, 1.0], [val_B, 1.0]])
    b_vec = np.array([depth_A, depth_B])
    det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]

    if abs(det) < 1e-6 or abs(val_A - val_B) < 0.05:
        # Anchors too close in relative space – use scale-only fit
        s = depth_A / (val_A + 1e-6)
        t = 0.0
    else:
        x = np.linalg.solve(A_mat, b_vec)
        s, t = float(x[0]), float(x[1])

    # Sanity-clamp scale
    s = float(np.clip(s, 0.3, 15.0))
    metric = (rel_norm * s + t).astype(np.float32)
    metric = np.clip(metric, METRIC_MIN_M, METRIC_MAX_M)
    return metric


# ---------------------------------------------------------------------------
# Point cloud unprojection
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    camera_height: float,
    step: int = 4,
    min_depth: float = METRIC_MIN_M,
    max_depth: float = METRIC_MAX_M,
) -> np.ndarray:
    """Back-project a metric depth map into a 3D point cloud in room frame.

    Room frame convention (Z-up, right-handed):
        X = right
        Y = forward (into room)
        Z = up

    Camera is at height `camera_height` above the floor (Z=0).

    Args:
        depth_m       : (H, W) float32 metric depth in metres.
        intrinsics    : {fx, fy, cx, cy}.
        camera_height : camera height above the floor in metres.
        step          : pixel stride for subsampling.
        min_depth     : minimum valid depth.
        max_depth     : maximum valid depth.

    Returns:
        pts_world : (N, 3) float32 array in room frame.
    """
    H, W = depth_m.shape
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    u_grid = np.arange(0, W, step, dtype=np.float32)
    v_grid = np.arange(0, H, step, dtype=np.float32)
    UU, VV = np.meshgrid(u_grid, v_grid)

    Z_cam = depth_m[::step, ::step].astype(np.float32)

    valid = (Z_cam >= min_depth) & (Z_cam <= max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    Z = Z_cam[valid]
    U = UU[valid]
    V = VV[valid]

    # Camera-frame back-projection (standard pinhole, -Z forward)
    X_cam = (U - cx) * Z / fx
    Y_cam = -(V - cy) * Z / fy    # flip V: image y+ is down, camera y+ is up
    Z_cam_bp = -Z                  # camera looks in -Z direction

    # Transform to room frame: camera sits at (0, 0, camera_height) looking
    # in the -Z_cam direction, which maps to +Y_room (forward into room).
    # Convention: X_cam → X_room, Y_cam → Z_room (up), -Z_cam → Y_room (forward)
    X_room = X_cam
    Y_room = -Z_cam_bp         # = Z   (distance in front of camera)
    Z_room = Y_cam + camera_height   # shift up by camera height

    pts_world = np.column_stack([
        X_room.astype(np.float32),
        Y_room.astype(np.float32),
        Z_room.astype(np.float32),
    ])

    # Remove physically implausible points
    z_valid = (pts_world[:, 2] >= -0.10) & (pts_world[:, 2] <= 4.0)
    return pts_world[z_valid]


# ---------------------------------------------------------------------------
# Main DepthEngine class
# ---------------------------------------------------------------------------

class DepthEngine:
    """Metric monocular depth engine with automatic backend selection.

    Priority:
        1. Depth Anything V2 (requires torch + transformers + internet access)
        2. MiDaS via OpenCV DNN (requires onnx model file in models/)
        3. Geometric triangulation (always available, no ML)

    Usage:
        engine = DepthEngine()
        depth_m = engine.predict(image_bgr, intrinsics, camera_height, seam_v, ceiling_height)
        pts = depth_to_pointcloud(depth_m, intrinsics, camera_height)
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: str = "cpu",
        force_backend: Optional[str] = None,
    ):
        """Initialize and auto-select best available backend.

        Args:
            models_dir   : directory for model weights (defaults to project models/).
            device       : 'cpu' or 'cuda'.
            force_backend: 'depth_anything_v2' | 'midas' | 'geometric' | None (auto).
        """
        self.device = device
        self._da_pipe = None
        self._midas_net = None
        self._midas_size = None
        self.backend: str = "geometric"

        if models_dir is None:
            # Locate project root models/ directory
            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "models"
                if candidate.exists():
                    models_dir = candidate
                    break
            else:
                models_dir = Path("models")
        self.models_dir = models_dir

        if force_backend:
            self._init_forced(force_backend)
        else:
            self._auto_init()

    def _auto_init(self):
        # Try Depth Anything V2
        pipe, name = _try_load_depth_anything_v2(self.device)
        if pipe is not None:
            self._da_pipe = pipe
            self.backend = name
            return

        # Try MiDaS ONNX
        net, size, name = _try_load_midas(self.models_dir)
        if net is not None:
            self._midas_net = net
            self._midas_size = size
            self.backend = name
            return

        # Geometric fallback
        self.backend = "geometric"
        log.info("DepthEngine using geometric (triangulation) backend")

    def _init_forced(self, backend: str):
        if backend == "depth_anything_v2":
            pipe, name = _try_load_depth_anything_v2(self.device)
            if pipe is None:
                raise RuntimeError("Depth Anything V2 forced but could not load")
            self._da_pipe = pipe
            self.backend = name
        elif backend == "midas":
            net, size, name = _try_load_midas(self.models_dir)
            if net is None:
                raise RuntimeError("MiDaS forced but could not load")
            self._midas_net = net
            self._midas_size = size
            self.backend = name
        elif backend == "geometric":
            self.backend = "geometric"
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(
        self,
        image_bgr: np.ndarray,
        intrinsics: Dict[str, float],
        camera_height: float,
        seam_v: float,
        ceiling_height: float = 2.50,
        pitch_rad: float = 0.0,
    ) -> np.ndarray:
        """Predict a metric depth map for a single RGB image.

        Args:
            image_bgr     : (H, W, 3) uint8 BGR array.
            intrinsics    : {fx, fy, cx, cy, width, height}.
            camera_height : metres (from scene_geometry.estimate_camera_height).
            seam_v        : y-pixel of floor-wall seam.
            ceiling_height: architectural prior in metres.
            pitch_rad     : camera optical pitch angle in radians.

        Returns:
            depth_m: (H, W) float32, metric metres.
        """
        if self.backend == "depth_anything_v2":
            rel = _predict_depth_anything(self._da_pipe, image_bgr)
            if rel is not None:
                return _affine_rescale_to_metric(
                    rel, camera_height, seam_v, ceiling_height, intrinsics,
                    invert=True,
                )

        if self.backend == "midas":
            rel = _predict_midas(self._midas_net, self._midas_size, image_bgr)
            if rel is not None:
                return _affine_rescale_to_metric(
                    rel, camera_height, seam_v, ceiling_height, intrinsics,
                    invert=True,
                )

        # Geometric fallback (always runs if ML backends fail)
        return _geometric_depth(
            image_bgr, intrinsics, camera_height, seam_v, ceiling_height, pitch_rad=pitch_rad,
        )

    def predict_raw(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return raw METRIC depth map in metres.

        For depth_anything_v2 (Metric-Indoor): returns metric depth directly.
        For midas: converts disparity to pseudo-metric (relative, not absolute metres).
        For geometric: returns None (no pixel-level depth available).

        Returns:
            (H, W) float32 array in metres, or None if unavailable.
        """
        if self.backend == "depth_anything_v2":
            # Metric-Indoor model: output is already in metres
            return _predict_depth_anything(self._da_pipe, image_bgr)

        if self.backend == "midas":
            # MiDaS is relative disparity — convert to pseudo-depth for ratio estimation
            rel = _predict_midas(self._midas_net, self._midas_size, image_bgr)
            if rel is not None:
                d_inv = rel.copy()
                d_min, d_max = d_inv.min(), d_inv.max()
                if d_max - d_min > 1e-6:
                    d_inv = (d_inv - d_min) / (d_max - d_min)
                return 1.0 / (d_inv + 0.01)  # pseudo-metric: only ratios are meaningful

        # geometric backend: no per-pixel depth available for scale estimation
        return None

    def predict_and_unproject(
        self,
        image_bgr: np.ndarray,
        intrinsics: Dict[str, float],
        camera_height: float,
        seam_v: float,
        ceiling_height: float = 2.50,
        pitch_rad: float = 0.0,
        pixel_step: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict depth and immediately unproject to 3D room-frame point cloud.

        Returns:
            depth_m   : (H, W) float32 metric depth map
            pts_world : (N, 3) float32 point cloud in room frame (Z-up)
        """
        depth_m = self.predict(
            image_bgr, intrinsics, camera_height, seam_v, ceiling_height, pitch_rad=pitch_rad,
        )
        pts_world = depth_to_pointcloud(
            depth_m, intrinsics, camera_height, step=pixel_step,
        )
        return depth_m, pts_world

    def __repr__(self) -> str:
        return f"DepthEngine(backend={self.backend!r}, device={self.device!r})"
