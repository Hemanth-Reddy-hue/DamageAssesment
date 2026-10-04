import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging

log = logging.getLogger(__name__)

@dataclass
class ScaleCue:
    name: str
    scale: float
    sigma: float
    confidence: float
    detail: str = ""

def fit_horizontal_plane_ransac(points: np.ndarray, distance_threshold: float = 0.04, min_inliers: int = 100) -> Optional[Dict[str, Any]]:
    """Fit a horizontal plane using RANSAC."""
    n_points = len(points)
    if n_points < 3:
        return None

    rng = np.random.default_rng(42)
    best_inliers = []
    best_normal = None
    best_d = None

    for _ in range(500):
        idx = rng.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if abs(normal[2]) < 0.85: # Must be roughly horizontal
            continue
            
        d = -float(np.dot(normal, p1))
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < distance_threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if len(best_inliers) < min_inliers:
        return None

    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d = -best_d

    inlier_pts = points[best_inliers]
    z_mean = float(np.mean(inlier_pts[:, 2]))
    return {"normal": best_normal, "d": best_d, "z_mean": z_mean, "inliers": best_inliers}

def estimate_floor_and_ceiling(points: np.ndarray, cam_zs: List[float]) -> Tuple[Optional[float], Optional[float], str, dict, dict]:
    """Estimate floor and ceiling Z in SfM units."""
    if len(points) < 100:
        return None, None, "insufficient_points", {}, {}
        
    med_cam_z = float(np.median(cam_zs)) if cam_zs else 0.0
    
    # Floor: lowest dense horizontal cluster below cameras
    floor_pts = points[points[:, 2] < med_cam_z]
    floor_plane = fit_horizontal_plane_ransac(floor_pts, distance_threshold=0.05, min_inliers=100)
    floor_z = floor_plane["z_mean"] if floor_plane else None
    
    # Ceiling: horizontal cluster above cameras
    ceil_pts = points[points[:, 2] > med_cam_z]
    ceil_plane = fit_horizontal_plane_ransac(ceil_pts, distance_threshold=0.05, min_inliers=100)
    
    ceil_z = None
    fallback_fired = ""
    
    if ceil_plane:
        ceil_z = ceil_plane["z_mean"]
        fallback_fired = "ceiling_plane"
    else:
        # Fallback 1: 95th-98th percentile of upper points
        if len(ceil_pts) > 50:
            ceil_z = float(np.percentile(ceil_pts[:, 2], 96))
            fallback_fired = "percentile_95_98"
        else:
            # Fallback to a default prior in the caller if ceil_z is None
            pass

    return floor_z, ceil_z, fallback_fired, floor_plane or {}, ceil_plane or {}

class ScaleResolver:
    def __init__(self, out_dir: Path):
        self.cues: List[ScaleCue] = []
        self.out_dir = out_dir

    def add_cue(self, cue: ScaleCue):
        self.cues.append(cue)

    def resolve(self) -> Tuple[float, str, float]:
        """Fuse cues using log-scale weighted median/mean. Returns (scale, method, confidence)"""
        if not self.cues:
            return 1.0, "none", 0.0

        # Overrides
        for cue in self.cues:
            if cue.name == "user_reference":
                return cue.scale, "user_reference", 0.99

        # Log scale fusion
        log_s = []
        weights = []
        for cue in self.cues:
            if cue.sigma <= 0:
                continue
            log_s.append(np.log(cue.scale))
            weights.append(cue.confidence / (cue.sigma ** 2))

        if not log_s:
            return 1.0, "none", 0.0

        log_s = np.array(log_s)
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Median
        sort_idx = np.argsort(log_s)
        cum_weights = np.cumsum(weights[sort_idx])
        median_idx = np.where(cum_weights >= 0.5)[0][0]
        median_log_s = log_s[sort_idx][median_idx]

        # Reject outliers (>25% diff in linear scale is ~0.22 in log scale)
        valid = np.abs(log_s - median_log_s) < 0.223
        if not np.any(valid):
            final_scale = np.exp(median_log_s)
            return float(final_scale), "median_fallback", 0.1

        # Weighted mean of survivors
        survivor_log_s = log_s[valid]
        survivor_weights = weights[valid]
        survivor_weights /= np.sum(survivor_weights)
        
        mean_log_s = np.sum(survivor_log_s * survivor_weights)
        final_scale = float(np.exp(mean_log_s))

        # Determine final confidence
        high_conf_survivors = sum(1 for i, v in enumerate(valid) if v and self.cues[i].confidence > 0.5)
        if high_conf_survivors > 0:
            final_conf = 0.8
            method = "fused_high_conf"
        else:
            final_conf = 0.3
            method = "fused_low_conf"

        # Report
        report = {
            "cues": [{"name": c.name, "scale": c.scale, "sigma": c.sigma, "confidence": c.confidence, "detail": c.detail} for c in self.cues],
            "final_scale": final_scale,
            "method": method,
            "confidence": final_conf
        }
        with open(self.out_dir / "scale_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return final_scale, method, final_conf
