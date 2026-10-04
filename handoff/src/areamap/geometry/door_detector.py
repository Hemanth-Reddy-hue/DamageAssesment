import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from areamap.state import Opening, Interval
from areamap.geometry.scene_geometry import estimate_per_image_geometry
from areamap.geometry.depth_engine import DepthEngine

def _make_iv(val: float, err: float = 0.05, tier: str = "photo") -> Interval:
    return Interval(value=val, lo=val - err, hi=val + err, confidence_level=0.9, method="conformal", tier=tier)

def detect_doors_in_image(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    seam_v: float
) -> List[Dict[str, float]]:
    h, w = rgb_image.shape[:2]
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    seam_int = int(seam_v)
    edges[max(0, seam_int-10):, :] = 0
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40, 
        minLineLength=int(h * 0.2), 
        maxLineGap=20
    )
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            pts = line[0] if len(line) == 1 else line
            x1, y1, x2, y2 = pts
            if abs(x2 - x1) < 15:
                if max(y1, y2) > seam_int - 150:
                    vertical_lines.append(int((x1 + x2) / 2))
                    
    vertical_lines = sorted(list(set(vertical_lines)))
    grouped_lines = []
    if vertical_lines:
        current_group = [vertical_lines[0]]
        for x in vertical_lines[1:]:
            if x - current_group[-1] < 20:
                current_group.append(x)
            else:
                grouped_lines.append(int(np.mean(current_group)))
                current_group = [x]
        grouped_lines.append(int(np.mean(current_group)))
        
    doors = []
    for i in range(len(grouped_lines)):
        for j in range(i+1, len(grouped_lines)):
            x_left = grouped_lines[i]
            x_right = grouped_lines[j]
            width = x_right - x_left
            
            if width < w * 0.05 or width > w * 0.8:
                continue
                
            y_sample = int(seam_v * 0.6)
            
            x_out_left = max(0, x_left - 10)
            x_in_left = min(w - 1, x_left + 10)
            x_in_right = max(0, x_right - 10)
            x_out_right = min(w - 1, x_right + 10)
            
            depth_wall_L = depth_map[y_sample, x_out_left]
            depth_door_L = depth_map[y_sample, x_in_left]
            depth_wall_R = depth_map[y_sample, x_out_right]
            depth_door_R = depth_map[y_sample, x_in_right]
            
            # Doorway should be further away than the wall (hole). 
            # Relaxed threshold for noisy depth maps
            if (depth_door_L > depth_wall_L + 0.05) and (depth_door_R > depth_wall_R + 0.05):
                confidence = 0.6 + 0.4 * min(1.0, (depth_door_L - depth_wall_L))
                doors.append({
                    "u_center": (x_left + x_right) / 2.0,
                    "width_pixels": width,
                    "confidence": float(confidence)
                })
                
    final_doors = []
    for d in sorted(doors, key=lambda x: x["confidence"], reverse=True):
        overlap = False
        for fd in final_doors:
            if abs(d["u_center"] - fd["u_center"]) < (d["width_pixels"] + fd["width_pixels"]) / 4:
                overlap = True
                break
        if not overlap:
            final_doors.append(d)
            
    return final_doors

def detect_doors_for_room_photo_tier(
    capture_path: str,
    room_id: str,
    primary_wall_id: str,
    primary_wall_center: List[float]
) -> List[Opening]:
    """Load the room's photos, run real visual door detection, and return Opening objects."""
    room_dir = Path(capture_path)
    # Handle single room flat structure vs multi-room subdirectory
    sub_dir = room_dir / room_id
    if sub_dir.is_dir():
        room_dir = sub_dir
        
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in room_dir.iterdir() if p.suffix.lower() in valid_exts]
    
    if not image_paths:
        return []
        
    # We only need to find one door for the connection, try the first few photos
    engine = DepthEngine(device="cpu")
    fake_intr = {"fx": 1500, "fy": 1500, "cx": 960, "cy": 540, "width": 1920, "height": 1080}
    
    for p in image_paths[:3]:
        img_color = cv2.imread(str(p))
        if img_color is None:
            continue
            
        h, w = img_color.shape[:2]
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        geom = estimate_per_image_geometry(gray, fake_intr)
        
        depth_map, points_3d = engine.predict_and_unproject(img_color, fake_intr, geom["camera_height"], geom["seam_v"], pixel_step=1)
        
        if depth_map is not None:
            doors = detect_doors_in_image(img_color, depth_map, geom["seam_v"])
            
            if doors:
                # Found a real door! Convert pixel width to metric width.
                # using similar triangles: metric_width = (pixel_width / fx) * median_depth
                best_door = doors[0] # Highest confidence
                u_center = best_door["u_center"]
                width_px = best_door["width_pixels"]
                
                # Sample depth at door frame for scale
                x_idx = int(u_center - width_px/2)
                y_idx = int(geom["seam_v"] * 0.6)
                if x_idx >= 0 and y_idx >= 0 and y_idx < h and x_idx < w:
                    wall_depth = depth_map[y_idx, x_idx]
                else:
                    wall_depth = 2.0
                    
                metric_width = (width_px / fake_intr["fx"]) * wall_depth
                metric_width = float(np.clip(metric_width, 0.6, 1.8)) # Sanity check for a door
                
                return [Opening(
                    opening_id=f"door_real_{primary_wall_id}_01",
                    wall_id=primary_wall_id,
                    type="door",
                    width=_make_iv(metric_width, tier="photo"),
                    height=_make_iv(2.05, tier="photo"),
                    sill_height=_make_iv(0.0, tier="photo"),
                    position=[primary_wall_center[0], primary_wall_center[1], 2.05 / 2.0],
                    confidence=best_door["confidence"]
                )]
                
    return []
