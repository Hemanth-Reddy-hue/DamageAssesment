"""Visual Place Recognition for loop closure detection (Fix for Issue #6)."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import torch
    import timm
    from torchvision import transforms
    from PIL import Image
    HAS_ML = True
except ImportError:
    HAS_ML = False

from areamap.state import CaptureState
from areamap.geometry.posegraph import make_se2_matrix
from areamap.geometry.registration import icp_align
try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)

class PlaceRecognizer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.transform = None
        
        if HAS_ML:
            self._init_model()
            
    def _init_model(self):
        """Initialize a lightweight timm model for global image embeddings."""
        try:
            # mobilenetv3_small_050 is extremely fast and lightweight
            self.model = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=0)
            self.model.to(self.device)
            self.model.eval()
            
            data_config = timm.data.resolve_model_data_config(self.model)
            self.transform = timm.data.create_transform(**data_config, is_training=False)
        except Exception as e:
            logger.warning(f"Failed to load timm model for place recognition: {e}")
            self.model = None

    def _get_images_for_room(self, capture_path: str, room_name: str) -> List[Path]:
        room_dir = Path(capture_path) / room_name
        if not room_dir.is_dir():
            return []
        return [p for p in room_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    def compute_room_embedding(self, image_paths: List[Path]) -> np.ndarray:
        """Compute the average embedding for a room from its images."""
        if not self.model or not image_paths:
            return np.zeros(0)
            
        embeddings = []
        with torch.no_grad():
            for p in image_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    tensor = self.transform(img).unsqueeze(0).to(self.device)
                    emb = self.model(tensor).cpu().numpy().flatten()
                    emb = emb / (np.linalg.norm(emb) + 1e-9)
                    embeddings.append(emb)
                except Exception:
                    continue
                    
        if not embeddings:
            return np.zeros(0)
            
        # Average pooling
        room_emb = np.mean(embeddings, axis=0)
        return room_emb / (np.linalg.norm(room_emb) + 1e-9)

def _load_point_cloud(ply_path: str) -> np.ndarray:
    """Load a point cloud from a PLY file (header-less fallback)."""
    try:
        # Simplistic PLY loader for XYZ if open3d is missing
        pts = []
        with open(ply_path, 'r') as f:
            in_vertex = False
            for line in f:
                line = line.strip()
                if line == "end_header":
                    in_vertex = True
                    continue
                if in_vertex:
                    parts = line.split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(pts)
    except Exception:
        return np.zeros((0, 3))

def detect_visual_loop_closures(state: CaptureState, similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Detect loop closures using deep visual features and ICP alignment."""
    if not HAS_ML:
        logger.warning("Visual loop closure requested, but ML stack (timm/torch) is missing.")
        return []
        
    recognizer = PlaceRecognizer()
    if recognizer.model is None:
        return []
        
    room_ids = list(state.room_geometry.keys())
    if len(room_ids) < 3:
        # Loop closures only make sense for 3+ rooms
        return []
        
    # 1. Compute visual embeddings for each room
    embeddings = {}
    for r_id in room_ids:
        # Photos are in capture_path / r_id
        img_paths = recognizer._get_images_for_room(state.capture_path, r_id)
        emb = recognizer.compute_room_embedding(img_paths)
        if len(emb) > 0:
            embeddings[r_id] = emb
            
    loop_closures = []
    
    # 2. Check all non-adjacent pairs for visual similarity
    # Assumes sequential adjacency is standard (e.g. 0-1, 1-2). Check pairs with index gap > 1
    for i in range(len(room_ids)):
        for j in range(i + 2, len(room_ids)):
            r1 = room_ids[i]
            r2 = room_ids[j]
            
            if r1 not in embeddings or r2 not in embeddings:
                continue
                
            sim = float(np.dot(embeddings[r1], embeddings[r2]))
            if sim > similarity_threshold:
                logger.info(f"Visual loop closure detected between {r1} and {r2} (sim={sim:.3f})")
                
                # 3. Estimate SE(2) transform via ICP
                pc1_path = state.point_clouds.get(r1)
                pc2_path = state.point_clouds.get(r2)
                
                if pc1_path and pc2_path:
                    pts1 = _load_point_cloud(pc1_path)
                    pts2 = _load_point_cloud(pc2_path)
                    
                    if len(pts1) > 20 and len(pts2) > 20:
                        T_4x4, rmse = icp_align(source=pts2, target=pts1) # Align r2 to r1
                        
                        if rmse < 0.5: # Valid alignment
                            # Extract SE(2) params from T_4x4
                            dx, dy = T_4x4[0, 3], T_4x4[1, 3]
                            theta = np.arctan2(T_4x4[1, 0], T_4x4[0, 0])
                            
                            loop_closures.append({
                                "from_room": r2,
                                "to_room": r1,
                                "transform": make_se2_matrix(dx, dy, theta),
                                "confidence": float(sim)
                            })
                            
    return loop_closures
