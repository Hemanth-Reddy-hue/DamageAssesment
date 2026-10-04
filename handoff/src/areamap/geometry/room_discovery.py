"""Room Discovery: Automatically cluster flat image datasets into rooms using visual & geometric cues (Issue #11)."""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple, Optional, Union
import networkx as nx
from networkx.algorithms import community

from areamap.geometry.place_recognition import PlaceRecognizer
from areamap.geometry.registration import estimate_relative_pose_essential
from areamap.tiers.photo import extract_exif_intrinsics

logger = logging.getLogger(__name__)

class RoomCluster:
    def __init__(self, room_id: str, images: List[Path], confidence: float):
        self.room_id = room_id
        self.images = images
        self.confidence = confidence

def discover_rooms(
    image_paths: List[Path],
    similarity_threshold: float = 0.15,
    min_geom_confidence: float = 0.12,
    output_dir: Optional[Union[Path, str]] = None
) -> Tuple[List[RoomCluster], List[Dict[str, Any]]]:
    """Cluster a flat list of images into semantic rooms and find transition bridges.
    
    Returns:
        clusters: List of RoomCluster objects
        transitions: List of dicts representing doorway bridges between rooms
    """
    if not image_paths:
        return [], []
        
    if len(image_paths) == 1:
        return [RoomCluster("room_00", image_paths, 1.0)], []

    logger.info(f"[ROOM DISCOVERY] Starting automatic discovery on {len(image_paths)} images.")

    # 1. Extract Global Embeddings
    recognizer = PlaceRecognizer(device="cpu")
    embeddings = {}
    for p in image_paths:
        emb = recognizer.compute_room_embedding([p])
        if emb.size > 0:
            embeddings[str(p)] = emb
            
    if not embeddings:
        return [RoomCluster("room_00", image_paths, 0.5)], []
        
    # 2. Build Candidate Pairs
    candidates = []
    path_strs = [str(p) for p in image_paths if str(p) in embeddings]
    n_paths = len(path_strs)
    
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            p_a, p_b = path_strs[i], path_strs[j]
            emb_a, emb_b = embeddings[p_a], embeddings[p_b]
            sim = float(np.dot(emb_a, emb_b))
            # Test pair if:
            # - High enough visual similarity (>= 0.15), OR
            # - Consecutive images in capture order (|i - j| <= 1), OR
            # - Small dataset (<= 12 images) where checking all pairs is fast
            if sim >= similarity_threshold or abs(i - j) <= 1 or n_paths <= 12:
                candidates.append((p_a, p_b, sim))
                
    logger.info(f"[ROOM DISCOVERY] Found {len(candidates)} candidate image relationships.")

    # 3. Geometric Verification (Build Graph)
    G = nx.Graph()
    for p in path_strs:
        G.add_node(p)
        
    # Preload and resize for fast matching
    gray_cache = {}
    intr_cache = {}
    for p in path_strs:
        img = cv2.imread(p)
        if img is not None:
            gray_cache[p] = cv2.cvtColor(cv2.resize(img, (640, 480)), cv2.COLOR_BGR2GRAY)
            intr_cache[p] = extract_exif_intrinsics(p)
            
    verified_edges = 0
    for p_a, p_b, sim in candidates:
        if p_a not in gray_cache or p_b not in gray_cache:
            continue
            
        _, _, conf = estimate_relative_pose_essential(
            gray_cache[p_a], gray_cache[p_b],
            intr_cache[p_a], intr_cache[p_b],
            intr_cache[p_a]["width"], intr_cache[p_b]["width"]
        )
        
        if conf >= min_geom_confidence or sim >= 0.35:
            weight = float(max(conf, sim * 0.5))
            G.add_edge(p_a, p_b, weight=weight, sim=sim, conf=conf)
            verified_edges += 1
            
    logger.info(f"[ROOM DISCOVERY] Strong geometric relationships established: {verified_edges}")

    # 4. Graph Clustering (Community Detection)
    # Check for genuine multi-room structure
    communities = []
    if verified_edges > 0:
        raw_communities = list(community.greedy_modularity_communities(G, weight='weight'))
        # Only keep communities with at least 2 images
        communities = [list(c) for c in raw_communities if len(c) >= 2]
        if not communities:
            # Check connected components with >= 2 images
            cc = [list(c) for c in nx.connected_components(G) if len(c) >= 2]
            if cc:
                communities = cc

    # Detect doorway transitions between candidate communities
    node_to_room = {}
    for idx, comm in enumerate(communities):
        room_id = f"room_{idx:02d}"
        for p in comm:
            node_to_room[p] = room_id

    transitions = []
    for p_a, p_b, data in G.edges(data=True):
        r_a = node_to_room.get(p_a)
        r_b = node_to_room.get(p_b)
        if r_a and r_b and r_a != r_b:
            transitions.append({
                "room_a": r_a,
                "room_b": r_b,
                "image_a": p_a,
                "image_b": p_b,
                "confidence": data.get("weight", 0.5)
            })

    # If small capture (<= 8 photos) and no doorway transitions, all images belong to one room!
    if (len(path_strs) <= 8 and len(transitions) == 0) or len(communities) <= 1:
        # Default single room
        clusters = [RoomCluster("room_00", [Path(p) for p in path_strs], 1.0)]
        logger.info(f"[ROOM DISCOVERY] Single room capture detected with {len(path_strs)} images.")
        return clusters, []

    # Assign remaining orphan images to the nearest/most similar community (NO 1-photo rooms)
    clusters = []
    for idx, comm in enumerate(communities):
        room_id = f"room_{idx:02d}"
        cluster_images = [Path(p) for p in comm]
        subgraph = G.subgraph(comm)
        internal_edges = subgraph.number_of_edges()
        max_edges = len(comm) * (len(comm) - 1) / 2
        conf = 0.5 + 0.5 * (internal_edges / max_edges) if max_edges > 0 else 1.0
        clusters.append(RoomCluster(room_id, cluster_images, round(conf, 3)))

    orphans = set(path_strs) - set(node_to_room.keys())
    for o in orphans:
        # Assign to community with highest embedding similarity
        best_comm_idx = 0
        best_sim = -1.0
        emb_o = embeddings.get(o)
        if emb_o is not None:
            for c_idx, comm in enumerate(communities):
                comm_sims = [float(np.dot(emb_o, embeddings[p])) for p in comm if p in embeddings]
                avg_sim = float(np.mean(comm_sims)) if comm_sims else 0.0
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_comm_idx = c_idx
        clusters[best_comm_idx].images.append(Path(o))
        node_to_room[o] = clusters[best_comm_idx].room_id

    logger.info(f"[ROOM DISCOVERY] Discovered {len(clusters)} rooms automatically.")

    # 5. Export Debug Graph
    try:
        import json
        out_dir = Path(output_dir) if output_dir else Path("out")
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_data = {
            "metadata": {"images": len(image_paths), "rooms": len(clusters)},
            "rooms": {c.room_id: [str(img.name) for img in c.images] for c in clusters},
            "transitions": transitions
        }
        with open(out_dir / "room_discovery_graph.json", "w") as f:
            json.dump(debug_data, f, indent=2)
        logger.info(f"[ROOM DISCOVERY] Exported debug graph to {out_dir / 'room_discovery_graph.json'}")
    except Exception as e:
        logger.warning(f"Failed to export room discovery debug graph: {e}")

    return clusters, transitions
