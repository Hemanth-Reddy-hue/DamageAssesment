"""Geometry processing: planes, openings, pose graphs, and uncertainty propagation."""

from .planes import fit_room_planes, extract_floor_polygon
from .openings import detect_openings_from_cutouts
from .posegraph import optimize_pose_graph
from .adjacency import infer_room_adjacency
from .uncertainty import calculate_interval

__all__ = [
    "fit_room_planes",
    "extract_floor_polygon",
    "detect_openings_from_cutouts",
    "optimize_pose_graph",
    "infer_room_adjacency",
    "calculate_interval",
]
