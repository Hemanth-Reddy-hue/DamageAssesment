"""Constrained room-placement solver for artificial collision resolution (Fix #8)."""

import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.affinity import translate

def _overlap_area(t, p1: Polygon, p2: Polygon) -> float:
    """Calculate overlap area when p2 is translated by t."""
    p2_t = translate(p2, xoff=t[0], yoff=t[1])
    try:
        return float(p1.intersection(p2_t).area)
    except Exception:
        # Shapely can sometimes raise topological errors on invalid polygons
        return 0.0

def resolve_room_collision(placed_pts: list[list[float]], new_pts: list[list[float]]) -> tuple[float, float]:
    """Find the minimal translation vector to resolve physical overlap between two rooms.
    
    Replaces the old 'shift_x' hardcode that blindly pushed rooms to the right.
    Uses COBYLA to minimize displacement distance subject to zero overlap area.
    
    Args:
        placed_pts: Polygon vertices of the already-placed room.
        new_pts: Polygon vertices of the newly added room.
        
    Returns:
        (dx, dy): The minimum translation required for new_pts to clear placed_pts.
    """
    if len(placed_pts) < 3 or len(new_pts) < 3:
        return 0.0, 0.0
        
    p1 = Polygon(placed_pts)
    p2 = Polygon(new_pts)
    
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
        
    # Check if there is actual overlap
    try:
        initial_overlap = p1.intersection(p2).area
    except Exception:
        initial_overlap = 0.0
        
    if initial_overlap < 0.05:
        return 0.0, 0.0
    
    # Objective: minimize dx^2 + dy^2 (shortest push out)
    def objective(t):
        return t[0]**2 + t[1]**2
    
    # Constraint: overlap area <= 0.01 (COBYLA requires constraint >= 0)
    def constraint(t):
        return 0.01 - _overlap_area(t, p1, p2)
        
    # Direction vector from p1 centroid to p2 centroid
    c1 = np.array(p1.centroid.coords[0])
    c2 = np.array(p2.centroid.coords[0])
    v = c2 - c1
    if np.linalg.norm(v) < 1e-3:
        # Fallback direction if centroids perfectly overlap
        v = np.array([1.0, 0.0])
    v = v / np.linalg.norm(v)
    
    # Line search to find a strictly feasible safe distance as initial guess
    safe_dist = 0.0
    for step in np.arange(0.0, 15.0, 0.5):
        if _overlap_area([v[0]*step, v[1]*step], p1, p2) < 0.01:
            safe_dist = step
            break
            
    # Constrained derivative-free optimization
    res = minimize(
        objective, 
        x0=np.array([v[0]*safe_dist, v[1]*safe_dist]),
        method='COBYLA',
        constraints=[{'type': 'ineq', 'fun': constraint}],
        options={'rhobeg': 0.5, 'maxiter': 100}
    )
    
    # Return optimized translation if successful, else fallback to safe line search
    if res.success and _overlap_area(res.x, p1, p2) < 0.05:
        return float(res.x[0]), float(res.x[1])
    return float(v[0]*safe_dist), float(v[1]*safe_dist)
