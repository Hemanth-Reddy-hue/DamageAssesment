"""Door-based room adjacency graph construction, SE(2) frame alignment, and overlap detection (Module M6)."""

from typing import List, Dict, Tuple, Optional
import numpy as np
from areamap.state import RoomGeometry, WallSegment, Opening, AdjacencyConnection
from areamap.geometry.posegraph import make_se2_matrix

def compute_wall_outward_normal(wall: WallSegment, room_polygon: List[List[float]]) -> np.ndarray:
    """Compute the 2D unit normal of a wall pointing outward from the room interior."""
    sx, sy = wall.start
    ex, ey = wall.end
    wall_vec = np.array([ex - sx, ey - sy], dtype=float)
    length = float(np.linalg.norm(wall_vec))
    if length < 1e-5:
        return np.array([0.0, 1.0])

    u_dir = wall_vec / length
    # Candidate perpendicular normals
    n1 = np.array([-u_dir[1], u_dir[0]])
    n2 = np.array([u_dir[1], -u_dir[0]])

    # Room centroid
    cx = float(np.mean([p[0] for p in room_polygon]))
    cy = float(np.mean([p[1] for p in room_polygon]))
    centroid = np.array([cx, cy])

    wall_mid = np.array([(sx + ex) / 2.0, (sy + ey) / 2.0])
    from_centroid = wall_mid - centroid

    # Outward normal should have positive dot product with vector from centroid to wall
    if np.dot(n1, from_centroid) >= 0:
        return n1
    return n2

def align_room_pair_se2(
    from_room: RoomGeometry,
    to_room: RoomGeometry,
    door_from: Opening,
    door_to: Opening
) -> np.ndarray:
    """Solve the 2D SE(2) transformation that snaps to_room onto from_room at their shared door.
    
    Enforces opposing outward wall normals and coinciding doorway centers.
    """
    # 1. Doorway positions in XY
    p_from = np.array(door_from.position[:2], dtype=float)
    p_to = np.array(door_to.position[:2], dtype=float)

    # 2. Find parent walls
    wall_from = next((w for w in from_room.walls if w.wall_id == door_from.wall_id), from_room.walls[0])
    wall_to = next((w for w in to_room.walls if w.wall_id == door_to.wall_id), to_room.walls[0])

    norm_from = compute_wall_outward_normal(wall_from, from_room.floor_polygon)
    norm_to = compute_wall_outward_normal(wall_to, to_room.floor_polygon)

    # 3. Solve rotation theta: R(theta) * norm_to = -norm_from
    ang_target = np.arctan2(-norm_from[1], -norm_from[0])
    ang_to = np.arctan2(norm_to[1], norm_to[0])
    theta = float(ang_target - ang_to)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # 4. Solve translation t: p_from = R * p_to + t  =>  t = p_from - R * p_to
    t = p_from - R @ p_to

    return make_se2_matrix(float(t[0]), float(t[1]), theta)

def apply_se2_transform_to_room(room: RoomGeometry, T: np.ndarray) -> RoomGeometry:
    """Apply a 3x3 SE(2) rigid transform to all vertices, walls, and openings of a room."""
    R = T[:2, :2]
    t = T[:2, 2]

    # Transform floor polygon vertices
    transformed_poly: List[List[float]] = []
    for pt in room.floor_polygon:
        v = np.array(pt[:2], dtype=float)
        new_v = R @ v + t
        transformed_poly.append([round(float(new_v[0]), 3), round(float(new_v[1]), 3)])

    # Transform walls
    transformed_walls: List[WallSegment] = []
    for w in room.walls:
        s = R @ np.array(w.start, dtype=float) + t
        e = R @ np.array(w.end, dtype=float) + t
        transformed_walls.append(
            WallSegment(
                wall_id=w.wall_id,
                start=[round(float(s[0]), 3), round(float(s[1]), 3)],
                end=[round(float(e[0]), 3), round(float(e[1]), 3)],
                length=w.length,
                confidence=w.confidence
            )
        )

    # Transform openings
    transformed_openings: List[Opening] = []
    for op in room.openings:
        p_xy = R @ np.array(op.position[:2], dtype=float) + t
        new_pos = [round(float(p_xy[0]), 3), round(float(p_xy[1]), 3), op.position[2] if len(op.position) > 2 else 1.0]
        transformed_openings.append(
            Opening(
                opening_id=op.opening_id,
                wall_id=op.wall_id,
                type=op.type,
                width=op.width,
                height=op.height,
                sill_height=op.sill_height,
                position=new_pos,
                confidence=op.confidence
            )
        )

    return RoomGeometry(
        room_id=room.room_id,
        room_name=room.room_name,
        ceiling_height=room.ceiling_height,
        floor_area=room.floor_area,
        walls=transformed_walls,
        floor_polygon=transformed_poly,
        openings=transformed_openings,
        is_rectilinear=room.is_rectilinear
    )

def infer_room_adjacency(rooms: Dict[str, RoomGeometry]) -> List[AdjacencyConnection]:
    """Infer room connectivity from matching doors or sequential topological order."""
    connections: List[AdjacencyConnection] = []
    room_ids = list(rooms.keys())

    if len(room_ids) < 2:
        return connections

    # Check for matching doors across rooms
    for i in range(len(room_ids)):
        r1 = room_ids[i]
        r1_doors = [op for op in rooms[r1].openings if op.type in ["door", "passageway"]]
        for j in range(i + 1, len(room_ids)):
            r2 = room_ids[j]
            r2_doors = [op for op in rooms[r2].openings if op.type in ["door", "passageway"]]

            matched = False
            for d1 in r1_doors:
                for d2 in r2_doors:
                    if abs(d1.width.value - d2.width.value) <= 0.15:
                        connections.append(
                            AdjacencyConnection(
                                from_room=r1,
                                to_room=r2,
                                opening_id=d1.opening_id,
                                confidence=0.95
                            )
                        )
                        matched = True
                        break
                if matched:
                    break

    # If no doors matched, fall back to sequential traversal order
    if not connections:
        for i in range(len(room_ids) - 1):
            r1 = room_ids[i]
            r2 = room_ids[i + 1]
            connections.append(
                AdjacencyConnection(
                    from_room=r1,
                    to_room=r2,
                    opening_id=f"connector_{r1}_to_{r2}",
                    confidence=0.90
                )
            )

    return connections

def _polygon_area(poly: List[List[float]]) -> float:
    """Calculate 2D polygon area via Shoelace formula."""
    if len(poly) < 3:
        return 0.0
    x = [p[0] for p in poly]
    y = [p[1] for p in poly]
    return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(poly) - 1)) + (x[-1] * y[0] - x[0] * y[-1]))

def _clip_polygon(subject: List[List[float]], clipper: List[List[float]]) -> List[List[float]]:
    """Sutherland-Hodgman polygon clipping algorithm."""
    output = list(subject)
    for i in range(len(clipper)):
        cp1 = clipper[i]
        cp2 = clipper[(i + 1) % len(clipper)]
        input_list = list(output)
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            def inside(p):
                return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= -1e-6

            def intersection(p1, p2):
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [p1[0] - p2[0], p1[1] - p2[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = p1[0] * p2[1] - p1[1] * p2[0]
                denom = dc[0] * dp[1] - dc[1] * dp[0]
                if abs(denom) < 1e-9:
                    return [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
                n3 = 1.0 / denom
                return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

            if inside(e):
                if not inside(s):
                    output.append(intersection(s, e))
                output.append(e)
            elif inside(s):
                output.append(intersection(s, e))
            s = e
    return output

def check_room_overlaps(rooms: Dict[str, RoomGeometry], max_allowed_overlap_m2: float = 0.05) -> List[str]:
    """Check for illegal physical overlaps between room polygons using polygon clipping.
    
    Returns list of warning messages for any detected overlap violations.
    """
    warnings: List[str] = []
    room_ids = list(rooms.keys())

    for i in range(len(room_ids)):
        r1 = room_ids[i]
        poly1 = rooms[r1].floor_polygon
        if len(poly1) < 3:
            continue
        for j in range(i + 1, len(room_ids)):
            r2 = room_ids[j]
            poly2 = rooms[r2].floor_polygon
            if len(poly2) < 3:
                continue

            clipped = _clip_polygon(poly1, poly2)
            overlap_area = _polygon_area(clipped)

            if overlap_area > max_allowed_overlap_m2:
                warnings.append(
                    f"Room overlap detected: {r1} and {r2} overlap by {overlap_area:.2f} m²"
                )

    return warnings
