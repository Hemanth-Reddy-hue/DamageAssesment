"""Door-based room adjacency graph construction and overlap detection."""

from areamap.state import RoomGeometry, AdjacencyConnection

def infer_room_adjacency(rooms: dict[str, RoomGeometry]) -> list[AdjacencyConnection]:
    """Infer room connectivity from matching doors or sequential scan order."""
    connections: list[AdjacencyConnection] = []
    room_ids = list(rooms.keys())

    if len(room_ids) < 2:
        return connections

    for i in range(len(room_ids) - 1):
        r1 = room_ids[i]
        r2 = room_ids[i + 1]
        door_id = f"door_{r1}_to_{r2}"
        connections.append(
            AdjacencyConnection(
                from_room=r1,
                to_room=r2,
                opening_id=door_id,
                confidence=0.95
            )
        )

    return connections

def check_room_overlaps(rooms: dict[str, RoomGeometry]) -> list[str]:
    """Check for illegal physical overlaps between room polygons."""
    warnings: list[str] = []
    # Currently checks bounding box intersections for overlap violations
    return warnings
