"""Test covisibility graph room community detection on synthetic two-room scene."""

import networkx as nx
import numpy as np
import pytest


def test_two_room_covisibility_segmentation():
    """Two point clusters joined by a doorway with a sequential camera path yield exactly 2 room segments."""
    # Room A: frames 0..19 observe points in Room A
    # Doorway: frames 18..21 observe points in doorway
    # Room B: frames 20..39 observe points in Room B
    G = nx.Graph()
    for f in range(40):
        G.add_node(f)

    # Dense intra-room edges for Room A: all frames share features
    for i in range(20):
        for j in range(i + 1, 20):
            G.add_edge(i, j, weight=30.0)

    # Dense intra-room edges for Room B: all frames share features
    for i in range(20, 40):
        for j in range(i + 1, 40):
            G.add_edge(i, j, weight=30.0)

    # Thin doorway bridge (few shared points between rooms)
    G.add_edge(19, 20, weight=2.0)
    G.add_edge(18, 20, weight=1.0)
    G.add_edge(19, 21, weight=1.0)

    # Run Louvain community detection
    communities = nx.algorithms.community.louvain_communities(G, seed=42)

    assert len(communities) == 2, f"Expected 2 communities, got {len(communities)}"

    # Check that Room A frames and Room B frames are cleanly separated
    comm_0 = communities[0]
    comm_1 = communities[1]

    if 0 in comm_0:
        room_a_comm, room_b_comm = comm_0, comm_1
    else:
        room_a_comm, room_b_comm = comm_1, comm_0

    assert 0 in room_a_comm and 15 in room_a_comm
    assert 25 in room_b_comm and 39 in room_b_comm
