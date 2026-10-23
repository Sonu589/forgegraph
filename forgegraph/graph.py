"""Curated troubleshooting graph used locally or seeded into AlloyDB tables."""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path
from typing import Any

GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "maintenance_graph.json"
_GRAPH = json.loads(GRAPH_PATH.read_text())
_NODES = {node["node_key"]: node for node in _GRAPH["nodes"]}
_EDGES = _GRAPH["edges"]


def start_nodes(flags: list[str]) -> list[str]:
    starts: list[str] = []
    text = " ".join(flags).lower()
    if "pressure" in text:
        starts.append("symptom:pressure_drop")
    if "temperature" in text or "oil temp" in text:
        starts.append("symptom:oil_overheat")
    if "vibration" in text:
        starts.append("symptom:vibration_spike")
    return starts


def local_graph_context(flags: list[str], max_depth: int = 2, max_nodes: int = 12) -> dict[str, Any]:
    starts = start_nodes(flags)
    if not starts:
        return {"nodes": [], "edges": [], "facts": []}
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for edge in _EDGES:
        adjacency.setdefault(edge["source_key"], []).append((edge["target_key"], edge))
        adjacency.setdefault(edge["target_key"], []).append((edge["source_key"], edge))
    depth_by_key = {key: 0 for key in starts}
    queue = deque(starts)
    traversed_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    while queue and len(depth_by_key) < max_nodes:
        current = queue.popleft()
        depth = depth_by_key[current]
        if depth >= max_depth:
            continue
        for neighbor, edge in adjacency.get(current, []):
            traversed_edges[(edge["source_key"], edge["target_key"], edge["relation"])] = edge
            if neighbor not in depth_by_key and len(depth_by_key) < max_nodes:
                depth_by_key[neighbor] = depth + 1
                queue.append(neighbor)
    nodes = [dict(_NODES[key], depth=depth) for key, depth in depth_by_key.items() if key in _NODES]
    edges = list(traversed_edges.values())
    facts = [
        f"{_NODES.get(edge['source_key'], {}).get('label', edge['source_key'])} --{edge['relation']}--> "
        f"{_NODES.get(edge['target_key'], {}).get('label', edge['target_key'])} "
        f"(confidence={edge['confidence']:.2f}; {edge['source_note']})"
        for edge in edges
    ]
    return {"nodes": nodes, "edges": edges, "facts": facts}
