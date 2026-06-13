from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eduplanbench.core.io import ensure_dir, parse_list_cell, read_json, write_json


@dataclass(slots=True)
class GraphNode:
    node_id: str
    type: str
    text: str = ""
    title: str = ""
    concepts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TextResourceGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    _resource_index: dict[str, list[GraphNode]] | None = field(default=None, init=False, repr=False)
    _prereq_index: dict[str, list[str]] | None = field(default=None, init=False, repr=False)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        self._resource_index = None

    def add_edge(self, source: str, target: str, edge_type: str, *, weight: float = 1.0) -> None:
        if source and target:
            self.edges.append(GraphEdge(source=source, target=target, type=edge_type, weight=weight))
            if edge_type == "prerequisite":
                self._prereq_index = None

    def resources_for_concepts(self, concepts: list[str], *, limit: int = 20, text_fallback: bool = True) -> list[GraphNode]:
        index = self._ensure_resource_index()
        matches: list[GraphNode] = []
        seen: set[str] = set()
        for concept in concepts:
            for node in index.get(concept, []):
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                matches.append(node)
                if len(matches) >= limit:
                    return matches
        if matches:
            return matches
        if not text_fallback:
            return matches
        wanted = set(concepts)
        for node in self.nodes.values():
            if node.type == "concept":
                continue
            if wanted.intersection(node.concepts) or any(item in node.text for item in wanted):
                matches.append(node)
                if len(matches) >= limit:
                    break
        return matches

    def prerequisites_of(self, concept: str) -> list[str]:
        return list(self._ensure_prereq_index().get(concept, []))

    def _ensure_resource_index(self) -> dict[str, list[GraphNode]]:
        if self._resource_index is None:
            index: dict[str, list[GraphNode]] = {}
            for node in self.nodes.values():
                if node.type == "concept":
                    continue
                for concept in node.concepts:
                    index.setdefault(concept, []).append(node)
            self._resource_index = index
        return self._resource_index

    def _ensure_prereq_index(self) -> dict[str, list[str]]:
        if self._prereq_index is None:
            index: dict[str, list[str]] = {}
            for edge in self.edges:
                if edge.type == "prerequisite":
                    index.setdefault(edge.target, []).append(edge.source)
            self._prereq_index = index
        return self._prereq_index

    def to_json(self, path: str | Path) -> None:
        write_json(
            path,
            {
                "nodes": [asdict(node) for node in self.nodes.values()],
                "edges": [asdict(edge) for edge in self.edges],
            },
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TextResourceGraph":
        payload = read_json(path)
        graph = cls()
        for node in payload.get("nodes", []):
            graph.add_node(GraphNode(**node))
        for edge in payload.get("edges", []):
            graph.edges.append(GraphEdge(**edge))
        return graph

    @classmethod
    def from_mooccubex(cls, root: Path, *, limit: int | None = 50_000) -> "TextResourceGraph":
        graph = cls()
        _load_concepts(root / "entities" / "concept.json", graph, limit=limit)
        _load_courses(root / "entities" / "course.json", graph, limit=limit)
        _load_relation_file(root / "relations" / "concept-course.txt", graph, "covers", "course", limit=limit)
        _load_relation_file(root / "relations" / "concept-video.txt", graph, "covers", "lecture_text", limit=limit)
        _load_relation_file(root / "relations" / "concept-problem.txt", graph, "requires", "exercise", limit=limit)
        for name in ("math.json", "cs.json", "psy.json"):
            _load_prerequisites(root / "prerequisites" / name, graph, limit=limit)
        return graph


def _iter_json_records(path: Path, *, limit: int | None) -> list[Any]:
    if not path.exists():
        return []
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            payload = json.load(fh)
            return payload[:limit] if limit is not None else payload
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_concepts(path: Path, graph: TextResourceGraph, *, limit: int | None) -> None:
    for item in _iter_json_records(path, limit=limit):
        cid = str(item.get("id") or item.get("concept_id") or item.get("name") or item.get("ccid") or "")
        name = str(item.get("name") or item.get("title") or cid)
        context = item.get("context") or []
        context_text = "\n".join(str(part) for part in context[:3]) if isinstance(context, list) else str(context)
        if cid:
            graph.add_node(GraphNode(node_id=cid, type="concept", title=name, text=f"{name}\n{context_text}", concepts=[cid], metadata=item))


def _load_courses(path: Path, graph: TextResourceGraph, *, limit: int | None) -> None:
    for item in _iter_json_records(path, limit=limit):
        cid = str(item.get("id") or item.get("course_id") or item.get("cid") or item.get("name") or "")
        title = str(item.get("name") or item.get("title") or cid)
        intro = str(item.get("about") or item.get("description") or item.get("prerequisites") or "")
        if cid:
            graph.add_node(GraphNode(node_id=cid, type="course", title=title, text=f"{title}\n{intro}", metadata=item))


def _load_relation_file(path: Path, graph: TextResourceGraph, edge_type: str, resource_type: str, *, limit: int | None) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            parts = line.strip().replace("\t", " ").split()
            if len(parts) < 2:
                continue
            concept, resource = parts[0], parts[1]
            if concept not in graph.nodes:
                graph.add_node(GraphNode(node_id=concept, type="concept", title=concept, text=concept, concepts=[concept]))
            if resource not in graph.nodes:
                graph.add_node(
                    GraphNode(
                        node_id=resource,
                        type=resource_type,
                        title=resource,
                        text=resource,
                        concepts=[concept],
                    )
                )
            else:
                node = graph.nodes[resource]
                if concept not in node.concepts:
                    node.concepts.append(concept)
            graph.add_edge(concept, resource, edge_type)


def _load_prerequisites(path: Path, graph: TextResourceGraph, *, limit: int | None) -> None:
    if not path.exists():
        return
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        payload = _iter_json_records(path, limit=limit)
    pairs: list[tuple[str, str]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                if "ground_truth" in item and int(item.get("ground_truth") or 0) != 1:
                    continue
                source = str(item.get("prerequisite") or item.get("source") or item.get("from") or item.get("c1") or "")
                target = str(item.get("concept") or item.get("target") or item.get("to") or item.get("c2") or "")
                if source and target:
                    pairs.append((source, target))
            elif isinstance(item, list) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
    elif isinstance(payload, dict):
        for target, sources in payload.items():
            for source in parse_list_cell(str(sources)) if not isinstance(sources, list) else sources:
                pairs.append((str(source), str(target)))
    for source, target in pairs[:limit]:
        for cid in (source, target):
            if cid not in graph.nodes:
                graph.add_node(GraphNode(node_id=cid, type="concept", title=cid, text=cid, concepts=[cid]))
        graph.add_edge(source, target, "prerequisite")
