from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import BBox
from .types import LayoutItem

_OVERLAP_PARAMS = {
    "regular": {"area_threshold": 1.3, "conf_threshold": 0.05},
    "picture": {"area_threshold": 2.0, "conf_threshold": 0.3},
    "wrapper": {"area_threshold": 2.0, "conf_threshold": 0.2},
}

_CONTAINER_TYPES = frozenset({"form", "key_value_region"})
_TABLE_TYPES = frozenset({"table", "document_index"})
_WRAPPER_TYPES = _CONTAINER_TYPES | _TABLE_TYPES
_SPECIAL_TYPES = _WRAPPER_TYPES | {"picture"}

_CONFIDENCE_THRESHOLDS = {
    "caption": 0.5,
    "footnote": 0.5,
    "formula": 0.5,
    "list_item": 0.5,
    "page_footer": 0.5,
    "page_header": 0.5,
    "picture": 0.5,
    "section_header": 0.45,
    "table": 0.5,
    "text": 0.5,
    "title": 0.45,
    "code": 0.45,
    "checkbox_selected": 0.45,
    "checkbox_unselected": 0.45,
    "form": 0.45,
    "key_value_region": 0.45,
    "document_index": 0.45,
}

_LABEL_REMAPPING = {"title": "section_header"}


@dataclass
class Cluster:
    id: int
    label: str
    bbox: BBox
    confidence: float
    children: list[Cluster] = field(default_factory=list)


class _UnionFind:
    def __init__(self, elements: Sequence[int]) -> None:
        self.parent = {e: e for e in elements}
        self.rank = dict.fromkeys(elements, 0)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        elif self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def get_groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for e in self.parent:
            groups[self.find(e)].append(e)
        return groups


def _ordered_bbox(b: BBox) -> BBox:
    return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))


def _area(b: BBox) -> float:
    b = _ordered_bbox(b)
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter_area(a: BBox, b: BBox) -> float:
    a, b = _ordered_bbox(a), _ordered_bbox(b)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _iou(a: BBox, b: BBox) -> float:
    inter = _inter_area(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _io_self(inner: BBox, outer: BBox) -> float:
    a = _area(inner)
    return _inter_area(inner, outer) / a if a > 0 else 0.0


def _check_overlap(b1: BBox, b2: BBox, overlap_thr: float, containment_thr: float) -> bool:
    if _area(b1) <= 0 or _area(b2) <= 0:
        return False
    return (
        _iou(b1, b2) > overlap_thr
        or _io_self(b1, b2) > containment_thr
        or _io_self(b2, b1) > containment_thr
    )


def _should_prefer(candidate: Cluster, other: Cluster, params: dict[str, float]) -> bool:
    cand_area = _area(candidate.bbox)
    other_area = _area(other.bbox)
    if candidate.label == "list_item" and other.label == "text" and other_area > 0:
        if abs(1.0 - cand_area / other_area) < 0.2:
            return True
    if candidate.label == "code" and _io_self(other.bbox, candidate.bbox) > 0.8:
        return True
    area_ratio = cand_area / other_area if other_area > 0 else float("inf")
    conf_diff = other.confidence - candidate.confidence
    if area_ratio <= params["area_threshold"] and conf_diff > params["conf_threshold"]:
        return False
    return True


def _select_best(group: list[Cluster], params: dict[str, float]) -> Cluster:
    best: Cluster | None = None
    for candidate in group:
        if all(_should_prefer(candidate, other, params) for other in group if other != candidate):
            if best is None or (
                _area(candidate.bbox) > _area(best.bbox)
                and best.confidence - candidate.confidence <= params["conf_threshold"]
            ):
                best = candidate
    return best if best is not None else group[0]


def _remove_overlapping(
    clusters: list[Cluster],
    cluster_type: str,
    overlap_thr: float = 0.8,
    containment_thr: float = 0.8,
) -> list[Cluster]:
    if not clusters:
        return []
    valid = {c.id: c for c in clusters}
    uf = _UnionFind([c.id for c in clusters])
    params = _OVERLAP_PARAMS[cluster_type]
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i + 1 :]:
            if _check_overlap(c1.bbox, c2.bbox, overlap_thr, containment_thr):
                uf.union(c1.id, c2.id)
    return [
        valid[grp[0]] if len(grp) == 1 else _select_best([valid[cid] for cid in grp], params)
        for grp in uf.get_groups().values()
    ]


def _resolve_coincident(
    losers: list[Cluster],
    winners: list[Cluster],
    iou_thr: float = 0.8,
    conf_tol: float = 0.1,
) -> set[int]:
    to_drop: set[int] = set()
    for loser in losers:
        for winner in winners:
            if _iou(loser.bbox, winner.bbox) <= iou_thr:
                continue
            if (loser.confidence - winner.confidence) < conf_tol:
                to_drop.add(loser.id)
                break
    return to_drop


def _handle_cross_type_overlaps(special: list[Cluster]) -> list[Cluster]:
    tables = [c for c in special if c.label == "table"]
    doc_indices = [c for c in special if c.label == "document_index"]
    pictures = [c for c in special if c.label == "picture"]
    containers = [c for c in special if c.label in _CONTAINER_TYPES]

    drop: set[int] = set()
    # Keep real tables unless a coincident document_index outscores them by >= 0.20.
    drop |= _resolve_coincident(tables, doc_indices, conf_tol=-0.20)
    surviving_tables = [c for c in tables if c.id not in drop]
    drop |= _resolve_coincident(doc_indices, surviving_tables, conf_tol=0.20)
    surviving_table_types = surviving_tables + [c for c in doc_indices if c.id not in drop]
    drop |= _resolve_coincident(pictures, surviving_table_types)
    surviving = surviving_table_types + [c for c in pictures if c.id not in drop]
    drop |= _resolve_coincident(containers, surviving)
    return [c for c in special if c.id not in drop]


def _sort_clusters(clusters: list[Cluster]) -> list[Cluster]:
    return sorted(clusters, key=lambda c: (c.bbox[1], c.bbox[0]))


def postprocess_layout(
    items: Sequence[LayoutItem],
    page_size: tuple[float, float],
    *,
    keep_empty_clusters: bool = False,
) -> list[Cluster]:
    # Ported from docling utils/layout_postprocessor.py (MIT) for cell-free image input.
    all_clusters = [
        Cluster(id=idx, label=it.label, bbox=it.bbox, confidence=it.score)
        for idx, it in enumerate(items)
        if it.label in _CONFIDENCE_THRESHOLDS
    ]
    regular = [c for c in all_clusters if c.label not in _SPECIAL_TYPES]
    special = [c for c in all_clusters if c.label in _SPECIAL_TYPES]

    regular = [c for c in regular if c.confidence >= _CONFIDENCE_THRESHOLDS[c.label]]
    for c in regular:
        c.label = _LABEL_REMAPPING.get(c.label, c.label)
    if not keep_empty_clusters:
        regular = [c for c in regular if c.label == "formula"]

    prev = len(regular) + 1
    for _ in range(3):
        if prev == len(regular):
            break
        prev = len(regular)
        regular = _remove_overlapping(regular, "regular")

    special = [c for c in special if c.confidence >= _CONFIDENCE_THRESHOLDS[c.label]]
    page_area = page_size[0] * page_size[1]
    if page_area > 0:
        special = [c for c in special if not (c.label == "picture" and _area(c.bbox) / page_area > 0.90)]

    pictures = _remove_overlapping([c for c in special if c.label == "picture"], "picture")
    tables = _remove_overlapping([c for c in special if c.label == "table"], "wrapper")
    doc_indices = _remove_overlapping([c for c in special if c.label == "document_index"], "wrapper")
    containers = _remove_overlapping([c for c in special if c.label in _CONTAINER_TYPES], "wrapper")

    special = _handle_cross_type_overlaps(pictures + tables + doc_indices + containers)
    pictures = [c for c in special if c.label == "picture"]
    tables = [c for c in special if c.label in _TABLE_TYPES]
    containers = [c for c in special if c.label in _CONTAINER_TYPES]

    nested = tables + pictures
    for cluster in nested:
        children = [r for r in regular if _io_self(r.bbox, cluster.bbox) > 0.8]
        if children:
            cluster.children = _sort_clusters(children)

    parent_by_nested: dict[int, int] = {}
    for child in nested:
        parents = [ct for ct in containers if _io_self(child.bbox, ct.bbox) > 0.8]
        if parents:
            parent = min(parents, key=lambda ct: (_area(ct.bbox), -ct.confidence, ct.id))
            parent_by_nested[child.id] = parent.id

    nested_reg_ids = {r.id for ch in nested for r in ch.children}
    parent_by_reg: dict[int, int] = {}
    for child in regular:
        if child.id in nested_reg_ids:
            continue
        parents = [ct for ct in containers if _io_self(child.bbox, ct.bbox) > 0.8]
        if parents:
            parent = min(parents, key=lambda ct: (_area(ct.bbox), -ct.confidence, ct.id))
            parent_by_reg[child.id] = parent.id

    for container in containers:
        children = [r for r in regular if parent_by_reg.get(r.id) == container.id] + [
            n for n in nested if parent_by_nested.get(n.id) == container.id
        ]
        if children:
            container.children = _sort_clusters(children)
            container.bbox = (
                min(ch.bbox[0] for ch in container.children),
                min(ch.bbox[1] for ch in container.children),
                max(ch.bbox[2] for ch in container.children),
                max(ch.bbox[3] for ch in container.children),
            )

    special = pictures + tables + containers
    contained_ids = {
        ch.id for w in special if w.label in _TABLE_TYPES or w.label == "picture" for ch in w.children
    }
    regular = [r for r in regular if r.id not in contained_ids]
    return _sort_clusters(regular + special)


def clusters_to_items(clusters: Sequence[Cluster]) -> list[dict[str, object]]:
    # Mirrors PageAssembleModel + ReadingOrderModel.iterate_leaves for cell-free image input.
    ordered = sorted(clusters, key=lambda c: (c.bbox[1], c.bbox[0]))
    out: list[dict[str, object]] = []
    for cluster in ordered:
        if cluster.label in _CONTAINER_TYPES:
            if cluster.children:
                for ch in cluster.children:
                    out.append({
                        "order": len(out),
                        "label": ch.label,
                        "bbox": list(ch.bbox),
                        "text": "",
                        "score": round(float(ch.confidence), 4),
                    })
            continue
        out.append({
            "order": len(out),
            "label": cluster.label,
            "bbox": list(cluster.bbox),
            "text": "",
            "score": round(float(cluster.confidence), 4),
        })
    return out
