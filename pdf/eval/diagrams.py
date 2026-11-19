from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdf._test_pdf import TestPdfBuilder  # noqa: E402
from pdf.eval.gt import append_run_log  # noqa: E402
from pdf.eval.metrics import iou, prf  # noqa: E402

_TEXT_NODE_MIN_IOU = 0.25
_DEFAULT_GT_CANDIDATES = (
    Path(__file__).with_name("diagrams_gt.json"),
    ROOT / "datasets" / "diagrams" / "diagrams_gt.json",
)


def normalize_label(text: str | None) -> str:
    return " ".join(str(text or "").split()).lower()


def _arrow_triangle(tip: tuple[float, float], dx: float, dy: float, size: float = 8.0) -> list[tuple[float, float]]:
    # Unit direction from tail toward tip defines arrowhead base perpendiculars.
    mag = (dx * dx + dy * dy) ** 0.5
    if mag <= 1e-6:
        return [tip, (tip[0] - size, tip[1] - size * 0.5), (tip[0] - size, tip[1] + size * 0.5)]
    ux, uy = dx / mag, dy / mag
    bx, by = tip[0] - ux * size, tip[1] - uy * size
    half_w = size * 0.45
    px, py = -uy * half_w, ux * half_w
    return [tip, (bx + px, by + py), (bx - px, by - py)]


def _draw_node_box(page: Any, bbox: tuple[float, float, float, float], label: str, *, width: float = 1.0) -> None:
    page.draw_rect(bbox, width=width)
    cx = (bbox[0] + bbox[2]) * 0.5
    cy = (bbox[1] + bbox[3]) * 0.5
    tx = max(bbox[0] + 6.0, cx - len(label) * 2.8)
    page.insert_text((tx, cy + 3.5), label, fontsize=10.0)


def _draw_connector(
    page: Any,
    points: Sequence[tuple[float, float]],
    *,
    dash: tuple[float, float] | None = None,
    arrow_start: bool = False,
    arrow_end: bool = True,
) -> None:
    for p1, p2 in zip(points, points[1:]):
        page.draw_line(p1, p2, width=1.0, dash=dash)
    if arrow_end and len(points) >= 2:
        p_prev, p_last = points[-2], points[-1]
        page.draw_polygon(_arrow_triangle(p_last, p_last[0] - p_prev[0], p_last[1] - p_prev[1]), fill=True, stroke=False)
    if arrow_start and len(points) >= 2:
        p_next, p_first = points[1], points[0]
        page.draw_polygon(_arrow_triangle(p_first, p_first[0] - p_next[0], p_first[1] - p_next[1]), fill=True, stroke=False)


def _gt_node(node_id: str, bbox: tuple[float, float, float, float], label: str, *, shape_type: str = "rect", parent_group_id: str | None = None) -> dict[str, Any]:
    return {
        "id": node_id,
        "shape_type": shape_type,
        "label": label,
        "bbox": list(bbox),
        "parent_group_id": parent_group_id,
    }


def _gt_edge(edge_id: str, src: str, dst: str, *, direction: str = "directed", style: str = "solid", label: str = "") -> dict[str, Any]:
    return {
        "id": edge_id,
        "source_node_id": src,
        "target_node_id": dst,
        "direction": direction,
        "style": style,
        "label": label,
    }


def _gt_group(group_id: str, bbox: tuple[float, float, float, float], label: str, *, parent_group_id: str | None = None, member_node_ids: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "id": group_id,
        "label": label,
        "bbox": list(bbox),
        "parent_group_id": parent_group_id,
        "member_node_ids": list(member_node_ids),
    }


def check_diagram_integrity(diagram_dict: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    nodes = diagram_dict.get("nodes") or ()
    edges = diagram_dict.get("edges") or ()
    groups = diagram_dict.get("groups") or ()

    node_ids: set[str] = set()
    for node in nodes:
        nid = node.get("id")
        if not nid:
            violations.append("node missing id")
            continue
        if nid in node_ids:
            violations.append(f"duplicate node id {nid!r}")
        node_ids.add(nid)

    group_ids: set[str] = set()
    parent_map: dict[str, str | None] = {}
    for group in groups:
        gid = group.get("id")
        if not gid:
            violations.append("group missing id")
            continue
        if gid in group_ids:
            violations.append(f"duplicate group id {gid!r}")
        group_ids.add(gid)
        parent_map[gid] = group.get("parent_group_id")

    edge_ids: set[str] = set()
    for edge in edges:
        eid = edge.get("id")
        if not eid:
            violations.append("edge missing id")
            eid = "?"
        else:
            if eid in edge_ids:
                violations.append(f"duplicate edge id {eid!r}")
            edge_ids.add(eid)
        src = edge.get("source_node_id")
        dst = edge.get("target_node_id")
        if src is not None and src not in node_ids:
            violations.append(f"edge {eid!r} source_node_id {src!r} missing")
        if dst is not None and dst not in node_ids:
            violations.append(f"edge {eid!r} target_node_id {dst!r} missing")

    node_parent: dict[str, str | None] = {}
    for node in nodes:
        nid = node.get("id", "?")
        pg = node.get("parent_group_id")
        if pg is not None and pg not in group_ids:
            violations.append(f"node {nid!r} parent_group_id {pg!r} missing")
        if nid in node_ids:
            node_parent[nid] = pg
        tids = node.get("text_element_ids") or ()
        seen_tids: set[str] = set()
        for tid in tids:
            if not isinstance(tid, str) or not tid:
                violations.append(f"node {nid!r} has invalid text_element_id {tid!r}")
            elif tid in seen_tids:
                violations.append(f"node {nid!r} has duplicate text_element_id {tid!r}")
            else:
                seen_tids.add(tid)

    group_members: dict[str, set[str]] = {}
    for group in groups:
        gid = group.get("id", "?")
        members: set[str] = set()
        for mid in group.get("member_node_ids") or ():
            if mid not in node_ids:
                violations.append(f"group {gid!r} member_node_id {mid!r} missing")
            else:
                members.add(mid)
                if node_parent.get(mid) != gid:
                    violations.append(f"group {gid!r} member_node_id {mid!r} parent_group_id mismatch")
        if gid in group_ids:
            group_members[gid] = members
        pg = group.get("parent_group_id")
        if pg is not None and pg not in group_ids:
            violations.append(f"group {gid!r} parent_group_id {pg!r} missing")

    for nid, pg in node_parent.items():
        if pg is not None and pg in group_members and nid not in group_members[pg]:
            violations.append(f"node {nid!r} parent_group_id {pg!r} missing from member_node_ids")

    reported_cycles: set[frozenset[str]] = set()
    for gid in parent_map:
        trail: list[str] = []
        seen: dict[str, int] = {}
        cur: str | None = gid
        while cur is not None and cur in parent_map:
            if cur in seen:
                cycle = frozenset(trail[seen[cur]:])
                if cycle not in reported_cycles:
                    reported_cycles.add(cycle)
                    violations.append(f"group parent cycle involving {sorted(cycle)}")
                break
            seen[cur] = len(trail)
            trail.append(cur)
            cur = parent_map.get(cur)

    return violations


def _match_nodes_greedy(
    pred_nodes: Sequence[dict[str, Any]],
    gt_nodes: Sequence[dict[str, Any]],
    min_iou: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used_p: set[int] = set()
    used_g: set[int] = set()

    strict_cands = sorted(
        (
            (-iou(p["bbox"], g["bbox"]), pi, gi)
            for pi, p in enumerate(pred_nodes)
            for gi, g in enumerate(gt_nodes)
            if iou(p["bbox"], g["bbox"]) >= min_iou
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for _, pi, gi in strict_cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))

    text_cands = sorted(
        (
            (-iou(p["bbox"], g["bbox"]), pi, gi)
            for pi, p in enumerate(pred_nodes)
            for gi, g in enumerate(gt_nodes)
            if pi not in used_p
            and gi not in used_g
            and (p.get("shape_type") == "text_node" or g.get("shape_type") == "text_node")
            and normalize_label(p.get("label"))
            and normalize_label(p.get("label")) == normalize_label(g.get("label"))
            and iou(p["bbox"], g["bbox"]) >= _TEXT_NODE_MIN_IOU
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for _, pi, gi in text_cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))

    return pairs


def _edge_matches(
    pred_edge: dict[str, Any],
    gt_edge: dict[str, Any],
    pred_to_gt_id: dict[str, str],
) -> bool:
    src = pred_edge.get("source_node_id")
    dst = pred_edge.get("target_node_id")
    if not src or not dst or src not in pred_to_gt_id or dst not in pred_to_gt_id:
        return False
    ms, md = pred_to_gt_id[src], pred_to_gt_id[dst]
    p_dir = str(pred_edge.get("direction", "directed")).lower()
    g_dir = str(gt_edge.get("direction", "directed")).lower()
    p_style = str(pred_edge.get("style", "solid")).lower()
    g_style = str(gt_edge.get("style", "solid")).lower()
    if p_dir != g_dir or p_style != g_style:
        return False
    gs = gt_edge.get("source_node_id")
    gd = gt_edge.get("target_node_id")
    if p_dir in ("undirected", "bidirectional"):
        return (ms, md) == (gs, gd) or (ms, md) == (gd, gs)
    return (ms, md) == (gs, gd)


def diagram_graph_metrics(
    pred_diagram_dict: dict[str, Any],
    gt_diagram_dict: dict[str, Any],
    *,
    min_iou: float = 0.5,
) -> dict[str, Any]:
    pred_nodes = list(pred_diagram_dict.get("nodes") or ())
    gt_nodes = list(gt_diagram_dict.get("nodes") or ())
    pred_edges = list(pred_diagram_dict.get("edges") or ())
    gt_edges = list(gt_diagram_dict.get("edges") or ())

    matched_pairs = _match_nodes_greedy(pred_nodes, gt_nodes, min_iou)
    node_tp = len(matched_pairs)
    node_fp = len(pred_nodes) - node_tp
    node_fn = len(gt_nodes) - node_tp
    if node_tp + node_fp + node_fn > 0:
        node_p, node_r, node_f1 = prf(node_tp, node_fp, node_fn)
    else:
        node_p, node_r, node_f1 = 1.0, 1.0, 1.0

    label_total = node_tp
    label_exact = sum(
        1
        for pi, gi in matched_pairs
        if normalize_label(pred_nodes[pi].get("label")) == normalize_label(gt_nodes[gi].get("label"))
    )
    if label_total > 0:
        text_to_shape_acc = label_exact / label_total
    else:
        text_to_shape_acc = 0.0

    pred_to_gt_id = {
        pred_nodes[pi]["id"]: gt_nodes[gi]["id"]
        for pi, gi in matched_pairs
        if pred_nodes[pi].get("id") and gt_nodes[gi].get("id")
    }
    used_gt_edges: set[int] = set()
    edge_tp = 0
    for pe in pred_edges:
        p_lbl = normalize_label(pe.get("label"))
        cands = [
            gi
            for gi, ge in enumerate(gt_edges)
            if gi not in used_gt_edges and _edge_matches(pe, ge, pred_to_gt_id)
        ]
        if not cands:
            continue
        exact_lbl = [gi for gi in cands if normalize_label(gt_edges[gi].get("label")) == p_lbl]
        chosen = exact_lbl[0] if exact_lbl else cands[0]
        used_gt_edges.add(chosen)
        edge_tp += 1

    edge_fp = len(pred_edges) - edge_tp
    edge_fn = len(gt_edges) - edge_tp
    if edge_tp + edge_fp + edge_fn > 0:
        edge_p, edge_r, edge_f1 = prf(edge_tp, edge_fp, edge_fn)
    else:
        edge_p, edge_r, edge_f1 = 1.0, 1.0, 1.0

    graph_tp = node_tp + edge_tp
    graph_fp = node_fp + edge_fp
    graph_fn = node_fn + edge_fn
    graph_f1 = prf(graph_tp, graph_fp, graph_fn)[2] if (graph_tp + graph_fp + graph_fn) > 0 else 1.0

    return {
        "node_tp": node_tp,
        "node_fp": node_fp,
        "node_fn": node_fn,
        "node_precision": round(node_p, 4),
        "node_recall": round(node_r, 4),
        "node_f1": round(node_f1, 4),
        "label_exact": label_exact,
        "label_total": label_total,
        "text_to_shape_acc": round(text_to_shape_acc, 4),
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "edge_precision": round(edge_p, 4),
        "edge_recall": round(edge_r, 4),
        "edge_f1": round(edge_f1, 4),
        "graph_f1": round(graph_f1, 4),
    }


def build_diagram_fixtures(dst_dir: Path) -> list[dict[str, Any]]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    fixtures: list[dict[str, Any]] = []

    def _save(name: str, fig_box: tuple[float, float, float, float], build_fn: Any, gt: dict[str, Any]) -> None:
        builder = TestPdfBuilder()
        page = builder.new_page(width=595.0, height=440.0)
        build_fn(page)
        pdf_path = dst_dir / f"{name}.pdf"
        builder.save(pdf_path)
        fixtures.append({"name": name, "pdf_path": pdf_path, "figure_bbox": fig_box, "gt_diagram": gt})

    def _build_flowchart(page: Any) -> None:
        _draw_node_box(page, (60.0, 90.0, 160.0, 140.0), "Start")
        _draw_node_box(page, (220.0, 90.0, 320.0, 140.0), "Validate")
        _draw_node_box(page, (220.0, 210.0, 320.0, 260.0), "Process")
        page.insert_text((395.0, 118.0), "Done", fontsize=10.0)
        _draw_connector(page, [(160.0, 115.0), (220.0, 115.0)])
        page.insert_text((182.0, 106.0), "ok", fontsize=9.0)
        _draw_connector(page, [(270.0, 140.0), (270.0, 210.0)])
        _draw_connector(page, [(320.0, 115.0), (390.0, 115.0)])

    _save(
        "flowchart",
        (45.0, 70.0, 455.0, 280.0),
        _build_flowchart,
        {
            "bbox": [45.0, 70.0, 455.0, 280.0],
            "diagram_type": "flowchart",
            "nodes": [
                _gt_node("n1", (60.0, 90.0, 160.0, 140.0), "Start"),
                _gt_node("n2", (220.0, 90.0, 320.0, 140.0), "Validate"),
                _gt_node("n3", (220.0, 210.0, 320.0, 260.0), "Process"),
                _gt_node("n4", (395.0, 108.0, 422.0, 121.0), "Done", shape_type="text_node"),
            ],
            "edges": [
                _gt_edge("e1", "n1", "n2", label="ok"),
                _gt_edge("e2", "n2", "n3"),
                _gt_edge("e3", "n2", "n4"),
            ],
            "groups": [],
        },
    )

    def _build_dashed_and_bidir(page: Any) -> None:
        _draw_node_box(page, (60.0, 90.0, 170.0, 145.0), "Client")
        _draw_node_box(page, (290.0, 90.0, 400.0, 145.0), "Broker")
        _draw_node_box(page, (290.0, 215.0, 400.0, 270.0), "Store")
        _draw_connector(page, [(170.0, 117.5), (290.0, 117.5)], arrow_start=True, arrow_end=True)
        _draw_connector(page, [(345.0, 145.0), (345.0, 215.0)], dash=(4.0, 3.0), arrow_end=True)
        _draw_connector(page, [(115.0, 145.0), (115.0, 242.5), (290.0, 242.5)], dash=(4.0, 3.0), arrow_end=True)

    _save(
        "dashed_and_bidir",
        (45.0, 70.0, 420.0, 290.0),
        _build_dashed_and_bidir,
        {
            "bbox": [45.0, 70.0, 420.0, 290.0],
            "diagram_type": "flowchart",
            "nodes": [
                _gt_node("n1", (60.0, 90.0, 170.0, 145.0), "Client"),
                _gt_node("n2", (290.0, 90.0, 400.0, 145.0), "Broker"),
                _gt_node("n3", (290.0, 215.0, 400.0, 270.0), "Store"),
            ],
            "edges": [
                _gt_edge("e1", "n1", "n2", direction="bidirectional", style="solid"),
                _gt_edge("e2", "n2", "n3", direction="directed", style="dashed"),
                _gt_edge("e3", "n1", "n3", direction="directed", style="dashed"),
            ],
            "groups": [],
        },
    )

    def _build_self_loop(page: Any) -> None:
        _draw_node_box(page, (110.0, 130.0, 230.0, 190.0), "WaitState")
        _draw_node_box(page, (320.0, 130.0, 440.0, 190.0), "Active")
        _draw_connector(page, [(195.0, 130.0), (195.0, 80.0), (145.0, 80.0), (145.0, 130.0)], arrow_end=True)
        _draw_connector(page, [(230.0, 160.0), (320.0, 160.0)], arrow_end=True)

    _save(
        "self_loop",
        (90.0, 60.0, 460.0, 210.0),
        _build_self_loop,
        {
            "bbox": [90.0, 60.0, 460.0, 210.0],
            "diagram_type": "flowchart",
            "nodes": [
                _gt_node("n1", (110.0, 130.0, 230.0, 190.0), "WaitState"),
                _gt_node("n2", (320.0, 130.0, 440.0, 190.0), "Active"),
            ],
            "edges": [
                _gt_edge("e1", "n1", "n1"),
                _gt_edge("e2", "n1", "n2"),
            ],
            "groups": [],
        },
    )

    def _build_swimlane_dag(page: Any) -> None:
        page.draw_rect((50.0, 50.0, 520.0, 330.0), width=1.2)
        page.insert_text((62.0, 70.0), "System", fontsize=11.0, bold=True)
        page.draw_rect((70.0, 90.0, 290.0, 305.0), width=1.0, dash=(5.0, 3.0))
        page.insert_text((82.0, 108.0), "SubsystemA", fontsize=10.0, bold=True)
        _draw_node_box(page, (95.0, 130.0, 200.0, 180.0), "TaskA")
        _draw_node_box(page, (95.0, 225.0, 200.0, 275.0), "TaskB")
        _draw_node_box(page, (365.0, 175.0, 475.0, 230.0), "Sink")
        _draw_connector(page, [(147.5, 180.0), (147.5, 225.0)], arrow_end=True)
        _draw_connector(page, [(200.0, 155.0), (315.0, 155.0), (315.0, 202.5), (365.0, 202.5)], arrow_end=True)

    _save(
        "swimlane_dag",
        (40.0, 40.0, 530.0, 340.0),
        _build_swimlane_dag,
        {
            "bbox": [40.0, 40.0, 530.0, 340.0],
            "diagram_type": "flowchart",
            "nodes": [
                _gt_node("n1", (95.0, 130.0, 200.0, 180.0), "TaskA", parent_group_id="g2"),
                _gt_node("n2", (95.0, 225.0, 200.0, 275.0), "TaskB", parent_group_id="g2"),
                _gt_node("n3", (365.0, 175.0, 475.0, 230.0), "Sink", parent_group_id="g1"),
            ],
            "edges": [
                _gt_edge("e1", "n1", "n2"),
                _gt_edge("e2", "n1", "n3"),
            ],
            "groups": [
                _gt_group("g1", (50.0, 50.0, 520.0, 330.0), "System", member_node_ids=("n3",)),
                _gt_group("g2", (70.0, 90.0, 290.0, 305.0), "SubsystemA", parent_group_id="g1", member_node_ids=("n1", "n2")),
            ],
        },
    )

    def _build_x_crossing(page: Any) -> None:
        _draw_node_box(page, (70.0, 70.0, 175.0, 120.0), "N1")
        _draw_node_box(page, (345.0, 70.0, 450.0, 120.0), "N2")
        _draw_node_box(page, (70.0, 235.0, 175.0, 285.0), "N3")
        _draw_node_box(page, (345.0, 235.0, 450.0, 285.0), "N4")
        _draw_connector(page, [(175.0, 105.0), (345.0, 250.0)], arrow_end=True)
        _draw_connector(page, [(345.0, 105.0), (175.0, 250.0)], arrow_end=True)

    _save(
        "x_crossing",
        (55.0, 55.0, 465.0, 300.0),
        _build_x_crossing,
        {
            "bbox": [55.0, 55.0, 465.0, 300.0],
            "diagram_type": "flowchart",
            "nodes": [
                _gt_node("n1", (70.0, 70.0, 175.0, 120.0), "N1"),
                _gt_node("n2", (345.0, 70.0, 450.0, 120.0), "N2"),
                _gt_node("n3", (70.0, 235.0, 175.0, 285.0), "N3"),
                _gt_node("n4", (345.0, 235.0, 450.0, 285.0), "N4"),
            ],
            "edges": [
                _gt_edge("e1", "n1", "n4"),
                _gt_edge("e2", "n2", "n3"),
            ],
            "groups": [],
        },
    )

    def _build_layered_autosar(page: Any) -> None:
        page.draw_rect((80.0, 60.0, 480.0, 315.0), width=1.2)
        page.insert_text((95.0, 82.0), "AUTOSAR Stack", fontsize=11.0, bold=True)
        _draw_node_box(page, (110.0, 105.0, 450.0, 150.0), "Application Layer")
        _draw_node_box(page, (110.0, 170.0, 450.0, 215.0), "Runtime Environment")
        _draw_node_box(page, (110.0, 235.0, 450.0, 280.0), "Basic Software")

    _save(
        "layered_autosar",
        (65.0, 45.0, 495.0, 330.0),
        _build_layered_autosar,
        {
            "bbox": [65.0, 45.0, 495.0, 330.0],
            "diagram_type": "layered",
            "nodes": [
                _gt_node("n1", (110.0, 105.0, 450.0, 150.0), "Application Layer", parent_group_id="g1"),
                _gt_node("n2", (110.0, 170.0, 450.0, 215.0), "Runtime Environment", parent_group_id="g1"),
                _gt_node("n3", (110.0, 235.0, 450.0, 280.0), "Basic Software", parent_group_id="g1"),
            ],
            "edges": [],
            "groups": [
                _gt_group("g1", (80.0, 60.0, 480.0, 315.0), "AUTOSAR Stack", member_node_ids=("n1", "n2", "n3")),
            ],
        },
    )

    return fixtures


def _aggregate_diagram_scores(rows: Sequence[dict[str, Any]], *, wall_sec: float, dataset: str, split: str, device: str) -> dict[str, Any]:
    from pdf.backends.rss import peak_rss_mb

    node_tp = sum(r["node_tp"] for r in rows)
    node_fp = sum(r["node_fp"] for r in rows)
    node_fn = sum(r["node_fn"] for r in rows)
    edge_tp = sum(r["edge_tp"] for r in rows)
    edge_fp = sum(r["edge_fp"] for r in rows)
    edge_fn = sum(r["edge_fn"] for r in rows)
    label_exact = sum(r["label_exact"] for r in rows)
    label_total = sum(r["label_total"] for r in rows)
    violations = sum(r["integrity_violations"] for r in rows)

    np, nr, nf1 = prf(node_tp, node_fp, node_fn) if (node_tp + node_fp + node_fn) > 0 else (1.0, 1.0, 1.0)
    ep, er, ef1 = prf(edge_tp, edge_fp, edge_fn) if (edge_tp + edge_fp + edge_fn) > 0 else (1.0, 1.0, 1.0)
    gtp, gfp, gfn = node_tp + edge_tp, node_fp + edge_fp, node_fn + edge_fn
    gf1 = prf(gtp, gfp, gfn)[2] if (gtp + gfp + gfn) > 0 else 1.0
    t2s = (label_exact / label_total) if label_total > 0 else 0.0

    return {
        "dataset": dataset,
        "split": split,
        "device": device,
        "diagrams": len(rows),
        "node_precision": round(np, 4),
        "node_recall": round(nr, 4),
        "node_f1": round(nf1, 4),
        "edge_precision": round(ep, 4),
        "edge_recall": round(er, 4),
        "edge_f1": round(ef1, 4),
        "graph_f1": round(gf1, 4),
        "text_to_shape_acc": round(t2s, 4),
        "integrity_violations": violations,
        "wall_sec": round(wall_sec, 4),
        "ram_mb": peak_rss_mb(),
        "fixtures": list(rows),
    }


def bench_synthetic_diagrams(*, device: str = "auto") -> dict[str, Any]:
    import pdfplumber

    from pdf.diagrams import diagram_to_ir_dict, extract_diagrams_on_page
    from pdf.text import extract_words

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        fixtures = build_diagram_fixtures(Path(tmp))
        for fix in fixtures:
            pdf_path = fix["pdf_path"]
            fig_box = tuple(float(v) for v in fix["figure_bbox"])
            with pdfplumber.open(str(pdf_path)) as doc:
                page = doc.pages[0]
                words = extract_words(page)
                parsed = extract_diagrams_on_page(
                    page,
                    [fig_box],
                    words=words,
                    is_raster=False,
                    extractor="vector_first",
                    device=device,
                )
            pred_dict = (
                diagram_to_ir_dict("diag_0001", "elem_0001", 1, pdf_path.name, parsed[0])
                if parsed
                else {"nodes": [], "edges": [], "groups": []}
            )
            violations = check_diagram_integrity(pred_dict)
            metrics = diagram_graph_metrics(pred_dict, fix["gt_diagram"])
            rows.append({
                "name": fix["name"],
                "diagram_type": pred_dict.get("diagram_type", ""),
                "integrity_violations": len(violations),
                "violations": violations,
                **metrics,
            })
    return _aggregate_diagram_scores(
        rows,
        wall_sec=time.perf_counter() - t0,
        dataset="synthetic_diagrams",
        split="synthetic",
        device=device,
    )


def resolve_diagram_gt_path(gt_path: str | Path | None = None) -> Path:
    if gt_path is not None:
        p = Path(gt_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"diagram GT JSON not found: {p}")
    for cand in _DEFAULT_GT_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError("diagram GT JSON not found; pass --gt <json_path>")


def load_diagram_gt(gt_path: str | Path | None = None, *, split: str = "dev") -> list[dict[str, Any]]:
    if split not in ("dev", "test"):
        raise ValueError(f"unknown split: {split!r}")
    resolved = resolve_diagram_gt_path(gt_path)
    raw = json.loads(resolved.read_text(encoding="utf-8-sig"))
    entries = raw["diagrams"] if isinstance(raw, dict) and "diagrams" in raw else raw
    if not isinstance(entries, list):
        raise ValueError(f"invalid diagram GT format in {resolved}")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(entries):
        item_split = item.get("split")
        if item_split is not None and item_split != split:
            continue
        pdf_raw = Path(item["pdf_path"])
        if not pdf_raw.is_absolute():
            rel_to_gt = resolved.parent / pdf_raw
            pdf_path = rel_to_gt if rel_to_gt.is_file() else (ROOT / pdf_raw)
        else:
            pdf_path = pdf_raw
        gt_diag = item["diagram"] if "diagram" in item else item
        out.append({
            "name": item.get("name", f"gt_{idx + 1}"),
            "pdf_path": pdf_path,
            "page": int(item.get("page", 1)),
            "gt_diagram": gt_diag,
        })
    return out


def bench_gt_diagrams(
    *,
    gt_path: str | Path | None = None,
    split: str = "dev",
    device: str = "auto",
) -> dict[str, Any]:
    from pdf import extract_ir

    entries = load_diagram_gt(gt_path, split=split)
    if not entries:
        raise FileNotFoundError(f"no diagram GT entries found for split={split!r}")

    rows: list[dict[str, Any]] = []
    ir_cache: dict[Path, dict[str, Any]] = {}
    t0 = time.perf_counter()
    for entry in entries:
        pdf_path = entry["pdf_path"]
        ir = ir_cache.get(pdf_path)
        if ir is None:
            ir = extract_ir(pdf_path, figures=True, ocr=False, tables=False, diagrams=True, device=device)
            ir_cache[pdf_path] = ir
        page_diags = [d for d in ir.get("diagrams", ()) if int(d.get("page", 0)) == entry["page"]]
        gt_diag = entry["gt_diagram"]
        if page_diags:
            gt_box = gt_diag.get("bbox")
            if gt_box and len(gt_box) == 4:
                best = max(page_diags, key=lambda d: iou(d["bbox"], gt_box))
                pred_dict = best if iou(best["bbox"], gt_box) >= 0.3 else {"nodes": [], "edges": [], "groups": []}
            else:
                pred_dict = page_diags[0]
        else:
            pred_dict = {"nodes": [], "edges": [], "groups": []}
        violations = check_diagram_integrity(pred_dict)
        metrics = diagram_graph_metrics(pred_dict, gt_diag)
        rows.append({
            "name": entry["name"],
            "integrity_violations": len(violations),
            "violations": violations,
            **metrics,
        })
    return _aggregate_diagram_scores(
        rows,
        wall_sec=time.perf_counter() - t0,
        dataset="golden_diagrams",
        split=split,
        device=device,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark vector diagram node-edge extraction.")
    ap.add_argument("--task", default="synthetic", choices=["synthetic", "gt", "all"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="Required when --split test")
    ap.add_argument("--gt", default=None, help="Path to golden diagram GT JSON")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--log-file", default=str(ROOT / "outputs" / "eval_runs.jsonl"))
    args = ap.parse_args(argv)

    if args.split == "test" and not args.final:
        print("REFUSED: --split test requires --final to prevent accidental test-split evaluation.", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    if args.task in ("synthetic", "all"):
        syn = bench_synthetic_diagrams(device=args.device)
        print(json.dumps(syn, ensure_ascii=False))
        results.append(syn)
    if args.task in ("gt", "all"):
        try:
            gt_res = bench_gt_diagrams(gt_path=args.gt, split=args.split, device=args.device)
        except FileNotFoundError as exc:
            if args.task == "gt" or args.gt is not None:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            gt_res = None
        if gt_res is not None:
            print(json.dumps(gt_res, ensure_ascii=False))
            results.append(gt_res)

    if results:
        append_run_log(
            Path(args.log_file),
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": f"diagrams:{args.task}",
                "split": args.split,
                "final": args.final,
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
