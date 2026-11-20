from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._pdf_io import clip_box, crop_page_rgb_with_bbox, page_bbox, pt_to_px, px_to_pt
from .backends import DiagramExtractor, get as get_backend
from .backends.base import RunStats
from .backends.rss import peak_rss_mb
from .models import BBox, DiagramEdge, DiagramGroup, DiagramNode, ExtractedDiagram, Word

_BOX_PAD_PT = 2.0
_LOOP_EPS_PT = 1.5
_VISUAL_LOOP_EPS_PT = 4.5
_JOINT_EPS_PT = 2.0
_VISUAL_JOINT_EPS_PT = 4.5
_SNAP_DIST_PT = 12.0
_ARROW_MAX_DIM_PT = 14.0
_ARROW_MATCH_DIST_PT = 10.0
_MIN_SHAPE_DIM_PT = 12.0
_BORDER_COVER_RATIO = 0.95
_EDGE_LABEL_DIST_PT = 16.0
_TEXT_NODE_SNAP_PT = 22.0


@dataclass
class _RawShape:
    bbox: BBox
    shape_type: str
    dashed: bool = False


@dataclass
class _Polyline:
    pts: list[tuple[float, float]]
    dashed: bool


@dataclass
class _ArrowPoly:
    centroid: tuple[float, float]
    verts: list[tuple[float, float]]
    bbox: BBox


def _round_bbox(bbox: BBox) -> list[float]:
    return [round(float(v), 3) + 0.0 for v in bbox]


def _round_conf(val: float) -> float:
    return round(float(val), 4) + 0.0


def _bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter_area(a: BBox, b: BBox) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter = _inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _unit_vec(p_from: tuple[float, float], p_to: tuple[float, float]) -> tuple[float, float] | None:
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return (dx / norm, dy / norm)


def _point_to_box_dist(pt: tuple[float, float], box: BBox) -> float:
    dx = max(box[0] - pt[0], 0.0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0.0, pt[1] - box[3])
    return math.hypot(dx, dy)


def _point_to_segment_dist(pt: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    len2 = vx * vx + vy * vy
    if len2 <= 1e-9:
        return _dist(pt, a)
    t = max(0.0, min(1.0, ((pt[0] - a[0]) * vx + (pt[1] - a[1]) * vy) / len2))
    proj = (a[0] + t * vx, a[1] + t * vy)
    return _dist(pt, proj)


def _point_to_polyline_dist(pt: tuple[float, float], pts: Sequence[tuple[float, float]]) -> float:
    if not pts:
        return float("inf")
    if len(pts) == 1:
        return _dist(pt, pts[0])
    return min(_point_to_segment_dist(pt, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _box_to_polyline_dist(box: BBox, pts: Sequence[tuple[float, float]]) -> float:
    if not pts:
        return float("inf")
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    # Sample box center, edge midpoints, and corners clipped against segment bounds.
    samples = (
        (cx, cy),
        (box[0], cy),
        (box[2], cy),
        (cx, box[1]),
        (cx, box[3]),
        (box[0], box[1]),
        (box[2], box[1]),
        (box[0], box[3]),
        (box[2], box[3]),
    )
    best = min(_point_to_polyline_dist(s, pts) for s in samples)
    for px, py in pts:
        best = min(best, _point_to_box_dist((px, py), box))
    return best


def _polyline_midpoint(pts: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    if len(pts) == 1:
        return pts[0]
    lengths = [_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(lengths)
    if total <= 1e-6:
        return pts[0]
    half = total * 0.5
    acc = 0.0
    for i, seg_len in enumerate(lengths):
        if acc + seg_len >= half and seg_len > 1e-9:
            t = (half - acc) / seg_len
            return (
                pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                pts[i][1] + t * (pts[i + 1][1] - pts[i][1]),
            )
        acc += seg_len
    return pts[-1]


def _waypoints_bbox(waypoints: Sequence[tuple[float, float]], fallback: BBox) -> BBox:
    if not waypoints:
        return fallback
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    return (min(xs), min(ys), max(xs), max(ys))


def _is_dashed(obj: Mapping[str, Any]) -> bool:
    dash = obj.get("dash")
    if isinstance(dash, bool):
        return dash
    if not dash:
        return False
    if isinstance(dash, Mapping):
        arr = dash.get("dash") or ()
        return any(isinstance(v, (int, float)) and v > 0 for v in arr)
    if isinstance(dash, (tuple, list)):
        if len(dash) == 2 and isinstance(dash[0], (tuple, list)):
            return any(isinstance(v, (int, float)) and v > 0 for v in dash[0])
        return any(isinstance(v, (int, float)) and v > 0 for v in dash)
    return False


def _obj_bbox(obj: Mapping[str, Any], page_h: float) -> BBox:
    x0 = float(obj.get("x0", 0.0))
    x1 = float(obj.get("x1", 0.0))
    if "top" in obj and "bottom" in obj:
        y0 = float(obj["top"])
        y1 = float(obj["bottom"])
    else:
        y0 = page_h - float(obj.get("y1", 0.0))
        y1 = page_h - float(obj.get("y0", 0.0))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _boxes_intersect(a: BBox, b: BBox) -> bool:
    return a[2] >= b[0] and a[0] <= b[2] and a[3] >= b[1] and a[1] <= b[3]


def _group_word_indices_into_lines(words: Sequence[Word], indices: Sequence[int]) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(indices, key=lambda i: ((words[i].bbox[1] + words[i].bbox[3]) * 0.5, words[i].bbox[0]))
    lines: list[list[int]] = []
    for wi in ordered:
        wy0, wy1 = words[wi].bbox[1], words[wi].bbox[3]
        wh = max(1.0, wy1 - wy0)
        placed = False
        for line in reversed(lines):
            ly0 = min(words[k].bbox[1] for k in line)
            ly1 = max(words[k].bbox[3] for k in line)
            lh = max(1.0, ly1 - ly0)
            if min(wy1, ly1) - max(wy0, ly0) >= 0.45 * min(wh, lh):
                line.append(wi)
                placed = True
                break
        if not placed:
            lines.append([wi])
    for line in lines:
        line.sort(key=lambda i: words[i].bbox[0])
    return lines


def _sort_words_reading_order(words: Sequence[Word]) -> list[Word]:
    if not words:
        return []
    out: list[Word] = []
    for line in _group_word_indices_into_lines(words, list(range(len(words)))):
        out.extend(words[i] for i in line)
    return out


def _join_words(words: Sequence[Word]) -> str:
    return " ".join(w.text.strip() for w in _sort_words_reading_order(words) if w.text.strip())


def _is_closed_curve(obj: Mapping[str, Any], pts: Sequence[tuple[float, float]]) -> bool:
    path = obj.get("path") or ()
    if any(op and op[0] == "h" for op in path):
        return True
    return len(pts) >= 3 and _dist(pts[0], pts[-1]) <= _LOOP_EPS_PT


def _curve_shape_type(obj: Mapping[str, Any], pts: Sequence[tuple[float, float]]) -> str:
    path = obj.get("path") or ()
    has_bezier = any(op and op[0] == "c" for op in path)
    if has_bezier:
        has_line = any(op and op[0] == "l" for op in path)
        return "round_rect" if has_line else "ellipse"
    uniq = _dedup_poly_pts(pts)
    if len(uniq) == 4:
        xs = sorted(p[0] for p in uniq)
        ys = sorted(p[1] for p in uniq)
        if abs(xs[0] - xs[1]) > 2.0 or abs(xs[2] - xs[3]) > 2.0:
            if abs(ys[0] - ys[1]) > 2.0 or abs(ys[2] - ys[3]) > 2.0:
                return "diamond"
    return "polygon"


def _dedup_poly_pts(pts: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in pts:
        if not out or _dist(out[-1], p) > 0.5:
            out.append(p)
    if len(out) > 1 and _dist(out[0], out[-1]) <= 0.5:
        out.pop()
    return out


def _reconstruct_rect_loops(
    lines: Sequence[tuple[tuple[float, float], tuple[float, float], bool]],
    *,
    loop_eps: float = _LOOP_EPS_PT,
) -> tuple[list[_RawShape], set[int]]:
    horiz: list[tuple[int, float, float, float, bool]] = []
    vert: list[tuple[int, float, float, float, bool]] = []
    for idx, (p1, p2, dashed) in enumerate(lines):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        if dy <= loop_eps and dx > 6.0:
            horiz.append((idx, min(p1[0], p2[0]), max(p1[0], p2[0]), (p1[1] + p2[1]) * 0.5, dashed))
        elif dx <= loop_eps and dy > 6.0:
            vert.append((idx, (p1[0] + p2[0]) * 0.5, min(p1[1], p2[1]), max(p1[1], p2[1]), dashed))

    shapes: list[_RawShape] = []
    consumed: set[int] = set()
    horiz.sort(key=lambda item: item[3])
    vert.sort(key=lambda item: item[1])

    for i, (hi_top, hx0, hx1, hy_top, hd_top) in enumerate(horiz):
        for hi_bot, bx0, bx1, hy_bot, hd_bot in horiz[i + 1 :]:
            if hy_bot - hy_top <= 8.0:
                continue
            if abs(hx0 - bx0) > loop_eps * 2.0 or abs(hx1 - bx1) > loop_eps * 2.0:
                continue
            x0 = (hx0 + bx0) * 0.5
            x1 = (hx1 + bx1) * 0.5
            left_match: tuple[int, bool] | None = None
            right_match: tuple[int, bool] | None = None
            for vi, vx, vy0, vy1, vd in vert:
                if abs(vy0 - hy_top) <= loop_eps and abs(vy1 - hy_bot) <= loop_eps:
                    if left_match is None and abs(vx - x0) <= loop_eps:
                        left_match = (vi, vd)
                    elif right_match is None and abs(vx - x1) <= loop_eps:
                        right_match = (vi, vd)
            if left_match is not None and right_match is not None:
                box = (x0, hy_top, x1, hy_bot)
                if max(x1 - x0, hy_bot - hy_top) > _MIN_SHAPE_DIM_PT:
                    shapes.append(
                        _RawShape(
                            bbox=box,
                            shape_type="rect",
                            dashed=(hd_top and hd_bot and left_match[1] and right_match[1]),
                        )
                    )
                    consumed.update((hi_top, hi_bot, left_match[0], right_match[0]))
                    break
    return shapes, consumed


def _extract_polyline_pts(obj: Mapping[str, Any], page_h: float) -> list[tuple[float, float]]:
    pts = [(float(px), float(py)) for px, py in (obj.get("pts") or ())]
    path = obj.get("path") or ()
    if len(pts) == 2 and any(op and op[0] == "c" for op in path):
        sampled: list[tuple[float, float]] = []
        cur = pts[0]
        for op in path:
            kind = op[0]
            if kind == "m":
                cur = (float(op[1][0]), float(op[1][1]))
                if not sampled:
                    sampled.append(cur)
            elif kind == "l":
                cur = (float(op[1][0]), float(op[1][1]))
                sampled.append(cur)
            elif kind == "c" and len(op) >= 4:
                c1 = (float(op[1][0]), float(op[1][1]))
                c2 = (float(op[2][0]), float(op[2][1]))
                p3 = (float(op[3][0]), float(op[3][1]))
                mid = (
                    0.125 * cur[0] + 0.375 * c1[0] + 0.375 * c2[0] + 0.125 * p3[0],
                    0.125 * cur[1] + 0.375 * c1[1] + 0.375 * c2[1] + 0.125 * p3[1],
                )
                sampled.append(mid)
                sampled.append(p3)
                cur = p3
        if len(sampled) >= 2:
            return sampled
    if pts:
        return pts
    b = _obj_bbox(obj, page_h)
    return [(b[0], b[1]), (b[2], b[3])]


def _dedup_shapes(shapes: Sequence[_RawShape]) -> list[_RawShape]:
    ordered = sorted(shapes, key=lambda s: _bbox_area(s.bbox), reverse=True)
    kept: list[_RawShape] = []
    for cand in ordered:
        if any(_bbox_iou(cand.bbox, prev.bbox) >= 0.90 for prev in kept):
            continue
        kept.append(cand)
    return kept


def _strictly_encloses(outer: BBox, inner: BBox) -> bool:
    outer_area = _bbox_area(outer)
    inner_area = _bbox_area(inner)
    if outer_area <= inner_area * 1.05 or inner_area <= 0.0:
        return False
    if (
        inner[0] < outer[0] - 2.0
        or inner[1] < outer[1] - 2.0
        or inner[2] > outer[2] + 2.0
        or inner[3] > outer[3] + 2.0
    ):
        return False
    return (_inter_area(outer, inner) / inner_area) >= 0.90


def _filter_border_frames(shapes: Sequence[_RawShape], page_box: BBox, fig_box: BBox) -> list[_RawShape]:
    page_area = max(1.0, _bbox_area(page_box))
    fig_area = max(1.0, _bbox_area(fig_box))
    page_w = max(1.0, page_box[2] - page_box[0])
    page_h = max(1.0, page_box[3] - page_box[1])
    fig_w = max(1.0, fig_box[2] - fig_box[0])
    fig_h = max(1.0, fig_box[3] - fig_box[1])

    out: list[_RawShape] = []
    for idx, sh in enumerate(shapes):
        b = sh.bbox
        w = b[2] - b[0]
        h = b[3] - b[1]
        area = _bbox_area(b)
        enclosed_count = sum(
            1 for j, other in enumerate(shapes) if j != idx and _strictly_encloses(b, other.bbox)
        )
        is_page_frame = (
            area >= _BORDER_COVER_RATIO * page_area
            and w >= _BORDER_COVER_RATIO * page_w
            and h >= _BORDER_COVER_RATIO * page_h
        )
        is_fig_frame = (
            area >= _BORDER_COVER_RATIO * fig_area
            and w >= _BORDER_COVER_RATIO * fig_w
            and h >= _BORDER_COVER_RATIO * fig_h
        )
        if is_page_frame and fig_area < 0.85 * page_area:
            continue
        if (is_page_frame or is_fig_frame) and enclosed_count < 2:
            continue
        out.append(sh)
    return out


def _chain_polylines(polylines: Sequence[_Polyline], *, joint_eps: float = _JOINT_EPS_PT) -> list[_Polyline]:
    if not polylines:
        return []
    work = [_Polyline(pts=list(p.pts), dashed=p.dashed) for p in polylines if len(p.pts) >= 2]
    if len(work) <= 1:
        return work

    clusters: list[tuple[float, float]] = []
    endpoints: list[tuple[int, int]] = []  # (poly_idx, end_flag: 0=start, 1=end)
    poly_clusters: list[list[int]] = [[-1, -1] for _ in work]

    def _cluster_for(pt: tuple[float, float]) -> int:
        for cid, cpt in enumerate(clusters):
            if _dist(pt, cpt) <= joint_eps:
                return cid
        clusters.append(pt)
        return len(clusters) - 1

    for pid, poly in enumerate(work):
        c0 = _cluster_for(poly.pts[0])
        c1 = _cluster_for(poly.pts[-1])
        poly_clusters[pid] = [c0, c1]
        endpoints.append((pid, 0))
        endpoints.append((pid, 1))

    incident: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pid, end_flag in endpoints:
        incident[poly_clusters[pid][end_flag]].append((pid, end_flag))

    def _outward_dir(pid: int, end_flag: int) -> tuple[float, float] | None:
        pts = work[pid].pts
        return _unit_vec(pts[1], pts[0]) if end_flag == 0 else _unit_vec(pts[-2], pts[-1])

    adj: dict[tuple[int, int], tuple[int, int]] = {}
    for cid, inc in incident.items():
        if len(inc) == 2:
            (p1, e1), (p2, e2) = inc
            if p1 == p2 or work[p1].dashed != work[p2].dashed:
                continue
            d1 = _outward_dir(p1, e1)
            d2 = _outward_dir(p2, e2)
            # Avoid folding two nearly identical overlapping strokes back onto each other.
            if d1 is not None and d2 is not None and (d1[0] * d2[0] + d1[1] * d2[1]) > 0.92:
                continue
            adj[(p1, e1)] = (p2, e2)
            adj[(p2, e2)] = (p1, e1)
        elif len(inc) >= 4:
            # At 4+ crossings, pair collinear opposite segments; leave degree-3 forks unmerged for fan-out.
            used_local: set[int] = set()
            for i in range(len(inc)):
                if i in used_local:
                    continue
                p1, e1 = inc[i]
                d1 = _outward_dir(p1, e1)
                if d1 is None:
                    continue
                best_j = -1
                best_dot = -0.95
                for j in range(i + 1, len(inc)):
                    if j in used_local:
                        continue
                    p2, e2 = inc[j]
                    if p1 == p2 or work[p1].dashed != work[p2].dashed:
                        continue
                    d2 = _outward_dir(p2, e2)
                    if d2 is None:
                        continue
                    dot = d1[0] * d2[0] + d1[1] * d2[1]
                    if dot < best_dot:
                        best_dot = dot
                        best_j = j
                if best_j >= 0:
                    p2, e2 = inc[best_j]
                    used_local.update((i, best_j))
                    adj[(p1, e1)] = (p2, e2)
                    adj[(p2, e2)] = (p1, e1)

    visited: set[int] = set()
    merged_out: list[_Polyline] = []

    def _walk_from(start_pid: int, start_enter_end: int) -> list[tuple[int, int]]:
        chain: list[tuple[int, int]] = []
        cur_pid, enter_end = start_pid, start_enter_end
        while cur_pid not in visited:
            visited.add(cur_pid)
            exit_end = 1 - enter_end
            chain.append((cur_pid, enter_end))
            nxt = adj.get((cur_pid, exit_end))
            if nxt is None or nxt[0] in visited:
                break
            cur_pid, enter_end = nxt
        return chain

    def _emit_chain(chain: Sequence[tuple[int, int]]) -> None:
        pts: list[tuple[float, float]] = []
        dashed = work[chain[0][0]].dashed
        for idx, (pid, enter_end) in enumerate(chain):
            seg = list(work[pid].pts)
            if enter_end == 1:
                seg.reverse()
            if idx == 0:
                pts.extend(seg)
            else:
                joint = ((pts[-1][0] + seg[0][0]) * 0.5, (pts[-1][1] + seg[0][1]) * 0.5)
                pts[-1] = joint
                pts.extend(seg[1:])
        merged_out.append(_Polyline(pts=pts, dashed=dashed))

    for pid in range(len(work)):
        if pid in visited:
            continue
        has0 = (pid, 0) in adj
        has1 = (pid, 1) in adj
        if not has0:
            _emit_chain(_walk_from(pid, 0))
        elif not has1:
            _emit_chain(_walk_from(pid, 1))

    for pid in range(len(work)):
        if pid not in visited:
            _emit_chain(_walk_from(pid, 0))

    return merged_out


def _matches_arrow_poly(
    end_pt: tuple[float, float],
    out_dir: tuple[float, float] | None,
    arrow: _ArrowPoly,
) -> bool:
    if out_dir is None:
        return False
    min_vert_dist = min((_dist(end_pt, v) for v in arrow.verts), default=float("inf"))
    if _dist(end_pt, arrow.centroid) > _ARROW_MATCH_DIST_PT and min_vert_dist > 6.0:
        return False
    projs = [
        (v[0] - arrow.centroid[0]) * out_dir[0] + (v[1] - arrow.centroid[1]) * out_dir[1]
        for v in arrow.verts
    ]
    max_p = max(projs)
    if max_p <= 0.5:
        return False
    front_count = sum(1 for p in projs if p >= max_p * 0.65)
    back_count = sum(1 for p in projs if p <= -0.1 * max_p)
    return front_count == 1 and back_count >= 2


def _extract_v_chevron_tips(
    polylines: Sequence[_Polyline],
) -> tuple[list[_Polyline], list[tuple[tuple[float, float], tuple[float, float], tuple[int, ...]]]]:
    short_indices: list[int] = []
    for idx, poly in enumerate(polylines):
        if len(poly.pts) == 2:
            seg_len = _dist(poly.pts[0], poly.pts[1])
            if 1.5 <= seg_len <= _ARROW_MAX_DIM_PT:
                short_indices.append(idx)

    chevrons: list[tuple[tuple[float, float], tuple[float, float], tuple[int, ...]]] = []
    for idx, poly in enumerate(polylines):
        if len(poly.pts) == 3:
            l1 = _dist(poly.pts[0], poly.pts[1])
            l2 = _dist(poly.pts[1], poly.pts[2])
            if 1.5 <= l1 <= _ARROW_MAX_DIM_PT and 1.5 <= l2 <= _ARROW_MAX_DIM_PT and max(l1, l2) <= 1.8 * min(l1, l2):
                u1 = _unit_vec(poly.pts[0], poly.pts[1])
                u2 = _unit_vec(poly.pts[2], poly.pts[1])
                if u1 is not None and u2 is not None:
                    dot = u1[0] * u2[0] + u1[1] * u2[1]
                    if -0.55 <= dot <= 0.96:
                        bis = _unit_vec((0.0, 0.0), (u1[0] + u2[0], u1[1] + u2[1]))
                        if bis is not None:
                            chevrons.append((poly.pts[1], bis, (idx,)))

    used_pairs: set[int] = set()
    for i in range(len(short_indices)):
        idx_a = short_indices[i]
        if idx_a in used_pairs:
            continue
        pa0, pa1 = polylines[idx_a].pts[0], polylines[idx_a].pts[1]
        la = _dist(pa0, pa1)
        for j in range(i + 1, len(short_indices)):
            idx_b = short_indices[j]
            if idx_b in used_pairs:
                continue
            pb0, pb1 = polylines[idx_b].pts[0], polylines[idx_b].pts[1]
            lb = _dist(pb0, pb1)
            if max(la, lb) > 1.8 * min(la, lb):
                continue
            matched: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
            for ea, wa in ((pa0, pa1), (pa1, pa0)):
                for eb, wb in ((pb0, pb1), (pb1, pb0)):
                    if _dist(ea, eb) <= _JOINT_EPS_PT:
                        matched = (((ea[0] + eb[0]) * 0.5, (ea[1] + eb[1]) * 0.5), wa, wb)
                        break
                if matched is not None:
                    break
            if matched is None:
                continue
            tip, w1, w2 = matched
            u1 = _unit_vec(w1, tip)
            u2 = _unit_vec(w2, tip)
            if u1 is None or u2 is None:
                continue
            dot = u1[0] * u2[0] + u1[1] * u2[1]
            if -0.75 <= dot <= 0.96:
                bis = _unit_vec((0.0, 0.0), (u1[0] + u2[0], u1[1] + u2[1]))
                if bis is not None:
                    chevrons.append((tip, bis, (idx_a, idx_b)))
                    used_pairs.update((idx_a, idx_b))
                    break

    return list(polylines), chevrons


def _collect_primitives(
    fig_box: BBox,
    page_box: BBox,
    clip: BBox,
    raw_rects: Sequence[Mapping[str, Any]],
    raw_lines: Sequence[Mapping[str, Any]],
    raw_curves: Sequence[Mapping[str, Any]],
) -> tuple[
    list[_RawShape],
    list[_ArrowPoly],
    list[tuple[tuple[float, float], tuple[float, float], bool]],
    list[_Polyline],
]:
    page_h = page_box[3]
    shapes: list[_RawShape] = []
    arrow_polys: list[_ArrowPoly] = []
    line_segs: list[tuple[tuple[float, float], tuple[float, float], bool]] = []
    open_polys: list[_Polyline] = []

    for r in raw_rects:
        if r.get("stroke") is False and r.get("fill") is False:
            continue
        b = _obj_bbox(r, page_h)
        if not _boxes_intersect(b, clip):
            continue
        if max(b[2] - b[0], b[3] - b[1]) <= _MIN_SHAPE_DIM_PT or min(b[2] - b[0], b[3] - b[1]) <= 2.0:
            continue
        shapes.append(_RawShape(bbox=b, shape_type="rect", dashed=_is_dashed(r)))

    for ln in raw_lines:
        b = _obj_bbox(ln, page_h)
        if not _boxes_intersect(b, clip):
            continue
        pts = _extract_polyline_pts(ln, page_h)
        if len(pts) >= 2:
            line_segs.append((pts[0], pts[-1], _is_dashed(ln)))

    for cv in raw_curves:
        b = _obj_bbox(cv, page_h)
        if not _boxes_intersect(b, clip):
            continue
        pts = _extract_polyline_pts(cv, page_h)
        w, h = b[2] - b[0], b[3] - b[1]
        dashed = _is_dashed(cv)
        if _is_closed_curve(cv, pts):
            uniq = _dedup_poly_pts(pts)
            if max(w, h) <= _ARROW_MAX_DIM_PT and 3 <= len(uniq) <= 4 and _bbox_area(b) > 0.5:
                cx = sum(p[0] for p in uniq) / len(uniq)
                cy = sum(p[1] for p in uniq) / len(uniq)
                arrow_polys.append(_ArrowPoly(centroid=(cx, cy), verts=uniq, bbox=b))
                continue
            if max(w, h) <= _MIN_SHAPE_DIM_PT or min(w, h) <= 2.0:
                continue
            shapes.append(_RawShape(bbox=b, shape_type=_curve_shape_type(cv, pts), dashed=dashed))
        elif len(pts) >= 2:
            open_polys.append(_Polyline(pts=pts, dashed=dashed))

    loop_shapes, consumed_lines = _reconstruct_rect_loops(line_segs)
    shapes.extend(loop_shapes)
    shapes = _filter_border_frames(_dedup_shapes(shapes), page_box, fig_box)
    remaining_lines = [seg for idx, seg in enumerate(line_segs) if idx not in consumed_lines]
    return shapes, arrow_polys, remaining_lines, open_polys


def _resolve_connectors(
    line_segs: Sequence[tuple[tuple[float, float], tuple[float, float], bool]],
    open_polys: Sequence[_Polyline],
) -> tuple[list[_Polyline], list[tuple[tuple[float, float], tuple[float, float]]]]:
    raw_connectors: list[_Polyline] = [_Polyline(pts=[p1, p2], dashed=d) for p1, p2, d in line_segs]
    raw_connectors.extend(open_polys)
    _, chevron_cands = _extract_v_chevron_tips(raw_connectors)
    chevron_consumed: set[int] = set()
    matched_chevrons: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for tip, bis, seg_idxs in chevron_cands:
        seg_set = set(seg_idxs)
        is_matched = any(
            out_dir is not None
            and _dist(end_pt, tip) <= 6.0
            and (out_dir[0] * bis[0] + out_dir[1] * bis[1]) > 0.3
            for c_idx, poly in enumerate(raw_connectors)
            if c_idx not in seg_set and len(poly.pts) >= 2
            for end_pt, prev_pt in ((poly.pts[0], poly.pts[1]), (poly.pts[-1], poly.pts[-2]))
            for out_dir in (_unit_vec(prev_pt, end_pt),)
        )
        if is_matched:
            chevron_consumed.update(seg_idxs)
            matched_chevrons.append((tip, bis))

    filtered = [p for idx, p in enumerate(raw_connectors) if idx not in chevron_consumed]
    return _chain_polylines(filtered), matched_chevrons


def _nearest_parent_group(
    child_bbox: BBox,
    group_entries: Sequence[dict[str, Any]],
    exclude_id: str | None = None,
) -> str | None:
    best_gid: str | None = None
    best_area = float("inf")
    for ge in group_entries:
        if ge["id"] == exclude_id:
            continue
        gb = ge["bbox"]
        if _strictly_encloses(gb, child_bbox):
            ga = _bbox_area(gb)
            if ga < best_area:
                best_area = ga
                best_gid = ge["id"]
    return best_gid


def _is_group_container(idx: int, shapes: Sequence[_RawShape]) -> bool:
    outer = shapes[idx].bbox
    outer_area = _bbox_area(outer)
    children = [shapes[j].bbox for j in range(len(shapes)) if j != idx and _strictly_encloses(outer, shapes[j].bbox)]
    if len(children) >= 2:
        return True
    if len(children) == 1:
        cb = children[0]
        return max(cb[2] - cb[0], cb[3] - cb[1]) >= 24.0 and _bbox_area(cb) <= 0.75 * outer_area
    return False


def _build_shape_hierarchy(
    shapes: Sequence[_RawShape],
    *,
    shape_conf: float = 0.95,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_shapes = len(shapes)
    group_indices = sorted(
        (i for i in range(n_shapes) if _is_group_container(i, shapes)),
        key=lambda i: (shapes[i].bbox[1], shapes[i].bbox[0], -_bbox_area(shapes[i].bbox)),
    )
    group_set = set(group_indices)
    node_indices = sorted(
        (i for i in range(n_shapes) if i not in group_set),
        key=lambda i: (shapes[i].bbox[1], shapes[i].bbox[0]),
    )

    group_entries: list[dict[str, Any]] = [
        {
            "id": f"group_{k + 1}",
            "bbox": shapes[s_idx].bbox,
            "parent_group_id": None,
            "words": [],
            "confidence": shape_conf,
        }
        for k, s_idx in enumerate(group_indices)
    ]
    for ge in group_entries:
        ge["parent_group_id"] = _nearest_parent_group(ge["bbox"], group_entries, exclude_id=ge["id"])

    node_entries: list[dict[str, Any]] = [
        {
            "id": f"node_{k + 1}",
            "shape_type": shapes[s_idx].shape_type,
            "bbox": shapes[s_idx].bbox,
            "parent_group_id": _nearest_parent_group(shapes[s_idx].bbox, group_entries),
            "words": [],
            "confidence": shape_conf,
        }
        for k, s_idx in enumerate(node_indices)
    ]
    return node_entries, group_entries


def _assign_node_words(fig_words: Sequence[Word], node_entries: Sequence[dict[str, Any]]) -> set[int]:
    used: set[int] = set()
    for w_idx, w in enumerate(fig_words):
        cx = (w.bbox[0] + w.bbox[2]) * 0.5
        cy = (w.bbox[1] + w.bbox[3]) * 0.5
        best_node: dict[str, Any] | None = None
        best_area = float("inf")
        for ne in node_entries:
            nb = ne["bbox"]
            if nb[0] - 1.5 <= cx <= nb[2] + 1.5 and nb[1] - 1.5 <= cy <= nb[3] + 1.5:
                na = _bbox_area(nb)
                if na < best_area:
                    best_area = na
                    best_node = ne
        if best_node is not None:
            best_node["words"].append(w)
            used.add(w_idx)
    return used


def _words_bbox(words: Sequence[Word], idxs: Sequence[int]) -> BBox:
    return (
        min(words[k].bbox[0] for k in idxs),
        min(words[k].bbox[1] for k in idxs),
        max(words[k].bbox[2] for k in idxs),
        max(words[k].bbox[3] for k in idxs),
    )


def _build_word_clusters(fig_words: Sequence[Word], used_word_ids: set[int]) -> list[tuple[list[int], BBox]]:
    unassigned = [i for i in range(len(fig_words)) if i not in used_word_ids]
    clusters: list[tuple[list[int], BBox]] = []
    for line in _group_word_indices_into_lines(fig_words, unassigned):
        cur = [line[0]]
        for wi in line[1:]:
            if fig_words[wi].bbox[0] - fig_words[cur[-1]].bbox[2] > 18.0:
                clusters.append((cur, _words_bbox(fig_words, cur)))
                cur = [wi]
            else:
                cur.append(wi)
        clusters.append((cur, _words_bbox(fig_words, cur)))
    return clusters


def _snap_node_id(pt: tuple[float, float], node_entries: Sequence[dict[str, Any]]) -> str | None:
    best_id: str | None = None
    best_dist = _SNAP_DIST_PT
    for ne in node_entries:
        d = _point_to_box_dist(pt, ne["bbox"])
        if d <= best_dist:
            best_dist = d
            best_id = ne["id"]
    return best_id


def _match_text_node_cluster(
    tip_pt: tuple[float, float],
    prev_pt: tuple[float, float],
    word_clusters: Sequence[tuple[list[int], BBox]],
    used_word_ids: set[int],
) -> int:
    out_dir = _unit_vec(prev_pt, tip_pt)
    best_cidx = -1
    best_score = float("inf")
    for cidx, (w_idxs, cbox) in enumerate(word_clusters):
        if any(wi in used_word_ids for wi in w_idxs):
            continue
        d = _point_to_box_dist(tip_pt, cbox)
        if d > _TEXT_NODE_SNAP_PT:
            continue
        forward_bonus = 0.0
        if out_dir is not None:
            cx = (cbox[0] + cbox[2]) * 0.5
            cy = (cbox[1] + cbox[3]) * 0.5
            proj = (cx - tip_pt[0]) * out_dir[0] + (cy - tip_pt[1]) * out_dir[1]
            if proj < -4.0:
                continue
            forward_bonus = -0.1 * min(proj, 20.0)
        if d + forward_bonus < best_score:
            best_score = d + forward_bonus
            best_cidx = cidx
    return best_cidx


def _fan_out_forks(
    connectors: Sequence[_Polyline],
    node_entries: Sequence[dict[str, Any]],
    has_arrow_fn: Any,
    *,
    joint_eps: float = _JOINT_EPS_PT,
) -> list[_Polyline]:
    items = [list(poly.pts) for poly in connectors if len(poly.pts) >= 2]
    dasheds = [poly.dashed for poly in connectors if len(poly.pts) >= 2]
    if len(items) <= 1:
        return [poly for poly in connectors if len(poly.pts) >= 2]

    info: list[dict[str, Any]] = []
    for pts in items:
        a0 = bool(has_arrow_fn(pts[0], pts[1]))
        a1 = bool(has_arrow_fn(pts[-1], pts[-2]))
        n0 = _snap_node_id(pts[0], node_entries)
        n1 = _snap_node_id(pts[-1], node_entries)
        info.append({"a0": a0, "a1": a1, "n0": n0, "n1": n1})

    consumed: set[int] = set()
    fanned: list[_Polyline] = []
    for si, stem in enumerate(items):
        sinfo = info[si]
        for stem_end in (0, 1):
            other_end = 1 - stem_end
            stem_node = sinfo[f"n{other_end}"]
            if sinfo[f"n{stem_end}"] is not None or sinfo[f"a{stem_end}"] or stem_node is None:
                continue
            jpt = stem[0] if stem_end == 0 else stem[-1]
            branches: list[tuple[int, int]] = []
            for bi, bpts in enumerate(items):
                if bi == si or dasheds[bi] != dasheds[si]:
                    continue
                binfo = info[bi]
                for b_end in (0, 1):
                    bfar = 1 - b_end
                    bpt = bpts[0] if b_end == 0 else bpts[-1]
                    if _dist(jpt, bpt) <= joint_eps and binfo[f"n{b_end}"] is None and not binfo[f"a{b_end}"]:
                        if binfo[f"n{bfar}"] is not None or binfo[f"a{bfar}"]:
                            branches.append((bi, b_end))
                            break
            if len(branches) >= 2:
                stem_ordered = list(reversed(stem)) if stem_end == 0 else list(stem)
                consumed.add(si)
                for bi, b_end in branches:
                    consumed.add(bi)
                    b_ordered = list(items[bi]) if b_end == 0 else list(reversed(items[bi]))
                    joint = ((stem_ordered[-1][0] + b_ordered[0][0]) * 0.5, (stem_ordered[-1][1] + b_ordered[0][1]) * 0.5)
                    fanned.append(_Polyline(pts=[*stem_ordered[:-1], joint, *b_ordered[1:]], dashed=dasheds[si]))
                break

    out = [_Polyline(pts=items[idx], dashed=dasheds[idx]) for idx in range(len(items)) if idx not in consumed]
    out.extend(fanned)
    return out


def _extract_edges_and_text_nodes(
    connectors: Sequence[_Polyline],
    arrow_polys: Sequence[_ArrowPoly],
    matched_chevrons: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    node_entries: list[dict[str, Any]],
    group_entries: Sequence[dict[str, Any]],
    fig_words: Sequence[Word],
    word_clusters: Sequence[tuple[list[int], BBox]],
    used_word_ids: set[int],
    *,
    visual_arrows: Sequence[tuple[tuple[float, float], BBox]] = (),
    joint_eps: float = _JOINT_EPS_PT,
    edge_conf_snapped: float = 0.95,
    edge_conf_unsnapped: float = 0.75,
    text_node_conf: float = 0.85,
) -> list[dict[str, Any]]:
    def _has_arrow_at(end_pt: tuple[float, float], prev_pt: tuple[float, float]) -> bool:
        out_dir = _unit_vec(prev_pt, end_pt)
        if out_dir is None:
            return False
        if any(_matches_arrow_poly(end_pt, out_dir, ap) for ap in arrow_polys):
            return True
        if any(
            _dist(end_pt, tip) <= 6.0 and (out_dir[0] * bis[0] + out_dir[1] * bis[1]) > 0.3
            for tip, bis in matched_chevrons
        ):
            return True
        for cent, abox in visual_arrows:
            if _point_to_box_dist(end_pt, abox) <= _ARROW_MATCH_DIST_PT:
                vec = _unit_vec(prev_pt, cent)
                if vec is not None and (out_dir[0] * vec[0] + out_dir[1] * vec[1]) > 0.35:
                    return True
        return False

    def _create_text_node_at(end_pt: tuple[float, float], prev_pt: tuple[float, float]) -> str | None:
        best_cidx = _match_text_node_cluster(end_pt, prev_pt, word_clusters, used_word_ids)
        if best_cidx < 0:
            return None
        w_idxs, cbox = word_clusters[best_cidx]
        used_word_ids.update(w_idxs)
        nid = f"node_{len(node_entries) + 1}"
        node_entries.append(
            {
                "id": nid,
                "shape_type": "text_node",
                "bbox": cbox,
                "parent_group_id": _nearest_parent_group(cbox, group_entries),
                "words": [fig_words[k] for k in w_idxs],
                "confidence": text_node_conf,
            }
        )
        return nid

    resolved_connectors = _fan_out_forks(connectors, node_entries, _has_arrow_at, joint_eps=joint_eps)
    edge_entries: list[dict[str, Any]] = []
    for poly in resolved_connectors:
        pts = list(poly.pts)
        if len(pts) < 2:
            continue
        arrow_start = _has_arrow_at(pts[0], pts[1])
        arrow_end = _has_arrow_at(pts[-1], pts[-2])
        if arrow_start and not arrow_end:
            pts.reverse()
            arrow_start, arrow_end = False, True

        direction = "bidirectional" if (arrow_start and arrow_end) else ("directed" if arrow_end else "undirected")
        src_id = _snap_node_id(pts[0], node_entries)
        dst_id = _snap_node_id(pts[-1], node_entries)

        if src_id is not None and src_id == dst_id:
            nb = next(ne["bbox"] for ne in node_entries if ne["id"] == src_id)
            max_outside = max((_point_to_box_dist(p, nb) for p in pts), default=0.0)
            if len(pts) < 3 or max_outside <= _SNAP_DIST_PT * 0.5:
                continue

        if dst_id is None and arrow_end:
            dst_id = _create_text_node_at(pts[-1], pts[-2])
            if dst_id is not None and src_id is None:
                src_id = _snap_node_id(pts[0], node_entries)

        if src_id is None and arrow_start:
            src_id = _create_text_node_at(pts[0], pts[1])
            if src_id is not None and dst_id is None:
                dst_id = _snap_node_id(pts[-1], node_entries)

        if (src_id is None or dst_id is None) and not (arrow_start or arrow_end):
            continue

        edge_entries.append(
            {
                "source_node_id": src_id,
                "target_node_id": dst_id,
                "direction": direction,
                "style": "dashed" if poly.dashed else "solid",
                "pts": pts,
                "words": [],
                "confidence": edge_conf_snapped if (src_id and dst_id) else edge_conf_unsnapped,
            }
        )

    for ee in edge_entries:
        if ee["source_node_id"] is None:
            ee["source_node_id"] = _snap_node_id(ee["pts"][0], node_entries)
        if ee["target_node_id"] is None:
            ee["target_node_id"] = _snap_node_id(ee["pts"][-1], node_entries)
    return edge_entries


def _assign_group_and_edge_labels(
    fig_words: Sequence[Word],
    word_clusters: Sequence[tuple[list[int], BBox]],
    used_word_ids: set[int],
    node_entries: Sequence[dict[str, Any]],
    group_entries: Sequence[dict[str, Any]],
    edge_entries: Sequence[dict[str, Any]],
) -> None:
    sorted_groups = sorted(group_entries, key=lambda g: _bbox_area(g["bbox"]))
    cluster_min_edge: dict[int, float] = {}
    if edge_entries:
        for w_idxs, cbox in word_clusters:
            d_min = min(_box_to_polyline_dist(cbox, ee["pts"]) for ee in edge_entries)
            for wi in w_idxs:
                cluster_min_edge[wi] = d_min

    for ge in sorted_groups:
        gb = ge["bbox"]
        child_tops = [ne["bbox"][1] for ne in node_entries if ne["parent_group_id"] == ge["id"]] + [
            other["bbox"][1] for other in group_entries if other["parent_group_id"] == ge["id"]
        ]
        header_limit_y = min(child_tops) + 2.0 if child_tops else (gb[1] + max(24.0, 0.30 * (gb[3] - gb[1])))
        for w_idx, w in enumerate(fig_words):
            if w_idx in used_word_ids:
                continue
            cx = (w.bbox[0] + w.bbox[2]) * 0.5
            cy = (w.bbox[1] + w.bbox[3]) * 0.5
            if gb[0] - 1.5 <= cx <= gb[2] + 1.5 and gb[1] - 1.5 <= cy <= min(gb[3] + 1.5, header_limit_y):
                d_edge = cluster_min_edge.get(w_idx, float("inf"))
                if d_edge <= _EDGE_LABEL_DIST_PT and d_edge < max(0.0, cy - gb[1]) * 0.6:
                    continue
                ge["words"].append(w)
                used_word_ids.add(w_idx)

    if edge_entries:
        for w_idxs, cbox in word_clusters:
            if any(wi in used_word_ids for wi in w_idxs):
                continue
            cx = (cbox[0] + cbox[2]) * 0.5
            cy = (cbox[1] + cbox[3]) * 0.5
            best_edge: dict[str, Any] | None = None
            best_score = float("inf")
            for ee in edge_entries:
                pts = ee["pts"]
                d_poly = _box_to_polyline_dist(cbox, pts)
                if d_poly <= _EDGE_LABEL_DIST_PT:
                    score = d_poly + 0.15 * _dist((cx, cy), _polyline_midpoint(pts))
                    if score < best_score:
                        best_score = score
                        best_edge = ee
            if best_edge is not None:
                best_edge["words"].extend(fig_words[wi] for wi in w_idxs)
                used_word_ids.update(w_idxs)

    for ge in sorted_groups:
        if ge["words"]:
            continue
        gb = ge["bbox"]
        for w_idx, w in enumerate(fig_words):
            if w_idx in used_word_ids:
                continue
            cx = (w.bbox[0] + w.bbox[2]) * 0.5
            cy = (w.bbox[1] + w.bbox[3]) * 0.5
            if gb[0] - 1.5 <= cx <= gb[2] + 1.5 and gb[1] - 1.5 <= cy <= gb[3] + 1.5:
                ge["words"].append(w)
                used_word_ids.add(w_idx)


def _build_diagram_from_parts(
    fig_box: BBox,
    shapes: Sequence[_RawShape],
    connectors: Sequence[_Polyline],
    arrow_polys: Sequence[_ArrowPoly],
    matched_chevrons: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    fig_words: Sequence[Word],
    *,
    visual_arrows: Sequence[tuple[tuple[float, float], BBox]] = (),
    joint_eps: float = _JOINT_EPS_PT,
    is_visual: bool = False,
) -> ExtractedDiagram:
    shape_conf = 0.78 if is_visual else 0.95
    node_entries, group_entries = _build_shape_hierarchy(shapes, shape_conf=shape_conf)

    used_word_ids = _assign_node_words(fig_words, node_entries)
    word_clusters = _build_word_clusters(fig_words, used_word_ids)
    edge_entries = _extract_edges_and_text_nodes(
        connectors,
        arrow_polys,
        matched_chevrons,
        node_entries,
        group_entries,
        fig_words,
        word_clusters,
        used_word_ids,
        visual_arrows=visual_arrows,
        joint_eps=joint_eps,
        edge_conf_snapped=0.78 if is_visual else 0.95,
        edge_conf_unsnapped=0.60 if is_visual else 0.75,
        text_node_conf=0.70 if is_visual else 0.85,
    )
    _assign_group_and_edge_labels(fig_words, word_clusters, used_word_ids, node_entries, group_entries, edge_entries)

    nodes_out = tuple(
        DiagramNode(
            id=ne["id"],
            shape_type=ne["shape_type"],
            label=_join_words(ne["words"]),
            bbox=ne["bbox"],
            parent_group_id=ne["parent_group_id"],
            text_element_ids=(),
            confidence=_round_conf(0.78 if (is_visual and ne["words"]) else (0.65 if is_visual else ne["confidence"]))
            if ne["shape_type"] != "text_node"
            else ne["confidence"],
        )
        for ne in node_entries
    )
    node_id_set = {n.id for n in nodes_out}

    edges_out = tuple(
        DiagramEdge(
            id=f"edge_{idx + 1}",
            source_node_id=ee["source_node_id"] if ee["source_node_id"] in node_id_set else None,
            target_node_id=ee["target_node_id"] if ee["target_node_id"] in node_id_set else None,
            direction=ee["direction"],
            style=ee["style"],
            label=_join_words(ee["words"]),
            waypoints=tuple((float(px), float(py)) for px, py in ee["pts"]),
            confidence=ee["confidence"],
        )
        for idx, ee in enumerate(edge_entries)
    )

    groups_out = tuple(
        DiagramGroup(
            id=ge["id"],
            label=_join_words(ge["words"]),
            bbox=ge["bbox"],
            parent_group_id=ge["parent_group_id"],
            member_node_ids=tuple(n.id for n in nodes_out if n.parent_group_id == ge["id"]),
            confidence=_round_conf(0.78 if (is_visual and ge["words"]) else (0.65 if is_visual else ge["confidence"])),
        )
        for ge in group_entries
    )

    if not (nodes_out or edges_out or groups_out):
        return ExtractedDiagram(
            bbox=fig_box,
            diagram_type="unknown",
            nodes=(),
            edges=(),
            groups=(),
            source="vector_graph",
            confidence=0.0,
        )

    diag = ExtractedDiagram(
        bbox=fig_box,
        diagram_type=_classify_diagram_type(nodes_out, edges_out, groups_out),
        nodes=nodes_out,
        edges=edges_out,
        groups=groups_out,
        source="vector_graph",
        confidence=_estimate_confidence(nodes_out, edges_out, is_visual=is_visual),
    )
    if is_visual:
        object.__setattr__(diag, "metadata", {"extraction_mode": "visual_fallback"})
    return diag


def _extract_diagram_from_box(
    fig_box: BBox,
    page_box: BBox,
    raw_rects: Sequence[Mapping[str, Any]],
    raw_lines: Sequence[Mapping[str, Any]],
    raw_curves: Sequence[Mapping[str, Any]],
    words: Sequence[Word],
) -> ExtractedDiagram:
    padded = (
        fig_box[0] - _BOX_PAD_PT,
        fig_box[1] - _BOX_PAD_PT,
        fig_box[2] + _BOX_PAD_PT,
        fig_box[3] + _BOX_PAD_PT,
    )
    clip = clip_box(padded, page_box) or fig_box

    shapes, arrow_polys, line_segs, open_polys = _collect_primitives(
        fig_box, page_box, clip, raw_rects, raw_lines, raw_curves
    )
    connectors, matched_chevrons = _resolve_connectors(line_segs, open_polys)
    fig_words = [
        w
        for w in words
        if fig_box[0] <= (w.bbox[0] + w.bbox[2]) * 0.5 <= fig_box[2]
        and fig_box[1] <= (w.bbox[1] + w.bbox[3]) * 0.5 <= fig_box[3]
    ]
    return _build_diagram_from_parts(fig_box, shapes, connectors, arrow_polys, matched_chevrons, fig_words)


def _has_raster_image_in_box(
    raw_images: Sequence[Mapping[str, Any]],
    clip: BBox,
    page_h: float,
) -> bool:
    clip_area = max(1.0, _bbox_area(clip))
    for img in raw_images:
        ib = _obj_bbox(img, page_h)
        if ib[2] - ib[0] < _MIN_SHAPE_DIM_PT or ib[3] - ib[1] < _MIN_SHAPE_DIM_PT:
            continue
        inter = _inter_area(ib, clip)
        if inter >= 0.15 * min(clip_area, max(1.0, _bbox_area(ib))):
            return True
    return False


def _pt_box_to_px_int(box: BBox, crop_box: BBox, img_size: tuple[int, int]) -> tuple[int, int, int, int]:
    px_box = pt_to_px(box, crop_box, img_size)
    iw, ih = img_size
    x0 = max(0, min(iw, int(round(px_box[0]))))
    y0 = max(0, min(ih, int(round(px_box[1]))))
    x1 = max(0, min(iw, int(round(px_box[2]))))
    y1 = max(0, min(ih, int(round(px_box[3]))))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _shares_edge_with(a: BBox, b: BBox, tol: float = 3.5) -> bool:
    ov_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ov_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aw = max(1.0, a[2] - a[0])
    ah = max(1.0, a[3] - a[1])
    if ov_x >= 0.35 * aw and (abs(a[1] - b[3]) <= tol or abs(a[3] - b[1]) <= tol):
        return True
    if ov_y >= 0.35 * ah and (abs(a[0] - b[2]) <= tol or abs(a[2] - b[0]) <= tol):
        return True
    return False


def _filter_visual_shapes(
    candidates: Sequence[tuple[_RawShape, bool]],
    fig_words: Sequence[Word],
    page_box: BBox,
    fig_box: BBox,
) -> list[_RawShape]:
    # Prefer clean inner-hole boxes when an outer contour wraps a single box plus attached arrowheads.
    hole_boxes = [sh.bbox for sh, is_hole in candidates if is_hole]
    kept_pairs: list[tuple[_RawShape, bool]] = []
    for sh, is_hole in candidates:
        if not is_hole:
            enclosed_holes = [hb for hb in hole_boxes if _strictly_encloses(sh.bbox, hb)]
            if len(enclosed_holes) == 1 and _bbox_area(enclosed_holes[0]) >= 0.65 * max(1.0, _bbox_area(sh.bbox)):
                continue
        kept_pairs.append((sh, is_hole))

    def _has_words(b: BBox) -> bool:
        return any(
            b[0] - 1.5 <= (w.bbox[0] + w.bbox[2]) * 0.5 <= b[2] + 1.5
            and b[1] - 1.5 <= (w.bbox[1] + w.bbox[3]) * 0.5 <= b[3] + 1.5
            for w in fig_words
        )

    all_boxes = [sh.bbox for sh, _ in kept_pairs]
    support = [
        _has_words(sh.bbox) or any(j != idx and _strictly_encloses(sh.bbox, ob) for j, ob in enumerate(all_boxes))
        for idx, (sh, _) in enumerate(kept_pairs)
    ]

    filtered: list[_RawShape] = []
    for idx, (sh, is_hole) in enumerate(kept_pairs):
        if is_hole and not support[idx]:
            if any(
                j != idx and support[j] and _shares_edge_with(sh.bbox, kept_pairs[j][0].bbox)
                for j in range(len(kept_pairs))
            ):
                continue
        filtered.append(sh)

    return _filter_border_frames(_dedup_shapes(filtered), page_box, fig_box)


def _merge_collinear_visual_segs(
    segs: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    leaf_boxes: Sequence[BBox],
    *,
    angle_tol_deg: float = 6.0,
    dist_tol_pt: float = 4.0,
    gap_tol_pt: float = 7.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if not segs:
        return []
    work = sorted(segs, key=lambda s: _dist(s[0], s[1]), reverse=True)
    used = [False] * len(work)
    merged: list[tuple[tuple[float, float], tuple[float, float]]] = []
    cos_tol = math.cos(math.radians(angle_tol_deg))

    for i in range(len(work)):
        if used[i]:
            continue
        used[i] = True
        p0, p1 = work[i]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        ux, uy = dx / seg_len, dy / seg_len
        nx, ny = -uy, ux
        min_t, max_t = 0.0, seg_len
        changed = True
        while changed:
            changed = False
            for j in range(len(work)):
                if used[j]:
                    continue
                q0, q1 = work[j]
                qdx, qdy = q1[0] - q0[0], q1[1] - q0[1]
                q_len = math.hypot(qdx, qdy)
                if q_len < 1e-6:
                    used[j] = True
                    continue
                if abs((qdx / q_len) * ux + (qdy / q_len) * uy) < cos_tol:
                    continue
                d0 = abs((q0[0] - p0[0]) * nx + (q0[1] - p0[1]) * ny)
                d1 = abs((q1[0] - p0[0]) * nx + (q1[1] - p0[1]) * ny)
                if max(d0, d1) > dist_tol_pt:
                    continue
                t0 = (q0[0] - p0[0]) * ux + (q0[1] - p0[1]) * uy
                t1 = (q1[0] - p0[0]) * ux + (q1[1] - p0[1]) * uy
                lo, hi = min(t0, t1), max(t0, t1)
                if hi < min_t - gap_tol_pt or lo > max_t + gap_tol_pt:
                    continue
                if hi < min_t or lo > max_t:
                    mid_t = 0.5 * (max(min_t, lo) + min(max_t, hi))
                    mid_pt = (p0[0] + mid_t * ux, p0[1] + mid_t * uy)
                    if any(lb[0] + 1.0 <= mid_pt[0] <= lb[2] - 1.0 and lb[1] + 1.0 <= mid_pt[1] <= lb[3] - 1.0 for lb in leaf_boxes):
                        continue
                min_t = min(min_t, lo)
                max_t = max(max_t, hi)
                used[j] = True
                changed = True
        merged.append(((p0[0] + min_t * ux, p0[1] + min_t * uy), (p0[0] + max_t * ux, p0[1] + max_t * uy)))
    return merged


def _binarize_visual_crop(crop: Any) -> Any | None:
    import cv2
    import numpy as np

    rgb = np.asarray(crop, dtype=np.uint8)
    ih, iw = rgb.shape[:2]
    if iw < 8 or ih < 8:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if int(gray.max()) - int(gray.min()) < 20:
        return None
    ink_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
    )
    bg_val = float(np.percentile(gray, 95))
    dark_mask = (gray < max(40, int(bg_val - 120.0))).astype(np.uint8) * 255
    ink = cv2.bitwise_or(ink_adapt, dark_mask)
    if int(cv2.countNonZero(ink)) < 16:
        return None
    return ink


def _collect_visual_words(
    crop: Any,
    crop_box: BBox,
    fig_box: BBox,
    page_words: Sequence[Word],
    ink: Any,
    *,
    is_raster: bool,
    get_ocr: Any,
) -> tuple[list[Word], list[tuple[int, int, int, int]], Any]:
    import cv2
    import numpy as np

    ih, iw = ink.shape[:2]
    native_fig_words = [
        w
        for w in page_words
        if fig_box[0] <= (w.bbox[0] + w.bbox[2]) * 0.5 <= fig_box[2]
        and fig_box[1] <= (w.bbox[1] + w.bbox[3]) * 0.5 <= fig_box[3]
    ]
    text_boxes_px: list[tuple[int, int, int, int]] = []
    if not is_raster and native_fig_words:
        fig_words = native_fig_words
        for w in fig_words:
            text_boxes_px.append(_pt_box_to_px_int(w.bbox, crop_box, (iw, ih)))
    else:
        ocr_rec = get_ocr()
        ocr_lines = ocr_rec.recognize(crop) if ocr_rec is not None else []
        # Reject low-confidence 1-2 char OCR hits on thick filled blobs (arrowheads).
        pre_dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5) if ocr_lines else None
        fig_words = []
        for ln in ocr_lines:
            txt = str(ln.text).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in ln.bbox)
            bx0, by0, bx1, by1 = max(0, min(iw, x0)), max(0, min(ih, y0)), max(0, min(iw, x1)), max(0, min(ih, y1))
            if (
                pre_dist is not None
                and len(txt) <= 2
                and float(getattr(ln, "confidence", 1.0)) < 0.85
                and bx1 > bx0
                and by1 > by0
                and bool(np.any(pre_dist[by0:by1, bx0:bx1] >= 2.2))
            ):
                continue
            b_pt = px_to_pt(ln.bbox, crop_box, (iw, ih))
            font_sz = max(6.0, round(b_pt[3] - b_pt[1], 2))
            fig_words.append(Word(text=txt, bbox=b_pt, font_size=font_sz, bold=False))
            text_boxes_px.append((bx0, by0, bx1, by1))

    text_mask = np.zeros((ih, iw), dtype=np.uint8)
    for x0, y0, x1, y1 in text_boxes_px:
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            text_mask[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = 255
        elif x1 > x0 and y1 > y0:
            text_mask[y0:y1, x0:x1] = 255
    return fig_words, text_boxes_px, cv2.bitwise_and(ink, cv2.bitwise_not(text_mask))


def _detect_visual_shapes(
    ink: Any,
    ink_notext: Any,
    text_boxes_px: Sequence[tuple[int, int, int, int]],
    crop_box: BBox,
    img_size: tuple[int, int],
    pt_per_px: float,
    fig_words: Sequence[Word],
    page_box: BBox,
    fig_box: BBox,
) -> tuple[list[_RawShape], Any]:
    import cv2
    import numpy as np

    iw, ih = img_size
    shape_cands: list[tuple[_RawShape, bool]] = []
    ink_shape = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, hier = cv2.findContours(ink_shape, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        h_arr = hier[0]
        for idx, cnt in enumerate(cnts):
            depth = 0
            parent = int(h_arr[idx][3])
            while parent != -1:
                depth += 1
                parent = int(h_arr[parent][3])
            x, y, bw, bh = cv2.boundingRect(cnt)
            c_area = float(cv2.contourArea(cnt))
            peri = float(cv2.arcLength(cnt, True))
            if peri <= 1e-3 or bw <= 0 or bh <= 0:
                continue
            b_px = (float(x), float(y), float(x + bw), float(y + bh))
            if any(_bbox_iou(b_px, (float(tx0), float(ty0), float(tx1), float(ty1))) >= 0.60 for tx0, ty0, tx1, ty1 in text_boxes_px):
                continue
            b_pt = px_to_pt(b_px, crop_box, img_size)
            w_pt, h_pt = b_pt[2] - b_pt[0], b_pt[3] - b_pt[1]
            if max(w_pt, h_pt) <= _MIN_SHAPE_DIM_PT or min(w_pt, h_pt) < 8.0:
                continue
            extent = c_area / float(bw * bh)
            if extent < 0.35:
                continue
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            shape_type: str | None = None
            if extent >= 0.80 and 4 <= len(approx) <= 8:
                mr_w, mr_h = cv2.minAreaRect(cnt)[1]
                if c_area / max(1.0, float(mr_w * mr_h)) >= 0.85:
                    shape_type = "rect" if len(approx) == 4 else "round_rect"
            elif len(approx) == 4 and 0.38 <= extent <= 0.65 and cv2.isContourConvex(approx):
                shape_type = "diamond"
            elif 0.62 <= extent <= 0.88 and len(approx) >= 6 and cv2.isContourConvex(approx):
                if (4.0 * math.pi * c_area / (peri * peri)) >= 0.65:
                    shape_type = "ellipse"
            if shape_type is not None:
                shape_cands.append((_RawShape(bbox=b_pt, shape_type=shape_type, dashed=False), (depth % 2 == 1)))

    # Reconstruct solid or dashed rectangles via directional morphology so dashed containers are recovered.
    dash_gap_px = max(5, int(round(6.0 / pt_per_px)))
    min_line_px = max(12, int(round(12.0 / pt_per_px)))
    h_bridge = cv2.morphologyEx(ink_notext, cv2.MORPH_CLOSE, np.ones((1, dash_gap_px), np.uint8))
    v_bridge = cv2.morphologyEx(ink_notext, cv2.MORPH_CLOSE, np.ones((dash_gap_px, 1), np.uint8))
    h_mask = cv2.morphologyEx(h_bridge, cv2.MORPH_OPEN, np.ones((1, min_line_px), np.uint8))
    v_mask = cv2.morphologyEx(v_bridge, cv2.MORPH_OPEN, np.ones((min_line_px, 1), np.uint8))

    hv_segs: list[tuple[tuple[float, float], tuple[float, float], bool]] = []
    nh, _, hstats, _ = cv2.connectedComponentsWithStats(h_mask)
    for k in range(1, nh):
        hx, hy, hw, hh, _ = (int(v) for v in hstats[k])
        if hh <= 8 and hw >= min_line_px:
            cov = float(np.any(ink_notext[max(0, hy - 1) : min(ih, hy + hh + 1), hx : hx + hw] > 0, axis=0).mean())
            p0 = px_to_pt((float(hx), float(hy) + hh * 0.5), crop_box, img_size)
            p1 = px_to_pt((float(hx + hw), float(hy) + hh * 0.5), crop_box, img_size)
            hv_segs.append((p0, p1, cov < 0.82))
    nv, _, vstats, _ = cv2.connectedComponentsWithStats(v_mask)
    for k in range(1, nv):
        vx, vy, vw, vh, _ = (int(v) for v in vstats[k])
        if vw <= 8 and vh >= min_line_px:
            cov = float(np.any(ink_notext[vy : vy + vh, max(0, vx - 1) : min(iw, vx + vw + 1)] > 0, axis=1).mean())
            p0 = px_to_pt((float(vx) + vw * 0.5, float(vy)), crop_box, img_size)
            p1 = px_to_pt((float(vx) + vw * 0.5, float(vy + vh)), crop_box, img_size)
            hv_segs.append((p0, p1, cov < 0.82))

    loop_shapes, _ = _reconstruct_rect_loops(hv_segs, loop_eps=_VISUAL_LOOP_EPS_PT)
    for lsh in loop_shapes:
        shape_cands.append((lsh, False))

    shapes = _filter_visual_shapes(shape_cands, fig_words, page_box, fig_box)
    shape_border_mask = np.zeros((ih, iw), dtype=np.uint8)
    for sh in shapes:
        px0, py0, px1, py1 = _pt_box_to_px_int(sh.bbox, crop_box, img_size)
        cx, cy = (px0 + px1) // 2, (py0 + py1) // 2
        if sh.shape_type == "diamond":
            pts = np.array([[cx, py0], [px1, cy], [cx, py1], [px0, cy]], dtype=np.int32)
            cv2.polylines(shape_border_mask, [pts], True, 255, thickness=5)
        elif sh.shape_type == "ellipse":
            cv2.ellipse(shape_border_mask, (cx, cy), (max(1, (px1 - px0) // 2), max(1, (py1 - py0) // 2)), 0, 0, 360, 255, thickness=5)
        else:
            cv2.rectangle(shape_border_mask, (px0, py0), (px1, py1), 255, thickness=4)
    return shapes, shape_border_mask


def _has_visual_chevron(
    conn_mask: Any,
    end_pt: tuple[float, float],
    prev_pt: tuple[float, float],
    crop_box: BBox,
    img_size: tuple[int, int],
) -> bool:
    out_dir = _unit_vec(prev_pt, end_pt)
    if out_dir is None:
        return False
    iw, ih = img_size
    ux, uy = out_dir
    nx, ny = -uy, ux
    pos_hits = 0
    neg_hits = 0
    perp_90_hits = 0
    for back in (2.5, 4.5, 6.5):
        for side in (2.5, 4.0):
            for sign in (1.0, -1.0):
                px_pt = (end_pt[0] - back * ux + sign * side * nx, end_pt[1] - back * uy + sign * side * ny)
                px_x, px_y = pt_to_px(px_pt, crop_box, img_size)
                ix, iy = int(round(px_x)), int(round(px_y))
                if 1 <= ix < iw - 1 and 1 <= iy < ih - 1 and int(conn_mask[iy - 1 : iy + 2, ix - 1 : ix + 2].max()) > 0:
                    if sign > 0:
                        pos_hits += 1
                    else:
                        neg_hits += 1
    for side in (5.0, 7.0):
        for sign in (1.0, -1.0):
            px_x, px_y = pt_to_px((end_pt[0] + sign * side * nx, end_pt[1] + sign * side * ny), crop_box, img_size)
            ix, iy = int(round(px_x)), int(round(px_y))
            if 0 <= ix < iw and 0 <= iy < ih and int(conn_mask[iy, ix]) > 0:
                perp_90_hits += 1
    return pos_hits >= 2 and neg_hits >= 2 and perp_90_hits == 0


def _detect_visual_arrows_and_connectors(
    conn_mask: Any,
    leaf_boxes: Sequence[BBox],
    crop_box: BBox,
    img_size: tuple[int, int],
    pt_per_px: float,
) -> tuple[list[_Polyline], list[tuple[tuple[float, float], BBox]]]:
    import cv2
    import numpy as np

    iw, ih = img_size
    # Distance transform separates thick filled arrowhead blobs from 1-3px connector stems.
    dist = cv2.distanceTransform(conn_mask, cv2.DIST_L2, 5)
    nz = dist[dist > 0]
    line_r = float(np.median(nz)) if nz.size else 1.0
    raw_arr = (dist >= max(2.2, line_r + 1.0)).astype(np.uint8) * 255
    arr_mask = cv2.dilate(raw_arr, np.ones((5, 5), np.uint8))

    visual_arrows: list[tuple[tuple[float, float], BBox]] = []
    arr_points: list[list[tuple[float, float]]] = []
    na, labels, astats, acent = cv2.connectedComponentsWithStats(arr_mask)
    for k in range(1, na):
        ax, ay, aw, ah, _ = (int(v) for v in astats[k])
        abox_pt = px_to_pt((float(ax), float(ay), float(ax + aw), float(ay + ah)), crop_box, img_size)
        if max(abox_pt[2] - abox_pt[0], abox_pt[3] - abox_pt[1]) <= 18.0:
            cent_pt = px_to_pt((float(acent[k][0]), float(acent[k][1])), crop_box, img_size)
            ys_px, xs_px = np.where((labels == k) & (conn_mask > 0))
            pts_pt = [px_to_pt((float(px_i), float(py_i)), crop_box, img_size) for py_i, px_i in zip(ys_px, xs_px)]
            visual_arrows.append((cent_pt, abox_pt))
            arr_points.append(pts_pt or [cent_pt])

    stem_mask = cv2.bitwise_and(conn_mask, cv2.bitwise_not(arr_mask))
    raw_h = cv2.HoughLinesP(
        stem_mask,
        1.0,
        np.pi / 180.0,
        threshold=max(10, int(round(8.0 / pt_per_px))),
        minLineLength=max(8, int(round(8.0 / pt_per_px))),
        maxLineGap=max(5, int(round(6.0 / pt_per_px))),
    )
    raw_segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if raw_h is not None:
        for hx0, hy0, hx1, hy1 in raw_h.reshape(-1, 4):
            p0 = px_to_pt((float(hx0), float(hy0)), crop_box, img_size)
            p1 = px_to_pt((float(hx1), float(hy1)), crop_box, img_size)
            if _dist(p0, p1) >= 4.0:
                raw_segs.append((p0, p1))

    merged_segs = _merge_collinear_visual_segs(raw_segs, leaf_boxes)

    def _extend_to_arrow(end_pt: tuple[float, float], prev_pt: tuple[float, float]) -> tuple[float, float]:
        out_dir = _unit_vec(prev_pt, end_pt)
        if out_dir is None:
            return end_pt
        best_pt = end_pt
        best_proj = 0.0
        for (cent, abox), pts_pt in zip(visual_arrows, arr_points):
            if _point_to_box_dist(end_pt, abox) > _ARROW_MATCH_DIST_PT:
                continue
            vec = _unit_vec(prev_pt, cent)
            if vec is None or (out_dir[0] * vec[0] + out_dir[1] * vec[1]) <= 0.35:
                continue
            for q in pts_pt:
                proj = (q[0] - end_pt[0]) * out_dir[0] + (q[1] - end_pt[1]) * out_dir[1]
                if 0.0 < proj <= 16.0 and proj > best_proj:
                    best_proj = proj
                    best_pt = (end_pt[0] + proj * out_dir[0], end_pt[1] + proj * out_dir[1])
        return best_pt

    def _is_seg_dashed(p0: tuple[float, float], p1: tuple[float, float]) -> bool:
        n_samples = max(10, int(round(_dist(p0, p1) / pt_per_px)))
        px0, py0 = pt_to_px(p0, crop_box, img_size)
        px1, py1 = pt_to_px(p1, crop_box, img_size)
        trim = min(0.15, 4.0 / max(1.0, float(n_samples)))
        hits: list[bool] = []
        for s_idx in range(n_samples):
            t = trim + (1.0 - 2.0 * trim) * (s_idx / max(1, n_samples - 1))
            ix = max(0, min(iw - 1, int(round(px0 + t * (px1 - px0)))))
            iy = max(0, min(ih - 1, int(round(py0 + t * (py1 - py0)))))
            hits.append(bool(np.any(stem_mask[max(0, iy - 1) : min(ih, iy + 2), max(0, ix - 1) : min(iw, ix + 2)] > 0)))
        if not hits:
            return False
        cov = sum(1 for h in hits if h) / len(hits)
        runs, gap_run, in_run = 0, 0, False
        for h in hits:
            if h:
                if not in_run:
                    runs += 1
                    in_run = True
                gap_run = 0
            else:
                gap_run += 1
                if gap_run >= 2:
                    in_run = False
        return 0.20 <= cov <= 0.80 and runs >= 2

    polylines: list[_Polyline] = []
    for p0, p1 in merged_segs:
        dashed = _is_seg_dashed(p0, p1)
        ep0 = _extend_to_arrow(p0, p1)
        ep1 = _extend_to_arrow(p1, p0)
        if ep0 == p0 and _has_visual_chevron(conn_mask, p0, p1, crop_box, img_size):
            visual_arrows.append((p0, (p0[0] - 2.0, p0[1] - 2.0, p0[0] + 2.0, p0[1] + 2.0)))
        if ep1 == p1 and _has_visual_chevron(conn_mask, p1, p0, crop_box, img_size):
            visual_arrows.append((p1, (p1[0] - 2.0, p1[1] - 2.0, p1[0] + 2.0, p1[1] + 2.0)))
        polylines.append(_Polyline(pts=[ep0, ep1], dashed=dashed))
    return _chain_polylines(polylines, joint_eps=_VISUAL_JOINT_EPS_PT), visual_arrows


def _extract_visual_diagram_from_box(
    page: Any,
    fig_box: BBox,
    page_box: BBox,
    *,
    page_words: Sequence[Word],
    is_raster: bool,
    get_ocr: Any,
    dpi: int,
) -> ExtractedDiagram:
    import cv2

    crop, crop_box = crop_page_rgb_with_bbox(page, fig_box, dpi, pad_pt=_BOX_PAD_PT)
    ink = _binarize_visual_crop(crop)
    if ink is None:
        return ExtractedDiagram(bbox=fig_box, diagram_type="unknown", source="vector_graph", confidence=0.0)

    ih, iw = ink.shape[:2]
    pt_per_px = max((crop_box[2] - crop_box[0]) / float(iw), (crop_box[3] - crop_box[1]) / float(ih), 1e-4)
    fig_words, text_boxes_px, ink_notext = _collect_visual_words(
        crop, crop_box, fig_box, page_words, ink, is_raster=is_raster, get_ocr=get_ocr
    )
    shapes, shape_border_mask = _detect_visual_shapes(
        ink, ink_notext, text_boxes_px, crop_box, (iw, ih), pt_per_px, fig_words, page_box, fig_box
    )
    leaf_boxes = [shapes[idx].bbox for idx in range(len(shapes)) if not _is_group_container(idx, shapes)]
    conn_mask = cv2.bitwise_and(ink_notext, cv2.bitwise_not(shape_border_mask))
    connectors, visual_arrows = _detect_visual_arrows_and_connectors(
        conn_mask, leaf_boxes, crop_box, (iw, ih), pt_per_px
    )
    return _build_diagram_from_parts(
        fig_box,
        shapes,
        connectors,
        (),
        (),
        fig_words,
        visual_arrows=visual_arrows,
        joint_eps=_VISUAL_JOINT_EPS_PT,
        is_visual=True,
    )


def _classify_diagram_type(
    nodes: Sequence[DiagramNode],
    edges: Sequence[DiagramEdge],
    groups: Sequence[DiagramGroup],
) -> str:
    if not edges:
        has_nested = any(g.parent_group_id is not None for g in groups)
        if has_nested or groups or len(nodes) >= 2:
            return "layered"
        return "block_diagram"
    if any(e.direction in ("directed", "bidirectional") for e in edges):
        return "flowchart"
    return "block_diagram"


def _estimate_confidence(
    nodes: Sequence[DiagramNode],
    edges: Sequence[DiagramEdge],
    *,
    is_visual: bool = False,
) -> float:
    labeled_ratio = (
        sum(1 for n in nodes if n.label.strip()) / len(nodes)
        if nodes
        else 1.0
    )
    snapped_ratio = (
        sum(1 for e in edges if e.source_node_id is not None and e.target_node_id is not None) / len(edges)
        if edges
        else 1.0
    )
    if is_visual:
        score = 0.55 + 0.15 * labeled_ratio + 0.12 * snapped_ratio
        return _round_conf(max(0.55, min(0.85, score)))
    score = 0.65 + 0.15 * labeled_ratio + 0.15 * snapped_ratio
    return _round_conf(max(0.5, min(0.95, score)))


def _make_run_stats(t0: float, *, infer_sec: float | None = None) -> RunStats:
    return RunStats(
        load_sec=0.0,
        infer_sec=round(time.perf_counter() - t0, 4) if infer_sec is None else infer_sec,
        pages=1,
        peak_rss_mb=peak_rss_mb(),
        in_process=True,
        warnings=(),
    )


class VectorFirstDiagramExtractor:
    name = "vector_first"

    def __init__(self) -> None:
        self.last_stats: RunStats | None = None

    def extract_from_page(
        self,
        page: Any,
        diagram_boxes_pt: Sequence[BBox],
        *,
        words: Sequence[Any] | None = None,
        is_raster: bool = False,
        ocr_backend: str = "rapidocr-onnx",
        device: str | None = None,
        dpi: int = 150,
    ) -> list[ExtractedDiagram]:
        t0 = time.perf_counter()
        if not diagram_boxes_pt:
            self.last_stats = _make_run_stats(t0, infer_sec=0.0)
            return []

        eff_device = "auto" if device is None else device
        ocr_rec: Any = None
        ocr_checked = False

        def _get_ocr() -> Any:
            nonlocal ocr_rec, ocr_checked
            if not ocr_checked:
                ocr_checked = True
                ocr_rec = get_backend("ocr", ocr_backend, device=eff_device)
            return ocr_rec

        pbox = page_bbox(page)
        if is_raster:
            results = [
                _extract_visual_diagram_from_box(
                    page,
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    pbox,
                    page_words=(),
                    is_raster=True,
                    get_ocr=_get_ocr,
                    dpi=dpi,
                )
                for box in diagram_boxes_pt
            ]
            self.last_stats = _make_run_stats(t0)
            return results

        if words is None:
            from .text import extract_words

            page_words: list[Word] = extract_words(page)
        else:
            page_words = list(words)

        raw_rects = tuple(getattr(page, "rects", ()) or ())
        raw_lines = tuple(getattr(page, "lines", ()) or ())
        raw_curves = tuple(getattr(page, "curves", ()) or ())
        raw_images = tuple(getattr(page, "images", ()) or ())

        results: list[ExtractedDiagram] = []
        for box in diagram_boxes_pt:
            fig_box: BBox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            diag = _extract_diagram_from_box(fig_box, pbox, raw_rects, raw_lines, raw_curves, page_words)
            if not (diag.nodes or diag.edges or diag.groups):
                padded = (
                    fig_box[0] - _BOX_PAD_PT,
                    fig_box[1] - _BOX_PAD_PT,
                    fig_box[2] + _BOX_PAD_PT,
                    fig_box[3] + _BOX_PAD_PT,
                )
                clip = clip_box(padded, pbox) or fig_box
                if _has_raster_image_in_box(raw_images, clip, pbox[3]):
                    diag = _extract_visual_diagram_from_box(
                        page,
                        fig_box,
                        pbox,
                        page_words=page_words,
                        is_raster=False,
                        get_ocr=_get_ocr,
                        dpi=dpi,
                    )
            results.append(diag)
        self.last_stats = _make_run_stats(t0)
        return results


def extract_diagrams_on_page(
    page: Any,
    figure_boxes_pt: Sequence[BBox],
    *,
    words: Sequence[Word] | None = None,
    is_raster: bool = False,
    extractor: str | DiagramExtractor = "vector_first",
    ocr_backend: str = "rapidocr-onnx",
    device: str = "auto",
    dpi: int = 150,
) -> list[ExtractedDiagram]:
    if not figure_boxes_pt:
        return []
    backend = get_backend("diagram", extractor, device=device) if isinstance(extractor, str) else extractor
    return list(
        backend.extract_from_page(
            page,
            figure_boxes_pt,
            words=words,
            is_raster=is_raster,
            ocr_backend=ocr_backend,
            device=device,
            dpi=dpi,
        )
    )


def diagram_to_ir_dict(
    diagram_id: str,
    element_id: str,
    page: int,
    source_file: str,
    diagram: ExtractedDiagram,
) -> dict[str, Any]:
    bbox = _round_bbox(diagram.bbox)
    conf = _round_conf(diagram.confidence)
    nodes = [
        {
            "id": n.id,
            "shape_type": n.shape_type,
            "label": n.label,
            "bbox": _round_bbox(n.bbox),
            "parent_group_id": n.parent_group_id,
            "text_element_ids": list(n.text_element_ids),
            "confidence": _round_conf(n.confidence),
            "source_ref": {"file": source_file, "page": page, "bbox": _round_bbox(n.bbox)},
        }
        for n in diagram.nodes
    ]
    edges = [
        {
            "id": e.id,
            "source_node_id": e.source_node_id,
            "target_node_id": e.target_node_id,
            "direction": e.direction,
            "style": e.style,
            "label": e.label,
            "waypoints": [[round(float(x), 3) + 0.0, round(float(y), 3) + 0.0] for x, y in e.waypoints],
            "confidence": _round_conf(e.confidence),
            "source_ref": {
                "file": source_file,
                "page": page,
                "bbox": _round_bbox(_waypoints_bbox(e.waypoints, diagram.bbox)),
            },
        }
        for e in diagram.edges
    ]
    groups = [
        {
            "id": g.id,
            "label": g.label,
            "bbox": _round_bbox(g.bbox),
            "parent_group_id": g.parent_group_id,
            "member_node_ids": list(g.member_node_ids),
            "confidence": _round_conf(g.confidence),
            "source_ref": {"file": source_file, "page": page, "bbox": _round_bbox(g.bbox)},
        }
        for g in diagram.groups
    ]
    meta: dict[str, Any] = {
        "element_id": element_id,
        "source": diagram.source,
        "confidence": conf,
    }
    extra_meta = getattr(diagram, "metadata", None)
    if isinstance(extra_meta, Mapping):
        for k, v in extra_meta.items():
            if k not in meta:
                meta[k] = v
    return {
        "id": diagram_id,
        "element_id": element_id,
        "page": page,
        "bbox": bbox,
        "diagram_type": diagram.diagram_type,
        "source": diagram.source,
        "confidence": conf,
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "source_ref": {"file": source_file, "page": page, "bbox": bbox},
        "metadata": meta,
    }
