from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from ._pdf_io import bbox_area, bbox_inter_area, bbox_iou
from .backends.base import FigureGrouper, PageLayout
from .models import FIGURE_MIN_AREA, FigureSource, GroupedFigure

__all__ = ["FigureGrouper", "RawPictureGrouper", "V1AnchorGrouper", "V2VectorVisionGrouper"]

_LEGACY_DEFAULT_SCORE = 0.95
_CANONICAL_LONG_PT = 800.0
_TABLE_BARRIER_GAP_PT = 4.0
_SEPARATOR_LABELS = frozenset(
    {"caption", "section_header", "title", "list_item", "page_header", "page_footer"}
)
_V1_SKIP_LABELS = frozenset(
    {"text", "page_header", "page_footer", "footnote", "list_item", "section_header", "title", "code"}
)


def _item_score(item: Mapping[str, Any], default: float = _LEGACY_DEFAULT_SCORE) -> float:
    val = item.get("score")
    return float(val) if val is not None else default


class _DisjointSetUnion:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            self.parent[ri] = rj
        elif self.rank[ri] > self.rank[rj]:
            self.parent[rj] = ri
        else:
            self.parent[rj] = ri
            self.rank[ri] += 1


def _box_inside(inner: Sequence[float], outer: Sequence[float], tolerance: float = 0.15) -> bool:
    inter = bbox_inter_area(inner, outer)
    inner_area = max(1.0, bbox_area(inner))
    return (inter / inner_area) >= (1.0 - tolerance)


def _boxes_close(b1: Sequence[float], b2: Sequence[float], pad_x: float, pad_y: float) -> bool:
    return not (
        b1[2] + pad_x < b2[0]
        or b2[2] + pad_x < b1[0]
        or b1[3] + pad_y < b2[1]
        or b2[3] + pad_y < b1[1]
    )


def _detect_visual_containers(
    bgr: np.ndarray,
    *,
    scale: float = 1.0,
    min_morph_len: int = 20,
    box_kernel: tuple[int, int] = (25, 10),
    min_w: float = 120.0,
    min_h: float = 60.0,
) -> list[list[float]]:
    ih, iw = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)

    h_len = max(min_morph_len, int(iw * 0.015) if scale == 1.0 else round(iw * 0.015))
    v_len = max(min_morph_len, int(ih * 0.015) if scale == 1.0 else round(ih * 0.015))
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 3))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_len))
    combined = cv2.bitwise_or(
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_k),
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_k),
    )
    box_k = cv2.getStructuringElement(cv2.MORPH_RECT, box_kernel)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, box_k)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = iw * ih * 0.008
    max_area = iw * ih * 0.85

    containers: list[list[float]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if min_area <= area <= max_area and cw * scale >= min_w and ch * scale >= min_h:
            containers.append([
                float(x) * scale,
                float(y) * scale,
                float(x + cw) * scale,
                float(y + ch) * scale,
            ])

    containers.sort(key=bbox_area, reverse=True)
    filtered: list[list[float]] = []
    for b in containers:
        if not any(bbox_iou(b, fb) > 0.85 for fb in filtered):
            filtered.append(b)
    return filtered


def _v1_group_boxes(
    bgr: np.ndarray,
    items: list[dict[str, Any]],
    img_size: tuple[int, int],
    page_size: tuple[float, float],
    min_area: float,
) -> list[GroupedFigure]:
    img_w, img_h = img_size
    dw, dh = page_size
    sx = img_w / dw if dw > 0 else 1.0
    sy = img_h / dh if dh > 0 else 1.0

    src = [
        it
        for it in items
        if it.get("bbox") and len(it["bbox"]) == 4 and str(it.get("label", "")).lower() not in _V1_SKIP_LABELS
    ]
    pixel_items = [
        {
            "orig_item": it,
            "label": it.get("label", "text"),
            "score": _item_score(it),
            "bbox": [
                round(it["bbox"][0] * sx, 1),
                round(it["bbox"][1] * sy, 1),
                round(it["bbox"][2] * sx, 1),
                round(it["bbox"][3] * sy, 1),
            ],
        }
        for it in src
    ]

    n_items = len(pixel_items)
    if n_items == 0:
        return []

    dsu = _DisjointSetUnion(n_items)
    visual_containers = _detect_visual_containers(bgr)

    def has_intervening_separator(b1: list[float], b2: list[float]) -> bool:
        y_top = min(b1[3], b2[3])
        y_bot = max(b1[1], b2[1])
        if y_bot - y_top < 20:
            return False
        for it in pixel_items:
            if it["label"] in ("section_header", "list_item"):
                ib = it["bbox"]
                if ib[1] >= y_top - 5 and ib[3] <= y_bot + 5:
                    return True
        return False

    def is_cross_column(b1: list[float], b2: list[float]) -> bool:
        cx1 = (b1[0] + b1[2]) / 2.0
        cx2 = (b2[0] + b2[2]) / 2.0
        return (cx1 < img_w * 0.48 and cx2 > img_w * 0.52) or (cx2 < img_w * 0.48 and cx1 > img_w * 0.52)

    for vc in visual_containers:
        vc_area = (vc[2] - vc[0]) * (vc[3] - vc[1])
        if vc_area > 0.35 * img_w * img_h:
            continue
        n_headers = sum(
            1
            for it in pixel_items
            if it["label"] in ("section_header", "title") and _box_inside(it["bbox"], vc, tolerance=0.15)
        )
        if n_headers >= 2:
            continue
        contained = [
            i
            for i, it in enumerate(pixel_items)
            if it["label"] not in ("section_header", "list_item", "title")
            and _box_inside(it["bbox"], vc, tolerance=0.15)
        ]
        if len(contained) >= 2:
            first = contained[0]
            for idx in contained[1:]:
                b_first, b_idx = pixel_items[first]["bbox"], pixel_items[idx]["bbox"]
                if not has_intervening_separator(b_first, b_idx) and not is_cross_column(b_first, b_idx):
                    dsu.union(first, idx)

    for i, it1 in enumerate(pixel_items):
        lbl1 = it1["label"].lower()
        b1 = it1["bbox"]
        if lbl1 not in ("picture", "figure", "caption", "formula") and (b1[2] - b1[0] >= 0.45 * img_w):
            continue
        for j in range(i + 1, n_items):
            it2 = pixel_items[j]
            lbl2 = it2["label"].lower()
            b2 = it2["bbox"]
            if has_intervening_separator(b1, b2) or is_cross_column(b1, b2):
                continue
            if lbl1 in ("picture", "figure") or lbl2 in ("picture", "figure"):
                gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                if gap_x <= 160 and _boxes_close(b1, b2, pad_x=45, pad_y=45):
                    dsu.union(i, j)
            elif abs(b1[1] - b2[1]) < 35 and abs(b1[3] - b2[3]) < 35:
                gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                if gap_x < 140 and (b1[2] - b1[0] < 0.45 * img_w) and (b2[2] - b2[0] < 0.45 * img_w):
                    dsu.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n_items):
        clusters.setdefault(dsu.find(i), []).append(i)

    boxes_px: list[GroupedFigure] = []
    for idx_list in clusters.values():
        if len(idx_list) < 2 and not any(pixel_items[idx]["label"] in ("picture", "figure") for idx in idx_list):
            continue
        if not any(
            pixel_items[idx]["orig_item"]["label"] in ("picture", "figure")
            and (pixel_items[idx]["orig_item"]["bbox"][2] - pixel_items[idx]["orig_item"]["bbox"][0])
            * (pixel_items[idx]["orig_item"]["bbox"][3] - pixel_items[idx]["orig_item"]["bbox"][1])
            >= min_area * dw * dh
            for idx in idx_list
        ):
            continue

        xs = [v for idx in idx_list for v in (pixel_items[idx]["bbox"][0], pixel_items[idx]["bbox"][2])]
        ys = [v for idx in idx_list for v in (pixel_items[idx]["bbox"][1], pixel_items[idx]["bbox"][3])]
        u_box = [min(xs), min(ys), max(xs), max(ys)]
        u_area = max(1.0, (u_box[2] - u_box[0]) * (u_box[3] - u_box[1]))
        for vc in visual_containers:
            vc_area = max(1.0, (vc[2] - vc[0]) * (vc[3] - vc[1]))
            if bbox_iou(u_box, vc) > 0.6 or (
                _box_inside(u_box, vc, tolerance=0.1) and vc_area < 1.35 * u_area
            ):
                u_box = [min(u_box[0], vc[0]), min(u_box[1], vc[1]), max(u_box[2], vc[2]), max(u_box[3], vc[3])]
                break

        total_w = 0.0
        weighted = 0.0
        for idx in idx_list:
            pb = pixel_items[idx]["bbox"]
            area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
            weighted += area * pixel_items[idx]["score"]
            total_w += area
        score = round(weighted / total_w, 4) if total_w > 0 else _LEGACY_DEFAULT_SCORE

        boxes_px.append(
            GroupedFigure(
                bbox=(
                    max(0.0, round(u_box[0] - 8, 1)),
                    max(0.0, round(u_box[1] - 8, 1)),
                    min(float(img_w), round(u_box[2] + 8, 1)),
                    min(float(img_h), round(u_box[3] + 8, 1)),
                ),
                score=score,
                source="v1_group",
            )
        )
    return boxes_px


class V1AnchorGrouper:
    name = "v1_anchor"
    source: FigureSource = "v1_group"

    def group(
        self,
        bgr: np.ndarray,
        layout: PageLayout,
        min_area: float = FIGURE_MIN_AREA,
    ) -> list[GroupedFigure]:
        ih, iw = bgr.shape[:2]
        return _v1_group_boxes(bgr, layout.items, (iw, ih), layout.frame, min_area)


class RawPictureGrouper:
    name = "raw_picture"
    source: FigureSource = "layout_picture"

    def group(
        self,
        bgr: np.ndarray,
        layout: PageLayout,
        min_area: float = FIGURE_MIN_AREA,
    ) -> list[GroupedFigure]:
        del min_area
        ih, iw = bgr.shape[:2]
        dw, dh = layout.frame
        sx = iw / dw if dw > 0 else 1.0
        sy = ih / dh if dh > 0 else 1.0
        boxes: list[GroupedFigure] = []
        for it in layout.items:
            if str(it.get("label", "")).lower() not in ("picture", "figure"):
                continue
            bbox = it.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0 = max(0.0, float(bbox[0]) * sx)
            y0 = max(0.0, float(bbox[1]) * sy)
            x1 = min(float(iw), float(bbox[2]) * sx)
            y1 = min(float(ih), float(bbox[3]) * sy)
            if x1 > x0 and y1 > y0:
                boxes.append(GroupedFigure(bbox=(x0, y0, x1, y1), score=round(_item_score(it), 4), source=self.source))
        return boxes


def _axis_gap(b1: Sequence[float], b2: Sequence[float]) -> tuple[float, float]:
    gx = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
    gy = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return gx, gy


def _count_line_peaks(coverage: np.ndarray, threshold: float, merge_dist: int) -> int:
    hits = np.flatnonzero(coverage >= threshold)
    if hits.size == 0:
        return 0
    count = 1
    prev = int(hits[0])
    for idx in hits[1:]:
        cur = int(idx)
        if cur - prev > merge_dist:
            count += 1
        prev = cur
    return count


def _has_bordered_grid(bgr: np.ndarray, box_px: Sequence[float], px_to_canon: float) -> bool:
    ih, iw = bgr.shape[:2]
    x0 = max(0, int(round(box_px[0])))
    y0 = max(0, int(round(box_px[1])))
    x1 = min(iw, int(round(box_px[2])))
    y1 = min(ih, int(round(box_px[3])))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return False
    crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    dark = crop < 210
    merge_dist = max(2, round(2.0 / px_to_canon))
    n_h = _count_line_peaks(dark.mean(axis=1), 0.55, merge_dist)
    n_v = _count_line_peaks(dark.mean(axis=0), 0.55, merge_dist)
    return n_h >= 3 and n_v >= 3


class V2VectorVisionGrouper:
    name = "v2_vector_vision"
    source: FigureSource = "v2_vector_vision"

    def group(
        self,
        bgr: np.ndarray,
        layout: PageLayout,
        min_area: float = FIGURE_MIN_AREA,
    ) -> list[GroupedFigure]:
        ih, iw = bgr.shape[:2]
        if ih <= 0 or iw <= 0:
            return []
        dw, dh = layout.frame
        px_to_canon = _CANONICAL_LONG_PT / float(max(iw, ih))
        canon_w = float(iw) * px_to_canon
        canon_h = float(ih) * px_to_canon
        sx_canon = canon_w / dw if dw > 0 else px_to_canon
        sy_canon = canon_h / dh if dh > 0 else px_to_canon
        sx_px = iw / dw if dw > 0 else 1.0
        sy_px = ih / dh if dh > 0 else 1.0

        canon_items: list[dict[str, Any]] = []
        for it in layout.items:
            raw = it.get("bbox")
            if not raw or len(raw) != 4:
                continue
            bx0, by0, bx1, by1 = float(raw[0]) * sx_canon, float(raw[1]) * sy_canon, float(raw[2]) * sx_canon, float(raw[3]) * sy_canon
            if bx1 <= bx0 or by1 <= by0:
                continue
            canon_items.append({
                "label": str(it.get("label", "text")).lower(),
                "score": _item_score(it),
                "bbox": [bx0, by0, bx1, by1],
                "bbox_px": [float(raw[0]) * sx_px, float(raw[1]) * sy_px, float(raw[2]) * sx_px, float(raw[3]) * sy_px],
            })

        n_items = len(canon_items)
        if n_items == 0:
            return []

        visual_containers = _detect_visual_containers(
            bgr,
            scale=px_to_canon,
            min_morph_len=5,
            box_kernel=(max(5, round(12.0 / px_to_canon)), max(3, round(5.0 / px_to_canon))),
            min_w=55.0,
            min_h=28.0,
        )
        pic_indices = [i for i, it in enumerate(canon_items) if it["label"] in ("picture", "figure")]
        if not pic_indices:
            return []

        # Mark standalone tables that act as Table-Barriers against figure merging.
        standalone_tables: set[int] = set()
        for i, it in enumerate(canon_items):
            if it["label"] != "table":
                continue
            tb = it["bbox"]
            inside_pic = any(_box_inside(tb, canon_items[pi]["bbox"], tolerance=0.15) for pi in pic_indices)
            if inside_pic:
                if _has_bordered_grid(bgr, it["bbox_px"], px_to_canon):
                    standalone_tables.add(i)
                continue
            min_gap = min(max(_axis_gap(tb, canon_items[pi]["bbox"])) for pi in pic_indices)
            if min_gap >= _TABLE_BARRIER_GAP_PT or _has_bordered_grid(bgr, it["bbox_px"], px_to_canon):
                standalone_tables.add(i)
                continue
            shared_vc = any(
                _box_inside(tb, vc, tolerance=0.15)
                and any(_box_inside(canon_items[pi]["bbox"], vc, tolerance=0.15) for pi in pic_indices)
                for vc in visual_containers
            )
            if not shared_vc:
                standalone_tables.add(i)

        separator_boxes = [
            it["bbox"]
            for idx, it in enumerate(canon_items)
            if it["label"] in _SEPARATOR_LABELS or idx in standalone_tables
        ]

        def is_blocked_item(idx: int) -> bool:
            return canon_items[idx]["label"] in _SEPARATOR_LABELS or idx in standalone_tables

        def has_intervening_separator(b1: Sequence[float], b2: Sequence[float]) -> bool:
            y_top = min(b1[3], b2[3])
            y_bot = max(b1[1], b2[1])
            x_min = min(b1[0], b2[0])
            x_max = max(b1[2], b2[2])
            if y_bot >= y_top - 2.0:
                for sb in separator_boxes:
                    cy = 0.5 * (sb[1] + sb[3])
                    if (y_top - 2.0) <= cy <= (y_bot + 2.0) and min(sb[2], x_max) > max(sb[0], x_min):
                        return True
            x_left = min(b1[2], b2[2])
            x_right = max(b1[0], b2[0])
            y_min = min(b1[1], b2[1])
            y_max = max(b1[3], b2[3])
            if x_right >= x_left - 2.0:
                for sb in separator_boxes:
                    cx = 0.5 * (sb[0] + sb[2])
                    if (x_left - 2.0) <= cx <= (x_right + 2.0) and min(sb[3], y_max) > max(sb[1], y_min):
                        return True
            return False

        def is_cross_column(b1: Sequence[float], b2: Sequence[float]) -> bool:
            cx1 = 0.5 * (b1[0] + b1[2])
            cx2 = 0.5 * (b2[0] + b2[2])
            return (cx1 < canon_w * 0.48 and cx2 > canon_w * 0.52) or (cx2 < canon_w * 0.48 and cx1 > canon_w * 0.52)

        dsu = _DisjointSetUnion(n_items)
        item_vcs: dict[int, list[list[float]]] = {}

        for vc in visual_containers:
            vc_area = bbox_area(vc)
            if vc_area > 0.35 * canon_w * canon_h or (vc[2] - vc[0]) >= 0.95 * canon_w or vc[1] >= 0.88 * canon_h:
                continue
            n_headers = sum(
                1
                for it in canon_items
                if it["label"] in ("section_header", "title") and _box_inside(it["bbox"], vc, tolerance=0.15)
            )
            if n_headers >= 2:
                continue
            if any(
                not _box_inside(canon_items[ti]["bbox"], vc, tolerance=0.15)
                and bbox_inter_area(canon_items[ti]["bbox"], vc) / max(1.0, bbox_area(canon_items[ti]["bbox"])) >= 0.30
                for ti in standalone_tables
            ):
                continue
            contained = [
                i
                for i, it in enumerate(canon_items)
                if not is_blocked_item(i)
                and it["bbox"][1] < 0.88 * canon_h
                and _box_inside(it["bbox"], vc, tolerance=0.15)
            ]
            overlapping_pics = [
                pi
                for pi in pic_indices
                if bbox_inter_area(canon_items[pi]["bbox"], vc) > 0.0
                and pi not in contained
                and bbox_inter_area(canon_items[pi]["bbox"], vc) / max(1.0, bbox_area(canon_items[pi]["bbox"])) >= 0.04
            ]
            members = contained + (overlapping_pics if contained else [])
            if len(members) >= 2:
                anchor = members[0]
                for idx in members[1:]:
                    b_anchor, b_idx = canon_items[anchor]["bbox"], canon_items[idx]["bbox"]
                    if not has_intervening_separator(b_anchor, b_idx) and not is_cross_column(b_anchor, b_idx):
                        dsu.union(anchor, idx)
                        item_vcs.setdefault(anchor, []).append(vc)
                        item_vcs.setdefault(idx, []).append(vc)

        for i in range(n_items):
            if is_blocked_item(i):
                continue
            it1 = canon_items[i]
            lbl1 = it1["label"]
            b1 = it1["bbox"]
            if lbl1 not in ("picture", "figure", "formula") and (b1[2] - b1[0] >= 0.45 * canon_w):
                continue
            for j in range(i + 1, n_items):
                if is_blocked_item(j):
                    continue
                it2 = canon_items[j]
                lbl2 = it2["label"]
                b2 = it2["bbox"]
                if has_intervening_separator(b1, b2) or is_cross_column(b1, b2):
                    continue
                gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                if lbl1 in ("picture", "figure") or lbl2 in ("picture", "figure"):
                    if gap_x <= 77.0 and _boxes_close(b1, b2, pad_x=22.0, pad_y=22.0):
                        dsu.union(i, j)
                elif abs(b1[1] - b2[1]) < 17.0 and abs(b1[3] - b2[3]) < 17.0:
                    if gap_x < 67.0 and (b1[2] - b1[0] < 0.45 * canon_w) and (b2[2] - b2[0] < 0.45 * canon_w):
                        dsu.union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n_items):
            clusters.setdefault(dsu.find(i), []).append(i)

        page_area_canon = canon_w * canon_h
        pad_canon = 4.0
        boxes_px: list[GroupedFigure] = []
        for idx_list in clusters.values():
            if not any(
                canon_items[idx]["label"] in ("picture", "figure")
                and bbox_area(canon_items[idx]["bbox"]) >= min_area * page_area_canon
                for idx in idx_list
            ):
                continue

            xs = [v for idx in idx_list for v in (canon_items[idx]["bbox"][0], canon_items[idx]["bbox"][2])]
            ys = [v for idx in idx_list for v in (canon_items[idx]["bbox"][1], canon_items[idx]["bbox"][3])]
            u_box = [min(xs), min(ys), max(xs), max(ys)]
            u_area = max(1.0, bbox_area(u_box))

            expanded_vcs: list[list[float]] = []
            for idx in idx_list:
                expanded_vcs.extend(item_vcs.get(idx, ()))
            for vc in expanded_vcs:
                if not has_intervening_separator(u_box, vc):
                    u_box = [min(u_box[0], vc[0]), min(u_box[1], vc[1]), max(u_box[2], vc[2]), max(u_box[3], vc[3])]

            for vc in visual_containers:
                vc_area = max(1.0, bbox_area(vc))
                if bbox_iou(u_box, vc) > 0.6 or (
                    _box_inside(u_box, vc, tolerance=0.1) and vc_area < 1.35 * u_area
                ):
                    if not any(
                        bbox_inter_area(canon_items[ti]["bbox"], vc) / max(1.0, bbox_area(canon_items[ti]["bbox"])) >= 0.30
                        for ti in standalone_tables
                    ):
                        u_box = [min(u_box[0], vc[0]), min(u_box[1], vc[1]), max(u_box[2], vc[2]), max(u_box[3], vc[3])]
                    break

            total_w = 0.0
            weighted = 0.0
            for idx in idx_list:
                area = max(1.0, bbox_area(canon_items[idx]["bbox"]))
                weighted += area * canon_items[idx]["score"]
                total_w += area
            score = round(weighted / total_w, 4) if total_w > 0 else _LEGACY_DEFAULT_SCORE

            boxes_px.append(
                GroupedFigure(
                    bbox=(
                        max(0.0, round((u_box[0] - pad_canon) / px_to_canon, 1)),
                        max(0.0, round((u_box[1] - pad_canon) / px_to_canon, 1)),
                        min(float(iw), round((u_box[2] + pad_canon) / px_to_canon, 1)),
                        min(float(ih), round((u_box[3] + pad_canon) / px_to_canon, 1)),
                    ),
                    score=score,
                    source=self.source,
                )
            )
        return boxes_px
