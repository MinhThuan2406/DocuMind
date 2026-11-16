from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image

from .layout import (
    LayoutCache,
    clamp_and_rescale,
    clusters_to_items,
    file_sha256,
    get_layout_detector,
    postprocess_layout,
)
from .models import FIGURE_FOOTER_FRAC, FIGURE_MIN_AREA, BBox, FigureBox

DEFAULT_DPI = 150
LAYOUT_BOX_CONFIDENCE = 0.95
_PNG_DEFAULT_DPI = 96.012

_REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    ix = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    iy = max(0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    inter = ix * iy
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _box_inside(inner: Sequence[float], outer: Sequence[float], tolerance: float = 0.15) -> bool:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (ix * iy / inner_area) >= (1.0 - tolerance)


def _boxes_close(b1: Sequence[float], b2: Sequence[float], pad_x: float, pad_y: float) -> bool:
    return not (
        b1[2] + pad_x < b2[0]
        or b2[2] + pad_x < b1[0]
        or b1[3] + pad_y < b2[1]
        or b2[3] + pad_y < b1[1]
    )


def _detect_visual_containers(bgr: np.ndarray) -> list[list[float]]:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)

    h_len = max(20, int(w * 0.015))
    v_len = max(20, int(h * 0.015))
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 3))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, v_len))
    combined = cv2.bitwise_or(
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_k),
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_k),
    )
    box_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, box_k)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = w * h * 0.008
    max_area = w * h * 0.85

    containers: list[list[float]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if min_area <= area <= max_area and cw >= 120 and ch >= 60:
            containers.append([float(x), float(y), float(x + cw), float(y + ch)])

    containers.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    filtered: list[list[float]] = []
    for b in containers:
        if not any(_compute_iou(b, fb) > 0.85 for fb in filtered):
            filtered.append(b)
    return filtered


def _run_layout_items(
    detector: Any,
    bgr: np.ndarray,
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    ih, iw = bgr.shape[:2]
    pil_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # Match the 72/96.012 PNG scale used when Docling's ImageDocumentBackend loaded PyMuPDF PNGs.
    page_w = iw * 72.0 / _PNG_DEFAULT_DPI
    page_h = ih * 72.0 / _PNG_DEFAULT_DPI
    target_wh = (max(1, round(page_w)), max(1, round(page_h)))
    pil_in = pil_rgb if pil_rgb.size == target_wh else pil_rgb.resize(target_wh, resample=Image.Resampling.LANCZOS)

    raw_items = detector.detect(pil_in)
    page_size = (page_w, page_h)
    scaled = clamp_and_rescale(
        [(it.label, it.bbox, it.score) for it in raw_items],
        (float(target_wh[0]), float(target_wh[1])),
        page_size,
    )
    clusters = postprocess_layout(scaled, page_size, keep_empty_clusters=False)
    return clusters_to_items(clusters), page_size


def _v1_group_boxes(
    bgr: np.ndarray,
    items: list[dict[str, Any]],
    img_size: tuple[int, int],
    page_size: tuple[float, float],
    min_area: float,
) -> list[list[float]]:
    img_w, img_h = img_size
    dw, dh = page_size
    sx = img_w / dw if dw > 0 else 1.0
    sy = img_h / dh if dh > 0 else 1.0

    src = [it for it in items if it.get("bbox") and len(it["bbox"]) == 4]
    pixel_items = [
        {
            "orig_item": it,
            "label": it.get("label", "text"),
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

    boxes_px: list[list[float]] = []
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
            if _compute_iou(u_box, vc) > 0.6 or (
                _box_inside(u_box, vc, tolerance=0.1) and vc_area < 1.35 * u_area
            ):
                u_box = [min(u_box[0], vc[0]), min(u_box[1], vc[1]), max(u_box[2], vc[2]), max(u_box[3], vc[3])]
                break

        boxes_px.append([
            max(0.0, round(u_box[0] - 8, 1)),
            max(0.0, round(u_box[1] - 8, 1)),
            min(float(img_w), round(u_box[2] + 8, 1)),
            min(float(img_h), round(u_box[3] + 8, 1)),
        ])
    return boxes_px


def _to_pt_box(box_px: Sequence[float], img_size: tuple[int, int], page_rect: pymupdf.Rect) -> BBox:
    iw, ih = img_size
    sx = float(page_rect.width) / iw
    sy = float(page_rect.height) / ih
    return (
        round(box_px[0] * sx, 3),
        round(box_px[1] * sy, 3),
        round(box_px[2] * sx, 3),
        round(box_px[3] * sy, 3),
    )


def _render_bgr(page: pymupdf.Page, dpi: int) -> tuple[np.ndarray, tuple[int, int]]:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, (pix.width, pix.height)


def _table_boxes_px(
    items: list[dict[str, Any]],
    img_size: tuple[int, int],
    page_size: tuple[float, float],
    footer_frac: float,
) -> list[list[float]]:
    iw, ih = img_size
    dw, dh = page_size
    sx = iw / dw if dw > 0 else 1.0
    sy = ih / dh if dh > 0 else 1.0
    tables: list[list[float]] = []
    for it in items:
        if it.get("label") != "table" or not it.get("bbox") or len(it["bbox"]) != 4:
            continue
        b = [it["bbox"][0] * sx, it["bbox"][1] * sy, it["bbox"][2] * sx, it["bbox"][3] * sy]
        if b[2] > b[0] and b[3] > b[1] and b[1] < footer_frac * ih:
            tables.append(b)
    return tables


def detect_figures(
    pdf_path: str | Path,
    pages: Sequence[int] | None = None,
    *,
    layout: str = "heron",
    dpi: int = DEFAULT_DPI,
    device: str = "auto",
    min_area: float = FIGURE_MIN_AREA,
    footer_frac: float = FIGURE_FOOTER_FRAC,
    cache_dir: str | Path | None = None,
    include_tables: bool = True,
) -> dict[int, list[FigureBox]]:
    path = Path(pdf_path)
    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else _REPO_ROOT / "outputs" / "08_pdf_figure_crops" / path.stem / "_layout_cache"
    )
    detector_holder: list[Any] = []

    def get_detector() -> Any:
        if not detector_holder:
            detector_holder.append(get_layout_detector(layout, device))
        return detector_holder[0]

    if layout == "heron":
        from .layout.heron_transformers import HERON_REVISION

        revision = HERON_REVISION
    else:
        revision = get_detector()[1]
    cache = LayoutCache(cache_root, backend=layout, revision=revision, dpi=dpi, pdf_hash=file_sha256(path))

    doc = pymupdf.open(str(path))
    results: dict[int, list[FigureBox]] = {}
    try:
        target_pages = list(range(1, len(doc) + 1)) if pages is None else sorted(set(pages))
        for pno in target_pages:
            if not (1 <= pno <= len(doc)):
                raise ValueError(f"page {pno} out of range 1..{len(doc)}")
            page = doc[pno - 1]
            bgr, img_size = _render_bgr(page, dpi)
            cached = cache.load(pno)
            if cached is not None:
                items, page_size = cached
            else:
                detector, _ = get_detector()
                items, page_size = _run_layout_items(detector, bgr)
                cache.save(pno, items, page_size)

            iw, ih = img_size
            raw_figs = _v1_group_boxes(bgr, items, img_size, page_size, min_area)
            kept_figs = [
                b for b in raw_figs if (b[2] - b[0]) * (b[3] - b[1]) >= min_area * iw * ih and b[1] < footer_frac * ih
            ]
            kept_figs.sort(key=lambda b: (b[1], b[0]))

            page_boxes: list[FigureBox] = [
                FigureBox(
                    bbox=_to_pt_box(b, img_size, page.rect),
                    kind="figure",
                    source="v1_group",
                    confidence=LAYOUT_BOX_CONFIDENCE,
                )
                for b in kept_figs
            ]
            if include_tables:
                raw_tables = _table_boxes_px(items, img_size, page_size, footer_frac)
                raw_tables.sort(key=lambda b: (b[1], b[0]))
                page_boxes.extend(
                    FigureBox(
                        bbox=_to_pt_box(b, img_size, page.rect),
                        kind="table",
                        source="layout_table",
                        confidence=LAYOUT_BOX_CONFIDENCE,
                    )
                    for b in raw_tables
                )
            results[pno] = page_boxes
    finally:
        doc.close()

    return results
