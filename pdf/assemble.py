from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import BBox, FigureBox, PageProfile, RasterPage, TextBlock, Word
from .text import order_blocks, xy_cut_order

_HEADER_BAND_FRAC = 0.10
_FOOTER_BAND_FRAC = 0.90
_MIN_MULTI_PAGE_DOCS = 3
_MIN_REPEAT_PAGES = 3
_MIN_REPEAT_FRAC = 0.5


@dataclass(frozen=True)
class AssembledItem:
    kind: str  # "text" | "figure" | "table"
    bbox: BBox
    text: str
    source: str
    confidence: float
    level: int
    parent_index: int | None
    font_size: float | None = None
    bold: bool | None = None
    label: str | None = None


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter_area(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def filter_overlapping_figures(figures: Sequence[FigureBox]) -> list[FigureBox]:
    fig_boxes = [f for f in figures if f.kind == "figure"]
    kept: list[FigureBox] = []
    for box in figures:
        if box.kind == "table":
            t_area = _area(box.bbox)
            if t_area > 0 and any(_inter_area(box.bbox, fb.bbox) / t_area >= 0.5 for fb in fig_boxes):
                continue
        kept.append(box)
    return kept


def _assign_words_to_boxes(
    words: Sequence[Word],
    boxes: Sequence[FigureBox],
) -> tuple[list[Word], dict[int, list[Word]]]:
    body_words: list[Word] = []
    box_words: dict[int, list[Word]] = {i: [] for i in range(len(boxes))}
    indexed = [(_area(box.bbox), idx, box) for idx, box in enumerate(boxes)]

    for word in words:
        cx = (word.bbox[0] + word.bbox[2]) * 0.5
        cy = (word.bbox[1] + word.bbox[3]) * 0.5
        best_idx: int | None = None
        best_key: tuple[float, int] | None = None
        for area, idx, box in indexed:
            bx0, by0, bx1, by1 = box.bbox
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                key = (area, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
        if best_idx is None:
            body_words.append(word)
        else:
            box_words[best_idx].append(word)

    return body_words, box_words


def _text_item(blk: TextBlock, *, level: int, parent_index: int | None) -> AssembledItem:
    return AssembledItem(
        kind="text",
        bbox=blk.bbox,
        text=blk.text,
        source=blk.source,
        confidence=blk.confidence,
        level=level,
        parent_index=parent_index,
        font_size=blk.font_size,
        bold=blk.bold,
        label=blk.label,
    )


def _assign_blocks_to_tables(
    blocks: Sequence[TextBlock],
    figures: Sequence[FigureBox],
) -> tuple[list[TextBlock], dict[int, list[TextBlock]]]:
    table_indices = [i for i, f in enumerate(figures) if f.kind == "table"]
    table_children: dict[int, list[TextBlock]] = {i: [] for i in table_indices}
    if not table_indices:
        return list(blocks), table_children

    body_blocks: list[TextBlock] = []
    for blk in blocks:
        cx = (blk.bbox[0] + blk.bbox[2]) * 0.5
        cy = (blk.bbox[1] + blk.bbox[3]) * 0.5
        best_idx: int | None = None
        best_key: tuple[float, int] | None = None
        for idx in table_indices:
            box = figures[idx].bbox
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                key = (_area(box), idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
        if best_idx is None and blk.label == "table":
            best_overlap = 0.0
            for idx in table_indices:
                overlap = _inter_area(blk.bbox, figures[idx].bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
        if best_idx is None:
            body_blocks.append(blk)
        else:
            table_children[best_idx].append(blk)

    return body_blocks, table_children


def _assemble_raster(raster_page: RasterPage) -> list[AssembledItem]:
    body_blocks, table_children = _assign_blocks_to_tables(raster_page.blocks, raster_page.figures)
    top_nodes: list[tuple[str, BBox, TextBlock | tuple[int, FigureBox]]] = [
        ("text", blk.bbox, blk) for blk in body_blocks
    ]
    for idx, fig in enumerate(raster_page.figures):
        top_nodes.append((fig.kind, fig.bbox, (idx, fig)))

    ordered = xy_cut_order(top_nodes, lambda node: node[1])
    items: list[AssembledItem] = []
    for kind, _bbox, payload in ordered:
        if kind == "text":
            blk = payload
            assert isinstance(blk, TextBlock)
            items.append(_text_item(blk, level=1, parent_index=None))
            continue

        fig_idx, fig = payload  # type: ignore[misc]
        parent_idx = len(items)
        items.append(
            AssembledItem(
                kind=fig.kind,
                bbox=fig.bbox,
                text="",
                source=fig.source,
                confidence=fig.confidence,
                level=1,
                parent_index=None,
            )
        )
        if fig.kind == "table":
            for child in xy_cut_order(table_children.get(fig_idx, ()), lambda b: b.bbox):
                items.append(_text_item(child, level=2, parent_index=parent_idx))
    return items


def assemble_page(
    profile: PageProfile,
    words: Sequence[Word],
    page_rect: BBox,
    figures: Sequence[FigureBox] = (),
    raster_page: RasterPage | None = None,
) -> list[AssembledItem]:
    if profile.kind == "raster":
        if raster_page is None:
            return []
        return _assemble_raster(raster_page)
    if profile.kind == "hybrid" and profile.suspect_ocr_layer and raster_page is not None:
        return _assemble_raster(raster_page)

    kept_figures = filter_overlapping_figures(figures)
    body_words, box_words = _assign_words_to_boxes(words, kept_figures)
    body_blocks = order_blocks(body_words, page_rect)

    top_nodes: list[tuple[str, BBox, TextBlock | tuple[int, FigureBox]]] = [
        ("text", blk.bbox, blk) for blk in body_blocks
    ]
    for idx, fig in enumerate(kept_figures):
        top_nodes.append((fig.kind, fig.bbox, (idx, fig)))

    ordered_nodes = xy_cut_order(top_nodes, lambda node: node[1])
    items: list[AssembledItem] = []

    for kind, bbox, payload in ordered_nodes:
        if kind == "text":
            blk = payload
            assert isinstance(blk, TextBlock)
            items.append(_text_item(blk, level=1, parent_index=None))
            continue

        fig_idx, fig = payload  # type: ignore[misc]
        parent_idx = len(items)
        items.append(
            AssembledItem(
                kind=fig.kind,
                bbox=fig.bbox,
                text="",
                source=fig.source,
                confidence=fig.confidence,
                level=1,
                parent_index=None,
            )
        )
        inner_words = box_words[fig_idx]
        if fig.kind == "figure":
            ordered_words = xy_cut_order(inner_words, lambda w: w.bbox)
            for w in ordered_words:
                items.append(
                    AssembledItem(
                        kind="text",
                        bbox=w.bbox,
                        text=w.text,
                        source="text_layer",
                        confidence=1.0,
                        level=2,
                        parent_index=parent_idx,
                        font_size=w.font_size,
                        bold=w.bold,
                    )
                )
        else:
            inner_blocks = order_blocks(inner_words, fig.bbox)
            for blk in inner_blocks:
                items.append(_text_item(blk, level=2, parent_index=parent_idx))

    return items


def _normalize_band_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip().lower()


def tag_running_headers_footers(
    elements: Sequence[dict[str, Any]],
    page_heights: Mapping[int, float],
) -> None:
    num_pages = len(page_heights)
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    candidate_pages: dict[tuple[str, str], set[int]] = {}

    for elem in elements:
        if elem.get("type") != "text":
            continue
        meta = elem.setdefault("metadata", {})
        layout_label = meta.get("layout_label")
        if layout_label in ("page_header", "page_footer"):
            meta["role"] = layout_label
        if num_pages < _MIN_MULTI_PAGE_DOCS or elem.get("level") != 1:
            continue
        page_no = elem.get("page")
        page_h = page_heights.get(page_no, 0.0) if isinstance(page_no, int) else 0.0
        bbox = elem.get("bbox")
        if page_h <= 0.0 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        cy = (float(bbox[1]) + float(bbox[3])) * 0.5
        if cy <= _HEADER_BAND_FRAC * page_h:
            zone = "page_header"
        elif cy >= _FOOTER_BAND_FRAC * page_h:
            zone = "page_footer"
        else:
            continue
        norm = _normalize_band_text(str(elem.get("text", "")))
        if not norm:
            continue
        key = (zone, norm)
        candidates.setdefault(key, []).append(elem)
        candidate_pages.setdefault(key, set()).add(page_no)

    for key, pages_seen in candidate_pages.items():
        if len(pages_seen) >= _MIN_REPEAT_PAGES or len(pages_seen) >= _MIN_REPEAT_FRAC * num_pages:
            role = key[0]
            for elem in candidates[key]:
                elem["metadata"].setdefault("role", role)
