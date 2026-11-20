from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any

from ._pdf_io import bbox_area, bbox_inter_area, bbox_iou
from .models import BBox, FigureBox, PageProfile, RasterPage, TextBlock, Word
from .text import order_blocks, xy_cut_order

_HEADER_BAND_FRAC = 0.10
_FOOTER_BAND_FRAC = 0.90
_MIN_MULTI_PAGE_DOCS = 3
_MIN_REPEAT_PAGES = 3
_MIN_REPEAT_FRAC = 0.5
_SUSPECT_OCR_MAX_CONF = 0.85
_GROUPER_SOURCES = frozenset({"v1_group", "v2_vector_vision"})
_ROMAN_NUMERAL_RE = re.compile(
    r"(?<![A-Za-z])(?:(?=[IVXLCD]+(?![A-Za-z]))D?C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))(?![A-Za-z])",
    re.IGNORECASE,
)
_ROMAN_LABEL_PREFIX_RE = re.compile(
    r"(?:appendix|section|table|figure|annex|part|chapter|exhibit)\s+$",
    re.IGNORECASE,
)


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


def _clip_figure_against_table(fig_bbox: BBox, tbl_bbox: BBox) -> BBox | None:
    fx0, fy0, fx1, fy1 = fig_bbox
    tx0, ty0, tx1, ty1 = tbl_bbox
    if tx0 >= fx1 or tx1 <= fx0 or ty0 >= fy1 or ty1 <= fy0:
        return fig_bbox
    cands: list[BBox] = []
    if tx0 > fx0:
        cands.append((fx0, fy0, tx0, fy1))
    if tx1 < fx1:
        cands.append((tx1, fy0, fx1, fy1))
    if ty0 > fy0:
        cands.append((fx0, fy0, fx1, ty0))
    if ty1 < fy1:
        cands.append((fx0, ty1, fx1, fy1))
    best = max(cands, key=bbox_area, default=None)
    return best if best is not None and bbox_area(best) > 0.0 else None


def filter_overlapping_figures(figures: Sequence[FigureBox]) -> list[FigureBox]:
    fig_indices = [i for i, f in enumerate(figures) if f.kind == "figure"]
    if not fig_indices:
        return list(figures)
    updated: list[FigureBox | None] = list(figures)
    for ti, box in enumerate(figures):
        if box.kind != "table":
            continue
        t_area = bbox_area(box.bbox)
        if t_area <= 0.0:
            continue
        overlapping_figs = [
            (fi, bbox_inter_area(box.bbox, updated[fi].bbox) / t_area)  # type: ignore[union-attr]
            for fi in fig_indices
            if updated[fi] is not None and bbox_inter_area(box.bbox, updated[fi].bbox) / t_area >= 0.50  # type: ignore[union-attr]
        ]
        if not overlapping_figs:
            continue
        drop_table = False
        pending_clips: list[tuple[int, BBox]] = []
        for fi, frac in overlapping_figs:
            fig_box = updated[fi]
            assert fig_box is not None
            if frac >= 0.85 or fig_box.source not in _GROUPER_SOURCES:
                drop_table = True
                break
            clipped = _clip_figure_against_table(fig_box.bbox, box.bbox)
            fig_area = bbox_area(fig_box.bbox)
            if clipped is None or (fig_area > 0.0 and bbox_area(clipped) < 0.15 * fig_area):
                drop_table = True
                break
            pending_clips.append((fi, clipped))
        if drop_table:
            updated[ti] = None
        else:
            for fi, clipped in pending_clips:
                fig_box = updated[fi]
                assert fig_box is not None
                updated[fi] = replace(fig_box, bbox=clipped)
    return [b for b in updated if b is not None]


def _assign_words_to_boxes(
    words: Sequence[Word],
    boxes: Sequence[FigureBox],
) -> tuple[list[Word], dict[int, list[Word]]]:
    body_words: list[Word] = []
    box_words: dict[int, list[Word]] = {i: [] for i in range(len(boxes))}
    indexed = [(bbox_area(box.bbox), idx, box) for idx, box in enumerate(boxes)]

    for word in words:
        w_area = bbox_area(word.bbox)
        best_idx: int | None = None
        best_key: tuple[float, float, int] | None = None
        if w_area <= 0.0:
            cx = (word.bbox[0] + word.bbox[2]) * 0.5
            cy = (word.bbox[1] + word.bbox[3]) * 0.5
            for area, idx, box in indexed:
                bx0, by0, bx1, by1 = box.bbox
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    key = (area, -1.0, idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = idx
        else:
            for area, idx, box in indexed:
                ioa = bbox_inter_area(word.bbox, box.bbox) / w_area
                if ioa >= 0.50:
                    key = (area, -ioa, idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = idx
        if best_idx is None:
            body_words.append(word)
        else:
            box_words[best_idx].append(word)

    return body_words, box_words


def _norm_text_sim(a: str, b: str) -> float:
    na = " ".join(a.lower().split())
    nb = " ".join(b.lower().split())
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def merge_hybrid_region_ocr(
    native_words: Sequence[Word],
    ocr_blocks: Sequence[TextBlock],
) -> tuple[list[Word], list[TextBlock]]:
    kept_words = list(native_words)
    if not native_words or not ocr_blocks:
        return kept_words, list(ocr_blocks)

    native_blocks = order_blocks(native_words)
    kept_ocr: list[TextBlock] = []
    for blk in ocr_blocks:
        is_dup = any(
            bbox_iou(blk.bbox, nb.bbox) >= 0.50 and _norm_text_sim(blk.text, nb.text) >= 0.60
            for nb in native_blocks
        )
        if not is_dup:
            overlapping: list[Word] = []
            for w in native_words:
                wa = bbox_area(w.bbox)
                if wa > 0.0 and bbox_inter_area(w.bbox, blk.bbox) / wa >= 0.50:
                    overlapping.append(w)
            if overlapping:
                u_box: BBox = (
                    min(w.bbox[0] for w in overlapping),
                    min(w.bbox[1] for w in overlapping),
                    max(w.bbox[2] for w in overlapping),
                    max(w.bbox[3] for w in overlapping),
                )
                overlapping.sort(key=lambda w: (round(w.bbox[1], 1), w.bbox[0]))
                joined = " ".join(w.text for w in overlapping)
                if bbox_iou(blk.bbox, u_box) >= 0.50 and _norm_text_sim(blk.text, joined) >= 0.60:
                    is_dup = True
        if not is_dup:
            kept_ocr.append(blk)
    return kept_words, kept_ocr


def _text_item(
    blk: TextBlock,
    *,
    level: int,
    parent_index: int | None,
    max_conf: float | None = None,
) -> AssembledItem:
    conf = blk.confidence if max_conf is None else round(min(float(blk.confidence), max_conf), 4)
    return AssembledItem(
        kind="text",
        bbox=blk.bbox,
        text=blk.text,
        source=blk.source,
        confidence=conf,
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
                key = (bbox_area(box), idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
        if best_idx is None and blk.label == "table":
            best_overlap = 0.0
            for idx in table_indices:
                overlap = bbox_inter_area(blk.bbox, figures[idx].bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
        if best_idx is None:
            body_blocks.append(blk)
        else:
            table_children[best_idx].append(blk)

    return body_blocks, table_children


def _assemble_raster(raster_page: RasterPage, *, max_text_conf: float | None = None) -> list[AssembledItem]:
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
            items.append(_text_item(blk, level=1, parent_index=None, max_conf=max_text_conf))
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
                items.append(_text_item(child, level=2, parent_index=parent_idx, max_conf=max_text_conf))
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
        return _assemble_raster(raster_page, max_text_conf=_SUSPECT_OCR_MAX_CONF)

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


def _mask_roman_numeral(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        tok = match.group(0)
        if len(tok) == 1:
            if tok.upper() in ("C", "D", "L"):
                return tok
            if tok.isupper() and _ROMAN_LABEL_PREFIX_RE.search(text[: match.start()]):
                return tok
        return "#"

    return _ROMAN_NUMERAL_RE.sub(_repl, text)


def _normalize_band_text(text: str) -> str:
    masked = _mask_roman_numeral(re.sub(r"\d+", "#", text))
    return re.sub(r"\s+", " ", masked).strip().lower()


def _norm_toc_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", " ".join(text.split()).lower()).strip()


def tag_running_headers_footers(
    elements: Sequence[dict[str, Any]],
    page_heights: Mapping[int, float],
    toc_titles: Collection[str] = (),
) -> None:
    num_pages = len(page_heights)
    protected_titles = {t for raw in toc_titles if (t := _norm_toc_title(raw))}
    even_pages = {p for p in page_heights if p % 2 == 0}
    odd_pages = {p for p in page_heights if p % 2 == 1}
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    candidate_pages: dict[tuple[str, str], set[int]] = {}

    for elem in elements:
        if elem.get("type") != "text":
            continue
        raw_text = str(elem.get("text", ""))
        if protected_titles and _norm_toc_title(raw_text) in protected_titles:
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
        norm = _normalize_band_text(raw_text)
        if not norm:
            continue
        key = (zone, norm)
        candidates.setdefault(key, []).append(elem)
        candidate_pages.setdefault(key, set()).add(page_no)

    for key, pages_seen in candidate_pages.items():
        role = key[0]
        if len(pages_seen) >= _MIN_REPEAT_PAGES or len(pages_seen) >= _MIN_REPEAT_FRAC * num_pages:
            for elem in candidates[key]:
                elem["metadata"].setdefault("role", role)
            continue
        if num_pages >= 4:
            even_seen = pages_seen & even_pages
            odd_seen = pages_seen & odd_pages
            even_match = len(even_pages) >= 2 and (
                len(even_seen) >= _MIN_REPEAT_PAGES or len(even_seen) >= 0.6 * len(even_pages)
            )
            odd_match = len(odd_pages) >= 2 and (
                len(odd_seen) >= _MIN_REPEAT_PAGES or len(odd_seen) >= 0.6 * len(odd_pages)
            )
            if even_match or odd_match:
                for elem in candidates[key]:
                    pno = elem.get("page")
                    if isinstance(pno, int) and (
                        (even_match and pno in even_pages) or (odd_match and pno in odd_pages)
                    ):
                        elem["metadata"].setdefault("role", role)
