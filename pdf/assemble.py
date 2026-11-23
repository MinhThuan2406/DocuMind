from __future__ import annotations

import re
import unicodedata
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
    heading_clean_text: str | None = None


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


def _strip_trailing_badges(
    blk: TextBlock,
    vector_boxes: Sequence[BBox],
) -> str | None:
    if blk.source != "text_layer" or "\n" in blk.text or len(blk.words) < 2:
        return None
    words = sorted(blk.words, key=lambda w: (w.bbox[0], w.bbox[1]))
    main_fs = max((w.font_size for w in words if w.font_size > 0.0), default=0.0)
    if main_fs <= 0.0:
        return None
    max_badge_fs = 0.84 * main_fs
    small_boxes = [
        b
        for b in vector_boxes
        if (b[3] - b[1]) <= 2.2 * main_fs and (b[2] - b[0]) <= 14.0 * main_fs
    ]

    def _word_in_badge_box(w: Word, left_x1: float) -> bool:
        cx = 0.5 * (w.bbox[0] + w.bbox[2])
        cy = 0.5 * (w.bbox[1] + w.bbox[3])
        for bx0, by0, bx1, by1 in small_boxes:
            if bx0 - 1.5 <= cx <= bx1 + 1.5 and by0 - 1.5 <= cy <= by1 + 1.5 and bx0 >= left_x1 - 0.5 * main_fs:
                return True
        return False

    best_split: int | None = None
    for split_idx in range(len(words) - 1, 0, -1):
        suffix = words[split_idx:]
        if not all(0.0 < w.font_size <= max_badge_fs for w in suffix):
            break
        left_words = words[:split_idx]
        left_x1 = max(w.bbox[2] for w in left_words)
        left_fs = max((w.font_size for w in left_words if w.font_size > 0.0), default=0.0)
        if left_fs <= 0.0 or max(w.font_size for w in suffix) > 0.84 * left_fs:
            continue
        all_boxed = bool(small_boxes) and all(_word_in_badge_box(w, left_x1) for w in suffix)
        gap_ok = (suffix[0].bbox[0] - left_x1) >= 0.8 * left_fs
        if not all_boxed and not gap_ok:
            continue
        left_text = " ".join(w.text for w in left_words).strip()
        badge_text = " ".join(w.text for w in suffix).strip()
        if not left_text or not badge_text:
            continue
        if all_boxed or len(left_text) > len(badge_text):
            best_split = split_idx

    if best_split is None:
        return None
    return " ".join(w.text for w in words[:best_split]).strip()


def _text_item(
    blk: TextBlock,
    *,
    level: int,
    parent_index: int | None,
    max_conf: float | None = None,
    vector_boxes: Sequence[BBox] = (),
) -> AssembledItem:
    conf = blk.confidence if max_conf is None else round(min(float(blk.confidence), max_conf), 4)
    clean_heading = _strip_trailing_badges(blk, vector_boxes) if level == 1 else None
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
        heading_clean_text=clean_heading,
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
    *,
    horiz_rules: Sequence[BBox] = (),
    vector_boxes: Sequence[BBox] = (),
) -> list[AssembledItem]:
    if profile.kind == "raster":
        if raster_page is None:
            return []
        return _assemble_raster(raster_page)
    if profile.kind == "hybrid" and profile.suspect_ocr_layer and raster_page is not None:
        return _assemble_raster(raster_page, max_text_conf=_SUSPECT_OCR_MAX_CONF)

    kept_figures = filter_overlapping_figures(figures)
    body_words, box_words = _assign_words_to_boxes(words, kept_figures)
    body_blocks = order_blocks(body_words, page_rect, horiz_rules=horiz_rules)

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
            items.append(_text_item(blk, level=1, parent_index=None, vector_boxes=vector_boxes))
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
    normed = unicodedata.normalize("NFKC", text)
    masked = _mask_roman_numeral(re.sub(r"\d+", "#", normed))
    return re.sub(r"\s+", " ", masked).strip().casefold()


def _norm_toc_title(text: str) -> str:
    normed = unicodedata.normalize("NFKC", text).casefold()
    chars = [ch if unicodedata.category(ch)[0] in ("L", "N", "M") else " " for ch in normed]
    return " ".join("".join(chars).split())


def _is_matchable_toc_title(norm: str) -> bool:
    compact = norm.replace(" ", "")
    return len(compact) >= 2 and any(ch.isalpha() for ch in compact)


def _table_plain_text(tbl: Mapping[str, Any]) -> str:
    rows: dict[int, list[tuple[int, int, str]]] = {}
    for idx, cell in enumerate(tbl.get("cells") or ()):
        if not isinstance(cell, Mapping):
            continue
        txt = str(cell.get("text") or "").strip()
        if not txt:
            continue
        rows.setdefault(int(cell.get("row", 0)), []).append((int(cell.get("col", 0)), idx, txt))
    lines: list[str] = []
    for r in sorted(rows):
        row_cells = sorted(rows[r], key=lambda item: (item[0], item[1]))
        lines.append(" ".join(txt for _, _, txt in row_cells))
    return "\n".join(lines)


def extract_plain_text(
    ir: Mapping[str, Any],
    *,
    include_furniture: bool = False,
) -> dict[int, str]:
    tables_by_id: dict[str, Mapping[str, Any]] = {}
    for tbl in ir.get("tables") or ():
        if isinstance(tbl, Mapping) and tbl.get("id") is not None:
            tables_by_id[str(tbl["id"])] = tbl

    page_parts: dict[int, list[str]] = {}
    for pmeta in (ir.get("metadata") or {}).get("pages") or ():
        if isinstance(pmeta, Mapping) and "number" in pmeta:
            page_parts.setdefault(int(pmeta["number"]), [])

    level1_by_page: dict[int, list[tuple[int, int, Mapping[str, Any]]]] = {}
    children_by_parent: dict[str, list[tuple[int, int, Mapping[str, Any]]]] = {}
    for idx, elem in enumerate(ir.get("elements") or ()):
        if not isinstance(elem, Mapping):
            continue
        lvl = int(elem.get("level", 1))
        ro = int(elem.get("reading_order", idx))
        if lvl == 1:
            pno = int(elem.get("page", 0))
            page_parts.setdefault(pno, [])
            level1_by_page.setdefault(pno, []).append((ro, idx, elem))
        elif lvl == 2 and elem.get("parent_id") is not None:
            children_by_parent.setdefault(str(elem["parent_id"]), []).append((ro, idx, elem))

    for pno, entries in level1_by_page.items():
        entries.sort(key=lambda item: (item[0], item[1]))
        parts = page_parts[pno]
        for _, _, elem in entries:
            meta = elem.get("metadata") or {}
            if not include_furniture and (
                meta.get("role") in ("page_header", "page_footer")
                or meta.get("layout_label") in ("page_header", "page_footer")
            ):
                continue
            etype = elem.get("type")
            if etype == "text":
                txt = str(elem.get("text") or "")
                if txt.strip():
                    parts.append(txt)
            elif etype == "table":
                tid = meta.get("table_id") or elem.get("table_id")
                tbl = tables_by_id.get(str(tid)) if tid is not None else None
                tbl_txt = _table_plain_text(tbl) if tbl is not None else ""
                if tbl_txt:
                    parts.append(tbl_txt)
                    continue
                children = children_by_parent.get(str(elem.get("id") or ""), ())
                if children:
                    child_lines = [
                        str(c.get("text") or "")
                        for _, _, c in sorted(children, key=lambda item: (item[0], item[1]))
                        if str(c.get("text") or "").strip()
                    ]
                    if child_lines:
                        parts.append("\n".join(child_lines))

    return {pno: "\n".join(parts) for pno, parts in sorted(page_parts.items())}


def _local_body_font_size(elements: Sequence[Mapping[str, Any]]) -> float:
    weighted: list[tuple[float, int]] = []
    for elem in elements:
        if elem.get("type") != "text" or elem.get("level", 1) != 1:
            continue
        meta = elem.get("metadata") or {}
        fs = float(meta.get("font_size") or 0.0)
        if fs <= 0.0:
            continue
        weight = max(1, len(str(elem.get("text") or "").strip()))
        weighted.append((fs, weight))
    if not weighted:
        return 0.0
    weighted.sort(key=lambda item: item[0])
    half = sum(w for _, w in weighted) * 0.5
    acc = 0.0
    for fs, w in weighted:
        acc += w
        if acc >= half:
            return fs
    return weighted[-1][0]


def tag_running_headers_footers(
    elements: Sequence[dict[str, Any]],
    page_heights: Mapping[int, float],
    toc_titles: Collection[str] = (),
    horiz_rules_by_page: Mapping[int, Sequence[BBox]] | None = None,
) -> None:
    num_pages = len(page_heights)
    protected_titles = {
        t for raw in toc_titles if (t := _norm_toc_title(raw)) and _is_matchable_toc_title(t)
    }
    body_font = _local_body_font_size(elements)
    even_pages = {p for p in page_heights if p % 2 == 0}
    odd_pages = {p for p in page_heights if p % 2 == 1}

    by_page_lvl1: dict[int, list[tuple[BBox, dict[str, Any]]]] = {}
    heading_or_body_titles: set[str] = set()
    max_fs_for_title: dict[str, float] = {}
    title_font_pages: set[int] = set()

    for elem in elements:
        if elem.get("type") != "text":
            continue
        meta = elem.setdefault("metadata", {})
        raw_text = str(elem.get("text", ""))
        fs = float(meta.get("font_size") or 0.0)
        pno = elem.get("page")
        bbox = elem.get("bbox")
        has_box = isinstance(pno, int) and isinstance(bbox, (list, tuple)) and len(bbox) == 4
        page_h = page_heights.get(pno, 0.0) if isinstance(pno, int) else 0.0

        if elem.get("level") == 1 and has_box:
            box_t: BBox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            by_page_lvl1.setdefault(pno, []).append((box_t, elem))
            if body_font > 0.0 and fs >= 1.15 * body_font:
                title_font_pages.add(pno)
            tnorm = _norm_toc_title(meta.get("heading_clean_text") or raw_text)
            if tnorm:
                if page_h > 0.0 and box_t[1] > _HEADER_BAND_FRAC * page_h:
                    heading_or_body_titles.add(tnorm)
                if fs > max_fs_for_title.get(tnorm, 0.0):
                    max_fs_for_title[tnorm] = fs

        layout_label = meta.get("layout_label")
        if layout_label in ("page_header", "page_footer"):
            in_margin = False
            if has_box and page_h > 0.0:
                cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
                in_margin = cy <= _HEADER_BAND_FRAC * page_h or cy >= _FOOTER_BAND_FRAC * page_h
            is_protected = bool(protected_titles and _norm_toc_title(raw_text) in protected_titles)
            if not is_protected or (in_margin and (body_font <= 0.0 or fs <= 0.0 or fs <= body_font * 1.02)):
                meta["role"] = layout_label

    if num_pages < _MIN_MULTI_PAGE_DOCS:
        return

    # Identify the top-band baseline y0 across pages for strict positional running-header verification.
    top_cand_y0: dict[int, float] = {}
    top_separated: dict[int, bool] = {}
    for pno, items in by_page_lvl1.items():
        page_h = page_heights.get(pno, 0.0)
        if page_h <= 0.0:
            continue
        sorted_items = sorted(items, key=lambda it: (it[0][1], it[0][0]))
        first_box, first_elem = sorted_items[0]
        first_cy = 0.5 * (first_box[1] + first_box[3])
        if first_cy > 0.055 * page_h:
            continue
        first_fs = float((first_elem.get("metadata") or {}).get("font_size") or 0.0)
        em = first_fs if first_fs > 0.0 else (body_font if body_font > 0.0 else 10.0)
        top_cand_y0[pno] = first_box[1]
        next_boxes = [b for b, _ in sorted_items[1:] if b[1] > first_box[3] - 0.2 * em]
        if not next_boxes:
            top_separated[pno] = True
        else:
            next_y0 = min(b[1] for b in next_boxes)
            gap_ok = (next_y0 - first_box[3]) >= 1.5 * em
            rule_ok = False
            if not gap_ok and horiz_rules_by_page:
                for rx0, ry0, rx1, ry1 in horiz_rules_by_page.get(pno, ()):
                    if first_box[3] - 2.0 <= 0.5 * (ry0 + ry1) <= next_y0 + 2.0 and (rx1 - rx0) >= 0.25 * (first_box[2] - first_box[0]):
                        rule_ok = True
                        break
            top_separated[pno] = gap_ok or rule_ok

    med_y0_hdr: float | None = None
    distinct_top_band = False
    if top_cand_y0:
        sorted_y0s = sorted(top_cand_y0.values())
        med_y0_hdr = sorted_y0s[len(sorted_y0s) // 2]
        band_pages: set[int] = set()
        for pno, y0 in top_cand_y0.items():
            if not top_separated.get(pno, False):
                continue
            p_em = body_font if body_font > 0.0 else 10.0
            if abs(y0 - med_y0_hdr) <= 0.35 * p_em:
                band_pages.add(pno)
        even_band = len(band_pages & even_pages)
        odd_band = len(band_pages & odd_pages)
        distinct_top_band = (
            (len(band_pages) >= _MIN_REPEAT_PAGES and len(band_pages) >= _MIN_REPEAT_FRAC * num_pages)
            or (
                num_pages >= 4
                and (
                    (len(even_pages) >= 2 and (even_band >= _MIN_REPEAT_PAGES or even_band >= 0.6 * len(even_pages)))
                    or (len(odd_pages) >= 2 and (odd_band >= _MIN_REPEAT_PAGES or odd_band >= 0.6 * len(odd_pages)))
                )
            )
        )

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    candidate_pages: dict[tuple[str, str], set[int]] = {}

    for elem in elements:
        if elem.get("type") != "text" or elem.get("level") != 1:
            continue
        pno = elem.get("page")
        page_h = page_heights.get(pno, 0.0) if isinstance(pno, int) else 0.0
        bbox = elem.get("bbox")
        if page_h <= 0.0 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        raw_text = str(elem.get("text", ""))
        meta = elem.get("metadata") or {}
        fs = float(meta.get("font_size") or 0.0)
        y0, y1 = float(bbox[1]), float(bbox[3])
        cy = (y0 + y1) * 0.5
        is_protected = bool(protected_titles and _norm_toc_title(raw_text) in protected_titles)

        if cy <= _HEADER_BAND_FRAC * page_h:
            zone = "page_header"
            em = fs if fs > 0.0 else (body_font if body_font > 0.0 else 10.0)
            # Positional running headers must sit on the top-band baseline with single-line short text.
            if cy > 0.055 * page_h:
                continue
            if med_y0_hdr is None or abs(y0 - med_y0_hdr) > 0.35 * em:
                continue
            if len(raw_text.splitlines()) != 1 or len(" ".join(raw_text.split())) > 100:
                continue
            if body_font > 0.0 and fs > 0.0:
                if not distinct_top_band:
                    continue
                if fs >= 1.15 * body_font and len(title_font_pages) <= 1:
                    continue
                tnorm = _norm_toc_title(meta.get("heading_clean_text") or raw_text)
                matches_heading = bool(
                    tnorm
                    and (
                        tnorm in heading_or_body_titles
                        or tnorm in protected_titles
                        or max_fs_for_title.get(tnorm, 0.0) > fs * 1.05
                    )
                )
                if not (fs < 0.95 * body_font or matches_heading):
                    continue
            elif is_protected:
                continue
        elif cy >= _FOOTER_BAND_FRAC * page_h:
            if is_protected:
                continue
            zone = "page_footer"
        else:
            continue

        norm = _normalize_band_text(raw_text)
        if not norm:
            continue
        key = (zone, norm)
        candidates.setdefault(key, []).append(elem)
        candidate_pages.setdefault(key, set()).add(pno)

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
