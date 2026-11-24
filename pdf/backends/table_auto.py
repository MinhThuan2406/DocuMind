from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from PIL import Image
import numpy as np

from .._pdf_io import bbox_iou
from ..models import BBox, ExtractedTable, Word
from ..tables import DEFAULT_TABLE_DPI, crop_page_region, ocr_crop_tokens, tokens_in_box
from . import get as get_backend
from .base import RunStats
from .rss import peak_rss_mb
from .tatr_onnx import (
    DEFAULT_THRESHOLD,
    _HEADER_RESCUE_MIN,
    _band_ink_runs,
    _cluster_1d_tokens,
    _crop_to_gray,
    _has_valid_grid,
    _overlap_indices,
    objects_to_table,
)

_ROW_OR_HEADER_LABELS = frozenset({"table row", "table column header", "table projected row header"})
_MIN_TOKEN_COVERAGE = 0.95
_TIER2_MAX_CROPS = 8
_TIER2_PAD_PX = 16


def _split_unspaced_multicell_ocr(
    ocr_rec: Any,
    crop: Image.Image,
    crop_box: BBox,
    tokens: Sequence[tuple[str, BBox]],
    table: ExtractedTable,
) -> list[tuple[str, BBox]] | None:
    if not tokens or table.num_cols < 2 or not table.cells:
        return None
    cw, ch = crop.size
    if cw <= 0 or ch <= 0:
        return None
    sx = (crop_box[2] - crop_box[0]) / cw
    sy = (crop_box[3] - crop_box[1]) / ch
    if sx <= 0.0 or sy <= 0.0:
        return None

    crop_gray = None
    updated_tokens: list[tuple[str, BBox]] = []
    changed = False
    crops_used = 0

    for text, tb in tokens:
        stripped = text.strip()
        if not stripped or any(ch_t.isspace() for ch_t in stripped) or crops_used >= _TIER2_MAX_CROPS:
            updated_tokens.append((text, tb))
            continue
        tcy = (tb[1] + tb[3]) * 0.5
        row_cells = sorted(
            (
                c
                for c in table.cells
                if c.bbox is not None
                and c.bbox[1] <= tcy <= c.bbox[3]
                and min(tb[2], c.bbox[2]) - max(tb[0], c.bbox[0]) >= min(0.12 * max(1.0, c.bbox[2] - c.bbox[0]), 4.0)
            ),
            key=lambda c: c.col,
        )
        if len(row_cells) < 2:
            updated_tokens.append((text, tb))
            continue

        if crop_gray is None:
            crop_gray = _crop_to_gray(crop)
        px0 = max(0, min(cw, int(math.floor((tb[0] - crop_box[0]) / sx))))
        py0 = max(0, min(ch, int(math.floor((tb[1] - crop_box[1]) / sy))))
        px1 = max(0, min(cw, int(math.ceil((tb[2] - crop_box[0]) / sx))))
        py1 = max(0, min(ch, int(math.ceil((tb[3] - crop_box[1]) / sy))))
        runs, line_h = _band_ink_runs(crop_gray, px0, py0, px1, py1)
        max_gap = max((runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)), default=0)
        if len(runs) < 2 or max_gap < 1.5 * line_h:
            updated_tokens.append((text, tb))
            continue

        sub_regions: list[tuple[int, int]] = []
        for c in row_cells:
            assert c.bbox is not None
            cx0 = max(px0, int(math.floor((c.bbox[0] - crop_box[0]) / sx)))
            cx1 = min(px1, int(math.ceil((c.bbox[2] - crop_box[0]) / sx)))
            if cx1 <= cx0:
                continue
            c_runs = [(max(cx0, r0), min(cx1, r1)) for r0, r1 in runs if min(r1, cx1) - max(r0, cx0) >= 2]
            if c_runs:
                sub_regions.append((min(r0 for r0, _ in c_runs), max(r1 for _, r1 in c_runs)))

        if len(sub_regions) < 2 or crops_used + len(sub_regions) > _TIER2_MAX_CROPS:
            updated_tokens.append((text, tb))
            continue

        sub_tokens: list[tuple[str, BBox]] = []
        all_ok = True
        for sub_x0, sub_x1 in sub_regions:
            sub_w = sub_x1 - sub_x0
            sub_h = py1 - py0
            if sub_w <= 0 or sub_h <= 0:
                all_ok = False
                break
            padded = Image.new("RGB", (sub_w + 2 * _TIER2_PAD_PX, sub_h + 2 * _TIER2_PAD_PX), (255, 255, 255))
            padded.paste(crop.crop((sub_x0, py0, sub_x1, py1)).convert("RGB"), (_TIER2_PAD_PX, _TIER2_PAD_PX))
            crops_used += 1
            ocr_lines = [ln for ln in ocr_rec.recognize(padded) if ln.text.strip()]
            if not ocr_lines:
                all_ok = False
                break
            for ln in ocr_lines:
                lx0 = sub_x0 + max(0.0, min(float(sub_w), ln.bbox[0] - _TIER2_PAD_PX))
                ly0 = py0 + max(0.0, min(float(sub_h), ln.bbox[1] - _TIER2_PAD_PX))
                lx1 = sub_x0 + max(0.0, min(float(sub_w), ln.bbox[2] - _TIER2_PAD_PX))
                ly1 = py0 + max(0.0, min(float(sub_h), ln.bbox[3] - _TIER2_PAD_PX))
                if lx1 <= lx0 or ly1 <= ly0:
                    lx0, ly0, lx1, ly1 = float(sub_x0), float(py0), float(sub_x1), float(py1)
                sub_tokens.append(
                    (
                        ln.text.strip(),
                        (
                            crop_box[0] + lx0 * sx,
                            crop_box[1] + ly0 * sy,
                            crop_box[0] + lx1 * sx,
                            crop_box[1] + ly1 * sy,
                        ),
                    )
                )
        if all_ok and sub_tokens:
            updated_tokens.extend(sub_tokens)
            changed = True
        else:
            updated_tokens.append((text, tb))

    return updated_tokens if changed else None


def _matches_overlapping_cells(norm_tok: str, cells: Sequence[str]) -> bool:
    if any(norm_tok in ctext for ctext in cells):
        return True
    if len(cells) < 2:
        return False
    matched_pos = {0}
    tok_len = len(norm_tok)
    for ctext in cells:
        next_pos = set(matched_pos)
        for pos in matched_pos:
            for end in range(pos + 1, tok_len + 1):
                if norm_tok[pos:end] in ctext:
                    next_pos.add(end)
                else:
                    break
        if tok_len in next_pos:
            return True
        matched_pos = next_pos
    return False


def _token_coverage(table: ExtractedTable, tokens: Sequence[tuple[str, BBox]]) -> float:
    non_empty = [(txt.strip(), tb) for txt, tb in tokens if txt.strip()]
    if not non_empty:
        return 1.0
    if not table.cells:
        return 0.0
    sorted_cells = sorted(table.cells, key=lambda c: (c.row, c.col))
    norm_cells = [(c.bbox, "".join(c.text.split()).lower()) for c in sorted_cells]
    has_bboxes = any(cb is not None for cb, _ in norm_cells)
    all_text = "".join(t for _, t in norm_cells)
    covered = 0
    for txt, tb in non_empty:
        norm_tok = "".join(txt.split()).lower()
        if not norm_tok:
            covered += 1
            continue
        if not has_bboxes:
            if norm_tok in all_text:
                covered += 1
            continue
        tcx = (tb[0] + tb[2]) * 0.5
        tcy = (tb[1] + tb[3]) * 0.5
        th = max(1.0, tb[3] - tb[1])
        if any(
            cb is not None
            and cb[0] - 1.5 <= tcx <= cb[2] + 1.5
            and cb[1] - 1.5 <= tcy <= cb[3] + 1.5
            and bool(ctext)
            and norm_tok in ctext
            for cb, ctext in norm_cells
        ):
            covered += 1
            continue
        overlapping_texts = [
            ctext
            for cb, ctext in norm_cells
            if cb is not None
            and ctext
            and cb[0] - 1.5 <= tb[2]
            and tb[0] <= cb[2] + 1.5
            and ((cb[1] - 1.5 <= tcy <= cb[3] + 1.5) or (min(tb[3], cb[3]) - max(tb[1], cb[1]) >= 0.3 * th))
        ]
        if overlapping_texts and _matches_overlapping_cells(norm_tok, overlapping_texts):
            covered += 1
    return covered / len(non_empty)


def _row_y_spans_pt(table: ExtractedTable) -> list[tuple[float, float]]:
    n = table.num_rows
    if n <= 0:
        return []
    edge_samples: list[list[float]] = [[] for _ in range(n + 1)]
    for cell in table.cells:
        if cell.bbox is None:
            continue
        if 0 <= cell.row <= n:
            edge_samples[cell.row].append(float(cell.bbox[1]))
        end_r = min(n, cell.row + max(1, cell.rowspan))
        if 0 <= end_r <= n:
            edge_samples[end_r].append(float(cell.bbox[3]))

    bounds: list[float | None] = [sum(vs) / len(vs) if vs else None for vs in edge_samples]
    if bounds[0] is None:
        bounds[0] = float(table.bbox[1])
    if bounds[n] is None:
        bounds[n] = float(table.bbox[3])

    # Linearly interpolate missing horizontal grid edges from staggered rowspan cells.
    i = 0
    while i < n:
        j = i + 1
        while j <= n and bounds[j] is None:
            j += 1
        lo_y = bounds[i]
        hi_y = bounds[j]
        assert lo_y is not None and hi_y is not None
        span = j - i
        for k in range(i + 1, j):
            bounds[k] = lo_y + (hi_y - lo_y) * (k - i) / span
        i = j

    resolved = [float(b) for b in bounds if b is not None]
    return [(resolved[r], max(resolved[r], resolved[r + 1])) for r in range(n)]


def _empty_track_fractions(table: ExtractedTable) -> tuple[float, float]:
    if table.num_rows <= 0 or table.num_cols <= 0:
        return 0.0, 0.0
    nonempty_rows = {
        r
        for c in table.cells
        if c.text.strip()
        for r in range(c.row, min(table.num_rows, c.row + max(1, c.rowspan)))
    }
    nonempty_cols = {
        col
        for c in table.cells
        if c.text.strip()
        for col in range(c.col, min(table.num_cols, c.col + max(1, c.colspan)))
    }
    return 1.0 - (len(nonempty_rows) / table.num_rows), 1.0 - (len(nonempty_cols) / table.num_cols)


def _is_undersegmented_booktabs(
    pl_tbl: ExtractedTable,
    tatr_tbl: ExtractedTable,
    tokens: Sequence[tuple[str, BBox]],
) -> bool:
    if tatr_tbl.num_rows <= pl_tbl.num_rows or tatr_tbl.num_rows < 2 or tatr_tbl.num_cols < 2 or not tokens:
        return False
    pl_row_spans = _row_y_spans_pt(pl_tbl)
    tatr_row_spans = _row_y_spans_pt(tatr_tbl)
    for r in range(1, pl_tbl.num_rows):
        py0, py1 = pl_row_spans[r]
        if py1 <= py0:
            continue
        matched_tatr = _overlap_indices(tatr_row_spans, py0, py1, min_frac=0.5)
        if len(matched_tatr) < 2:
            continue
        row_cells = sorted(
            (c for c in pl_tbl.cells if c.row <= r < c.row + max(1, c.rowspan) and c.bbox is not None),
            key=lambda c: c.col,
        )
        col_x_spans = [(c.bbox[0], c.bbox[2]) for c in row_cells if c.bbox is not None and c.bbox[2] > c.bbox[0]]
        multi_cluster_cols = 0
        for cx0, cx1 in col_x_spans:
            col_toks = [
                tb
                for _, tb in tokens
                if cx0 <= (tb[0] + tb[2]) * 0.5 <= cx1 and py0 <= (tb[1] + tb[3]) * 0.5 <= py1
            ]
            if len(col_toks) < 2:
                continue
            y_cls = _cluster_1d_tokens(
                [(tb[0], tb[1] - py0, tb[2], tb[3] - py0) for tb in col_toks],
                axis=1,
                limit=py1 - py0,
            )
            if len(y_cls) < 2:
                continue
            matched_for_col = {
                tr
                for y0_rel, y1_rel, _ in y_cls
                for tr in matched_tatr
                if min(py0 + y1_rel, tatr_row_spans[tr][1]) - max(py0 + y0_rel, tatr_row_spans[tr][0]) > 0.0
            }
            if len(matched_for_col) >= 2:
                multi_cluster_cols += 1
        if multi_cluster_cols >= 2:
            return True
    return False


def _is_row0_styled(
    table: ExtractedTable,
    page_words: Sequence[Word],
    crop: Image.Image | None,
    crop_box_pt: BBox,
) -> bool:
    if table.num_rows < 2:
        return False
    row_spans = _row_y_spans_pt(table)
    r0_y0, r0_y1 = row_spans[0]
    tx0, _, tx1, ty1 = table.bbox
    if page_words:
        tbl_words = [
            w
            for w in page_words
            if tx0 - 1.0 <= (w.bbox[0] + w.bbox[2]) * 0.5 <= tx1 + 1.0
            and r0_y0 - 1.0 <= (w.bbox[1] + w.bbox[3]) * 0.5 <= ty1 + 1.0
        ]
        r0_words = [w for w in tbl_words if (w.bbox[1] + w.bbox[3]) * 0.5 <= r0_y1]
        body_words = [w for w in tbl_words if (w.bbox[1] + w.bbox[3]) * 0.5 > r0_y1]
        if r0_words and body_words:
            r0_bold = sum(1 for w in r0_words if w.bold) / len(r0_words)
            body_bold = sum(1 for w in body_words if w.bold) / len(body_words)
            if r0_bold >= 0.70 and body_bold <= 0.30:
                return True

    if crop is not None and crop.width >= 4 and crop.height >= 4:
        sx = (crop_box_pt[2] - crop_box_pt[0]) / crop.width
        sy = (crop_box_pt[3] - crop_box_pt[1]) / crop.height
        if sx > 0.0 and sy > 0.0:
            arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
            ch, cw = arr.shape[:2]
            px0 = max(0, min(cw - 1, int(round((tx0 - crop_box_pt[0]) / sx)) + 2))
            px1 = max(px0 + 1, min(cw, int(round((tx1 - crop_box_pt[0]) / sx)) - 2))
            row_bgs: list[tuple[int, int, int]] = []
            for ry0, ry1 in row_spans:
                inset = 0.15 * max(0.0, ry1 - ry0)
                py0 = max(0, min(ch - 1, int(round((ry0 + inset - crop_box_pt[1]) / sy))))
                py1 = max(py0 + 1, min(ch, int(round((ry1 - inset - crop_box_pt[1]) / sy))))
                med = np.median(arr[py0:py1, px0:px1], axis=(0, 1))
                row_bgs.append((int(round(float(med[0]))), int(round(float(med[1]))), int(round(float(med[2])))))
            if len(row_bgs) >= 2 and min(row_bgs[0]) < 248:
                r0_bg = row_bgs[0]
                body_bgs = row_bgs[1:]

                def _bg_diff(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
                    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))

                # Exclude zebra-striped tables where row 0 matches row 2 while row 1 differs.
                is_zebra = (
                    len(body_bgs) >= 2
                    and _bg_diff(r0_bg, body_bgs[1]) < 10
                    and _bg_diff(r0_bg, body_bgs[0]) >= 10
                    and (len(body_bgs) == 2 or _bg_diff(body_bgs[0], body_bgs[2]) < 10)
                )
                if not is_zebra and _bg_diff(r0_bg, body_bgs[0]) >= 10:
                    diff_cnt = sum(1 for bg in body_bgs if _bg_diff(r0_bg, bg) >= 10)
                    if diff_cnt > len(body_bgs) / 2:
                        return True
    return False


def refine_table_headers(
    table: ExtractedTable,
    objects: Sequence[tuple[str, BBox, float]],
    crop_size: tuple[int, int],
    crop_box_pt: BBox,
    *,
    row0_styled: bool = False,
) -> ExtractedTable:
    cw, ch = float(crop_size[0]), float(crop_size[1])
    if (
        table.num_rows == 0
        or table.num_cols == 0
        or cw <= 0.0
        or ch <= 0.0
        or not any(lbl in _ROW_OR_HEADER_LABELS for lbl, _, _ in objects)
    ):
        return replace(table, source="auto_refined")

    row_spans = _row_y_spans_pt(table)
    sy = (crop_box_pt[3] - crop_box_pt[1]) / ch
    header_rows: set[int] = set()
    proj_header_rows: set[int] = set()
    for lbl, box, score in objects:
        if lbl == "table column header":
            ov = _overlap_indices(row_spans, crop_box_pt[1] + box[1] * sy, crop_box_pt[1] + box[3] * sy)
            if score >= DEFAULT_THRESHOLD or (row0_styled and score >= _HEADER_RESCUE_MIN and 0 in ov):
                header_rows.update(ov)
        elif lbl == "table projected row header":
            proj_header_rows.update(
                _overlap_indices(row_spans, crop_box_pt[1] + box[1] * sy, crop_box_pt[1] + box[3] * sy)
            )

    all_hdr_rows = header_rows | proj_header_rows
    new_cells = tuple(
        replace(
            c,
            is_header=(c.row in header_rows) or all(r in all_hdr_rows for r in range(c.row, c.row + c.rowspan)),
        )
        for c in table.cells
    )
    return replace(table, cells=new_cells, source="auto_refined")


class AutoTableExtractor:
    name = "auto"

    def __init__(
        self,
        *,
        device: str = "auto",
        pdfplumber_extractor: Any | None = None,
        tatr_extractor: Any | None = None,
        ocr_recognizer: Any | None = None,
    ) -> None:
        self.device_name = device
        self._pdfplumber = pdfplumber_extractor
        self._tatr = tatr_extractor
        self._ocr = ocr_recognizer
        self.last_stats: RunStats | None = None

    def _get_plumber(self) -> Any:
        return self._pdfplumber if self._pdfplumber is not None else get_backend("table", "pdfplumber")

    def _get_tatr(self, device: str | None = None) -> Any:
        if self._tatr is not None:
            return self._tatr
        eff_device = self.device_name if device is None else device
        return get_backend("table", "tatr-onnx", device=eff_device)

    def _get_ocr(self, ocr_backend: str, device: str | None = None, ocr_lang: str | None = None) -> Any:
        if self._ocr is not None:
            return self._ocr
        eff_device = self.device_name if device is None else device
        return get_backend("ocr", ocr_backend, device=eff_device, **({"lang": ocr_lang} if ocr_lang is not None else {}))

    def ensure_loaded(self) -> bool:
        tatr = self._get_tatr()
        if hasattr(tatr, "ensure_loaded"):
            return bool(tatr.ensure_loaded())
        return False

    def extract_from_page(
        self,
        page: Any,
        table_boxes_pt: Sequence[BBox],
        *,
        words: Sequence[Word] | None = None,
        is_raster: bool = False,
        ocr_backend: str = "rapidocr-onnx",
        ocr_lang: str | None = None,
        device: str | None = None,
        dpi: int = DEFAULT_TABLE_DPI,
    ) -> list[ExtractedTable]:
        if not table_boxes_pt:
            self.last_stats = RunStats(
                load_sec=0.0,
                infer_sec=0.0,
                pages=1,
                peak_rss_mb=peak_rss_mb(),
                in_process=True,
            )
            return []

        tatr = self._get_tatr(device)
        t_load0 = time.perf_counter()
        loaded = bool(tatr.ensure_loaded()) if hasattr(tatr, "ensure_loaded") else False
        load_sec = time.perf_counter() - t_load0 if loaded else 0.0

        t0 = time.perf_counter()
        out: list[ExtractedTable] = []
        warnings: list[dict[str, Any]] = []
        page_no = getattr(page, "page_number", None)

        if is_raster:
            ocr_rec = self._get_ocr(ocr_backend, device, ocr_lang=ocr_lang)
            for box in table_boxes_pt:
                crop, crop_box = crop_page_region(page, box, dpi)
                box_tokens = ocr_crop_tokens(ocr_rec, crop, crop_box)
                if not box_tokens:
                    warnings.append({
                        "page": page_no,
                        "code": "table_ocr_empty_tokens",
                        "severity": "warning",
                        "reason": "ocr_returned_no_tokens_in_table_crop",
                        "fallback": "tatr_structure_only",
                    })
                tbl = tatr.extract(crop, crop_box, box_tokens)
                refined_tokens = _split_unspaced_multicell_ocr(ocr_rec, crop, crop_box, box_tokens, tbl)
                if refined_tokens is not None:
                    tbl = tatr.extract(crop, crop_box, refined_tokens)
                out.append(replace(tbl, source="auto_tatr"))
        else:
            plumber = self._get_plumber()
            pl_tables = plumber.extract_from_page(page, table_boxes_pt)
            strict_flags = getattr(plumber, "last_used_strict_lines", None)
            cached_words: list[Word] | None = None if words is None else list(words)
            cached_tokens: list[tuple[str, BBox]] | None = None if words is None else [(w.text, w.bbox) for w in words]

            def get_native_words() -> list[Word]:
                nonlocal cached_words, cached_tokens
                if cached_words is None:
                    from ..text import extract_words

                    cached_words = list(extract_words(page))
                    cached_tokens = [(w.text, w.bbox) for w in cached_words]
                return cached_words

            def get_native_tokens() -> list[tuple[str, BBox]]:
                nonlocal cached_tokens
                if cached_tokens is None:
                    get_native_words()
                assert cached_tokens is not None
                return cached_tokens

            for idx, (box, pl_tbl) in enumerate(zip(table_boxes_pt, pl_tables)):
                used_strict = (
                    bool(strict_flags[idx])
                    if isinstance(strict_flags, Sequence) and idx < len(strict_flags)
                    else True
                )
                if pl_tbl.num_rows == 0 or pl_tbl.num_cols == 0:
                    crop, crop_box = crop_page_region(page, box, dpi)
                    box_tokens = tokens_in_box(get_native_tokens(), crop_box)
                    out.append(replace(tatr.extract(crop, crop_box, box_tokens), source="auto_tatr"))
                elif pl_tbl.num_rows == 1 and pl_tbl.num_cols == 1:
                    crop, crop_box = crop_page_region(page, box, dpi)
                    crop_tokens = tokens_in_box(get_native_tokens(), crop_box)
                    tatr_tbl = tatr.extract(crop, crop_box, crop_tokens)
                    if tatr_tbl.num_rows * tatr_tbl.num_cols >= 2:
                        out.append(replace(tatr_tbl, source="auto_tatr"))
                    elif (
                        _has_valid_grid(pl_tbl)
                        and bbox_iou(pl_tbl.bbox, box) >= 0.50
                        and _token_coverage(pl_tbl, tokens_in_box(get_native_tokens(), box)) >= _MIN_TOKEN_COVERAGE
                    ):
                        callout_cells = tuple(replace(c, is_header=False) for c in pl_tbl.cells)
                        out.append(
                            replace(
                                pl_tbl,
                                cells=callout_cells,
                                source="auto_refined",
                                confidence=min(pl_tbl.confidence, 0.75),
                            )
                        )
                    else:
                        callout_cells = tuple(replace(c, is_header=False) for c in tatr_tbl.cells)
                        out.append(replace(tatr_tbl, cells=callout_cells, source="auto_tatr"))
                else:
                    box_tokens = tokens_in_box(get_native_tokens(), box)
                    if (
                        not _has_valid_grid(pl_tbl)
                        or bbox_iou(pl_tbl.bbox, box) < 0.50
                        or _token_coverage(pl_tbl, box_tokens) < _MIN_TOKEN_COVERAGE
                    ):
                        crop, crop_box = crop_page_region(page, box, dpi)
                        crop_tokens = tokens_in_box(get_native_tokens(), crop_box)
                        out.append(replace(tatr.extract(crop, crop_box, crop_tokens), source="auto_tatr"))
                    elif (pl_tbl.num_cols == 1 or pl_tbl.num_rows == 1) and pl_tbl.num_rows * pl_tbl.num_cols >= 2:
                        crop, crop_box = crop_page_region(page, box, dpi)
                        crop_tokens = tokens_in_box(get_native_tokens(), crop_box)
                        objects = tatr.detect_objects(crop)
                        tatr_tbl = objects_to_table(objects, crop.size, crop_box, crop_tokens, crop=crop)
                        if tatr_tbl.num_cols >= 2 and tatr_tbl.num_rows >= 2:
                            out.append(replace(tatr_tbl, source="auto_tatr"))
                        else:
                            row0_styled = _is_row0_styled(pl_tbl, get_native_words(), crop, crop_box)
                            out.append(refine_table_headers(pl_tbl, objects, crop.size, crop_box, row0_styled=row0_styled))
                    else:
                        eval_box = (
                            min(box[0], pl_tbl.bbox[0]),
                            min(box[1], pl_tbl.bbox[1]),
                            max(box[2], pl_tbl.bbox[2]),
                            max(box[3], pl_tbl.bbox[3]),
                        )
                        crop, crop_box = crop_page_region(page, eval_box, dpi)
                        objects = tatr.detect_objects(crop)
                        empty_row_frac, empty_col_frac = _empty_track_fractions(pl_tbl)
                        overseg_candidate = (
                            empty_row_frac >= 0.25 or empty_col_frac >= 0.25
                        ) or (not used_strict and empty_row_frac + empty_col_frac >= 0.30)
                        has_row_objs = any(lbl == "table row" for lbl, _, _ in objects)
                        switched_to_tatr = False
                        if overseg_candidate or has_row_objs:
                            eval_tokens = tokens_in_box(get_native_tokens(), eval_box)
                            crop_tokens = tokens_in_box(get_native_tokens(), crop_box)
                            tatr_tbl = objects_to_table(objects, crop.size, crop_box, crop_tokens, crop=crop)
                            tatr_valid = (
                                tatr_tbl.num_rows >= 2
                                and tatr_tbl.num_cols >= 2
                                and _has_valid_grid(tatr_tbl)
                                and _token_coverage(tatr_tbl, eval_tokens) >= 0.90
                            )
                            if tatr_valid:
                                tatr_cells = tatr_tbl.num_rows * tatr_tbl.num_cols
                                pl_cells = pl_tbl.num_rows * pl_tbl.num_cols
                                if (overseg_candidate and tatr_cells < 0.75 * pl_cells) or _is_undersegmented_booktabs(
                                    pl_tbl, tatr_tbl, eval_tokens
                                ):
                                    out.append(replace(tatr_tbl, source="auto_tatr"))
                                    switched_to_tatr = True
                        if not switched_to_tatr:
                            row0_styled = _is_row0_styled(pl_tbl, get_native_words(), crop, crop_box)
                            out.append(
                                refine_table_headers(pl_tbl, objects, crop.size, crop_box, row0_styled=row0_styled)
                            )

        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=1,
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
            warnings=tuple(warnings),
        )
        return out

    def extract(
        self,
        image: Image.Image,
        table_bbox_pt: BBox,
        tokens: Sequence[tuple[str, BBox]] = (),
    ) -> ExtractedTable:
        t_load0 = time.perf_counter()
        loaded = self.ensure_loaded()
        load_sec = time.perf_counter() - t_load0 if loaded else 0.0

        t0 = time.perf_counter()
        tbl = self._get_tatr().extract(image, table_bbox_pt, tokens)
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=1,
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return replace(tbl, source="auto_tatr")
