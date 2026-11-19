from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from PIL import Image

from ..models import BBox, ExtractedTable, Word
from ..tables import DEFAULT_TABLE_DPI, crop_page_region, ocr_crop_tokens, tokens_in_box
from . import get as get_backend
from .base import RunStats
from .rss import peak_rss_mb
from .tatr_onnx import _overlap_indices, objects_to_table

_ROW_OR_HEADER_LABELS = frozenset({"table row", "table column header", "table projected row header"})


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


def refine_table_headers(
    table: ExtractedTable,
    objects: Sequence[tuple[str, BBox, float]],
    crop_size: tuple[int, int],
    crop_box_pt: BBox,
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
    for lbl, box, _ in objects:
        if lbl == "table column header":
            header_rows.update(_overlap_indices(row_spans, crop_box_pt[1] + box[1] * sy, crop_box_pt[1] + box[3] * sy))
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

    def _get_ocr(self, ocr_backend: str, device: str | None = None) -> Any:
        if self._ocr is not None:
            return self._ocr
        eff_device = self.device_name if device is None else device
        return get_backend("ocr", ocr_backend, device=eff_device)

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
            ocr_rec = self._get_ocr(ocr_backend, device)
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
                out.append(replace(tatr.extract(crop, crop_box, box_tokens), source="auto_tatr"))
        else:
            pl_tables = self._get_plumber().extract_from_page(page, table_boxes_pt)
            cached_tokens: list[tuple[str, BBox]] | None = None if words is None else [(w.text, w.bbox) for w in words]

            def get_native_tokens() -> list[tuple[str, BBox]]:
                nonlocal cached_tokens
                if cached_tokens is None:
                    from ..text import extract_words

                    cached_tokens = [(w.text, w.bbox) for w in extract_words(page)]
                return cached_tokens

            for box, pl_tbl in zip(table_boxes_pt, pl_tables):
                if pl_tbl.num_rows == 0 or pl_tbl.num_cols == 0 or (pl_tbl.num_rows == 1 and pl_tbl.num_cols == 1):
                    crop, crop_box = crop_page_region(page, box, dpi)
                    box_tokens = tokens_in_box(get_native_tokens(), crop_box)
                    out.append(replace(tatr.extract(crop, crop_box, box_tokens), source="auto_tatr"))
                elif pl_tbl.num_cols == 1 and pl_tbl.num_rows >= 2:
                    crop, crop_box = crop_page_region(page, box, dpi)
                    box_tokens = tokens_in_box(get_native_tokens(), crop_box)
                    objects = tatr.detect_objects(crop)
                    tatr_tbl = objects_to_table(objects, crop.size, crop_box, box_tokens)
                    if tatr_tbl.num_cols >= 2 and tatr_tbl.num_rows >= 2:
                        out.append(replace(tatr_tbl, source="auto_tatr"))
                    else:
                        out.append(refine_table_headers(pl_tbl, objects, crop.size, crop_box))
                else:
                    crop, crop_box = crop_page_region(page, pl_tbl.bbox, dpi)
                    objects = tatr.detect_objects(crop)
                    out.append(refine_table_headers(pl_tbl, objects, crop.size, crop_box))

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
