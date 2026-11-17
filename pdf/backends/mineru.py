from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pymupdf

from ..layout.mineru_pipeline import (
    MINERU_PIPELINE_BACKEND,
    MINERU_PIPELINE_REVISION,
    mineru_middle_json_to_cache_pages,
    mineru_middle_json_to_raster_pages,
    run_mineru_pdf,
)
from ..models import RasterPage
from .base import PageLayout, RunStats


class MinerULayoutBackend:
    isolated = True

    def __init__(self, *, device: str = "auto", python_exe: str | Path | None = None) -> None:
        self.device = device
        self.python_exe = python_exe
        self.name = MINERU_PIPELINE_BACKEND
        self.revision = MINERU_PIPELINE_REVISION
        self.cache_dpi = 0
        self.last_stats: RunStats | None = None
        self.last_unmapped: dict[str, int] = {}

    def analyze(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, PageLayout]:
        path = Path(pdf_path)
        doc = pymupdf.open(str(path))
        try:
            page_sizes = {i + 1: (float(doc[i].rect.width), float(doc[i].rect.height)) for i in range(len(doc))}
        finally:
            doc.close()

        for pno in pages:
            if pno not in page_sizes:
                raise ValueError(f"page {pno} out of range 1..{len(page_sizes)}")

        middle_json, metrics = run_mineru_pdf(path, device=self.device, python_exe=self.python_exe)
        self.last_stats = RunStats(
            load_sec=float(metrics.get("load_sec", 0.0)),
            infer_sec=float(metrics.get("inference_sec", 0.0)),
            pages=int(metrics.get("pages_processed", len(page_sizes))),
            peak_rss_mb=float(metrics.get("peak_rss_mb", 0.0)),
            in_process=False,
        )
        pages_out, unmapped, _ = mineru_middle_json_to_cache_pages(middle_json, page_sizes)
        self.last_unmapped = dict(unmapped)
        return {pno: PageLayout(items=items, frame=frame) for pno, (items, frame) in pages_out.items()}


class MinerUReader:
    isolated = True

    def __init__(self, *, device: str = "auto", python_exe: str | Path | None = None) -> None:
        self.device = device
        self.python_exe = python_exe
        self.name = MINERU_PIPELINE_BACKEND
        self.revision = MINERU_PIPELINE_REVISION
        self.last_stats: RunStats | None = None

    def read(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, RasterPage]:
        page_list = list(pages)
        if not page_list:
            return {}
        path = Path(pdf_path)
        doc_pdf = pymupdf.open(str(path))
        try:
            page_sizes = {idx: (float(p.rect.width), float(p.rect.height)) for idx, p in enumerate(doc_pdf, start=1)}
        finally:
            doc_pdf.close()

        for pno in page_list:
            if pno not in page_sizes:
                raise ValueError(f"page {pno} out of range 1..{len(page_sizes)}")

        middle_json, metrics = run_mineru_pdf(
            path,
            device=self.device,
            python_exe=self.python_exe,
            strip_content=False,
            ocr_mode="auto",
        )
        self.last_stats = RunStats(
            load_sec=float(metrics.get("load_sec", 0.0)),
            infer_sec=float(metrics.get("inference_sec", 0.0)),
            pages=int(metrics.get("pages_processed", len(page_list))),
            peak_rss_mb=float(metrics.get("peak_rss_mb", 0.0)),
            in_process=False,
        )
        return mineru_middle_json_to_raster_pages(middle_json, page_sizes, target_pages=page_list)
