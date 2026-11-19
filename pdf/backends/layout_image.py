from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .._pdf_io import render_bgr
from ..layout.detector import clamp_and_rescale
from ..layout.postprocess import clusters_to_items, postprocess_layout
from .base import LayoutDetector, PageLayout, RunStats
from .rss import peak_rss_mb

_PNG_DEFAULT_DPI = 96.012


def run_layout_items(
    detector: LayoutDetector,
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


class ImageLayoutBackend:
    def __init__(
        self,
        detector: LayoutDetector,
        *,
        name: str,
        revision: str,
        dpi: int = 150,
    ) -> None:
        self.detector = detector
        self.name = name
        self.revision = revision
        self.cache_dpi = dpi
        self.last_stats: RunStats | None = None
        self.last_rendered: dict[int, np.ndarray] = {}
        self._shared_pdfium_doc: tuple[Path, Any] | None = None

    @contextmanager
    def _use_pdfium_doc(self, pdf_path: str | Path, pdfium_doc: Any) -> Iterator[None]:
        prev = self._shared_pdfium_doc
        self._shared_pdfium_doc = (Path(pdf_path).resolve(), pdfium_doc)
        try:
            yield
        finally:
            self._shared_pdfium_doc = prev

    def analyze(
        self,
        pdf_path: str | Path,
        pages: Sequence[int],
        *,
        pdfium_doc: Any = None,
    ) -> dict[int, PageLayout]:
        target_pages = sorted(set(pages))
        self.last_rendered.clear()
        if not target_pages:
            return {}
        load_sec = 0.0
        ensure = getattr(self.detector, "ensure_loaded", None)
        if callable(ensure):
            t_load0 = time.perf_counter()
            if ensure():
                load_sec = time.perf_counter() - t_load0

        resolved = Path(pdf_path).resolve()
        shared = (
            self._shared_pdfium_doc[1]
            if pdfium_doc is None and self._shared_pdfium_doc is not None and self._shared_pdfium_doc[0] == resolved
            else pdfium_doc
        )
        owns_doc = shared is None
        doc = pdfium.PdfDocument(str(pdf_path)) if owns_doc else shared
        t0 = time.perf_counter()
        out: dict[int, PageLayout] = {}
        try:
            for pno in target_pages:
                if not (1 <= pno <= len(doc)):
                    raise ValueError(f"page {pno} out of range 1..{len(doc)}")
                bgr, _ = render_bgr(doc[pno - 1], self.cache_dpi)
                self.last_rendered[pno] = bgr
                items, frame = run_layout_items(self.detector, bgr)
                out[pno] = PageLayout(items=items, frame=frame)
        finally:
            if owns_doc:
                doc.close()
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=len(target_pages),
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return out

    def analyze_with_images(
        self,
        pdf_path: str | Path,
        pages: Sequence[int],
        *,
        pdfium_doc: Any = None,
    ) -> tuple[dict[int, PageLayout], dict[int, np.ndarray]]:
        layouts = self.analyze(pdf_path, pages, pdfium_doc=pdfium_doc)
        images = dict(self.last_rendered)
        self.last_rendered.clear()
        return layouts, images
