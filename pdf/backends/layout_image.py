from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image

from ..layout.detector import clamp_and_rescale
from ..layout.postprocess import clusters_to_items, postprocess_layout
from .base import LayoutDetector, PageLayout, RunStats
from .rss import peak_rss_mb

_PNG_DEFAULT_DPI = 96.012


def render_bgr(page: pymupdf.Page, dpi: int) -> tuple[np.ndarray, tuple[int, int]]:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, (pix.width, pix.height)


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

    def analyze(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, PageLayout]:
        target_pages = sorted(set(pages))
        if not target_pages:
            return {}
        load_sec = 0.0
        ensure = getattr(self.detector, "ensure_loaded", None)
        if callable(ensure):
            t_load0 = time.perf_counter()
            if ensure():
                load_sec = time.perf_counter() - t_load0

        doc = pymupdf.open(str(pdf_path))
        t0 = time.perf_counter()
        out: dict[int, PageLayout] = {}
        try:
            for pno in target_pages:
                if not (1 <= pno <= len(doc)):
                    raise ValueError(f"page {pno} out of range 1..{len(doc)}")
                bgr, _ = render_bgr(doc[pno - 1], self.cache_dpi)
                items, frame = run_layout_items(self.detector, bgr)
                out[pno] = PageLayout(items=items, frame=frame)
        finally:
            doc.close()
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=len(target_pages),
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return out
