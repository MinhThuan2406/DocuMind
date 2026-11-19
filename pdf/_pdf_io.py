from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from .models import BBox

WORD_X_TOLERANCE_RATIO = 0.15


class PdfContext:
    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        self._plumber: Any = None
        self._pdfium: pdfium.PdfDocument | None = None
        self._pdf_hash: str | None = None

    @property
    def plumber(self) -> Any:
        if self._plumber is None:
            self._plumber = pdfplumber.open(str(self.pdf_path))
        return self._plumber

    @property
    def pdfium(self) -> pdfium.PdfDocument:
        if self._pdfium is None:
            self._pdfium = pdfium.PdfDocument(str(self.pdf_path))
        return self._pdfium

    @property
    def pdf_hash(self) -> str:
        if self._pdf_hash is None:
            from .backends.layout_cache import file_sha256

            self._pdf_hash = file_sha256(self.pdf_path)
        return self._pdf_hash

    def close(self) -> None:
        if self._plumber is not None:
            try:
                self._plumber.close()
            finally:
                self._plumber = None
        if self._pdfium is not None:
            try:
                self._pdfium.close()
            finally:
                self._pdfium = None

    def __enter__(self) -> PdfContext:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def page_bbox(page: Any) -> BBox:
    b = getattr(page, "bbox", None)
    if b is not None and len(b) == 4:
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return (0.0, 0.0, float(page.width), float(page.height))


def clip_box(box: Sequence[float], bounds: BBox) -> BBox | None:
    x0 = max(float(box[0]), bounds[0])
    y0 = max(float(box[1]), bounds[1])
    x1 = min(float(box[2]), bounds[2])
    y1 = min(float(box[3]), bounds[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def extract_words_raw(page: Any, *, with_style: bool = False) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"x_tolerance_ratio": WORD_X_TOLERANCE_RATIO}
    if with_style:
        kwargs["extra_attrs"] = ["fontname", "size"]
    return list(page.extract_words(**kwargs))


def extract_toc(pdf_path: str | Path, *, pdfium_doc: Any = None) -> list[tuple[int, str, int]]:
    owns_doc = pdfium_doc is None
    doc = pdfium.PdfDocument(str(pdf_path)) if owns_doc else pdfium_doc
    try:
        out: list[tuple[int, str, int]] = []
        for bm in doc.get_toc():
            dest = bm.get_dest()
            idx = dest.get_index() if dest is not None else None
            if idx is None:
                continue
            out.append((int(bm.level) + 1, str(bm.get_title()), int(idx) + 1))
        return out
    finally:
        if owns_doc:
            doc.close()


def render_bgr(page: Any, dpi: int) -> tuple[np.ndarray, tuple[int, int]]:
    zoom = dpi / 72.0
    if hasattr(page, "render"):
        pil_rgb = page.render(scale=zoom).to_pil().convert("RGB")
    else:
        pil_rgb = page.to_image(resolution=dpi).original.convert("RGB")
    rgb = np.asarray(pil_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ih, iw = bgr.shape[:2]
    return bgr, (iw, ih)


def crop_page_rgb(page: Any, clip_bbox: BBox, dpi: int) -> Image.Image:
    return page.crop(clip_bbox).to_image(resolution=dpi).original.convert("RGB")


def crop_page_rgb_with_bbox(
    page: Any,
    clip_bbox: BBox,
    dpi: int,
    *,
    pad_pt: float = 0.0,
    max_pixels: int = 25_000_000,
) -> tuple[Image.Image, BBox]:
    bounds = page_bbox(page)
    padded = (
        (clip_bbox[0] - pad_pt, clip_bbox[1] - pad_pt, clip_bbox[2] + pad_pt, clip_bbox[3] + pad_pt)
        if pad_pt
        else clip_bbox
    )
    clip = clip_box(padded, bounds) or bounds
    eff_dpi = dpi
    if max_pixels > 0:
        est_pixels = ((clip[2] - clip[0]) * dpi / 72.0) * ((clip[3] - clip[1]) * dpi / 72.0)
        if est_pixels > max_pixels:
            eff_dpi = max(1, int(dpi * math.sqrt(max_pixels / est_pixels)))
    return crop_page_rgb(page, clip, eff_dpi), clip
