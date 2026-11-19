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
    if hasattr(page, "get_size"):
        w, h = page.get_size()
        return (0.0, 0.0, float(w), float(h))
    return (0.0, 0.0, float(page.width), float(page.height))


def page_rotation(page: Any) -> int:
    if hasattr(page, "get_rotation"):
        rot = page.get_rotation()
    else:
        rot = getattr(page, "rotation", None)
        if rot is None:
            rot = getattr(getattr(page, "page_obj", None), "rotate", 0)
    norm = int(rot or 0) % 360
    return norm if norm in (0, 90, 180, 270) else 0


def _cropbox_in_page_space(page: Any) -> BBox:
    if hasattr(page, "get_size"):
        w, h = page.get_size()
        return (0.0, 0.0, float(w), float(h))
    po = getattr(page, "page_obj", None)
    root = getattr(page, "root_page", page)
    if po is None or root is not page or not hasattr(po, "mediabox") or not hasattr(po, "cropbox"):
        return page_bbox(page)
    mx0, my0, mx1, my1 = (float(v) for v in po.mediabox)
    cx0, cy0, cx1, cy1 = (float(v) for v in po.cropbox)
    mx0, mx1 = min(mx0, mx1), max(mx0, mx1)
    my0, my1 = min(my0, my1), max(my0, my1)
    cx0, cx1 = min(cx0, cx1), max(cx0, cx1)
    cy0, cy1 = min(cy0, cy1), max(cy0, cy1)
    rot = page_rotation(page)
    wm, hm = mx1 - mx0, my1 - my0
    if rot == 0:
        return (cx0, hm - cy1, cx1, hm - cy0)
    if rot == 90:
        return (cy0, cx0 - 2.0 * mx0, cy1, cx1 - 2.0 * mx0)
    if rot == 180:
        return (mx1 + mx0 - cx1, cy0 - 2.0 * my0, mx1 + mx0 - cx0, cy1 - 2.0 * my0)
    return (my1 + my0 - cy1, wm - cx1, my1 + my0 - cy0, wm - cx0)


def normalize_page_bbox(page: Any) -> BBox:
    cb = _cropbox_in_page_space(page)
    return (0.0, 0.0, max(0.0, cb[2] - cb[0]), max(0.0, cb[3] - cb[1]))


def normalize_point_or_box(
    coords: Sequence[float],
    page: Any,
    *,
    inverse: bool = False,
) -> tuple[float, float] | BBox:
    ox, oy = _cropbox_in_page_space(page)[:2]
    dx, dy = (ox, oy) if inverse else (-ox, -oy)
    if len(coords) == 2:
        return (float(coords[0]) + dx, float(coords[1]) + dy)
    if len(coords) == 4:
        return (
            float(coords[0]) + dx,
            float(coords[1]) + dy,
            float(coords[2]) + dx,
            float(coords[3]) + dy,
        )
    raise ValueError(f"expected 2 or 4 coordinates, got {len(coords)}")


def _scale_point_or_box(
    coords: Sequence[float],
    ox: float,
    oy: float,
    sx: float,
    sy: float,
    *,
    shift_before_scale: bool,
) -> tuple[float, float] | BBox:
    def _x(v: float) -> float:
        return (v - ox) * sx if shift_before_scale else ox + v * sx

    def _y(v: float) -> float:
        return (v - oy) * sy if shift_before_scale else oy + v * sy

    if len(coords) == 2:
        return (_x(float(coords[0])), _y(float(coords[1])))
    if len(coords) == 4:
        return (_x(float(coords[0])), _y(float(coords[1])), _x(float(coords[2])), _y(float(coords[3])))
    raise ValueError(f"expected 2 or 4 coordinates, got {len(coords)}")


def pt_to_px(
    coords: Sequence[float],
    page_bounds: BBox,
    img_size: tuple[int, int],
) -> tuple[float, float] | BBox:
    pw = page_bounds[2] - page_bounds[0]
    ph = page_bounds[3] - page_bounds[1]
    if pw <= 0 or ph <= 0:
        raise ValueError(f"invalid page_bounds {page_bounds}")
    return _scale_point_or_box(
        coords,
        page_bounds[0],
        page_bounds[1],
        float(img_size[0]) / pw,
        float(img_size[1]) / ph,
        shift_before_scale=True,
    )


def px_to_pt(
    coords: Sequence[float],
    page_bounds: BBox,
    img_size: tuple[int, int],
) -> tuple[float, float] | BBox:
    iw, ih = float(img_size[0]), float(img_size[1])
    if iw <= 0 or ih <= 0:
        raise ValueError(f"invalid img_size {img_size}")
    return _scale_point_or_box(
        coords,
        page_bounds[0],
        page_bounds[1],
        (page_bounds[2] - page_bounds[0]) / iw,
        (page_bounds[3] - page_bounds[1]) / ih,
        shift_before_scale=False,
    )


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
        kwargs["extra_attrs"] = ["fontname", "size", "upright"]
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
