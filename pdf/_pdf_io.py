from __future__ import annotations

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


def extract_toc(pdf_path: str | Path) -> list[tuple[int, str, int]]:
    doc = pdfium.PdfDocument(str(pdf_path))
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
