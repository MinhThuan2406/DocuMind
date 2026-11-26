from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from PIL import Image, ImageOps

from ._pdf_io import clip_box, crop_page_rgb_with_bbox, page_bbox, pdfium_page_box, pt_to_px, round_bbox
from .backends import get as get_backend
from .models import BBox

DEFAULT_FORMULA_DPI = 200
DEFAULT_PAD_FRAC = 0.04
_MAX_RENDER_PIXELS = 25_000_000


@dataclass(frozen=True)
class ParsedFormula:
    latex: str
    confidence: float
    source: str
    backend: str


@runtime_checkable
class FormulaExtractor(Protocol):
    def extract_formula(self, crop: Image.Image) -> tuple[str, float]: ...


def pad_formula_crop(crop: Image.Image, *, pad_frac: float = DEFAULT_PAD_FRAC) -> Image.Image:
    w, h = crop.size
    pad = max(8, min(48, round(pad_frac * max(w, h))))
    return ImageOps.expand(crop.convert("RGB"), border=pad, fill=(255, 255, 255))


def _crop_from_image(page_img: Image.Image, page_bounds: BBox, bbox: BBox) -> Image.Image | None:
    clipped = clip_box(bbox, page_bounds)
    if clipped is None:
        return None
    px0, py0, px1, py1 = pt_to_px(clipped, page_bounds, page_img.size)
    iw, ih = page_img.size
    ix0 = max(0, min(iw, int(math.floor(px0))))
    iy0 = max(0, min(ih, int(math.floor(py0))))
    ix1 = max(0, min(iw, int(math.ceil(px1))))
    iy1 = max(0, min(ih, int(math.ceil(py1))))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return page_img.crop((ix0, iy0, ix1, iy1))


def _render_pdfium_page(pdf_page: Any, dpi: int) -> tuple[Image.Image, BBox]:
    page_bounds = pdfium_page_box(pdf_page)
    pw = max(1.0, page_bounds[2] - page_bounds[0])
    ph = max(1.0, page_bounds[3] - page_bounds[1])
    est_pixels = (pw * dpi / 72.0) * (ph * dpi / 72.0)
    eff_dpi = max(1, int(dpi * math.sqrt(_MAX_RENDER_PIXELS / est_pixels))) if est_pixels > _MAX_RENDER_PIXELS else dpi
    page_img = pdf_page.render(scale=eff_dpi / 72.0).to_pil().convert("RGB")
    return page_img, page_bounds


def _resolve_raster_bounds(raster_page: Any, page_or_pdfium: Any, img_size: tuple[int, int], dpi: int) -> BBox:
    explicit = getattr(raster_page, "page_box", None) or getattr(raster_page, "bbox", None)
    if explicit is not None and len(explicit) == 4:
        return (float(explicit[0]), float(explicit[1]), float(explicit[2]), float(explicit[3]))
    if page_or_pdfium is not None:
        if hasattr(page_or_pdfium, "get_size"):
            return pdfium_page_box(page_or_pdfium)
        if hasattr(page_or_pdfium, "crop"):
            return page_bbox(page_or_pdfium)
    return (0.0, 0.0, float(img_size[0]) * 72.0 / dpi, float(img_size[1]) * 72.0 / dpi)


def extract_formulas_on_page(
    page_or_pdfium: Any,
    bboxes: Sequence[BBox],
    *,
    extractor: str | FormulaExtractor = "auto",
    dpi: int = DEFAULT_FORMULA_DPI,
    pad_frac: float = DEFAULT_PAD_FRAC,
    raster_page: Any = None,
    device: str = "auto",
) -> list[ParsedFormula]:
    if not bboxes:
        return []
    resolved = get_backend("formula", extractor, device=device) if isinstance(extractor, str) else extractor
    backend_name = str(getattr(resolved, "name", "") or type(resolved).__name__)
    source_name = str(getattr(resolved, "source", "") or backend_name.replace("-", "_"))

    raster_img = getattr(raster_page, "image", None) if raster_page is not None else None
    page_img: Image.Image | None = None
    page_bounds: BBox | None = None
    use_plumber_crop = False

    if raster_img is not None:
        page_img = raster_img.convert("RGB")
        page_bounds = _resolve_raster_bounds(raster_page, page_or_pdfium, page_img.size, dpi)
    elif isinstance(page_or_pdfium, Image.Image):
        page_img = page_or_pdfium.convert("RGB")
        page_bounds = (0.0, 0.0, float(page_img.width) * 72.0 / dpi, float(page_img.height) * 72.0 / dpi)
    elif hasattr(page_or_pdfium, "render") and hasattr(page_or_pdfium, "get_size"):
        page_img, page_bounds = _render_pdfium_page(page_or_pdfium, dpi)
    elif hasattr(page_or_pdfium, "crop"):
        use_plumber_crop = True
        page_bounds = page_bbox(page_or_pdfium)
    else:
        raise TypeError(f"unsupported page object for formula extraction: {type(page_or_pdfium).__name__}")

    results: list[ParsedFormula] = []
    for bbox in bboxes:
        if use_plumber_crop:
            assert page_bounds is not None
            clipped = clip_box(bbox, page_bounds)
            crop = None if clipped is None else crop_page_rgb_with_bbox(page_or_pdfium, clipped, dpi)[0]
        else:
            assert page_img is not None and page_bounds is not None
            crop = _crop_from_image(page_img, page_bounds, bbox)

        if crop is None or crop.width <= 0 or crop.height <= 0:
            results.append(ParsedFormula(latex="", confidence=0.0, source=source_name, backend=backend_name))
            continue

        padded = pad_formula_crop(crop, pad_frac=pad_frac)
        raw_latex, raw_conf = resolved.extract_formula(padded)
        latex = str(raw_latex or "").strip()
        conf = round(float(raw_conf), 4) if latex else 0.0
        results.append(ParsedFormula(latex=latex, confidence=conf, source=source_name, backend=backend_name))
    return results


def formula_to_ir_dict(
    formula_id: str,
    element_id: str,
    page: int,
    source_file: str,
    bbox: BBox,
    parsed: ParsedFormula,
) -> dict[str, Any]:
    rounded = [v + 0.0 for v in round_bbox(bbox)]
    return {
        "id": formula_id,
        "element_id": element_id,
        "page": page,
        "source_file": source_file,
        "bbox": rounded,
        "source": parsed.source,
        "backend": parsed.backend,
        "confidence": round(float(parsed.confidence), 4),
        "latex": parsed.latex,
    }
