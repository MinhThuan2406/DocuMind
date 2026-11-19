from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import PageReader, get as get_backend
from .backends.reader_composed import DEFAULT_HIGH_DPI, DEFAULT_RASTER_DPI
from .models import BBox, RasterPage, TextBlock


def _resolve_reader(
    reader: str | PageReader,
    *,
    layout: str,
    ocr_backend: str,
    device: str,
    dpi: int,
) -> Any:
    if not isinstance(reader, str):
        return reader
    return get_backend("reader", reader, layout=layout, ocr_backend=ocr_backend, device=device, dpi=dpi)


def read_raster_pages(
    pdf_path: str | Path,
    pages: Sequence[int],
    *,
    reader: str | PageReader = "composed",
    layout: str = "heron-onnx",
    ocr_backend: str = "rapidocr-onnx",
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
    pdfium_doc: Any = None,
    page_boxes: Mapping[int, BBox] | None = None,
) -> dict[int, RasterPage]:
    page_list = list(pages)
    if not page_list:
        return {}
    page_reader = _resolve_reader(reader, layout=layout, ocr_backend=ocr_backend, device=device, dpi=dpi)
    params = inspect.signature(page_reader.read).parameters
    kwargs: dict[str, Any] = {}
    if pdfium_doc is not None and "pdfium_doc" in params:
        kwargs["pdfium_doc"] = pdfium_doc
    if page_boxes is not None and "page_boxes" in params:
        kwargs["page_boxes"] = page_boxes
    return page_reader.read(pdf_path, page_list, **kwargs)


def read_raster_page(
    pdf_path: str | Path,
    page: int,
    *,
    reader: str | PageReader = "composed",
    layout: str = "heron-onnx",
    ocr_backend: str = "rapidocr-onnx",
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
    pdfium_doc: Any = None,
    page_box: BBox | None = None,
) -> RasterPage:
    return read_raster_pages(
        pdf_path,
        [page],
        reader=reader,
        layout=layout,
        ocr_backend=ocr_backend,
        device=device,
        dpi=dpi,
        pdfium_doc=pdfium_doc,
        page_boxes={page: page_box} if page_box is not None else None,
    )[page]


def read_raster_regions(
    pdf_path: str | Path,
    regions_by_page: Mapping[int, Sequence[BBox]],
    *,
    reader: str | PageReader = "composed",
    layout: str = "heron-onnx",
    ocr_backend: str = "rapidocr-onnx",
    device: str = "auto",
    dpi: int = DEFAULT_HIGH_DPI,
    pdfium_doc: Any = None,
    page_boxes: Mapping[int, BBox] | None = None,
) -> dict[int, list[TextBlock]]:
    if not regions_by_page:
        return {}
    page_reader = _resolve_reader(
        reader, layout=layout, ocr_backend=ocr_backend, device=device, dpi=DEFAULT_RASTER_DPI
    )
    fn = getattr(page_reader, "read_raster_regions", None)
    if not callable(fn):
        raise NotImplementedError(f"reader {type(page_reader).__name__} does not support read_raster_regions")
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {}
    if "dpi" in params:
        kwargs["dpi"] = dpi
    if pdfium_doc is not None and "pdfium_doc" in params:
        kwargs["pdfium_doc"] = pdfium_doc
    if page_boxes is not None and "page_boxes" in params:
        kwargs["page_boxes"] = page_boxes
    return fn(pdf_path, regions_by_page, **kwargs)
