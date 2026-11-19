from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .backends import PageReader, get as get_backend
from .backends.reader_composed import DEFAULT_RASTER_DPI
from .models import RasterPage


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
) -> dict[int, RasterPage]:
    page_list = list(pages)
    if not page_list:
        return {}
    page_reader = (
        get_backend("reader", reader, layout=layout, ocr_backend=ocr_backend, device=device, dpi=dpi)
        if isinstance(reader, str)
        else reader
    )
    if pdfium_doc is not None and "pdfium_doc" in inspect.signature(page_reader.read).parameters:
        return page_reader.read(pdf_path, page_list, pdfium_doc=pdfium_doc)
    return page_reader.read(pdf_path, page_list)


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
    )[page]
