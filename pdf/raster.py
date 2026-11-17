from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .backends import PageReader, get as get_backend
from .backends.reader_composed import DEFAULT_RASTER_DPI
from .models import RasterPage


def read_raster_pages(
    pdf_path: str | Path,
    pages: Sequence[int],
    *,
    reader: str | PageReader = "composed",
    layout: str = "heron",
    ocr_backend: str = "rapidocr",
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
) -> dict[int, RasterPage]:
    page_list = list(pages)
    if not page_list:
        return {}
    page_reader = (
        get_backend("reader", reader, layout=layout, ocr_backend=ocr_backend, device=device, dpi=dpi)
        if isinstance(reader, str)
        else reader
    )
    return page_reader.read(pdf_path, page_list)


def read_raster_page(
    pdf_path: str | Path,
    page: int,
    *,
    reader: str | PageReader = "composed",
    layout: str = "heron",
    ocr_backend: str = "rapidocr",
    device: str = "auto",
    dpi: int = DEFAULT_RASTER_DPI,
) -> RasterPage:
    return read_raster_pages(
        pdf_path,
        [page],
        reader=reader,
        layout=layout,
        ocr_backend=ocr_backend,
        device=device,
        dpi=dpi,
    )[page]
