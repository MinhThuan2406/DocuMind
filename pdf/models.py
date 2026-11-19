from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BBox = tuple[float, float, float, float]
PageKind = Literal["native", "raster", "hybrid"]
TextSource = Literal["text_layer", "ocr"]
FigureKind = Literal["figure", "table"]
FigureSource = Literal["layout_picture", "v1_group", "vector_only", "layout_table"]

FIGURE_MIN_AREA = 0.015
FIGURE_FOOTER_FRAC = 0.88


@dataclass(frozen=True)
class PageProfile:
    page: int
    kind: PageKind
    word_count: int
    text_area_ratio: float
    image_area_ratio: float
    suspect_ocr_layer: bool
    figure_candidate: bool = True


@dataclass(frozen=True)
class Word:
    text: str
    bbox: BBox
    font_size: float
    bold: bool


@dataclass(frozen=True)
class TextBlock:
    text: str
    bbox: BBox
    font_size: float
    bold: bool
    source: TextSource
    confidence: float
    words: tuple[Word, ...] = ()
    label: str | None = None


@dataclass(frozen=True)
class FigureBox:
    bbox: BBox
    kind: FigureKind
    source: FigureSource
    confidence: float


@dataclass(frozen=True)
class RasterPage:
    blocks: tuple[TextBlock, ...]
    figures: tuple[FigureBox, ...]
    reading_order: tuple[int, ...]


@dataclass(frozen=True)
class TableCell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str
    bbox: BBox | None
    is_header: bool = False


@dataclass(frozen=True)
class ExtractedTable:
    bbox: BBox
    num_rows: int
    num_cols: int
    cells: tuple[TableCell, ...]
    source: str
    confidence: float
