from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BBox = tuple[float, float, float, float]
PageKind = Literal["native", "raster", "hybrid"]
TextSource = Literal["text_layer", "ocr", "ocr_high_dpi"]
FigureKind = Literal["figure", "table"]
FigureSource = Literal["layout_picture", "v1_group", "vector_only", "layout_table", "v2_vector_vision"]
TableSource = Literal["pdfplumber", "tatr_onnx", "auto", "auto_refined", "auto_tatr"]
DiagramSource = Literal["vector_graph"]

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
    suspect_reasons: tuple[str, ...] = ()
    partial_raster_regions: tuple[BBox, ...] = ()
    watermark_word_count: int = 0
    table_candidate: bool = False


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    bbox: BBox
    font_size: float
    bold: bool
    upright: bool = True
    direction: str = "ltr"
    monospace: bool = False


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
class GroupedFigure:
    bbox: BBox
    score: float
    source: FigureSource


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
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedTable:
    bbox: BBox
    num_rows: int
    num_cols: int
    cells: tuple[TableCell, ...]
    source: str
    confidence: float
    continuation_of: str | None = None


@dataclass(frozen=True)
class SemanticRegion:
    bbox: BBox
    label: str
    confidence: float = 1.0


@dataclass(frozen=True)
class DiagramNode:
    id: str
    shape_type: str
    label: str
    bbox: BBox
    parent_group_id: str | None = None
    text_element_ids: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class DiagramEdge:
    id: str
    source_node_id: str | None
    target_node_id: str | None
    direction: str
    style: str
    label: str = ""
    waypoints: tuple[tuple[float, float], ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class DiagramGroup:
    id: str
    label: str
    bbox: BBox
    parent_group_id: str | None = None
    member_node_ids: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedDiagram:
    bbox: BBox
    diagram_type: str
    nodes: tuple[DiagramNode, ...] = ()
    edges: tuple[DiagramEdge, ...] = ()
    groups: tuple[DiagramGroup, ...] = ()
    source: str = "vector_graph"
    confidence: float = 0.0
