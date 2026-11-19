from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ..layout.types import LayoutItem
from ..models import BBox, ExtractedTable, RasterPage
from ..ocr.types import OcrLine


@dataclass(frozen=True)
class RunStats:
    load_sec: float = 0.0
    infer_sec: float = 0.0
    pages: int = 0
    peak_rss_mb: float = 0.0
    in_process: bool = True


@dataclass(frozen=True)
class PageLayout:
    items: list[dict[str, Any]]
    frame: tuple[float, float]


@runtime_checkable
class LayoutDetector(Protocol):
    def detect(self, image: Image.Image) -> list[LayoutItem]: ...


@runtime_checkable
class LayoutBackend(Protocol):
    name: str
    revision: str
    cache_dpi: int
    last_stats: RunStats | None

    def analyze(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, PageLayout]: ...


@runtime_checkable
class TextRecognizer(Protocol):
    def recognize(self, image: Image.Image, regions: Sequence[BBox] | None = None) -> list[OcrLine]: ...


@runtime_checkable
class PageReader(Protocol):
    name: str
    last_stats: RunStats | None

    def read(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, RasterPage]: ...


@runtime_checkable
class FigureGrouper(Protocol):
    name: str

    def group(self, bgr: Any, layout: PageLayout, min_area: float) -> list[list[float]]: ...


@runtime_checkable
class TableExtractor(Protocol):
    name: str
    last_stats: RunStats | None

    def extract(
        self,
        image: Image.Image,
        table_bbox_pt: BBox,
        tokens: Sequence[tuple[str, BBox]] = (),
    ) -> ExtractedTable: ...
