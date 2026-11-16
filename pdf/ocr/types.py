from __future__ import annotations

from dataclasses import dataclass

from ..models import BBox


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox: BBox
    confidence: float
