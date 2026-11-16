from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from PIL import Image

from ..models import BBox
from .types import OcrLine


@runtime_checkable
class TextRecognizer(Protocol):
    def recognize(self, image: Image.Image, regions: Sequence[BBox] | None = None) -> list[OcrLine]: ...
