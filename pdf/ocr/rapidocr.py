from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from ..models import BBox
from .types import OcrLine


class RapidOcrRecognizer:
    # Lazy wrapper around rapidocr.RapidOCR (ONNXRuntime, no PyTorch required).
    def __init__(self, *, text_score: float = 0.5) -> None:
        self.text_score = text_score
        self._reader: Any = None

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        from rapidocr import RapidOCR

        self._reader = RapidOCR(
            params={
                "Global.text_score": self.text_score,
                "Global.log_level": "warning",
            }
        )

    def _run_array(self, image: Image.Image, offset: tuple[float, float] = (0.0, 0.0)) -> list[OcrLine]:
        import numpy as np

        self._ensure_loaded()
        rgb = image.convert("RGB")
        res = self._reader(np.asarray(rgb))
        if res is None or res.boxes is None or not res.txts:
            return []

        ox, oy = offset
        w, h = float(rgb.width), float(rgb.height)
        lines: list[OcrLine] = []
        for quad, txt, score in zip(res.boxes.tolist(), res.txts, res.scores):
            text = str(txt).strip()
            if not text:
                continue
            xs = [float(pt[0]) for pt in quad]
            ys = [float(pt[1]) for pt in quad]
            x0 = min(max(min(xs), 0.0), w) + ox
            y0 = min(max(min(ys), 0.0), h) + oy
            x1 = min(max(max(xs), 0.0), w) + ox
            y1 = min(max(max(ys), 0.0), h) + oy
            if x1 > x0 and y1 > y0:
                lines.append(OcrLine(text=text, bbox=(x0, y0, x1, y1), confidence=round(float(score), 4)))
        return lines

    def recognize(self, image: Image.Image, regions: Sequence[BBox] | None = None) -> list[OcrLine]:
        if regions is None:
            return self._run_array(image)
        lines: list[OcrLine] = []
        iw, ih = image.size
        for region in regions:
            x0 = max(0, int(region[0]))
            y0 = max(0, int(region[1]))
            x1 = min(iw, int(round(region[2])))
            y1 = min(ih, int(round(region[3])))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image.crop((x0, y0, x1, y1))
            lines.extend(self._run_array(crop, offset=(float(x0), float(y0))))
        return lines
