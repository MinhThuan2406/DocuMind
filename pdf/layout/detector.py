from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from ..models import BBox
from .types import LAYOUT_LABELS, LayoutItem


@runtime_checkable
class LayoutDetector(Protocol):
    def detect(self, image: Image.Image) -> list[LayoutItem]: ...


def get_layout_detector(backend: str = "heron", device: str = "auto") -> tuple[LayoutDetector, str]:
    from ..backends.registry import get

    det = get("detector", backend, device=device)
    return det, str(getattr(det, "revision", "local"))


def clamp_and_rescale(
    items: list[tuple[str, BBox, float]],
    src_size: tuple[float, float],
    dst_size: tuple[float, float],
) -> list[LayoutItem]:
    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / sw if sw > 0 else 1.0
    sy = dh / sh if sh > 0 else 1.0
    out: list[LayoutItem] = []
    for label, bbox, score in items:
        if label not in LAYOUT_LABELS:
            continue
        x0 = min(max(bbox[0] * sx, 0.0), dw)
        y0 = min(max(bbox[1] * sy, 0.0), dh)
        x1 = min(max(bbox[2] * sx, 0.0), dw)
        y1 = min(max(bbox[3] * sy, 0.0), dh)
        if x1 > x0 and y1 > y0:
            out.append(LayoutItem(label=label, bbox=(x0, y0, x1, y1), score=float(score)))
    return out
