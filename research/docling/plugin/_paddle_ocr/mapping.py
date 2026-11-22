from collections.abc import Mapping
from typing import Any

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell


def _payload(result: Any) -> Mapping[str, Any]:
    value = result.json if hasattr(result, "json") else result
    if callable(value):
        value = value()
    if isinstance(value, Mapping) and isinstance(value.get("res"), Mapping):
        value = value["res"]
    if not isinstance(value, Mapping):
        raise TypeError("PaddleOCR result must be a mapping or expose a JSON mapping")
    return value


def result_to_cells(
    result: Any,
    *,
    crop: BoundingBox,
    scale: float,
    threshold: float,
) -> list[TextCell]:
    data = _payload(result)
    polygons = data.get("rec_polys") or data.get("dt_polys") or []
    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []
    cells: list[TextCell] = []
    for polygon, text, score in zip(polygons, texts, scores):
        confidence = float(score)
        if confidence < threshold or not str(text).strip():
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        bbox = BoundingBox(
            l=crop.l + min(xs) / scale,
            t=crop.t + min(ys) / scale,
            r=crop.l + max(xs) / scale,
            b=crop.t + max(ys) / scale,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        cells.append(
            TextCell(
                index=len(cells),
                text=str(text),
                orig=str(text),
                confidence=confidence,
                from_ocr=True,
                rect=BoundingRectangle.from_bounding_box(bbox),
            )
        )
    return cells
