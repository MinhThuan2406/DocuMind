from collections.abc import Mapping, Sequence
from typing import Any

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling.datamodel.base_models import Cluster


LABEL_MAP = {
    "abstract": DocItemLabel.TEXT,
    "algorithm": DocItemLabel.CODE,
    "aside_text": DocItemLabel.TEXT,
    "chart": DocItemLabel.PICTURE,
    "content": DocItemLabel.DOCUMENT_INDEX,
    "display_formula": DocItemLabel.FORMULA,
    "doc_title": DocItemLabel.TITLE,
    "figure_title": DocItemLabel.CAPTION,
    "footer": DocItemLabel.PAGE_FOOTER,
    "footer_image": DocItemLabel.PAGE_FOOTER,
    "footnote": DocItemLabel.FOOTNOTE,
    "formula_number": DocItemLabel.FORMULA,
    "header": DocItemLabel.PAGE_HEADER,
    "header_image": DocItemLabel.PAGE_HEADER,
    "image": DocItemLabel.PICTURE,
    "inline_formula": DocItemLabel.FORMULA,
    "number": DocItemLabel.PAGE_FOOTER,
    "paragraph_title": DocItemLabel.SECTION_HEADER,
    "reference": DocItemLabel.TEXT,
    "reference_content": DocItemLabel.TEXT,
    "seal": DocItemLabel.PICTURE,
    "table": DocItemLabel.TABLE,
    "text": DocItemLabel.TEXT,
    "vertical_text": DocItemLabel.TEXT,
    "vision_footnote": DocItemLabel.FOOTNOTE,
}


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _payload(result: Any) -> Mapping[str, Any]:
    value = result.json if hasattr(result, "json") else result
    if callable(value):
        value = value()
    if isinstance(value, Mapping) and isinstance(value.get("res"), Mapping):
        value = value["res"]
    if not isinstance(value, Mapping):
        raise TypeError("Paddle layout result must be a mapping or expose JSON")
    return value


def v3_result_to_clusters(
    result: Any,
    *,
    image_size: tuple[int, int],
    page_size: tuple[float, float],
    label_map: Mapping[str, DocItemLabel] = LABEL_MAP,
    model_name: str = "PP-DocLayoutV3",
) -> list[Cluster]:
    data = _payload(result)
    boxes: Sequence[Mapping[str, Any]] = data.get("boxes") or []
    image_width, image_height = image_size
    page_width, page_height = page_size
    scale_x, scale_y = page_width / image_width, page_height / image_height
    clusters: list[Cluster] = []
    for box in boxes:
        name = normalize_label(box.get("label", ""))
        if name not in label_map:
            raise RuntimeError(f"{model_name} emitted an unmapped label: {name!r}")
        coordinates = box.get("coordinate")
        if not isinstance(coordinates, Sequence) or len(coordinates) != 4:
            raise ValueError("Paddle layout coordinate must be [xmin, ymin, xmax, ymax]")
        l, t, r, b = [float(value) for value in coordinates]
        bbox = BoundingBox(
            l=min(max(l * scale_x, 0.0), page_width),
            t=min(max(t * scale_y, 0.0), page_height),
            r=min(max(r * scale_x, 0.0), page_width),
            b=min(max(b * scale_y, 0.0), page_height),
            coord_origin=CoordOrigin.TOPLEFT,
        )
        if bbox.r <= bbox.l or bbox.b <= bbox.t:
            continue
        clusters.append(
            Cluster(
                id=len(clusters),
                label=label_map[name],
                confidence=float(box.get("score", 0.0)),
                bbox=bbox,
                cells=[],
            )
        )
    return clusters
