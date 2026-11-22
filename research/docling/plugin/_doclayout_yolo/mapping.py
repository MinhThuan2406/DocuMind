from collections.abc import Mapping, Sequence

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling.datamodel.base_models import Cluster


LABEL_MAP = {
    "title": DocItemLabel.TITLE,
    "plain text": DocItemLabel.TEXT,
    "text": DocItemLabel.TEXT,
    "abandon": DocItemLabel.PAGE_HEADER,
    "figure": DocItemLabel.PICTURE,
    "picture": DocItemLabel.PICTURE,
    "figure_caption": DocItemLabel.CAPTION,
    "figure caption": DocItemLabel.CAPTION,
    "table": DocItemLabel.TABLE,
    "table_caption": DocItemLabel.CAPTION,
    "table caption": DocItemLabel.CAPTION,
    "table_footnote": DocItemLabel.FOOTNOTE,
    "table footnote": DocItemLabel.FOOTNOTE,
    "isolate_formula": DocItemLabel.FORMULA,
    "isolate formula": DocItemLabel.FORMULA,
    "formula": DocItemLabel.FORMULA,
    "formula_caption": DocItemLabel.CAPTION,
    "formula caption": DocItemLabel.CAPTION,
}


def detections_to_clusters(
    *,
    boxes: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    scores: Sequence[float],
    names: Mapping[int, str],
    image_size: tuple[int, int],
    page_size: tuple[float, float],
) -> list[Cluster]:
    image_width, image_height = image_size
    page_width, page_height = page_size
    scale_x, scale_y = page_width / image_width, page_height / image_height
    clusters: list[Cluster] = []
    for box, class_id, score in zip(boxes, class_ids, scores):
        name = str(names.get(int(class_id), "")).strip().lower().replace("-", "_")
        label = LABEL_MAP.get(name)
        if label is None:
            continue
        l, t, r, b = [float(value) for value in box]
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
                label=label,
                confidence=float(score),
                bbox=bbox,
                cells=[],
            )
        )
    return clusters
