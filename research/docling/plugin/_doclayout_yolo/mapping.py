from collections.abc import Mapping, Sequence

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
from docling.datamodel.base_models import Cluster


LABEL_MAP = {
    # Universal checkpoint (22 labels).
    "text": DocItemLabel.TEXT,
    "title": DocItemLabel.TITLE,
    "header": DocItemLabel.PAGE_HEADER,
    "footer": DocItemLabel.PAGE_FOOTER,
    "page_number": DocItemLabel.PAGE_FOOTER,
    "image": DocItemLabel.PICTURE,
    "image_caption": DocItemLabel.CAPTION,
    "image_footnote": DocItemLabel.FOOTNOTE,
    "image_block": DocItemLabel.PICTURE,
    "table": DocItemLabel.TABLE,
    "table_caption": DocItemLabel.CAPTION,
    "table_footnote": DocItemLabel.FOOTNOTE,
    "equation": DocItemLabel.FORMULA,
    "equation_block": DocItemLabel.FORMULA,
    "list": DocItemLabel.LIST_ITEM,
    # REFERENCE exists in DocItemLabel but is not accepted by Docling's
    # layout postprocessor, so preserve reference regions as regular text.
    "ref_text": DocItemLabel.TEXT,
    "aside_text": DocItemLabel.TEXT,
    "page_footnote": DocItemLabel.FOOTNOTE,
    "code": DocItemLabel.CODE,
    "code_caption": DocItemLabel.CAPTION,
    "algorithm": DocItemLabel.CODE,
    "phonetic": DocItemLabel.TEXT,

    # Legacy DocStructBench checkpoint (10 labels).
    "plain_text": DocItemLabel.TEXT,
    "figure": DocItemLabel.PICTURE,
    "picture": DocItemLabel.PICTURE,
    "figure_caption": DocItemLabel.CAPTION,
    "isolate_formula": DocItemLabel.FORMULA,
    "formula": DocItemLabel.FORMULA,
    "formula_caption": DocItemLabel.CAPTION,
}

IGNORED_LABELS = {"abandon"}


def normalize_label(name: object) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def get_unmapped_labels(names: Mapping[int, str]) -> list[str]:
    return sorted(
        {
            str(name)
            for name in names.values()
            if normalize_label(name) not in LABEL_MAP
            and normalize_label(name) not in IGNORED_LABELS
        }
    )


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
        name = normalize_label(names.get(int(class_id), ""))
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
