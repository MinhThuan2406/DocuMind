from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..models import BBox

LayoutLabel = Literal[
    "caption",
    "footnote",
    "formula",
    "list_item",
    "page_footer",
    "page_header",
    "picture",
    "section_header",
    "table",
    "text",
    "title",
    "document_index",
    "code",
    "checkbox_selected",
    "checkbox_unselected",
    "form",
    "key_value_region",
]

LAYOUT_LABELS: frozenset[str] = frozenset(
    {
        "caption",
        "footnote",
        "formula",
        "list_item",
        "page_footer",
        "page_header",
        "picture",
        "section_header",
        "table",
        "text",
        "title",
        "document_index",
        "code",
        "checkbox_selected",
        "checkbox_unselected",
        "form",
        "key_value_region",
    }
)


@dataclass(frozen=True)
class LayoutItem:
    label: str
    bbox: BBox
    score: float
