from __future__ import annotations

from importlib import import_module
from typing import Any

from .gt import OMNIDOC_SAMPLE_PAGES, AutosarDocGT, build_raster_pdf, load_autosar_gt, load_omnidoc_gt
from .metrics import area, char_counts, inter, iou, match_count, median, prf, strip_markup

_LAZY_ATTRS = {
    "build_reading_order_fixtures": ".reading_order",
    "kendall_tau_matched": ".reading_order",
    "kendall_tau_penalized": ".reading_order",
    "load_omnidoc_reading_order": ".reading_order",
    "load_omnidoc_tables": ".tables",
    "merged_cell_span_f1": ".tables",
    "parse_table_html": ".tables",
    "teds_score": ".tables",
}


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_ATTRS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(mod_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AutosarDocGT",
    "OMNIDOC_SAMPLE_PAGES",
    "area",
    "build_raster_pdf",
    "build_reading_order_fixtures",
    "char_counts",
    "inter",
    "iou",
    "kendall_tau_matched",
    "kendall_tau_penalized",
    "load_autosar_gt",
    "load_omnidoc_gt",
    "load_omnidoc_reading_order",
    "load_omnidoc_tables",
    "match_count",
    "median",
    "merged_cell_span_f1",
    "parse_table_html",
    "prf",
    "strip_markup",
    "teds_score",
]
