from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .models import BBox, FigureBox, PageProfile, RasterPage, TextBlock, Word
from .text import order_blocks, xy_cut_order


@dataclass(frozen=True)
class AssembledItem:
    kind: str  # "text" | "figure" | "table"
    bbox: BBox
    text: str
    source: str
    confidence: float
    level: int
    parent_index: int | None
    font_size: float | None = None
    bold: bool | None = None
    label: str | None = None


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter_area(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def filter_overlapping_figures(figures: Sequence[FigureBox]) -> list[FigureBox]:
    fig_boxes = [f for f in figures if f.kind == "figure"]
    kept: list[FigureBox] = []
    for box in figures:
        if box.kind == "table":
            t_area = _area(box.bbox)
            if t_area > 0 and any(_inter_area(box.bbox, fb.bbox) / t_area >= 0.5 for fb in fig_boxes):
                continue
        kept.append(box)
    return kept


def _assign_words_to_boxes(
    words: Sequence[Word],
    boxes: Sequence[FigureBox],
) -> tuple[list[Word], dict[int, list[Word]]]:
    body_words: list[Word] = []
    box_words: dict[int, list[Word]] = {i: [] for i in range(len(boxes))}
    indexed = [(_area(box.bbox), idx, box) for idx, box in enumerate(boxes)]

    for word in words:
        cx = (word.bbox[0] + word.bbox[2]) * 0.5
        cy = (word.bbox[1] + word.bbox[3]) * 0.5
        best_idx: int | None = None
        best_key: tuple[float, int] | None = None
        for area, idx, box in indexed:
            bx0, by0, bx1, by1 = box.bbox
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                key = (area, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
        if best_idx is None:
            body_words.append(word)
        else:
            box_words[best_idx].append(word)

    return body_words, box_words


def _assemble_raster(raster_page: RasterPage) -> list[AssembledItem]:
    # Order raster blocks and figures together as top-level atomic items via XY-cut.
    top_nodes: list[tuple[str, BBox, TextBlock | FigureBox]] = []
    for blk in raster_page.blocks:
        top_nodes.append(("text", blk.bbox, blk))
    for fig in raster_page.figures:
        top_nodes.append((fig.kind, fig.bbox, fig))

    ordered = xy_cut_order(top_nodes, lambda node: node[1])
    items: list[AssembledItem] = []
    for kind, bbox, obj in ordered:
        if isinstance(obj, TextBlock):
            items.append(
                AssembledItem(
                    kind="text",
                    bbox=bbox,
                    text=obj.text,
                    source=obj.source,
                    confidence=obj.confidence,
                    level=1,
                    parent_index=None,
                    font_size=obj.font_size,
                    bold=obj.bold,
                    label=obj.label,
                )
            )
        else:
            items.append(
                AssembledItem(
                    kind=obj.kind,
                    bbox=bbox,
                    text="",
                    source=obj.source,
                    confidence=obj.confidence,
                    level=1,
                    parent_index=None,
                )
            )
    return items


def assemble_page(
    profile: PageProfile,
    words: Sequence[Word],
    page_rect: BBox,
    figures: Sequence[FigureBox] = (),
    raster_page: RasterPage | None = None,
) -> list[AssembledItem]:
    if profile.kind == "raster":
        if raster_page is None:
            return []
        return _assemble_raster(raster_page)

    kept_figures = filter_overlapping_figures(figures)
    body_words, box_words = _assign_words_to_boxes(words, kept_figures)
    body_blocks = order_blocks(body_words, page_rect)

    top_nodes: list[tuple[str, BBox, TextBlock | tuple[int, FigureBox]]] = [
        ("text", blk.bbox, blk) for blk in body_blocks
    ]
    for idx, fig in enumerate(kept_figures):
        top_nodes.append((fig.kind, fig.bbox, (idx, fig)))

    ordered_nodes = xy_cut_order(top_nodes, lambda node: node[1])
    items: list[AssembledItem] = []

    for kind, bbox, payload in ordered_nodes:
        if kind == "text":
            blk = payload
            assert isinstance(blk, TextBlock)
            items.append(
                AssembledItem(
                    kind="text",
                    bbox=blk.bbox,
                    text=blk.text,
                    source=blk.source,
                    confidence=blk.confidence,
                    level=1,
                    parent_index=None,
                    font_size=blk.font_size,
                    bold=blk.bold,
                )
            )
            continue

        fig_idx, fig = payload  # type: ignore[misc]
        parent_idx = len(items)
        items.append(
            AssembledItem(
                kind=fig.kind,
                bbox=fig.bbox,
                text="",
                source=fig.source,
                confidence=fig.confidence,
                level=1,
                parent_index=None,
            )
        )
        inner_words = box_words[fig_idx]
        if fig.kind == "figure":
            ordered_words = xy_cut_order(inner_words, lambda w: w.bbox)
            for w in ordered_words:
                items.append(
                    AssembledItem(
                        kind="text",
                        bbox=w.bbox,
                        text=w.text,
                        source="text_layer",
                        confidence=1.0,
                        level=2,
                        parent_index=parent_idx,
                        font_size=w.font_size,
                        bold=w.bold,
                    )
                )
        else:
            inner_blocks = order_blocks(inner_words, fig.bbox)
            for blk in inner_blocks:
                items.append(
                    AssembledItem(
                        kind="text",
                        bbox=blk.bbox,
                        text=blk.text,
                        source=blk.source,
                        confidence=blk.confidence,
                        level=2,
                        parent_index=parent_idx,
                        font_size=blk.font_size,
                        bold=blk.bold,
                    )
                )

    return items
