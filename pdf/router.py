from __future__ import annotations

from typing import Any

from ._pdf_io import clip_box, extract_words_raw, page_bbox
from .models import FIGURE_FOOTER_FRAC, FIGURE_MIN_AREA, BBox, PageKind, PageProfile

FULL_PAGE_IMAGE_RATIO = 0.96
MIN_CANDIDATE_DRAWINGS = 8
MIN_CANDIDATE_THICK_DRAWINGS = 3
THICK_DRAWING_MIN_SIDE = 1.5
MIN_NATIVE_WORDS = 5
MIN_NATIVE_TEXT_AREA_RATIO = 0.002
SUSPECT_INVALID_CHAR_RATIO = 0.05
SUSPECT_IMAGE_RATIO = 0.80
SUSPECT_TINY_TEXT_RATIO = 0.01


def _box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _union_area(boxes: list[BBox]) -> float:
    # Coordinate-compression sweep computes exact union area without grid approximations.
    if not boxes:
        return 0.0
    xs = sorted({b[0] for b in boxes} | {b[2] for b in boxes})
    total = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x1 <= x0:
            continue
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] <= x0 and b[2] >= x1)
        if not intervals:
            continue
        covered_y = 0.0
        cur_y0, cur_y1 = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= cur_y1:
                cur_y1 = max(cur_y1, y1)
            else:
                covered_y += cur_y1 - cur_y0
                cur_y0, cur_y1 = y0, y1
        covered_y += cur_y1 - cur_y0
        total += (x1 - x0) * covered_y
    return total


def _is_invalid_char(ch: str) -> bool:
    return ch == "\ufffd" or (ord(ch) < 32 and ch not in "\t\n\r")


def _is_figure_candidate(page: Any, bounds: BBox, image_boxes: list[BBox]) -> bool:
    page_w = bounds[2] - bounds[0]
    page_h = bounds[3] - bounds[1]
    page_area = page_w * page_h
    footer_y = bounds[1] + FIGURE_FOOTER_FRAC * page_h
    if any(
        FIGURE_MIN_AREA * page_area <= _box_area(b) < FULL_PAGE_IMAGE_RATIO * page_area and b[1] < footer_y
        for b in image_boxes
    ):
        return True
    drawings = thick = 0
    for obj in (*page.lines, *page.rects, *page.curves):
        x0, top, x1, bottom = float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"])
        w = max(0.0, x1 - x0)
        h = max(0.0, bottom - top)
        if (w >= FULL_PAGE_IMAGE_RATIO * page_w and h >= FULL_PAGE_IMAGE_RATIO * page_h) or top >= footer_y:
            continue
        drawings += 1
        thick += min(w, h) >= THICK_DRAWING_MIN_SIDE
    return drawings >= MIN_CANDIDATE_DRAWINGS or thick >= MIN_CANDIDATE_THICK_DRAWINGS


def classify_page(
    page: Any,
    page_number: int | None = None,
    *,
    raw_words: list[dict[str, Any]] | None = None,
) -> PageProfile:
    number = int(getattr(page, "page_number", 1)) if page_number is None else page_number
    bounds = page_bbox(page)
    page_w = bounds[2] - bounds[0]
    page_h = bounds[3] - bounds[1]
    page_area = page_w * page_h
    if page_area <= 0:
        return PageProfile(number, "native", 0, 0.0, 0.0, False)

    words = extract_words_raw(page) if raw_words is None else raw_words
    word_count = len(words)
    word_boxes = [
        b for w in words if (b := clip_box((w["x0"], w["top"], w["x1"], w["bottom"]), bounds)) is not None
    ]
    text_area_ratio = min(1.0, sum(_box_area(b) for b in word_boxes) / page_area)

    image_boxes = [
        b for im in page.images if (b := clip_box((im["x0"], im["top"], im["x1"], im["bottom"]), bounds)) is not None
    ]
    max_image_cover = max((_box_area(b) / page_area for b in image_boxes), default=0.0)
    image_area_ratio = min(1.0, _union_area(image_boxes) / page_area)

    total_chars = sum(len(w["text"]) for w in words)
    invalid_chars = sum(1 for w in words for ch in w["text"] if _is_invalid_char(ch))
    invalid_ratio = invalid_chars / total_chars if total_chars > 0 else 0.0

    suspect_ocr_layer = (
        (word_count > 0 and invalid_ratio >= SUSPECT_INVALID_CHAR_RATIO)
        or (word_count > 0 and max_image_cover >= SUSPECT_IMAGE_RATIO and text_area_ratio < SUSPECT_TINY_TEXT_RATIO)
    )

    kind: PageKind
    if word_count < MIN_NATIVE_WORDS and max_image_cover >= FULL_PAGE_IMAGE_RATIO:
        kind = "raster"
    elif (
        word_count >= MIN_NATIVE_WORDS
        and text_area_ratio >= MIN_NATIVE_TEXT_AREA_RATIO
        and not suspect_ocr_layer
        and max_image_cover < FULL_PAGE_IMAGE_RATIO
    ):
        kind = "native"
    elif word_count == 0 and image_area_ratio == 0.0:
        kind = "native"
    else:
        kind = "hybrid"

    return PageProfile(
        page=number,
        kind=kind,
        word_count=word_count,
        text_area_ratio=round(text_area_ratio, 6),
        image_area_ratio=round(image_area_ratio, 6),
        suspect_ocr_layer=suspect_ocr_layer,
        figure_candidate=_is_figure_candidate(page, bounds, image_boxes),
    )
