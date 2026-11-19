from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from ._pdf_io import clip_box, extract_words_raw, page_bbox
from .models import BBox, TextBlock, TextSource, Word

_SCALE = 72.0 / 150.0
_X_INSET_MAX = 8.0 * _SCALE
_X_GAP_TOL = 1.0 * _SCALE
_Y_OVERLAP_MIN = 10.0 * _SCALE
_Y_GAP_TOL = 2.0 * _SCALE
_Y_BUCKET = 12.0 * _SCALE

T = TypeVar("T")


def _is_bold_font(fontname: str) -> bool:
    lower = fontname.lower()
    return any(tok in lower for tok in ("bold", "black", "heavy", "semibold", "demi"))


def extract_words(
    page: Any,
    *,
    min_font_size: float = 0.0,
    raw_words: list[dict[str, Any]] | None = None,
) -> list[Word]:
    bounds = page_bbox(page)
    items = extract_words_raw(page, with_style=True) if raw_words is None else raw_words
    if not items:
        return []
    words: list[Word] = []
    for item in items:
        raw_box = (float(item["x0"]), float(item["top"]), float(item["x1"]), float(item["bottom"]))
        box = clip_box(raw_box, bounds) or raw_box
        font_size = float(item.get("size", 0.0))
        bold = _is_bold_font(str(item.get("fontname", "")))
        if min_font_size > 0.0 and font_size > 0.0 and font_size < min_font_size:
            continue
        words.append(Word(text=str(item["text"]), bbox=box, font_size=round(font_size, 2), bold=bold))
    return words


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _words_to_line_segments(words: Sequence[Word]) -> list[tuple[BBox, list[Word]]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) * 0.5, w.bbox[0]))
    lines: list[list[Word]] = []
    for word in ordered:
        wy0, wy1 = word.bbox[1], word.bbox[3]
        wh = max(1.0, wy1 - wy0)
        placed = False
        for line in reversed(lines):
            ly0 = min(w.bbox[1] for w in line)
            ly1 = max(w.bbox[3] for w in line)
            lh = max(1.0, ly1 - ly0)
            overlap = min(wy1, ly1) - max(wy0, ly0)
            if overlap >= 0.45 * min(wh, lh):
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    segments: list[tuple[BBox, list[Word]]] = []
    for line in lines:
        line.sort(key=lambda w: w.bbox[0])
        cur: list[Word] = [line[0]]
        for word in line[1:]:
            prev = cur[-1]
            ref_size = max(prev.font_size, word.font_size, prev.bbox[3] - prev.bbox[1], 6.0)
            gap_thr = max(1.8 * ref_size, 18.0)
            if word.bbox[0] - prev.bbox[2] > gap_thr:
                segments.append((_union_bbox([w.bbox for w in cur]), cur))
                cur = [word]
            else:
                cur.append(word)
        segments.append((_union_bbox([w.bbox for w in cur]), cur))
    return segments


def _segment_style(words: Sequence[Word]) -> tuple[float, bool]:
    if not words:
        return 0.0, False
    sizes = sorted(w.font_size for w in words)
    median_size = sizes[len(sizes) // 2]
    bold_chars = sum(len(w.text) for w in words if w.bold)
    total_chars = sum(len(w.text) for w in words)
    return round(median_size, 2), (total_chars > 0 and bold_chars >= 0.5 * total_chars)


def _horiz_overlap(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0])


def _segments_to_blocks(
    segments: list[tuple[BBox, list[Word]]],
    *,
    source: TextSource,
    confidence: float,
) -> list[TextBlock]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s[0][1], s[0][0]))
    groups: list[list[tuple[BBox, list[Word], float, bool]]] = []
    for bbox, line_words in ordered:
        fs, bold = _segment_style(line_words)
        lh = max(1.0, bbox[3] - bbox[1])
        merged = False
        for group in reversed(groups):
            last_box, _, last_fs, last_bold = group[-1]
            if not _horiz_overlap(last_box, bbox):
                continue
            gap_y = bbox[1] - last_box[3]
            max_gap = max(0.8 * max(lh, last_box[3] - last_box[1]), 6.0)
            if -0.3 * lh <= gap_y <= max_gap and abs(fs - last_fs) <= 1.0 and bold == last_bold:
                group.append((bbox, line_words, fs, bold))
                merged = True
                break
        if not merged:
            groups.append([(bbox, line_words, fs, bold)])

    blocks: list[TextBlock] = []
    for group in groups:
        all_words = tuple(w for _, seg_words, _, _ in group for w in seg_words)
        text = "\n".join(" ".join(w.text for w in seg_words) for _, seg_words, _, _ in group)
        block_box = _union_bbox([b for b, _, _, _ in group])
        fs, bold = _segment_style(all_words)
        blocks.append(
            TextBlock(
                text=text,
                bbox=block_box,
                font_size=fs,
                bold=bold,
                source=source,
                confidence=confidence,
                words=all_words,
            )
        )
    return blocks


def xy_cut_order(items: Sequence[T], get_bbox: Callable[[T], BBox]) -> list[T]:
    if len(items) <= 1:
        return list(items)

    def _recurse(group: list[T]) -> list[T]:
        if len(group) <= 1:
            return group

        min_x = min(get_bbox(it)[0] for it in group)
        max_x = max(get_bbox(it)[2] for it in group)
        group_w = max_x - min_x

        x_sorted = sorted(group, key=lambda it: (get_bbox(it)[0], get_bbox(it)[1]))
        cols: list[list[T]] = []
        cur_col = [x_sorted[0]]
        b0 = get_bbox(x_sorted[0])
        ins0 = min(_X_INSET_MAX, 0.05 * max(0.0, b0[2] - b0[0]))
        cur_max_x = b0[2] - ins0

        for item in x_sorted[1:]:
            b = get_bbox(item)
            ins = min(_X_INSET_MAX, 0.05 * max(0.0, b[2] - b[0]))
            item_min_x = b[0] + ins
            item_max_x = b[2] - ins
            if item_min_x >= cur_max_x - _X_GAP_TOL:
                cols.append(cur_col)
                cur_col = [item]
                cur_max_x = item_max_x
            else:
                cur_col.append(item)
                cur_max_x = max(cur_max_x, item_max_x)
        if cur_col:
            cols.append(cur_col)

        if len(cols) >= 2:
            has_parallel = False
            for idx in range(len(cols) - 1):
                col_a, col_b = cols[idx], cols[idx + 1]
                min_ya = min(get_bbox(it)[1] for it in col_a)
                max_ya = max(get_bbox(it)[3] for it in col_a)
                min_yb = min(get_bbox(it)[1] for it in col_b)
                max_yb = max(get_bbox(it)[3] for it in col_b)
                if min(max_ya, max_yb) - max(min_ya, min_yb) > _Y_OVERLAP_MIN:
                    has_parallel = True
                    break
            if has_parallel:
                out: list[T] = []
                for col in cols:
                    out.extend(_recurse(col))
                return out

        if len(group) >= 3 and group_w > 0:
            top_item = min(group, key=lambda it: (get_bbox(it)[1], get_bbox(it)[0]))
            top_b = get_bbox(top_item)
            if (top_b[2] - top_b[0]) >= 0.55 * group_w:
                rest = [it for it in group if it is not top_item]
                return [top_item] + _recurse(rest)

            bot_item = max(group, key=lambda it: (get_bbox(it)[3], -get_bbox(it)[0]))
            bot_b = get_bbox(bot_item)
            if (bot_b[2] - bot_b[0]) >= 0.55 * group_w:
                rest = [it for it in group if it is not bot_item]
                return _recurse(rest) + [bot_item]

        y_sorted = sorted(group, key=lambda it: (get_bbox(it)[1], get_bbox(it)[0]))
        bands: list[list[T]] = []
        cur_band = [y_sorted[0]]
        cur_max_y = get_bbox(y_sorted[0])[3]

        for item in y_sorted[1:]:
            b = get_bbox(item)
            if b[1] >= cur_max_y - _Y_GAP_TOL:
                bands.append(cur_band)
                cur_band = [item]
                cur_max_y = b[3]
            else:
                cur_band.append(item)
                cur_max_y = max(cur_max_y, b[3])
        if cur_band:
            bands.append(cur_band)

        if len(bands) >= 2:
            out = []
            for band in bands:
                out.extend(_recurse(band))
            return out

        return sorted(group, key=lambda it: (round(get_bbox(it)[1] / _Y_BUCKET), get_bbox(it)[0]))

    return _recurse(list(items))


def order_blocks(
    words: Sequence[Word],
    page_rect: BBox | None = None,
    *,
    source: TextSource = "text_layer",
    confidence: float = 1.0,
) -> list[TextBlock]:
    del page_rect
    segments = _words_to_line_segments(words)
    blocks = _segments_to_blocks(segments, source=source, confidence=confidence)
    return xy_cut_order(blocks, lambda b: b.bbox)


def extract_text(page: Any, *, min_font_size: float = 0.0) -> list[TextBlock]:
    words = extract_words(page, min_font_size=min_font_size)
    return order_blocks(words, page_bbox(page))
