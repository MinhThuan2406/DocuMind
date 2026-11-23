from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from ._pdf_io import WORD_X_TOLERANCE_RATIO, clip_box, extract_words_raw, page_bbox
from .models import BBox, TextBlock, TextSource, Word

T = TypeVar("T")

_MAX_XY_DEPTH = 64
_UNICODE_BULLETS = frozenset("•●○◦▪▫‣⁃・･")
_ASCII_BULLETS = frozenset(("-", "–", "—", "*", "+"))
_LIST_MARKER_RE = re.compile(
    r"^(?:"
    r"\d{1,3}[.)、:]"
    r"|\(\d{1,3}\)"
    r"|\([^\W\d_]\)"
    r"|(?:[ivx]{1,4}|[IVX]{1,4})[.)]"
    r"|\[[A-Za-z0-9_.:-]{1,40}\]"
    r")$"
)
_ROT_CHAR_GAP_RATIO = 0.35


def _is_bold_font(fontname: str) -> bool:
    lower = fontname.lower()
    return any(tok in lower for tok in ("bold", "black", "heavy", "semibold", "demi"))


def _is_vert_matrix_char(char: Mapping[str, Any]) -> bool:
    m = char.get("matrix")
    return (
        isinstance(m, (tuple, list))
        and len(m) >= 4
        and abs(float(m[1])) > abs(float(m[0]))
        and abs(float(m[2])) > abs(float(m[3]))
    )


def _word_direction(word: Word) -> str:
    return getattr(word, "direction", "ltr")


def cluster_by_overlap(
    items: Sequence[T],
    get_span: Callable[[T], tuple[float, float]],
    get_cross: Callable[[T], float],
    *,
    min_overlap: float = 0.45,
) -> list[list[T]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda it: ((get_span(it)[0] + get_span(it)[1]) * 0.5, get_cross(it)))
    groups: list[list[T]] = []
    for item in ordered:
        s0, s1 = get_span(item)
        sw = max(1.0, s1 - s0)
        placed = False
        for group in reversed(groups):
            g0 = min(get_span(x)[0] for x in group)
            g1 = max(get_span(x)[1] for x in group)
            gw = max(1.0, g1 - g0)
            if min(s1, g1) - max(s0, g0) >= min_overlap * min(sw, gw):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def _rot_chars_to_words(
    rot_chars: Sequence[Mapping[str, Any]],
    bounds: BBox,
    *,
    min_font_size: float = 0.0,
) -> list[Word]:
    words: list[Word] = []
    for is_btt in (True, False):
        group = [c for c in rot_chars if (float(c["matrix"][1]) > 0.0) == is_btt]
        if not group:
            continue
        cols = cluster_by_overlap(
            group,
            lambda c: (float(c["x0"]), float(c["x1"])),
            lambda c: float(c["top"]),
        )
        direction = "btt" if is_btt else "ttb"
        for col in cols:
            col.sort(key=(lambda c: -float(c["bottom"])) if is_btt else (lambda c: float(c["top"])))
            cur: list[Mapping[str, Any]] = []

            def _flush() -> None:
                if not cur:
                    return
                text = "".join(str(c["text"]) for c in cur)
                if not text:
                    return
                widths = sorted(max(0.0, float(c["x1"]) - float(c["x0"])) for c in cur)
                font_size = widths[len(widths) // 2] if widths else 0.0
                if min_font_size > 0.0 and font_size > 0.0 and font_size < min_font_size:
                    return
                raw_box = (
                    min(float(c["x0"]) for c in cur),
                    min(float(c["top"]) for c in cur),
                    max(float(c["x1"]) for c in cur),
                    max(float(c["bottom"]) for c in cur),
                )
                box = clip_box(raw_box, bounds) or raw_box
                bold = any(_is_bold_font(str(c.get("fontname", ""))) for c in cur)
                words.append(
                    Word(
                        text=text,
                        bbox=box,
                        font_size=round(font_size, 2),
                        bold=bold,
                        upright=False,
                        direction=direction,
                    )
                )

            for c in col:
                ch = str(c.get("text", ""))
                if not ch or ch.isspace():
                    _flush()
                    cur = []
                    continue
                if cur:
                    prev = cur[-1]
                    gap = (
                        float(prev["top"]) - float(c["bottom"])
                        if is_btt
                        else float(c["top"]) - float(prev["bottom"])
                    )
                    ph = max(1.0, float(prev["bottom"]) - float(prev["top"]))
                    ch_h = max(1.0, float(c["bottom"]) - float(c["top"]))
                    if gap > _ROT_CHAR_GAP_RATIO * max(ph, ch_h):
                        _flush()
                        cur = [c]
                        continue
                cur.append(c)
            _flush()
    return words


def raw_words_to_words(
    items: Sequence[Mapping[str, Any]],
    bounds: BBox,
    *,
    min_font_size: float = 0.0,
) -> list[Word]:
    words: list[Word] = []
    for item in items:
        raw_box = (float(item["x0"]), float(item["top"]), float(item["x1"]), float(item["bottom"]))
        box = clip_box(raw_box, bounds) or raw_box
        font_size = float(item.get("size", 0.0) or 0.0)
        bold = _is_bold_font(str(item.get("fontname", "")))
        upright = bool(item.get("upright", True))
        if min_font_size > 0.0 and font_size > 0.0 and font_size < min_font_size:
            continue
        words.append(
            Word(
                text=str(item["text"]),
                bbox=box,
                font_size=round(font_size, 2),
                bold=bold,
                upright=upright,
                direction=str(item.get("direction", "ltr")),
            )
        )
    return words


def extract_horiz_rules(
    page: Any,
    bounds: BBox,
    *,
    max_thickness: float = 6.0,
) -> tuple[BBox, ...]:
    pw = max(0.0, bounds[2] - bounds[0])
    ph = max(0.0, bounds[3] - bounds[1])
    if pw <= 0.0 or ph <= 0.0:
        return ()
    rules: list[BBox] = []
    for obj in (
        *getattr(page, "lines", ()),
        *getattr(page, "rects", ()),
        *getattr(page, "curves", ()),
    ):
        raw_box = (float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"]))
        box = clip_box(raw_box, bounds)
        if box is None:
            continue
        w = box[2] - box[0]
        h = box[3] - box[1]
        if w <= 0.0 or h > max_thickness:
            continue
        if w >= 0.96 * pw and h >= 0.96 * ph:
            continue
        rules.append(box)
    return tuple(rules)


def extract_words(
    page: Any,
    *,
    min_font_size: float = 0.0,
    raw_words: list[dict[str, Any]] | None = None,
) -> list[Word]:
    bounds = page_bbox(page)
    items = extract_words_raw(page, with_style=True) if raw_words is None else raw_words
    objects = getattr(page, "objects", None)
    page_chars = objects.get("char") if isinstance(objects, dict) else None
    rot_words: list[Word] = []
    if isinstance(page_chars, list) and any(_is_vert_matrix_char(c) for c in page_chars):
        horiz_chars = [c for c in page_chars if not _is_vert_matrix_char(c)]
        rot_chars = [c for c in page_chars if _is_vert_matrix_char(c)]
        if horiz_chars:
            from pdfplumber.utils.text import WordExtractor

            items = WordExtractor(
                x_tolerance_ratio=WORD_X_TOLERANCE_RATIO,
                extra_attrs=["fontname", "size", "upright"],
            ).extract_words(horiz_chars)
        else:
            items = []
        rot_words = _rot_chars_to_words(rot_chars, bounds, min_font_size=min_font_size)

    if not items and not rot_words:
        return []
    words = raw_words_to_words(items, bounds, min_font_size=min_font_size)
    words.extend(rot_words)
    return words


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _median_font_size(words: Sequence[Word], page_rect: BBox | None = None) -> float:
    sizes = sorted(w.font_size for w in words if w.font_size > 0.0)
    if sizes:
        return sizes[len(sizes) // 2]
    heights = sorted(w.bbox[3] - w.bbox[1] for w in words if w.bbox[3] > w.bbox[1])
    if heights:
        return heights[len(heights) // 2]
    if page_rect is not None:
        pw = max(0.0, page_rect[2] - page_rect[0])
        ph = max(0.0, page_rect[3] - page_rect[1])
        min_dim = min(pw, ph) if (pw > 0.0 and ph > 0.0) else max(pw, ph)
        if min_dim > 0.0:
            return max(1.0, min_dim / 60.0)
    return 10.0


def _xy_tolerances(
    page_rect: BBox | None = None,
    font_size: float = 0.0,
) -> tuple[float, float, float, float, float]:
    if font_size > 0.0:
        unit = max(0.5, float(font_size))
    elif page_rect is not None:
        pw = max(0.0, page_rect[2] - page_rect[0])
        ph = max(0.0, page_rect[3] - page_rect[1])
        min_dim = min(pw, ph) if (pw > 0.0 and ph > 0.0) else max(pw, ph)
        unit = max(0.5, min_dim / 60.0) if min_dim > 0.0 else 10.0
    else:
        unit = 10.0
    return (
        0.384 * unit,
        0.048 * unit,
        0.48 * unit,
        0.096 * unit,
        max(0.5, 0.576 * unit),
    )


def _partition_orientation(words: Sequence[Word]) -> tuple[list[Word], list[Word]]:
    horiz: list[Word] = []
    vert: list[Word] = []
    needs_check = [
        i
        for i, w in enumerate(words)
        if not w.upright or (w.bbox[3] - w.bbox[1]) >= 2.2 * max(0.5, w.bbox[2] - w.bbox[0])
    ]
    if not needs_check:
        return list(words), vert

    by_y0 = sorted(range(len(words)), key=lambda i: words[i].bbox[1])
    y0_vals = [words[i].bbox[1] for i in by_y0]
    max_bh = max(max(0.5, w.bbox[3] - w.bbox[1]) for w in words)

    def _is_vert_box(w: Word) -> bool:
        bw = max(0.5, w.bbox[2] - w.bbox[0])
        bh = max(0.5, w.bbox[3] - w.bbox[1])
        return not w.upright and bh >= 1.35 * bw

    def _has_horiz_neighbor(idx: int) -> bool:
        b = words[idx].bbox
        bh = max(0.5, b[3] - b[1])
        gap_factor = 0.65 if words[idx].upright else 0.45
        lo = bisect_left(y0_vals, b[1] - max_bh)
        hi = bisect_right(y0_vals, b[3])
        for k in range(lo, hi):
            j = by_y0[k]
            if j == idx or _is_vert_box(words[j]):
                continue
            ob = words[j].bbox
            oh = max(0.5, ob[3] - ob[1])
            if max(bh, oh) > 1.8 * min(bh, oh):
                continue
            y_ov = min(b[3], ob[3]) - max(b[1], ob[1])
            if y_ov < 0.55 * min(bh, oh):
                continue
            x_gap = max(b[0] - ob[2], ob[0] - b[2])
            if x_gap <= gap_factor * min(bh, oh):
                return True
        return False

    def _has_vert_neighbor(idx: int) -> bool:
        b = words[idx].bbox
        bw = max(0.5, b[2] - b[0])
        bh = max(0.5, b[3] - b[1])
        pad = max(1.5 * max(bw, bh), 1.5 * bh)
        lo = bisect_left(y0_vals, b[1] - max_bh - pad)
        hi = bisect_right(y0_vals, b[3] + pad)
        for k in range(lo, hi):
            j = by_y0[k]
            if j == idx:
                continue
            ob = words[j].bbox
            ow = max(0.5, ob[2] - ob[0])
            x_ov = min(b[2], ob[2]) - max(b[0], ob[0])
            if x_ov < 0.5 * min(bw, ow):
                continue
            y_gap = max(b[1] - ob[3], ob[1] - b[3])
            if -0.25 * max(bw, ow) <= y_gap <= max(1.5 * max(bw, ow), 1.5 * bh):
                return True
        return False

    for idx, w in enumerate(words):
        bw = max(0.5, w.bbox[2] - w.bbox[0])
        bh = max(0.5, w.bbox[3] - w.bbox[1])
        if not w.upright:
            if _word_direction(w) in ("btt", "ttb") or (bh >= 1.35 * bw and not _has_horiz_neighbor(idx)):
                vert.append(w)
            elif not _has_horiz_neighbor(idx) and _has_vert_neighbor(idx):
                vert.append(w)
            else:
                horiz.append(w)
        elif len(w.text) >= 3 and bh >= 2.2 * bw and any(ch.isalpha() for ch in w.text):
            vert.append(w)
        else:
            horiz.append(w)
    return horiz, vert


def _words_to_raw_lines(
    words: Sequence[Word],
    *,
    median_font_size: float = 0.0,
) -> list[list[tuple[BBox, list[Word]]]]:
    if not words:
        return []
    ref_fs = median_font_size if median_font_size > 0.0 else _median_font_size(words)
    lines = cluster_by_overlap(words, lambda w: (w.bbox[1], w.bbox[3]), lambda w: w.bbox[0])

    raw_lines: list[list[tuple[BBox, list[Word]]]] = []
    for line in lines:
        line.sort(key=lambda w: w.bbox[0])
        segs: list[tuple[BBox, list[Word]]] = []
        cur: list[Word] = [line[0]]
        for word in line[1:]:
            prev = cur[-1]
            ref_size = max(prev.font_size, word.font_size, prev.bbox[3] - prev.bbox[1], 0.6 * ref_fs, 1.0)
            gap_thr = max(1.8 * ref_size, 1.8 * ref_fs)
            if word.bbox[0] - prev.bbox[2] > gap_thr:
                segs.append((_union_bbox([w.bbox for w in cur]), cur))
                cur = [word]
            else:
                cur.append(word)
        segs.append((_union_bbox([w.bbox for w in cur]), cur))
        raw_lines.append(segs)
    raw_lines.sort(key=lambda segs: min(s[0][1] for s in segs))
    return raw_lines


def extract_raw_line_segments(
    words: Sequence[Word],
    page_rect: BBox | None = None,
) -> list[list[tuple[BBox, list[Word]]]]:
    if not words:
        return []
    horiz_words, _ = _partition_orientation(words)
    if not horiz_words:
        return []
    median_fs = _median_font_size(horiz_words, page_rect)
    return _words_to_raw_lines(horiz_words, median_font_size=median_fs)


def _words_to_line_segments(
    words: Sequence[Word],
    *,
    median_font_size: float = 0.0,
) -> list[tuple[BBox, list[Word]]]:
    return [seg for line in _words_to_raw_lines(words, median_font_size=median_font_size) for seg in line]


def _vertical_words_to_blocks(
    words: Sequence[Word],
    *,
    source: TextSource,
    confidence: float,
    median_font_size: float = 0.0,
) -> list[TextBlock]:
    if not words:
        return []
    ref_fs = median_font_size if median_font_size > 0.0 else _median_font_size(words)
    btt_words = [w for w in words if _word_direction(w) == "btt"]
    other_words = [w for w in words if _word_direction(w) != "btt"]

    blocks: list[TextBlock] = []
    for group, is_btt in ((btt_words, True), (other_words, False)):
        if not group:
            continue
        cols = cluster_by_overlap(group, lambda w: (w.bbox[0], w.bbox[2]), lambda w: w.bbox[1])
        for col in cols:
            if is_btt:
                col.sort(key=lambda w: (-w.bbox[3], w.bbox[0]))
            else:
                col.sort(key=lambda w: (w.bbox[1], w.bbox[0]))
            cur: list[Word] = [col[0]]
            for word in col[1:]:
                prev = cur[-1]
                ref_size = max(prev.font_size, word.font_size, prev.bbox[2] - prev.bbox[0], 0.6 * ref_fs, 1.0)
                gap_thr = max(1.5 * ref_size, 1.2 * ref_fs)
                gap = (prev.bbox[1] - word.bbox[3]) if is_btt else (word.bbox[1] - prev.bbox[3])
                if gap > gap_thr:
                    blocks.append(_make_text_block([cur], source=source, confidence=confidence))
                    cur = [word]
                else:
                    cur.append(word)
            blocks.append(_make_text_block([cur], source=source, confidence=confidence))
    return blocks


def _segment_style(words: Sequence[Word]) -> tuple[float, bool]:
    if not words:
        return 0.0, False
    sizes = sorted(w.font_size for w in words)
    median_size = sizes[len(sizes) // 2]
    bold_chars = sum(len(w.text) for w in words if w.bold)
    total_chars = sum(len(w.text) for w in words)
    return round(median_size, 2), (total_chars > 0 and bold_chars >= 0.5 * total_chars)


def _make_text_block(
    lines: Sequence[Sequence[Word]],
    *,
    source: TextSource,
    confidence: float,
) -> TextBlock:
    all_words = tuple(w for line in lines for w in line)
    text = "\n".join(" ".join(w.text for w in line) for line in lines)
    block_box = _union_bbox([w.bbox for w in all_words])
    fs, bold = _segment_style(all_words)
    return TextBlock(
        text=text,
        bbox=block_box,
        font_size=fs,
        bold=bold,
        source=source,
        confidence=confidence,
        words=all_words,
    )


def _horiz_overlap(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0])


def _is_list_item_start(line_words: Sequence[Word], *, has_trailing: bool = False) -> bool:
    if not line_words:
        return False
    raw = line_words[0].text.strip()
    if not raw:
        return False
    if raw[0] in _UNICODE_BULLETS:
        return True
    has_body = has_trailing or len(line_words) >= 2
    if len(raw) == 1:
        if raw in _ASCII_BULLETS and has_body:
            return True
        if unicodedata.category(raw) in ("No", "So") and has_body:
            return True
    nfkc = unicodedata.normalize("NFKC", raw)
    if not nfkc:
        return False
    if _LIST_MARKER_RE.match(nfkc):
        return True
    return bool(
        len(nfkc) == 2
        and nfkc[0].isalpha()
        and unicodedata.category(nfkc[1]).startswith("P")
        and has_body
    )


def _is_leader_token(text: str, *, min_chars: int = 3) -> bool:
    s = unicodedata.normalize("NFKC", text).strip()
    chars = [ch for ch in s if not ch.isspace()]
    if len(chars) < min_chars:
        return False
    return all(unicodedata.category(ch)[0] in ("P", "S") for ch in chars) and len(set(chars)) <= 2


def _is_leader_segment(words: Sequence[Word]) -> bool:
    chars = [
        ch
        for w in words
        for ch in unicodedata.normalize("NFKC", w.text).strip()
        if not ch.isspace()
    ]
    if not chars:
        return False
    return all(unicodedata.category(ch)[0] in ("P", "S") for ch in chars) and len(set(chars)) <= 2


def _is_short_right_token(words: Sequence[Word], box: BBox, ref_fs: float) -> bool:
    if len(words) > 2 or (box[2] - box[0]) > 5.0 * ref_fs:
        return False
    total_chars = sum(len(unicodedata.normalize("NFKC", w.text).strip()) for w in words)
    return 0 < total_chars <= 12


def _has_stroke_leader(
    lb: BBox,
    rb: BBox,
    horiz_rules: Sequence[BBox],
    ref_fs: float,
    page_w: float,
) -> bool:
    if not horiz_rules:
        return False
    gap_w = rb[0] - lb[2]
    if gap_w <= 0.0:
        return False
    y_lo = min(lb[1], rb[1]) - 0.2 * ref_fs
    y_hi = max(lb[3], rb[3]) + 0.5 * ref_fs
    max_h = max(3.5, 0.4 * ref_fs)
    for rx0, ry0, rx1, ry1 in horiz_rules:
        rw = rx1 - rx0
        if (ry1 - ry0) > max_h or (page_w > 0.0 and rw >= 0.9 * page_w):
            continue
        if ry0 < y_lo or ry1 > y_hi:
            continue
        if rx0 >= lb[2] - 0.8 * ref_fs and rx1 <= rb[0] + 0.8 * ref_fs and rw >= 0.4 * gap_w:
            return True
    return False


def _merge_line_segments_horizontally(
    raw_lines: list[list[tuple[BBox, list[Word]]]],
    *,
    page_rect: BBox | None,
    median_font_size: float,
    horiz_rules: Sequence[BBox] = (),
) -> list[tuple[BBox, list[Word], float | None]]:
    if not raw_lines:
        return []
    ref_fs = median_font_size if median_font_size > 0.0 else 10.0
    align_tol = max(0.8 * ref_fs, 2.0)
    page_w = max(0.0, page_rect[2] - page_rect[0]) if page_rect is not None else 0.0

    line_bounds = [(min(s[0][1] for s in ln), max(s[0][3] for s in ln)) for ln in raw_lines]

    # Collect repeating list-marker tab-stops (A2) and right-aligned short token edges (B2).
    tab_pairs: list[tuple[float, float]] = []
    right_token_lines: list[tuple[int, BBox]] = []
    wide_segs: list[tuple[int, BBox]] = []
    for li, ln in enumerate(raw_lines):
        for si, (sb, sw) in enumerate(ln):
            w_seg = sb[2] - sb[0]
            if w_seg > 5.0 * ref_fs:
                wide_segs.append((li, sb))
            if si + 1 < len(ln):
                s_fs, _ = _segment_style(sw)
                if w_seg <= 3.5 * max(s_fs, ref_fs) and _is_list_item_start(sw, has_trailing=True):
                    tab_pairs.append((sb[0], ln[si + 1][0][0]))
            if si >= 1 and _is_short_right_token(sw, sb, ref_fs):
                right_token_lines.append((li, sb))

    def _is_repeating_tab_pair(lx0: float, rx0: float) -> bool:
        return sum(1 for px, py in tab_pairs if abs(px - lx0) <= align_tol and abs(py - rx0) <= align_tol) >= 2

    def _is_repeating_right_edge(li: int, lb: BBox, rb: BBox) -> bool:
        matching_lines = {
            idx for idx, b in right_token_lines if abs(b[2] - rb[2]) <= align_tol
        }
        if len(matching_lines) < 3:
            return False
        for _, wb in wide_segs:
            if (wb[2] - wb[0]) > 6.0 * ref_fs and abs(wb[2] - rb[2]) <= align_tol and wb[0] < rb[0] - 2.0 * ref_fs:
                return False
            if abs(wb[0] - rb[0]) <= align_tol:
                return False
        band_y0 = min(line_bounds[idx][0] for idx in matching_lines) - 3.0 * ref_fs
        band_y1 = max(line_bounds[idx][1] for idx in matching_lines) + 3.0 * ref_fs
        gutter_hits = 0
        for idx in range(len(raw_lines)):
            if line_bounds[idx][1] < band_y0 or line_bounds[idx][0] > band_y1:
                continue
            has_wide_left = any(
                (b[2] - b[0]) > 3.5 * ref_fs and b[2] <= rb[0] + 0.5 * ref_fs for b, _ in raw_lines[idx]
            )
            has_wide_right = any(
                (b[2] - b[0]) > 3.5 * ref_fs and b[0] >= lb[2] - 0.5 * ref_fs for b, _ in raw_lines[idx]
            )
            if has_wide_left and has_wide_right:
                gutter_hits += 1
        return gutter_hits < 2

    def _has_prose_left_column(li: int, lx0: float, rx0: float, max_label_w: float) -> bool:
        ly0, ly1 = line_bounds[li]
        for idx in range(len(raw_lines)):
            if idx == li:
                continue
            if line_bounds[idx][1] < ly0 - 5.0 * ref_fs or line_bounds[idx][0] > ly1 + 5.0 * ref_fs:
                continue
            for b, _ in raw_lines[idx]:
                if abs(b[0] - lx0) <= align_tol and (b[2] - b[0]) > max_label_w and b[2] <= rx0 + 0.5 * ref_fs:
                    return True
        return False

    def _has_hanging_indent_next_line(li: int, lb: BBox, rb: BBox) -> bool:
        if li + 1 >= len(raw_lines):
            return False
        if line_bounds[li + 1][0] - line_bounds[li][1] > 1.8 * ref_fs:
            return False
        next_first_box = raw_lines[li + 1][0][0]
        return next_first_box[0] > lb[2] - 0.2 * ref_fs and abs(next_first_box[0] - rb[0]) <= align_tol

    out: list[tuple[BBox, list[Word], float | None]] = []
    for li, ln in enumerate(raw_lines):
        i = 0
        while i < len(ln):
            cur_box, cur_words = ln[i]
            cur_body_x0: float | None = None
            j = i + 1
            while j < len(ln):
                nxt_box, nxt_words = ln[j]
                l_fs, _ = _segment_style(cur_words)
                r_fs, _ = _segment_style(nxt_words)
                fs_tol = max(0.25 * max(l_fs, r_fs, ref_fs), 1.5)
                merged_this = False

                # B1a: intermediate leader-only segments between cur and a later segment.
                if _is_leader_segment(nxt_words):
                    k = j
                    leader_chars = 0
                    while k < len(ln) and _is_leader_segment(ln[k][1]):
                        leader_chars += sum(len(unicodedata.normalize("NFKC", w.text).strip()) for w in ln[k][1])
                        k += 1
                    if k < len(ln) and leader_chars >= 3:
                        t_box, t_words = ln[k]
                        t_fs, _ = _segment_style(t_words)
                        if abs(l_fs - t_fs) <= max(0.25 * max(l_fs, t_fs, ref_fs), 1.5):
                            combined = sorted(
                                [w for idx_m in range(i, k + 1) for w in (cur_words if idx_m == i else ln[idx_m][1])],
                                key=lambda w: w.bbox[0],
                            )
                            cur_box = _union_bbox([w.bbox for w in combined])
                            cur_words = combined
                            j = k + 1
                            merged_this = True
                if merged_this:
                    continue

                if abs(l_fs - r_fs) > fs_tol:
                    break

                # B1b: leader attached to end of left or start of right, or stroke leader in gap.
                has_char_leader = (
                    (len(cur_words) >= 2 and _is_leader_token(cur_words[-1].text))
                    or (len(nxt_words) >= 2 and _is_leader_token(nxt_words[0].text))
                )
                has_stroke = _has_stroke_leader(cur_box, nxt_box, horiz_rules, ref_fs, page_w)
                if has_char_leader or has_stroke:
                    combined = sorted(cur_words + nxt_words, key=lambda w: w.bbox[0])
                    cur_box = _union_bbox([w.bbox for w in combined])
                    cur_words = combined
                    j += 1
                    continue

                # A: hanging-indent label-body merge (only on the initial unmerged left label).
                if j == i + 1:
                    max_label_w = 3.5 * max(l_fs, ref_fs)
                    if (
                        (cur_box[2] - cur_box[0]) <= max_label_w
                        and abs(l_fs - r_fs) <= max(0.2 * max(l_fs, r_fs, ref_fs), 1.5)
                        and not _has_prose_left_column(li, cur_box[0], nxt_box[0], max_label_w)
                    ):
                        has_indent = _has_hanging_indent_next_line(li, cur_box, nxt_box)
                        has_list_tab = _is_list_item_start(cur_words, has_trailing=True) and _is_repeating_tab_pair(
                            cur_box[0], nxt_box[0]
                        )
                        if has_indent or has_list_tab:
                            cur_body_x0 = nxt_box[0]
                            combined = sorted(cur_words + nxt_words, key=lambda w: w.bbox[0])
                            cur_box = _union_bbox([w.bbox for w in combined])
                            cur_words = combined
                            j += 1
                            continue

                # B2: repeating short right-aligned token along shared right edge.
                if _is_short_right_token(nxt_words, nxt_box, ref_fs) and _is_repeating_right_edge(
                    li, cur_box, nxt_box
                ):
                    combined = sorted(cur_words + nxt_words, key=lambda w: w.bbox[0])
                    cur_box = _union_bbox([w.bbox for w in combined])
                    cur_words = combined
                    j += 1
                    continue

                break

            out.append((cur_box, cur_words, cur_body_x0))
            i = j
    return out


def _segments_to_blocks(
    segments: Sequence[tuple[BBox, list[Word]] | tuple[BBox, list[Word], float | None]],
    *,
    source: TextSource,
    confidence: float,
    median_font_size: float = 0.0,
) -> list[TextBlock]:
    if not segments:
        return []
    ref_fs = median_font_size if median_font_size > 0.0 else 10.0
    norm_segs: list[tuple[BBox, list[Word], float | None]] = [
        (s[0], s[1], s[2]) if len(s) == 3 else (s[0], s[1], None) for s in segments
    ]
    ordered = sorted(norm_segs, key=lambda s: (s[0][1], s[0][0]))
    groups: list[list[tuple[BBox, list[Word], float, bool, bool, float | None]]] = []
    for bbox, line_words, body_x0 in ordered:
        fs, bold = _segment_style(line_words)
        all_bold = bool(line_words) and all(w.bold for w in line_words)
        lh = max(1.0, bbox[3] - bbox[1])
        is_list_start = _is_list_item_start(line_words) or body_x0 is not None
        merged = False
        if not is_list_start:
            for group in reversed(groups):
                first_box, _, _, _, _, first_body_x0 = group[0]
                last_box, _, last_fs, last_bold, _, _ = group[-1]
                if not _horiz_overlap(last_box, bbox):
                    continue
                # Prevent merging across a vertical outdent back to a label/bullet margin.
                has_hanging = first_body_x0 is not None or (
                    len(group) >= 2 and last_box[0] > first_box[0] + 0.5 * ref_fs
                )
                if has_hanging:
                    expected_x0 = first_body_x0 if first_body_x0 is not None else last_box[0]
                    if bbox[0] < expected_x0 - 0.5 * ref_fs:
                        continue
                gap_y = bbox[1] - last_box[3]
                max_gap = max(0.8 * max(lh, last_box[3] - last_box[1]), 0.6 * ref_fs, 1.0)
                fs_tol = max(0.1 * max(fs, last_fs, ref_fs), 0.5)
                group_all_bold = all(item[4] for item in group)
                bold_ok = (bold == last_bold) or (not group_all_bold and not all_bold)
                if -0.3 * lh <= gap_y <= max_gap and abs(fs - last_fs) <= fs_tol and bold_ok:
                    group.append((bbox, line_words, fs, bold, all_bold, body_x0))
                    merged = True
                    break
        if not merged:
            groups.append([(bbox, line_words, fs, bold, all_bold, body_x0)])

    return [
        _make_text_block([seg_words for _, seg_words, _, _, _, _ in group], source=source, confidence=confidence)
        for group in groups
    ]


def xy_cut_order(
    items: Sequence[T],
    get_bbox: Callable[[T], BBox],
    *,
    page_rect: BBox | None = None,
    font_size: float = 0.0,
) -> list[T]:
    if len(items) <= 1:
        return list(items)

    x_inset_max, x_gap_tol, y_overlap_min, y_gap_tol, y_bucket = _xy_tolerances(page_rect, font_size)
    pairs = [(item, get_bbox(item)) for item in items]

    def _fallback_sort(group: list[tuple[T, BBox]]) -> list[tuple[T, BBox]]:
        return sorted(group, key=lambda p: (round(p[1][1] / y_bucket), p[1][0], p[1][1]))

    def _split_x(group: list[tuple[T, BBox]]) -> list[list[tuple[T, BBox]]] | None:
        x_sorted = sorted(group, key=lambda p: (p[1][0], p[1][1]))
        cols: list[list[tuple[T, BBox]]] = []
        cur_col = [x_sorted[0]]
        b0 = x_sorted[0][1]
        ins0 = min(x_inset_max, 0.05 * max(0.0, b0[2] - b0[0]))
        cur_max_x = b0[2] - ins0

        for pair in x_sorted[1:]:
            b = pair[1]
            ins = min(x_inset_max, 0.05 * max(0.0, b[2] - b[0]))
            item_min_x = b[0] + ins
            item_max_x = b[2] - ins
            if item_min_x >= cur_max_x - x_gap_tol:
                cols.append(cur_col)
                cur_col = [pair]
                cur_max_x = item_max_x
            else:
                cur_col.append(pair)
                cur_max_x = max(cur_max_x, item_max_x)
        if cur_col:
            cols.append(cur_col)

        if len(cols) < 2:
            return None
        for idx in range(len(cols) - 1):
            col_a, col_b = cols[idx], cols[idx + 1]
            min_ya = min(p[1][1] for p in col_a)
            max_ya = max(p[1][3] for p in col_a)
            min_yb = min(p[1][1] for p in col_b)
            max_yb = max(p[1][3] for p in col_b)
            if min(max_ya, max_yb) - max(min_ya, min_yb) > y_overlap_min:
                return cols
        return None

    def _has_multi_row_cols(group: list[tuple[T, BBox]]) -> bool:
        cols = _split_x(group)
        if cols is None:
            return False
        for idx in range(len(cols) - 1):
            col_a, col_b = cols[idx], cols[idx + 1]
            min_ya, max_ya = min(p[1][1] for p in col_a), max(p[1][3] for p in col_a)
            min_yb, max_yb = min(p[1][1] for p in col_b), max(p[1][3] for p in col_b)
            na = sum(1 for p in col_a if min(p[1][3], max_yb) - max(p[1][1], min_yb) > y_overlap_min)
            nb = sum(1 for p in col_b if min(p[1][3], max_ya) - max(p[1][1], min_ya) > y_overlap_min)
            if na >= 1 and nb >= 1 and (na + nb) >= 3:
                return True
        return False

    def _find_mid_span(
        group: list[tuple[T, BBox]],
        group_w: float,
    ) -> tuple[list[tuple[T, BBox]], list[tuple[T, BBox]], list[tuple[T, BBox]]] | None:
        y1s = sorted(p[1][3] for p in group)
        y0s = sorted(p[1][1] for p in group)
        n = len(group)
        best_score: tuple[int, float, float] | None = None
        best_cand: tuple[T, BBox] | None = None
        for cand in group:
            cb = cand[1]
            cw = cb[2] - cb[0]
            if cw < 0.60 * group_w:
                continue
            n_upper = bisect_right(y1s, cb[1] + y_gap_tol)
            if cb[3] <= cb[1] + y_gap_tol:
                n_upper = max(0, n_upper - 1)
            n_lower = n - bisect_left(y0s, cb[3] - y_gap_tol)
            if cb[1] >= cb[3] - y_gap_tol and cb[3] > cb[1] + y_gap_tol:
                n_lower = max(0, n_lower - 1)
            balance = min(n_upper, n_lower)
            if balance == 0:
                continue
            score = (balance, cw, -cb[1])
            if best_score is None or score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is None:
            return None
        cb = best_cand[1]
        upper: list[tuple[T, BBox]] = []
        middle: list[tuple[T, BBox]] = [best_cand]
        lower: list[tuple[T, BBox]] = []
        for p in group:
            if p is best_cand:
                continue
            pb = p[1]
            if pb[3] <= cb[1] + y_gap_tol:
                upper.append(p)
            elif pb[1] >= cb[3] - y_gap_tol:
                lower.append(p)
            else:
                middle.append(p)
        if not upper or not lower:
            return None
        return upper, middle, lower

    def _recurse(group: list[tuple[T, BBox]], depth: int) -> list[tuple[T, BBox]]:
        if len(group) <= 1:
            return group
        if depth >= _MAX_XY_DEPTH:
            return _fallback_sort(group)

        cols = _split_x(group)
        if cols is not None and all(len(col) < len(group) for col in cols):
            out: list[tuple[T, BBox]] = []
            for col in cols:
                out.extend(_recurse(col, depth + 1))
            return out

        min_x = min(p[1][0] for p in group)
        max_x = max(p[1][2] for p in group)
        group_w = max_x - min_x

        if len(group) >= 3 and group_w > 0.0:
            top_item = min(group, key=lambda p: (p[1][1], p[1][0]))
            rest_top = [p for p in group if p is not top_item]
            if len(rest_top) < len(group) and top_item[1][3] <= min(p[1][1] for p in rest_top) + y_gap_tol:
                if _has_multi_row_cols(rest_top):
                    return [top_item] + _recurse(rest_top, depth + 1)

            bot_item = max(group, key=lambda p: (p[1][3], -p[1][0]))
            rest_bot = [p for p in group if p is not bot_item]
            if len(rest_bot) < len(group) and bot_item[1][1] >= max(p[1][3] for p in rest_bot) - y_gap_tol:
                if _has_multi_row_cols(rest_bot):
                    return _recurse(rest_bot, depth + 1) + [bot_item]

            mid_split = _find_mid_span(group, group_w)
            if mid_split is not None:
                upper, middle, lower = mid_split
                if (
                    len(upper) >= 2
                    and len(lower) >= 2
                    and (_split_x(upper) is not None and _split_x(lower) is not None)
                    and all(len(part) < len(group) for part in (upper, middle, lower))
                ):
                    return (
                        _recurse(upper, depth + 1)
                        + _recurse(middle, depth + 1)
                        + _recurse(lower, depth + 1)
                    )

        y_sorted = sorted(group, key=lambda p: (p[1][1], p[1][0]))
        bands: list[list[tuple[T, BBox]]] = []
        cur_band = [y_sorted[0]]
        cur_max_y = y_sorted[0][1][3]

        for pair in y_sorted[1:]:
            b = pair[1]
            if b[1] >= cur_max_y - y_gap_tol:
                bands.append(cur_band)
                cur_band = [pair]
                cur_max_y = b[3]
            else:
                cur_band.append(pair)
                cur_max_y = max(cur_max_y, b[3])
        if cur_band:
            bands.append(cur_band)

        if len(bands) >= 2 and all(len(band) < len(group) for band in bands):
            # Peel top/bottom margin bands first when doing so unlocks a column split on the body.
            rest_after_top = [p for b in bands[1:] for p in b]
            if len(rest_after_top) >= 2 and _has_multi_row_cols(rest_after_top):
                return _recurse(bands[0], depth + 1) + _recurse(rest_after_top, depth + 1)
            rest_before_bot = [p for b in bands[:-1] for p in b]
            if len(rest_before_bot) >= 2 and _has_multi_row_cols(rest_before_bot):
                return _recurse(rest_before_bot, depth + 1) + _recurse(bands[-1], depth + 1)
            if len(bands) >= 3:
                mid_items = [p for b in bands[1:-1] for p in b]
                if len(mid_items) >= 2 and _has_multi_row_cols(mid_items):
                    return (
                        _recurse(bands[0], depth + 1)
                        + _recurse(mid_items, depth + 1)
                        + _recurse(bands[-1], depth + 1)
                    )

            out = []
            for band in bands:
                out.extend(_recurse(band, depth + 1))
            return out

        if len(group) >= 3 and group_w > 0.0:
            if (top_item[1][2] - top_item[1][0]) >= 0.55 * group_w:
                if len(rest_top) < len(group) and top_item[1][3] <= min(p[1][1] for p in rest_top) + y_gap_tol:
                    return [top_item] + _recurse(rest_top, depth + 1)

            if (bot_item[1][2] - bot_item[1][0]) >= 0.55 * group_w:
                if len(rest_bot) < len(group) and bot_item[1][1] >= max(p[1][3] for p in rest_bot) - y_gap_tol:
                    return _recurse(rest_bot, depth + 1) + [bot_item]

            mid_split = _find_mid_span(group, group_w)
            if mid_split is not None:
                upper, middle, lower = mid_split
                if all(len(part) < len(group) for part in (upper, middle, lower)):
                    return (
                        _recurse(upper, depth + 1)
                        + _recurse(middle, depth + 1)
                        + _recurse(lower, depth + 1)
                    )

        return _fallback_sort(group)

    return [pair[0] for pair in _recurse(pairs, 0)]


def order_blocks(
    words: Sequence[Word],
    page_rect: BBox | None = None,
    *,
    source: TextSource = "text_layer",
    confidence: float = 1.0,
    horiz_rules: Sequence[BBox] = (),
) -> list[TextBlock]:
    if not words:
        return []
    median_fs = _median_font_size(words, page_rect)
    horiz_words, vert_words = _partition_orientation(words)
    raw_lines = _words_to_raw_lines(horiz_words, median_font_size=median_fs)
    segments = _merge_line_segments_horizontally(
        raw_lines,
        page_rect=page_rect,
        median_font_size=median_fs,
        horiz_rules=horiz_rules,
    )
    blocks = _segments_to_blocks(
        segments,
        source=source,
        confidence=confidence,
        median_font_size=median_fs,
    )
    blocks.extend(
        _vertical_words_to_blocks(
            vert_words,
            source=source,
            confidence=confidence,
            median_font_size=median_fs,
        )
    )
    return xy_cut_order(blocks, lambda b: b.bbox, page_rect=page_rect, font_size=median_fs)


def extract_text(page: Any, *, min_font_size: float = 0.0) -> list[TextBlock]:
    bounds = page_bbox(page)
    words = extract_words(page, min_font_size=min_font_size)
    return order_blocks(words, bounds, horiz_rules=extract_horiz_rules(page, bounds))
