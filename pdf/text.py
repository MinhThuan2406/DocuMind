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
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
_BOLD_TOKEN_RE = re.compile(r"(?:bold|black|heavy|semibold|demi)")
_BOLD_SHORT_RE = re.compile(r"(?:^|[-_])(?:bd|bo|w[6-9])(?:$|[-_\d])")
_TEX_BOLD_RE = re.compile(
    r"(?:^|[-_])(?:cm|lm|sf|ec|tc|cs|la|tx|px|ptm|phv|bch|put|pnc|ppl|pbk|pag|pcr)(?:ss|tt|rm)?bx\d*"
)
_MEDIUM_FONT_RE = re.compile(r"(?:^|[-_])(?:medi(?:um)?|med|w5)(?:ital(?:ic)?|oblique|obl)?(?:$|[-_\d])")
_FAMILY_SPLIT_RE = re.compile(
    r"[-_](?=(?:regu(?:lar)?|roman|book|light|normal|medi(?:um)?|med|bold|bd|bo|black|heavy|semibold|demi|ital(?:ic)?|oblique|obl|w\d)(?:$|[-_\d]|ital|obl))"
)
_MONOSPACE_TOKENS = (
    "courier",
    "consola",
    "menlo",
    "monaco",
    "inconsolata",
    "firacode",
    "cascadiacode",
    "sourcecode",
    "jetbrains",
    "lucidaconsole",
    "andale",
    "mono",
    "fixedsys",
    "typewriter",
)
_TEX_MONO_RE = re.compile(r"(?:^|[-_])(?:cm|lm|ec|tc|tx|px|pcr)(?:ss)?tt\d*|(?:^|[-_])pcr[rb]")


def _strip_subset_prefix(fontname: str) -> str:
    return _SUBSET_PREFIX_RE.sub("", fontname).strip().lower()


def _font_family_key(fontname: str) -> str:
    base = _strip_subset_prefix(fontname)
    if not base:
        return ""
    return _FAMILY_SPLIT_RE.split(base, maxsplit=1)[0]


def _is_medium_font(fontname: str) -> bool:
    base = _strip_subset_prefix(fontname)
    return bool(base and _MEDIUM_FONT_RE.search(base))


def _is_bold_font(
    fontname: str,
    medium_bold_families: frozenset[str] = frozenset(),
) -> bool:
    base = _strip_subset_prefix(fontname)
    if not base:
        return False
    if _BOLD_TOKEN_RE.search(base) or _BOLD_SHORT_RE.search(base) or _TEX_BOLD_RE.search(base):
        return True
    if medium_bold_families and _MEDIUM_FONT_RE.search(base):
        return _font_family_key(fontname) in medium_bold_families
    return False


def _is_monospace_font(fontname: str) -> bool:
    base = _strip_subset_prefix(fontname)
    if not base:
        return False
    return any(tok in base for tok in _MONOSPACE_TOKENS) or bool(_TEX_MONO_RE.search(base))


def infer_medium_bold_families(raw_words: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    if not raw_words:
        return frozenset()
    fam_kind_chars: dict[tuple[str, str], int] = {}
    fam_med_chars: dict[str, int] = {}
    fam_light_chars: dict[str, int] = {}
    total_chars = 0
    for item in raw_words:
        fname = str(item.get("fontname") or "")
        fam = _font_family_key(fname)
        if not fam:
            continue
        n_chars = len(str(item.get("text") or "").strip())
        if n_chars <= 0:
            continue
        total_chars += n_chars
        if _is_bold_font(fname):
            kind = "bold"
        elif _is_medium_font(fname):
            kind = "medium"
            fam_med_chars[fam] = fam_med_chars.get(fam, 0) + n_chars
        else:
            kind = "light"
            fam_light_chars[fam] = fam_light_chars.get(fam, 0) + n_chars
        fam_kind_chars[(fam, kind)] = fam_kind_chars.get((fam, kind), 0) + n_chars
    if not fam_kind_chars or not fam_med_chars:
        return frozenset()
    (dom_fam, dom_kind), _ = max(fam_kind_chars.items(), key=lambda kv: kv[1])
    if dom_kind != "light":
        return frozenset()
    promoted = {
        fam
        for fam, med_cnt in fam_med_chars.items()
        if fam_light_chars.get(fam, 0) > med_cnt
        or (fam != dom_fam and fam_light_chars.get(fam, 0) == 0 and med_cnt <= 0.40 * max(1, total_chars))
    }
    return frozenset(promoted)


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
    medium_bold_families: frozenset[str] = frozenset(),
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
                bold = any(_is_bold_font(str(c.get("fontname", "")), medium_bold_families) for c in cur)
                monospace = any(_is_monospace_font(str(c.get("fontname", ""))) for c in cur)
                words.append(
                    Word(
                        text=text,
                        bbox=box,
                        font_size=round(font_size, 2),
                        bold=bold,
                        upright=False,
                        direction=direction,
                        monospace=monospace,
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
    medium_bold_families: frozenset[str] | None = None,
) -> list[Word]:
    resolved_mbf = infer_medium_bold_families(items) if medium_bold_families is None else medium_bold_families
    words: list[Word] = []
    for item in items:
        raw_box = (float(item["x0"]), float(item["top"]), float(item["x1"]), float(item["bottom"]))
        box = clip_box(raw_box, bounds) or raw_box
        font_size = float(item.get("size", 0.0) or 0.0)
        fontname = str(item.get("fontname", ""))
        bold = _is_bold_font(fontname, resolved_mbf)
        monospace = _is_monospace_font(fontname)
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
                monospace=monospace,
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
    raw_rules: list[BBox] = []
    for obj in (
        *getattr(page, "lines", ()),
        *getattr(page, "rects", ()),
        *getattr(page, "curves", ()),
    ):
        rx0, ry0, rx1, ry1 = float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"])
        if rx0 > rx1:
            rx0, rx1 = rx1, rx0
        if ry0 > ry1:
            ry0, ry1 = ry1, ry0
        if ry1 < bounds[1] or ry0 > bounds[3]:
            continue
        x0 = max(rx0, bounds[0])
        y0 = max(ry0, bounds[1])
        x1 = min(rx1, bounds[2])
        y1 = min(ry1, bounds[3])
        w = x1 - x0
        h = y1 - y0
        if w <= 0.0 or h > max_thickness:
            continue
        if w >= 0.96 * pw and h >= 0.96 * ph:
            continue
        raw_rules.append((x0, y0, x1, y1))

    if len(raw_rules) <= 1:
        return tuple(raw_rules)

    raw_rules.sort(key=lambda r: (r[0], r[2], r[1]))
    # Track [x0, y0, x1, y1, anchor_cy, anchor_h] to prevent vertical drift and merging rects of differing thickness.
    active: list[list[float]] = []
    finished: list[BBox] = []
    for sx0, sy0, sx1, sy1 in raw_rules:
        scy = 0.5 * (sy0 + sy1)
        sh = sy1 - sy0
        still_active: list[list[float]] = []
        merged = False
        for grp in active:
            if grp[2] < sx0 - 2.0:
                finished.append((grp[0], grp[1], grp[2], grp[3]))
                continue
            thick_tol = 0.5 if max(sh, grp[5]) > 1.5 else 1.0
            if (
                not merged
                and abs(scy - grp[4]) <= 1.0
                and abs(sh - grp[5]) <= thick_tol
                and max(grp[3], sy1) - min(grp[1], sy0) <= max_thickness
            ):
                grp[1] = min(grp[1], sy0)
                grp[2] = max(grp[2], sx1)
                grp[3] = max(grp[3], sy1)
                merged = True
            still_active.append(grp)
        if not merged:
            still_active.append([sx0, sy0, sx1, sy1, scy, sh])
        active = still_active

    for grp in active:
        finished.append((grp[0], grp[1], grp[2], grp[3]))
    finished.sort(key=lambda r: (r[1], r[0]))
    return tuple(finished)


def extract_words(
    page: Any,
    *,
    min_font_size: float = 0.0,
    raw_words: list[dict[str, Any]] | None = None,
    medium_bold_families: frozenset[str] | None = None,
) -> list[Word]:
    bounds = page_bbox(page)
    items = extract_words_raw(page, with_style=True) if raw_words is None else raw_words
    objects = getattr(page, "objects", None)
    page_chars = objects.get("char") if isinstance(objects, dict) else None
    rot_words: list[Word] = []
    resolved_mbf = medium_bold_families
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
        if resolved_mbf is None:
            resolved_mbf = infer_medium_bold_families(items or rot_chars)
        rot_words = _rot_chars_to_words(
            rot_chars,
            bounds,
            min_font_size=min_font_size,
            medium_bold_families=resolved_mbf,
        )

    if not items and not rot_words:
        return []
    words = raw_words_to_words(
        items,
        bounds,
        min_font_size=min_font_size,
        medium_bold_families=resolved_mbf,
    )
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


def _detect_vertical_gutters(
    sorted_lines: Sequence[tuple[float, float, list[Word]]],
    ref_fs: float,
    gutter_min_w: float,
) -> list[tuple[float, float, float]]:
    if len(sorted_lines) < 4:
        return []
    line_chunks: list[list[tuple[float, float]]] = []
    for _, _, line in sorted_lines:
        chunks: list[tuple[float, float]] = []
        cx0, cx1 = line[0].bbox[0], line[0].bbox[2]
        for w in line[1:]:
            if w.bbox[0] - cx1 < gutter_min_w:
                cx1 = max(cx1, w.bbox[2])
            else:
                chunks.append((cx0, cx1))
                cx0, cx1 = w.bbox[0], w.bbox[2]
        chunks.append((cx0, cx1))
        line_chunks.append(chunks)

    xs = sorted({x for chs in line_chunks for cx0, cx1 in chs for x in (cx0, cx1)})
    if len(xs) < 2:
        return []

    gutters: list[tuple[float, float, float]] = []
    seen_runs: set[tuple[float, float, int, int]] = set()
    min_side_w = 4.0 * ref_fs
    n_lines = len(sorted_lines)

    def _emit_run(mx: float, s_idx: int, e_idx: int) -> None:
        if e_idx - s_idx + 1 < 4:
            return
        gy0 = min(sorted_lines[i][0] for i in range(s_idx, e_idx + 1))
        gy1 = max(sorted_lines[i][1] for i in range(s_idx, e_idx + 1))
        if gy1 - gy0 < 5.0 * ref_fs:
            return
        left_wide = right_wide = 0
        ly0_min, ly1_max = float("inf"), float("-inf")
        ry0_min, ry1_max = float("inf"), float("-inf")
        band_x0, band_x1 = float("-inf"), float("inf")
        for i in range(s_idx, e_idx + 1):
            y0, y1, _ = sorted_lines[i]
            l_segs = [c for c in line_chunks[i] if c[1] <= mx]
            r_segs = [c for c in line_chunks[i] if c[0] >= mx]
            if l_segs:
                ly0_min = min(ly0_min, y0)
                ly1_max = max(ly1_max, y1)
                band_x0 = max(band_x0, max(c[1] for c in l_segs))
                if any(c[1] - c[0] >= min_side_w for c in l_segs):
                    left_wide += 1
            if r_segs:
                ry0_min = min(ry0_min, y0)
                ry1_max = max(ry1_max, y1)
                band_x1 = min(band_x1, min(c[0] for c in r_segs))
                if any(c[1] - c[0] >= min_side_w for c in r_segs):
                    right_wide += 1
        if left_wide < 4 or right_wide < 4:
            return
        if min(ly1_max, ry1_max) - max(ly0_min, ry0_min) < 3.0 * ref_fs:
            return
        if band_x1 - band_x0 < gutter_min_w:
            return
        key = (round(band_x0, 2), round(band_x1, 2), s_idx, e_idx)
        if key not in seen_runs:
            seen_runs.add(key)
            gutters.append((0.5 * (band_x0 + band_x1), gy0, gy1))

    for k in range(len(xs) - 1):
        if xs[k + 1] - xs[k] <= 1e-4:
            continue
        mx = 0.5 * (xs[k] + xs[k + 1])
        run_start: int | None = None
        for li in range(n_lines):
            bridged = any(cx0 < mx < cx1 for cx0, cx1 in line_chunks[li])
            if not bridged:
                if run_start is None:
                    run_start = li
            else:
                if run_start is not None:
                    _emit_run(mx, run_start, li - 1)
                    run_start = None
        if run_start is not None:
            _emit_run(mx, run_start, n_lines - 1)
    return gutters


def _words_to_raw_lines(
    words: Sequence[Word],
    *,
    median_font_size: float = 0.0,
) -> list[list[tuple[BBox, list[Word]]]]:
    if not words:
        return []
    ref_fs = median_font_size if median_font_size > 0.0 else _median_font_size(words)
    lines = cluster_by_overlap(words, lambda w: (w.bbox[1], w.bbox[3]), lambda w: w.bbox[0])

    sorted_lines: list[tuple[float, float, list[Word]]] = []
    space_samples: list[float] = []
    for line in lines:
        line.sort(key=lambda w: w.bbox[0])
        ly0 = min(w.bbox[1] for w in line)
        ly1 = max(w.bbox[3] for w in line)
        sorted_lines.append((ly0, ly1, line))
        for idx in range(1, len(line)):
            gap = line[idx].bbox[0] - line[idx - 1].bbox[2]
            if 0.0 < gap <= 1.0 * ref_fs:
                space_samples.append(gap)
    sorted_lines.sort(key=lambda it: (it[0], it[2][0].bbox[0]))

    if len(space_samples) >= 10:
        space_samples.sort()
        med_space = min(space_samples[len(space_samples) // 2], 0.45 * ref_fs)
    else:
        med_space = 0.28 * ref_fs
    gutter_min_w = max(0.95 * ref_fs, 2.8 * med_space)
    gutters = _detect_vertical_gutters(sorted_lines, ref_fs, gutter_min_w)

    raw_lines: list[list[tuple[BBox, list[Word]]]] = []
    for ly0, ly1, line in sorted_lines:
        line_cy = 0.5 * (ly0 + ly1)
        active_gutters = [
            gx for gx, gy0, gy1 in gutters if gy0 - 1.5 * ref_fs <= line_cy <= gy1 + 1.5 * ref_fs
        ]
        segs: list[tuple[BBox, list[Word]]] = []
        cur: list[Word] = [line[0]]
        for word in line[1:]:
            prev = cur[-1]
            ref_size = max(prev.font_size, word.font_size, prev.bbox[3] - prev.bbox[1], 0.6 * ref_fs, 1.0)
            gap_thr = max(1.8 * ref_size, 1.8 * ref_fs)
            gap = word.bbox[0] - prev.bbox[2]
            hit_gutter = bool(
                active_gutters
                and gap >= gutter_min_w
                and any(prev.bbox[2] <= gx <= word.bbox[0] for gx in active_gutters)
            )
            if gap > gap_thr or hit_gutter:
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
    label: str | None = None,
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
                    blocks.append(_make_text_block([cur], source=source, confidence=confidence, label=label))
                    cur = [word]
                else:
                    cur.append(word)
            blocks.append(_make_text_block([cur], source=source, confidence=confidence, label=label))
    return blocks


def _segment_style(words: Sequence[Word]) -> tuple[float, bool]:
    if not words:
        return 0.0, False
    sizes = sorted(w.font_size for w in words)
    median_size = sizes[len(sizes) // 2]
    bold_chars = sum(len(w.text) for w in words if w.bold)
    total_chars = sum(len(w.text) for w in words)
    return round(median_size, 2), (total_chars > 0 and bold_chars >= 0.5 * total_chars)


def _is_monospace_line(words: Sequence[Word]) -> bool:
    total_chars = sum(len(w.text) for w in words)
    if total_chars <= 0:
        return False
    mono_chars = sum(len(w.text) for w in words if getattr(w, "monospace", False))
    return mono_chars >= 0.5 * total_chars


def _make_text_block(
    lines: Sequence[Sequence[Word]],
    *,
    source: TextSource,
    confidence: float,
    label: str | None = None,
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
        label=label,
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
    cands: list[BBox] = []
    for rx0, ry0, rx1, ry1 in horiz_rules:
        rw = rx1 - rx0
        if (ry1 - ry0) > max_h or (page_w > 0.0 and rw >= 0.9 * page_w):
            continue
        if ry0 < y_lo or ry1 > y_hi:
            continue
        if rx0 >= lb[2] - 0.8 * ref_fs and rx1 <= rb[0] + 0.8 * ref_fs:
            if rw >= 0.4 * gap_w:
                return True
            cands.append((rx0, ry0, rx1, ry1))
    if len(cands) < 3:
        return False
    cands.sort(key=lambda r: (r[0], r[2]))
    dy_tol = 0.25 * ref_fs
    gap_tol = 0.6 * ref_fs
    chains: list[list[float]] = []
    for rx0, ry0, rx1, ry1 in cands:
        rcy = 0.5 * (ry0 + ry1)
        extended = False
        for ch in chains:
            if abs(rcy - ch[2]) <= dy_tol and rx0 - ch[1] <= gap_tol:
                ch[1] = max(ch[1], rx1)
                ch[3] += 1
                if ch[3] >= 3 and (ch[1] - ch[0]) >= 0.4 * gap_w:
                    return True
                extended = True
                break
        if not extended:
            chains.append([rx0, rx1, rcy, 1])
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
        band_y0 = min(line_bounds[idx][0] for idx in matching_lines) - 3.0 * ref_fs
        band_y1 = max(line_bounds[idx][1] for idx in matching_lines) + 3.0 * ref_fs
        for w_idx, wb in wide_segs:
            if line_bounds[w_idx][1] < band_y0 or line_bounds[w_idx][0] > band_y1:
                continue
            if (wb[2] - wb[0]) > 6.0 * ref_fs and abs(wb[2] - rb[2]) <= align_tol and wb[0] < rb[0] - 2.0 * ref_fs:
                return False
            if abs(wb[0] - rb[0]) <= align_tol:
                return False
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


def _ends_sentence(line_words: Sequence[Word]) -> bool:
    if not line_words:
        return False
    tail = unicodedata.normalize("NFKC", "".join(w.text for w in line_words[-2:])).rstrip(" \t\r\n\"'”’」』）)]")
    return bool(tail) and tail[-1] in ".?!。？！"


def _split_into_column_groups(
    items: Sequence[tuple[int, BBox, list[Word], float | None]],
    ref_fs: float,
) -> list[list[tuple[int, BBox, list[Word], float | None]]]:
    if len(items) <= 2:
        return [list(items)]
    x_inset_max = 0.384 * ref_fs
    x_gap_tol = 0.048 * ref_fs
    y_overlap_min = 0.48 * ref_fs
    y_gap_tol = 0.096 * ref_fs

    def _split_x(
        grp: list[tuple[int, BBox, list[Word], float | None]],
    ) -> list[list[tuple[int, BBox, list[Word], float | None]]] | None:
        x_sorted = sorted(grp, key=lambda p: (p[1][0], p[1][1]))
        cols: list[list[tuple[int, BBox, list[Word], float | None]]] = []
        cur_col = [x_sorted[0]]
        b0 = x_sorted[0][1]
        cur_max_x = b0[2] - min(x_inset_max, 0.05 * max(0.0, b0[2] - b0[0]))
        for item in x_sorted[1:]:
            b = item[1]
            ins = min(x_inset_max, 0.05 * max(0.0, b[2] - b[0]))
            if b[0] + ins >= cur_max_x - x_gap_tol:
                cols.append(cur_col)
                cur_col = [item]
                cur_max_x = b[2] - ins
            else:
                cur_col.append(item)
                cur_max_x = max(cur_max_x, b[2] - ins)
        cols.append(cur_col)
        if len(cols) < 2:
            return None
        for idx in range(len(cols) - 1):
            col_a, col_b = cols[idx], cols[idx + 1]
            min_ya, max_ya = min(p[1][1] for p in col_a), max(p[1][3] for p in col_a)
            min_yb, max_yb = min(p[1][1] for p in col_b), max(p[1][3] for p in col_b)
            na = sum(1 for p in col_a if min(p[1][3], max_yb) - max(p[1][1], min_yb) > y_overlap_min)
            nb = sum(1 for p in col_b if min(p[1][3], max_ya) - max(p[1][1], min_ya) > y_overlap_min)
            if na >= 1 and nb >= 1 and (na + nb) >= 3:
                return cols
        return None

    def _find_mid_span(
        grp: list[tuple[int, BBox, list[Word], float | None]],
        group_w: float,
    ) -> tuple[
        list[tuple[int, BBox, list[Word], float | None]],
        list[tuple[int, BBox, list[Word], float | None]],
        list[tuple[int, BBox, list[Word], float | None]],
    ] | None:
        y1s = sorted(p[1][3] for p in grp)
        y0s = sorted(p[1][1] for p in grp)
        n = len(grp)
        best_score: tuple[int, float, float] | None = None
        best_cand: tuple[int, BBox, list[Word], float | None] | None = None
        for cand in grp:
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
        upper: list[tuple[int, BBox, list[Word], float | None]] = []
        middle: list[tuple[int, BBox, list[Word], float | None]] = [best_cand]
        lower: list[tuple[int, BBox, list[Word], float | None]] = []
        for p in grp:
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

    def _recurse(
        grp: list[tuple[int, BBox, list[Word], float | None]],
        depth: int,
    ) -> list[list[tuple[int, BBox, list[Word], float | None]]]:
        if len(grp) <= 2 or depth >= 16:
            return [grp]
        cols = _split_x(grp)
        if cols is not None and all(len(c) < len(grp) for c in cols):
            out: list[list[tuple[int, BBox, list[Word], float | None]]] = []
            for c in cols:
                out.extend(_recurse(c, depth + 1))
            return out
        group_w = max(p[1][2] for p in grp) - min(p[1][0] for p in grp)
        if len(grp) >= 3 and group_w > 0.0:
            mid_split = _find_mid_span(grp, group_w)
            if mid_split is not None:
                upper, middle, lower = mid_split
                if (
                    all(len(part) < len(grp) for part in (upper, middle, lower))
                    and ((len(upper) >= 3 and _split_x(upper) is not None) or (len(lower) >= 3 and _split_x(lower) is not None))
                ):
                    return _recurse(upper, depth + 1) + _recurse(middle, depth + 1) + _recurse(lower, depth + 1)
        y_sorted = sorted(grp, key=lambda p: (p[1][1], p[1][0]))
        bands: list[list[tuple[int, BBox, list[Word], float | None]]] = []
        cur_band = [y_sorted[0]]
        cur_max_y = y_sorted[0][1][3]
        for item in y_sorted[1:]:
            if item[1][1] >= cur_max_y - y_gap_tol:
                bands.append(cur_band)
                cur_band = [item]
                cur_max_y = item[1][3]
            else:
                cur_band.append(item)
                cur_max_y = max(cur_max_y, item[1][3])
        bands.append(cur_band)
        if len(bands) >= 2:
            rest_after_top = [p for b in bands[1:] for p in b]
            if len(rest_after_top) >= 3 and _split_x(rest_after_top) is not None:
                return _recurse(bands[0], depth + 1) + _recurse(rest_after_top, depth + 1)
            rest_before_bot = [p for b in bands[:-1] for p in b]
            if len(rest_before_bot) >= 3 and _split_x(rest_before_bot) is not None:
                return _recurse(rest_before_bot, depth + 1) + _recurse(bands[-1], depth + 1)
            if len(bands) >= 3:
                mid_items = [p for b in bands[1:-1] for p in b]
                if len(mid_items) >= 3 and _split_x(mid_items) is not None:
                    return _recurse(bands[0], depth + 1) + _recurse(mid_items, depth + 1) + _recurse(bands[-1], depth + 1)
        return [grp]

    return _recurse(list(items), 0)


def _compute_column_stats(
    col_groups: Sequence[Sequence[tuple[int, BBox, list[Word], float | None]]],
    ref_fs: float,
) -> tuple[dict[int, int], dict[int, tuple[float, float, float, float, float]]]:
    seg_col_id: dict[int, int] = {}
    raw_gaps_by_col: dict[int, list[float]] = {}
    all_gaps: list[float] = []
    geom_by_col: dict[int, tuple[float, float, float, float]] = {}

    for cid, grp in enumerate(col_groups):
        for idx, _, _, _ in grp:
            seg_col_id[idx] = cid
        x0s = sorted(it[1][0] for it in grp)
        x1s = sorted(it[1][2] for it in grp)
        widths = sorted(it[1][2] - it[1][0] for it in grp)
        n = len(grp)
        col_x0 = x0s[round(0.15 * (n - 1))]
        col_x1 = x1s[round(0.90 * (n - 1))]
        col_w = widths[round(0.90 * (n - 1))]
        geom_by_col[cid] = (col_x0, col_x1, col_w, 0.5 * (col_x0 + col_x1))

        y_sorted = sorted(grp, key=lambda it: (it[1][1], it[1][0]))
        c_gaps: list[float] = []
        for k in range(1, len(y_sorted)):
            prev_box, prev_words = y_sorted[k - 1][1], y_sorted[k - 1][2]
            cur_box, cur_words = y_sorted[k][1], y_sorted[k][2]
            prev_fs, _ = _segment_style(prev_words)
            cur_fs, _ = _segment_style(cur_words)
            fs_tol = max(0.1 * max(cur_fs, prev_fs, ref_fs), 0.5)
            gap_y = cur_box[1] - prev_box[3]
            if 0.0 < gap_y <= 2.5 * ref_fs and abs(cur_fs - prev_fs) <= fs_tol and _horiz_overlap(prev_box, cur_box):
                c_gaps.append(gap_y)
                all_gaps.append(gap_y)
        raw_gaps_by_col[cid] = c_gaps

    if all_gaps:
        all_gaps.sort()
        page_gap = max(0.15 * ref_fs, min(1.25 * ref_fs, all_gaps[round(0.35 * (len(all_gaps) - 1))]))
    else:
        page_gap = 0.25 * ref_fs

    stats: dict[int, tuple[float, float, float, float, float]] = {}
    for cid, geom in geom_by_col.items():
        c_gaps = raw_gaps_by_col[cid]
        if c_gaps:
            c_gaps.sort()
            t_gap = max(0.15 * ref_fs, min(1.25 * ref_fs, c_gaps[round(0.35 * (len(c_gaps) - 1))]))
        else:
            t_gap = page_gap
        stats[cid] = (*geom, t_gap)
    return seg_col_id, stats


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
    indexed = [(idx, s[0], s[1], s[2]) for idx, s in enumerate(ordered)]
    col_groups = _split_into_column_groups(indexed, ref_fs)
    seg_col_id, col_stats = _compute_column_stats(col_groups, ref_fs)

    groups: list[list[tuple[BBox, list[Word], float, bool, bool, float | None]]] = []
    for idx, bbox, line_words, body_x0 in indexed:
        col_x0, col_x1, col_w, col_cx, typical_gap = col_stats[seg_col_id[idx]]
        fs, bold = _segment_style(line_words)
        all_bold = bool(line_words) and all(w.bold for w in line_words)
        lh = max(1.0, bbox[3] - bbox[1])
        is_list_start = _is_list_item_start(line_words) or body_x0 is not None
        merged = False
        if not is_list_start:
            cur_cx = 0.5 * (bbox[0] + bbox[2])
            is_centered = (
                abs(cur_cx - col_cx) <= 0.6 * ref_fs
                and bbox[0] > col_x0 + 1.5 * ref_fs
                and bbox[2] < col_x1 - 1.5 * ref_fs
            )
            for group in reversed(groups):
                first_box, first_words, _, _, _, first_body_x0 = group[0]
                last_box, last_words, last_fs, last_bold, _, _ = group[-1]
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
                max_lh = max(lh, last_box[3] - last_box[1])
                max_gap = max(0.8 * max_lh, min(1.30 * max_lh, 1.25 * typical_gap), 0.6 * ref_fs, 1.0)
                fs_tol = max(0.1 * max(fs, last_fs, ref_fs), 0.5)
                group_all_bold = all(item[4] for item in group)
                bold_ok = (bold == last_bold) or (not group_all_bold and not all_bold)
                if not (-0.3 * lh <= gap_y <= max_gap and abs(fs - last_fs) <= fs_tol and bold_ok):
                    continue
                last_w = last_box[2] - last_box[0]
                last_ends = _ends_sentence(last_words)
                both_mono = _is_monospace_line(last_words) and _is_monospace_line(line_words)
                allow_indented_cont = (
                    (both_mono and gap_y <= max(1.25 * typical_gap, 0.35 * max_lh))
                    or (last_box[0] >= col_x0 + 0.6 * ref_fs and not last_ends)
                )
                hanging_first_wrap = (
                    len(group) == 1
                    and last_w >= 0.80 * col_w
                    and ((not last_ends) or _is_list_item_start(first_words))
                )
                first_line_indent_break = (
                    not allow_indented_cont
                    and not has_hanging
                    and not hanging_first_wrap
                    and not is_centered
                    and bbox[0] >= col_x0 + 0.6 * ref_fs
                    and (last_ends or last_box[2] < col_x1 - 1.5 * ref_fs)
                )
                short_last_line_break = (
                    not allow_indented_cont
                    and last_ends
                    and last_w < 0.78 * col_w
                    and gap_y > 1.05 * typical_gap
                )
                if first_line_indent_break or short_last_line_break:
                    continue
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


def _region_words_to_blocks(
    words: Sequence[Word],
    regions: Sequence[Any],
    page_rect: BBox | None,
    *,
    source: TextSource,
    confidence: float,
    median_font_size: float,
    horiz_rules: Sequence[BBox] = (),
) -> list[TextBlock]:
    parsed: list[tuple[int, float, BBox, str, float]] = []
    for idx, reg in enumerate(regions):
        if isinstance(reg, Mapping):
            rb = reg.get("bbox")
            lbl = reg.get("label")
            conf = float(reg.get("confidence", 1.0))
        else:
            rb = getattr(reg, "bbox", None)
            lbl = getattr(reg, "label", None)
            conf = float(getattr(reg, "confidence", 1.0))
        if not rb or len(rb) != 4 or not lbl:
            continue
        box: BBox = (float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3]))
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area <= 0.0:
            continue
        parsed.append((idx, area, box, str(lbl).lower(), conf))

    if not parsed:
        return order_blocks(
            words,
            page_rect,
            source=source,
            confidence=confidence,
            horiz_rules=horiz_rules,
        )

    assigned: dict[int, list[Word]] = {r[0]: [] for r in parsed}
    orphan_words: list[Word] = []
    for word in words:
        wx0, wy0, wx1, wy1 = word.bbox
        w_area = max(0.0, wx1 - wx0) * max(0.0, wy1 - wy0)
        cx = 0.5 * (wx0 + wx1)
        cy = 0.5 * (wy0 + wy1)
        best_idx: int | None = None
        best_key: tuple[float, float, float, int] | None = None
        for r_idx, r_area, (rx0, ry0, rx1, ry1), _lbl, r_conf in parsed:
            center_in = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
            if w_area > 0.0:
                ix = max(0.0, min(wx1, rx1) - max(wx0, rx0))
                iy = max(0.0, min(wy1, ry1) - max(wy0, ry0))
                ioa = (ix * iy) / w_area
                if ioa < 0.50 and not center_in:
                    continue
            elif not center_in:
                continue
            else:
                ioa = 1.0
            key = (r_area, -ioa, -r_conf, r_idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = r_idx
        if best_idx is None:
            orphan_words.append(word)
        else:
            assigned[best_idx].append(word)

    blocks: list[TextBlock] = []
    for r_idx, _area, _box, r_label, _conf in parsed:
        reg_words = assigned[r_idx]
        if not reg_words:
            continue
        h_words, v_words = _partition_orientation(reg_words)
        if h_words:
            raw_lines = cluster_by_overlap(h_words, lambda w: (w.bbox[1], w.bbox[3]), lambda w: w.bbox[0])
            lines = [sorted(ln, key=lambda w: w.bbox[0]) for ln in raw_lines]
            lines.sort(key=lambda ln: (min(w.bbox[1] for w in ln), ln[0].bbox[0]))
            blocks.append(_make_text_block(lines, source=source, confidence=confidence, label=r_label))
        if v_words:
            blocks.extend(
                _vertical_words_to_blocks(
                    v_words,
                    source=source,
                    confidence=confidence,
                    median_font_size=median_font_size,
                    label=r_label,
                )
            )

    if orphan_words:
        blocks.extend(
            order_blocks(
                orphan_words,
                page_rect,
                source=source,
                confidence=confidence,
                horiz_rules=horiz_rules,
            )
        )
    return xy_cut_order(blocks, lambda b: b.bbox, page_rect=page_rect, font_size=median_font_size)


def order_blocks(
    words: Sequence[Word],
    page_rect: BBox | None = None,
    *,
    source: TextSource = "text_layer",
    confidence: float = 1.0,
    horiz_rules: Sequence[BBox] = (),
    regions: Sequence[Any] | None = None,
) -> list[TextBlock]:
    if not words:
        return []
    median_fs = _median_font_size(words, page_rect)
    if regions:
        return _region_words_to_blocks(
            words,
            regions,
            page_rect,
            source=source,
            confidence=confidence,
            median_font_size=median_fs,
            horiz_rules=horiz_rules,
        )
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
