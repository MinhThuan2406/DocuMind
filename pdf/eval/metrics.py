from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Sequence

_HTML_TAG_RE = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col|br|hr|p|div|span|sup|sub|b|i|em|strong|u|ul|ol|li|pre|code|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)


def area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def inter(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def match_count(preds: Sequence[Sequence[float]], gts: Sequence[Sequence[float]], mode: str) -> int:
    if mode == "strict":
        cand = sorted(
            (-iou(p, g), i, j)
            for i, p in enumerate(preds)
            for j, g in enumerate(gts)
            if iou(p, g) >= 0.5
        )
    elif mode == "lenient":
        cand = sorted(
            (area(p) / area(g), i, j)
            for i, p in enumerate(preds)
            for j, g in enumerate(gts)
            if area(g) > 0 and inter(p, g) >= 0.95 * area(g) and area(p) <= 2.0 * area(g)
        )
    else:
        raise ValueError(f"unknown match mode: {mode!r}")

    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, i, j in cand:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
    return len(used_p)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2.0 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def strip_markup(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\[a-zA-Z]+\*?", " ", cleaned)
    cleaned = re.sub(r"\\[\[\]\(\)\\\{\}\$]", " ", cleaned)
    return cleaned.replace("$", " ").replace("{", " ").replace("}", " ")


def char_counts(text: str) -> Counter[str]:
    return Counter("".join(text.split()))


def median(vals: Sequence[float]) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) * 0.5
