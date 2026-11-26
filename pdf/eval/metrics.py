from __future__ import annotations

import html
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

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


def matched_pred_indices(
    preds: Sequence[Sequence[float]],
    gts: Sequence[Sequence[float]],
    mode: str = "strict",
) -> set[int]:
    if mode == "strict":
        cand = sorted(
            (-score, i, j)
            for i, p in enumerate(preds)
            for j, g in enumerate(gts)
            if (score := iou(p, g)) >= 0.5
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
    return used_p


def match_count(preds: Sequence[Sequence[float]], gts: Sequence[Sequence[float]], mode: str) -> int:
    return len(matched_pred_indices(preds, gts, mode))


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


def _accumulate_calibration_bins(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    n_bins: int,
) -> list[tuple[int, float, float]]:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(confidences) != len(labels):
        raise ValueError(f"length mismatch: {len(confidences)} confidences vs {len(labels)} labels")

    counts = [0] * n_bins
    sum_confs = [0.0] * n_bins
    sum_labels = [0.0] * n_bins
    for raw_c, raw_y in zip(confidences, labels):
        if isinstance(raw_c, bool):
            raise ValueError(f"confidence must be a number, got {raw_c!r}")
        c = float(raw_c)
        if not math.isfinite(c):
            raise ValueError(f"non-finite confidence: {raw_c!r}")
        c = min(1.0, max(0.0, c))
        if isinstance(raw_y, bool):
            y = 1.0 if raw_y else 0.0
        elif isinstance(raw_y, int) and raw_y in (0, 1):
            y = float(raw_y)
        else:
            raise ValueError(f"label must be 0/1 or bool, got {raw_y!r}")
        idx = min(n_bins - 1, max(0, int(c * n_bins)))
        counts[idx] += 1
        sum_confs[idx] += c
        sum_labels[idx] += y
    return list(zip(counts, sum_confs, sum_labels))


def calibration_bins(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    raw_bins = _accumulate_calibration_bins(confidences, labels, n_bins)
    out: list[dict[str, Any]] = []
    for idx, (count, sum_c, sum_y) in enumerate(raw_bins):
        if count > 0:
            acc = sum_y / count
            avg_c = sum_c / count
            gap = abs(acc - avg_c)
            out.append({
                "bin": idx,
                "lower": round(idx / n_bins, 6),
                "upper": round((idx + 1) / n_bins, 6),
                "count": count,
                "accuracy": round(acc, 6),
                "avg_confidence": round(avg_c, 6),
                "gap": round(gap, 6),
            })
        else:
            out.append({
                "bin": idx,
                "lower": round(idx / n_bins, 6),
                "upper": round((idx + 1) / n_bins, 6),
                "count": 0,
                "accuracy": None,
                "avg_confidence": None,
                "gap": 0.0,
            })
    return out


def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    n_bins: int = 10,
) -> float:
    raw_bins = _accumulate_calibration_bins(confidences, labels, n_bins)
    total = len(confidences)
    if total == 0:
        return 0.0
    ece = sum(abs(sum_y - sum_c) for count, sum_c, sum_y in raw_bins if count > 0) / total
    return round(ece, 6)


def confidence_separation_summary(
    tp_confs: Sequence[float],
    fp_confs: Sequence[float],
    *,
    high_conf_threshold: float = 0.95,
    low_conf_threshold: float = 0.70,
) -> dict[str, Any]:
    tp: list[float] = []
    fp: list[float] = []
    for raw_c in tp_confs:
        if isinstance(raw_c, bool) or not math.isfinite(float(raw_c)):
            raise ValueError(f"invalid confidence: {raw_c!r}")
        tp.append(float(raw_c))
    for raw_c in fp_confs:
        if isinstance(raw_c, bool) or not math.isfinite(float(raw_c)):
            raise ValueError(f"invalid confidence: {raw_c!r}")
        fp.append(float(raw_c))

    raw_mean_tp = (sum(tp) / len(tp)) if tp else None
    raw_mean_fp = (sum(fp) / len(fp)) if fp else None
    mean_tp = round(raw_mean_tp, 6) if raw_mean_tp is not None else None
    mean_fp = round(raw_mean_fp, 6) if raw_mean_fp is not None else None
    mean_tp_gt_mean_fp = (raw_mean_tp > raw_mean_fp) if (raw_mean_tp is not None and raw_mean_fp is not None) else None
    fp_high_conf_count = sum(1 for c in fp if c >= high_conf_threshold)
    fp_high_conf_rate = round(fp_high_conf_count / len(fp), 6) if fp else 0.0
    tp_low_conf_count = sum(1 for c in tp if c < low_conf_threshold)
    fp_low_conf_count = sum(1 for c in fp if c < low_conf_threshold)
    return {
        "tp_count": len(tp),
        "fp_count": len(fp),
        "mean_tp": mean_tp,
        "mean_fp": mean_fp,
        "mean_tp_gt_mean_fp": mean_tp_gt_mean_fp,
        "fp_high_conf_count": fp_high_conf_count,
        "fp_high_conf_rate": fp_high_conf_rate,
        "low_conf_count": tp_low_conf_count + fp_low_conf_count,
        "tp_low_conf_count": tp_low_conf_count,
        "fp_low_conf_count": fp_low_conf_count,
        "high_conf_threshold": high_conf_threshold,
        "low_conf_threshold": low_conf_threshold,
    }
