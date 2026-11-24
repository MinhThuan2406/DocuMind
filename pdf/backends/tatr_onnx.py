from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import BBox, ExtractedTable, TableCell, Word
from .base import RunStats
from .rss import peak_rss_mb

TATR_ONNX_REPO_ID = "Xenova/table-transformer-structure-recognition-v1.1-all"
TATR_ONNX_REVISION = "76c3afa174a0f36bafa9438934f57dcdb6e486a5"
DEFAULT_THRESHOLD = 0.60
_SPAN_THRESHOLD = 0.72
_SPAN_GUARD_MIN = 0.58
_HEADER_RESCUE_MIN = 0.35
_RESCUE_THRESHOLD = 0.30
_SPAN_OVERLAP_FRAC = 0.55
_LONGEST_EDGE = 800
_MAX_TATR_CONF = 0.96
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
_MULTICELL_SPLIT_RE = re.compile(r"\s+|(?<=\.\.)(?=[^.\s])|(?<=…)(?=\S)")
_NUMERIC_VALUE_RE = re.compile(r"^[-+~$€£¥]?\d[\d,./:%()-]*$")
_CATEGORICAL_VALUES = frozenset({
    "yes", "no", "true", "false", "pass", "fail", "ok", "ng", "n/a", "na", "none", "high", "low", "medium", "on", "off",
})


def _resolve_model_folder(model_path: str | Path | None, repo_id: str, revision: str) -> Path:
    if model_path is not None:
        return Path(model_path)
    from huggingface_hub import snapshot_download

    patterns = ["config.json", "preprocessor_config.json", "onnx/model.onnx"]
    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    except Exception:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=patterns))


def _load_id2label(model_folder: Path) -> dict[int, str]:
    raw = json.loads((model_folder / "config.json").read_text(encoding="utf-8")).get("id2label", {})
    return {int(k): str(v).lower() for k, v in raw.items()}


def _dedupe_1d_intervals(
    boxes: list[tuple[BBox, float]],
    axis: int,
) -> list[tuple[float, float, float]]:
    # Deduplicate overlapping row (axis=1) or column (axis=0) spans by confidence, then snap gaps.
    sorted_by_conf = sorted(boxes, key=lambda item: item[1], reverse=True)
    kept: list[tuple[float, float, float]] = []
    for box, score in sorted_by_conf:
        a0, a1 = box[axis], box[axis + 2]
        alen = max(0.0, a1 - a0)
        if alen <= 0.0:
            continue
        overlaps = False
        for k0, k1, _ in kept:
            inter = min(a1, k1) - max(a0, k0)
            if inter >= 0.5 * min(alen, k1 - k0):
                overlaps = True
                break
        if not overlaps:
            kept.append((a0, a1, score))
    kept.sort(key=lambda item: item[0])
    return kept


def _cluster_1d_tokens(
    tok_px: Sequence[BBox],
    axis: int,
    limit: float,
) -> list[tuple[float, float, float]]:
    if limit <= 0.0:
        return []
    items = sorted(
        (max(0.0, float(b[axis])), min(float(limit), float(b[axis + 2])))
        for b in tok_px
        if min(float(limit), float(b[axis + 2])) > max(0.0, float(b[axis]))
    )
    clusters: list[list[float]] = []
    for a0, a1 in items:
        if not clusters:
            clusters.append([a0, a1])
            continue
        c0, c1 = clusters[-1]
        ov = min(a1, c1) - max(a0, c0)
        if ov >= 0.35 * min(a1 - a0, c1 - c0):
            clusters[-1][0] = min(c0, a0)
            clusters[-1][1] = max(c1, a1)
        else:
            clusters.append([a0, a1])
    return [(c[0], c[1], 0.6) for c in clusters if c[1] > c[0]]


def _snap_intervals(
    intervals: list[tuple[float, float, float]],
    limit: float,
    tok_spans: Sequence[tuple[float, float]] = (),
    *,
    ignore_cross_spans: bool = False,
) -> list[tuple[float, float]]:
    if not intervals or limit <= 0.0:
        return []
    cleaned = sorted(
        (max(0.0, float(a0)), min(float(limit), float(a1)))
        for a0, a1, _ in intervals
        if min(float(limit), float(a1)) > max(0.0, float(a0))
    )
    if not cleaned:
        return []
    n = len(cleaned)
    min_step = min(1.0, float(limit) / (2.0 * n))
    bounds = [0.0]
    for idx in range(n - 1):
        rem = n - 1 - idx
        min_b = bounds[-1] + min_step
        max_b = float(limit) - rem * min_step
        mid = max(min_b, min(max_b, (cleaned[idx][1] + cleaned[idx + 1][0]) * 0.5))
        if ignore_cross_spans and tok_spans:
            c0_mid = (cleaned[idx][0] + cleaned[idx][1]) * 0.5
            c1_mid = (cleaned[idx + 1][0] + cleaned[idx + 1][1]) * 0.5
            local_spans: Sequence[tuple[float, float]] = [
                (t0, t1) for t0, t1 in tok_spans if not (t0 < c0_mid and t1 > c1_mid)
            ]
        else:
            local_spans = tok_spans
        if local_spans and any(t0 + 1.0 < mid < t1 - 1.0 for t0, t1 in local_spans):
            w_prev = max(1.0, cleaned[idx][1] - cleaned[idx][0])
            w_next = max(1.0, cleaned[idx + 1][1] - cleaned[idx + 1][0])
            lo_win = max(min_b, cleaned[idx][1] - 0.35 * w_prev, mid - 0.45 * min(w_prev, w_next))
            hi_win = min(max_b, cleaned[idx + 1][0] + 0.35 * w_next, mid + 0.45 * min(w_prev, w_next))
            if hi_win > lo_win:
                pts = sorted(
                    {mid, lo_win, hi_win}
                    | {edge for t0, t1 in local_spans for edge in (t0, t1) if lo_win <= edge <= hi_win}
                )
                cands = list(pts) + [(pts[k] + pts[k + 1]) * 0.5 for k in range(len(pts) - 1)]
                best_x = mid
                best_key = (sum(1 for t0, t1 in local_spans if t0 + 1.0 < mid < t1 - 1.0), 0.0)
                for x in cands:
                    cuts = sum(1 for t0, t1 in local_spans if t0 + 1.0 < x < t1 - 1.0)
                    key = (cuts, abs(x - mid))
                    if key < best_key:
                        best_key = key
                        best_x = x
                mid = max(min_b, min(max_b, best_x))
        bounds.append(mid)
    bounds.append(float(limit))
    return [(bounds[idx], bounds[idx + 1]) for idx in range(n)]


def _overlap_indices(
    spans: Sequence[tuple[float, float]],
    lo: float,
    hi: float,
    min_frac: float = 0.5,
) -> list[int]:
    matched: list[int] = []
    for idx, (s0, s1) in enumerate(spans):
        slen = s1 - s0
        if slen <= 0.0:
            continue
        inter = min(hi, s1) - max(lo, s0)
        if inter >= min_frac * slen:
            matched.append(idx)
    return matched


def _sort_cell_tokens(tokens: list[tuple[str, BBox]]) -> str:
    if not tokens:
        return ""
    multi_char = [tb for text, tb in tokens if len(text.strip()) >= 2]
    is_vertical = (
        len(tokens) >= 2
        and bool(multi_char)
        and sum(1 for tb in multi_char if (tb[3] - tb[1]) > 1.25 * max(1.0, tb[2] - tb[0])) >= 0.7 * len(multi_char)
    )
    if is_vertical:
        ordered_v = sorted(tokens, key=lambda item: ((item[1][0] + item[1][2]) * 0.5, -(item[1][1] + item[1][3]) * 0.5))
        cols: list[list[tuple[str, BBox]]] = []
        for tok in ordered_v:
            tx0, tx1 = tok[1][0], tok[1][2]
            tw = max(1.0, tx1 - tx0)
            placed = False
            for col in reversed(cols):
                cx0 = min(t[1][0] for t in col)
                cx1 = max(t[1][2] for t in col)
                if min(tx1, cx1) - max(tx0, cx0) >= 0.45 * min(tw, max(1.0, cx1 - cx0)):
                    col.append(tok)
                    placed = True
                    break
            if not placed:
                cols.append([tok])
        parts_v: list[str] = []
        for col in cols:
            col.sort(key=lambda item: -((item[1][1] + item[1][3]) * 0.5))
            parts_v.append(" ".join(t[0] for t in col if t[0]))
        return " ".join(p for p in parts_v if p).strip()

    ordered = sorted(tokens, key=lambda item: ((item[1][1] + item[1][3]) * 0.5, item[1][0]))
    lines: list[list[tuple[str, BBox]]] = []
    for tok in ordered:
        ty0, ty1 = tok[1][1], tok[1][3]
        th = max(1.0, ty1 - ty0)
        placed = False
        for line in reversed(lines):
            ly0 = min(t[1][1] for t in line)
            ly1 = max(t[1][3] for t in line)
            lh = max(1.0, ly1 - ly0)
            if min(ty1, ly1) - max(ty0, ly0) >= 0.45 * min(th, lh):
                line.append(tok)
                placed = True
                break
        if not placed:
            lines.append([tok])
    parts: list[str] = []
    for line in lines:
        line.sort(key=lambda item: item[1][0])
        parts.append(" ".join(t[0] for t in line if t[0]))
    return " ".join(p for p in parts if p).strip()


def _char_display_width(ch: str) -> float:
    return 2.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 1.0


def _crop_to_gray(crop: Image.Image) -> np.ndarray:
    return np.asarray(crop.convert("L"), dtype=np.uint8)


def _band_ink_runs(
    crop_gray: np.ndarray,
    px0: int,
    py0: int,
    px1: int,
    py1: int,
) -> tuple[list[tuple[int, int]], float]:
    ch, cw = crop_gray.shape[:2]
    x0 = max(0, min(cw, px0))
    x1 = max(0, min(cw, px1))
    y0 = max(0, min(ch, py0))
    y1 = max(0, min(ch, py1))
    line_h = float(max(1, y1 - y0))
    if x1 - x0 < 4 or y1 - y0 < 2:
        return [], line_h
    band = crop_gray[y0:y1, x0:x1].astype(np.int16)
    bg = int(round(float(np.median(band))))
    ink = np.abs(band - bg) >= 35
    keep_rows = ink.mean(axis=1) < 0.80
    if int(keep_rows.sum()) >= max(2, int(0.3 * ink.shape[0])):
        ink = ink[keep_rows]
    # Mask thin continuous vertical border lines extending across row margins.
    margin = max(3, int(round(0.5 * line_h)))
    ey0, ey1 = max(0, y0 - margin), min(ch, y1 + margin)
    if ey1 - ey0 > (y1 - y0) + 2:
        ext_band = crop_gray[ey0:ey1, x0:x1].astype(np.int16)
        ext_bg = np.median(ext_band, axis=1, keepdims=True)
        ext_cols = (np.abs(ext_band - ext_bg) >= 35).mean(axis=0) >= 0.85
        max_rule_w = max(3, int(round(0.2 * line_h)))
        w = ink.shape[1]
        i = 0
        while i < w:
            if ext_cols[i]:
                j = i + 1
                while j < w and ext_cols[j]:
                    j += 1
                if j - i <= max_rule_w:
                    ink[:, i:j] = False
                i = j
            else:
                i += 1
    col_ink = ink.any(axis=0)
    raw_runs: list[tuple[int, int]] = []
    i = 0
    w = len(col_ink)
    while i < w:
        if col_ink[i]:
            j = i + 1
            while j < w and col_ink[j]:
                j += 1
            raw_runs.append((x0 + i, x0 + j))
            i = j
        else:
            i += 1
    if not raw_runs:
        return [], line_h
    merge_gap = max(1, int(round(0.15 * line_h)))
    merged: list[list[int]] = [[raw_runs[0][0], raw_runs[0][1]]]
    for rx0, rx1 in raw_runs[1:]:
        if rx0 - merged[-1][1] <= merge_gap:
            merged[-1][1] = rx1
        else:
            merged.append([rx0, rx1])
    return [(r[0], r[1]) for r in merged], line_h


def _display_width_spans(text: str, pieces: Sequence[str], x0: float, x1: float) -> list[tuple[float, float]]:
    cum_w = [0.0]
    for ch in text:
        cum_w.append(cum_w[-1] + _char_display_width(ch))
    total_w = max(1.0, cum_w[-1])
    tw = x1 - x0
    spans: list[tuple[float, float]] = []
    pos = 0
    for piece in pieces:
        start = text.find(piece, pos)
        end = start + len(piece)
        pos = end
        wx0 = x0 + (cum_w[start] / total_w) * tw
        wx1 = x0 + (cum_w[end] / total_w) * tw
        spans.append((wx0, max(wx0 + 0.5, wx1)))
    return spans


def _assign_pieces_to_ink_runs(
    text: str,
    pieces: Sequence[str],
    runs: Sequence[tuple[int, int]],
    tb: BBox,
    tx0: float,
    sx: float,
) -> list[tuple[float, float]] | None:
    m, n = len(runs), len(pieces)
    if m < 2 or n < 2 or sx <= 0.0:
        return None
    if m == n:
        groups: list[tuple[int, int]] = [(r[0], r[1]) for r in runs]
    elif m > n:
        gaps = sorted(((runs[k + 1][0] - runs[k][1], k) for k in range(m - 1)), reverse=True)
        splits = sorted(idx for _, idx in gaps[: n - 1])
        groups = []
        prev = 0
        for sp in splits:
            groups.append((runs[prev][0], runs[sp][1]))
            prev = sp + 1
        groups.append((runs[prev][0], runs[-1][1]))
    else:
        if n > 24:
            return None
        dws = [max(0.5, sum(_char_display_width(c) for c in p)) for p in pieces]
        total_dw = sum(dws)
        run_ws = [max(1.0, float(r1 - r0)) for r0, r1 in runs]
        total_rw = sum(run_ws)
        dw_pref = [0.0]
        for w in dws:
            dw_pref.append(dw_pref[-1] + w)
        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        back = [[-1] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for mi in range(1, m + 1):
            rw_frac = run_ws[mi - 1] / total_rw
            for ni in range(mi, n - (m - mi) + 1):
                for prev_ni in range(mi - 1, ni):
                    cost = dp[mi - 1][prev_ni] + abs(((dw_pref[ni] - dw_pref[prev_ni]) / total_dw) - rw_frac)
                    if cost < dp[mi][ni]:
                        dp[mi][ni] = cost
                        back[mi][ni] = prev_ni
        bounds: list[int] = [n]
        cur_n = n
        for mi in range(m, 0, -1):
            cur_n = back[mi][cur_n]
            bounds.append(cur_n)
        bounds.reverse()
        out_spans: list[tuple[float, float]] = []
        for mi, (gx0, gx1) in enumerate(runs):
            sub_pieces = pieces[bounds[mi] : bounds[mi + 1]]
            px_lo = max(tb[0], min(tb[2], tx0 + gx0 * sx))
            px_hi = max(px_lo + 0.5, min(tb[2], tx0 + gx1 * sx))
            if len(sub_pieces) == 1:
                out_spans.append((px_lo, px_hi))
            else:
                sub_text = " ".join(sub_pieces)
                out_spans.extend(_display_width_spans(sub_text, sub_pieces, px_lo, px_hi))
        return out_spans
    out_pt: list[tuple[float, float]] = []
    for gx0, gx1 in groups:
        wx0 = max(tb[0], min(tb[2], tx0 + gx0 * sx))
        wx1 = max(wx0 + 0.5, min(tb[2], tx0 + gx1 * sx))
        out_pt.append((wx0, wx1))
    return out_pt


def _split_multicell_tokens(
    tokens: Sequence[tuple[str, BBox]],
    cell_defs: Sequence[tuple[int, int, int, int, BBox, bool, float]],
    *,
    crop_gray: np.ndarray | None = None,
    table_bbox_pt: BBox | None = None,
) -> tuple[list[tuple[str, BBox]], set[tuple[str, BBox]]]:
    out: list[tuple[str, BBox]] = []
    split_pieces: set[tuple[str, BBox]] = set()
    sx = 0.0
    sy = 0.0
    tx0 = ty0 = 0.0
    if crop_gray is not None and table_bbox_pt is not None:
        ch, cw = crop_gray.shape[:2]
        tx0, ty0, tx1, ty1 = table_bbox_pt
        sx = (tx1 - tx0) / cw if cw > 0 else 0.0
        sy = (ty1 - ty0) / ch if ch > 0 else 0.0
    for text, tb in tokens:
        pieces = [p for p in _MULTICELL_SPLIT_RE.split(text) if p]
        if len(pieces) >= 2 and (tb[2] - tb[0]) > 0.0:
            tcy = (tb[1] + tb[3]) * 0.5
            row_hits = sum(
                1
                for _, _, _, _, cb, _, _ in cell_defs
                if cb[1] <= tcy <= cb[3]
                and min(tb[2], cb[2]) - max(tb[0], cb[0]) >= min(0.12 * max(1.0, cb[2] - cb[0]), 4.0)
            )
            if row_hits >= 2:
                piece_spans: list[tuple[float, float]] | None = None
                if crop_gray is not None and sx > 0.0 and sy > 0.0:
                    px0 = int(np.floor((tb[0] - tx0) / sx))
                    py0 = int(np.floor((tb[1] - ty0) / sy))
                    px1 = int(np.ceil((tb[2] - tx0) / sx))
                    py1 = int(np.ceil((tb[3] - ty0) / sy))
                    runs, _ = _band_ink_runs(crop_gray, px0, py0, px1, py1)
                    piece_spans = _assign_pieces_to_ink_runs(text, pieces, runs, tb, tx0, sx)
                if piece_spans is None:
                    piece_spans = _display_width_spans(text, pieces, tb[0], tb[2])
                for piece, (wx0, wx1) in zip(pieces, piece_spans):
                    item = (piece, (wx0, tb[1], max(wx0 + 0.5, wx1), tb[3]))
                    out.append(item)
                    split_pieces.add(item)
                continue
        out.append((text, tb))
    return out, split_pieces


def _split_undersegmented_cols(
    col_intervals: list[tuple[float, float]],
    col_scores: list[float],
    tok_px: Sequence[BBox],
    cw: float,
    hdr_objs: Sequence[BBox],
    span_boxes: Sequence[BBox],
) -> tuple[list[tuple[float, float]], list[float], bool]:
    if not tok_px or not col_intervals:
        return col_intervals, col_scores, False
    min_gutter = max(4.0, 0.015 * cw)
    hdr_y_max = max((hb[3] for hb in hdr_objs), default=-1.0)
    out_cols: list[tuple[float, float]] = []
    out_scores: list[float] = []
    did_split = False
    for (cx0, cx1), score in zip(col_intervals, col_scores):
        cwidth = cx1 - cx0
        inside = [
            b
            for b in tok_px
            if b[0] >= cx0 - 2.0
            and b[2] <= cx1 + 2.0
            and (b[2] - b[0]) < 0.75 * cwidth
            and (b[1] + b[3]) * 0.5 > hdr_y_max
            and not any(
                sb[0] <= (b[0] + b[2]) * 0.5 <= sb[2] and sb[1] <= (b[1] + b[3]) * 0.5 <= sb[3]
                for sb in span_boxes
            )
        ]
        if len(inside) < 4:
            out_cols.append((cx0, cx1))
            out_scores.append(score)
            continue
        xs = sorted((max(cx0, b[0]), min(cx1, b[2])) for b in inside if min(cx1, b[2]) > max(cx0, b[0]))
        merged: list[list[float]] = []
        for a0, a1 in xs:
            if not merged or a0 > merged[-1][1]:
                merged.append([a0, a1])
            else:
                merged[-1][1] = max(merged[-1][1], a1)
        splits: list[float] = []
        for k in range(len(merged) - 1):
            if merged[k + 1][0] - merged[k][1] >= min_gutter:
                mid = (merged[k][1] + merged[k + 1][0]) * 0.5
                if sum(1 for b in inside if b[2] <= mid) >= 2 and sum(1 for b in inside if b[0] >= mid) >= 2:
                    splits.append(mid)
        if not splits:
            out_cols.append((cx0, cx1))
            out_scores.append(score)
        else:
            did_split = True
            pts = [cx0, *splits, cx1]
            sub_score = min(score, 0.85)
            for k in range(len(pts) - 1):
                out_cols.append((pts[k], pts[k + 1]))
                out_scores.append(sub_score)
    return out_cols, out_scores, did_split


def _cells_grid_valid(
    cells: Sequence[TableCell],
    num_rows: int,
    num_cols: int,
    *,
    allow_empty: bool = False,
) -> bool:
    if num_rows <= 0 or num_cols <= 0 or not cells:
        return False
    occupied: set[tuple[int, int]] = set()
    non_empty = 0
    for c in cells:
        if c.row < 0 or c.col < 0 or c.rowspan < 1 or c.colspan < 1:
            return False
        if c.row + c.rowspan > num_rows or c.col + c.colspan > num_cols:
            return False
        for rr in range(c.row, c.row + c.rowspan):
            for cc in range(c.col, c.col + c.colspan):
                if (rr, cc) in occupied:
                    return False
                occupied.add((rr, cc))
        if c.text.strip():
            non_empty += 1
    if not allow_empty and non_empty == 0:
        return False
    return len(occupied) == num_rows * num_cols


def _has_valid_grid(table: ExtractedTable) -> bool:
    return _cells_grid_valid(table.cells, table.num_rows, table.num_cols, allow_empty=False)


def _is_discrete_value_text(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if _NUMERIC_VALUE_RE.fullmatch(s) or s.lower() in _CATEGORICAL_VALUES:
        return True
    if len(s) <= 2 and s.isalnum() and s.isupper():
        return True
    parts = s.split()
    return len(parts) >= 2 and all(bool(_NUMERIC_VALUE_RE.fullmatch(p)) for p in parts)


def _is_wrapped_prose_pair(upper_texts: Sequence[str], lower_texts: Sequence[str]) -> bool:
    u_str = " ".join(t.strip() for t in upper_texts if t.strip()).strip()
    l_str = " ".join(t.strip() for t in lower_texts if t.strip()).strip()
    if not u_str or not l_str:
        return False
    if _is_discrete_value_text(u_str) or _is_discrete_value_text(l_str):
        return False
    if u_str[-1] in ".!?:;":
        return False
    if not (l_str[0].islower() or l_str[0] in ",;)"):
        return False
    return len(u_str.split()) >= 2 or u_str.endswith("-")


def _horizontal_x_clusters(tokens: Sequence[BBox], line_h: float) -> list[tuple[float, float]]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda b: (b[0], b[1]))
    gap_thr = max(6.0, 0.9 * line_h)
    clusters: list[list[float]] = []
    for b in ordered:
        if not clusters or b[0] - clusters[-1][1] > gap_thr or min(b[3], clusters[-1][3]) - max(b[1], clusters[-1][2]) <= 0.0:
            clusters.append([b[0], b[2], b[1], b[3]])
        else:
            clusters[-1][1] = max(clusters[-1][1], b[2])
            clusters[-1][2] = min(clusters[-1][2], b[1])
            clusters[-1][3] = max(clusters[-1][3], b[3])
    return [(c[0], c[1]) for c in clusters]


def _upper_xcluster_spanned_cols(
    ux0: float,
    ux1: float,
    col_intervals: Sequence[tuple[float, float]],
    lower_cols: set[int],
    other_xclusters: Sequence[tuple[float, float]],
) -> list[int]:
    uw = max(1.0, ux1 - ux0)
    hit: list[int] = []
    for ci, (cx0, cx1) in enumerate(col_intervals):
        cw = max(1.0, cx1 - cx0)
        ov = min(ux1, cx1) - max(ux0, cx0)
        if ov >= max(2.0, min(0.15 * cw, 0.25 * uw)):
            hit.append(ci)
    if len(hit) < 2 or hit != list(range(hit[0], hit[-1] + 1)):
        return []
    if not all(ci in lower_cols for ci in hit):
        return []
    span_x0, span_x1 = col_intervals[hit[0]][0], col_intervals[hit[-1]][1]
    for ox0, ox1 in other_xclusters:
        ocx = (ox0 + ox1) * 0.5
        if span_x0 <= ocx <= span_x1:
            return []
    return hit


def _find_multitier_header_splits(
    rtoks: Sequence[BBox],
    ry0: float,
    ry1: float,
    col_intervals: Sequence[tuple[float, float]],
    span_boxes: Sequence[BBox],
    ch: float,
    line_h: float,
    header_bounds: Sequence[BBox] = (),
) -> list[float]:
    if len(rtoks) < 3 or len(col_intervals) < 2:
        return []
    row_h = ry1 - ry0
    if row_h <= 0.0:
        return []
    hb_y0, hb_y1 = ry0, ry1
    if header_bounds:
        ov_bounds = [(max(ry0, hb[1]), min(ry1, hb[3])) for hb in header_bounds if min(ry1, hb[3]) > max(ry0, hb[1])]
        if not ov_bounds:
            return []
        hb_y0 = min(lo for lo, _ in ov_bounds)
        hb_y1 = max(hi for _, hi in ov_bounds)
        rtoks = [b for b in rtoks if hb_y0 - 0.15 * line_h <= (b[1] + b[3]) * 0.5 <= hb_y1 + 0.15 * line_h]
        if len(rtoks) < 3:
            return []
    for sb in span_boxes:
        if (
            min(sb[3], ry1) - max(sb[1], ry0) >= 0.25 * row_h
            and (sb[3] - sb[1]) <= 0.85 * row_h
            and len(_overlap_indices(col_intervals, sb[0], sb[2], 0.3)) >= 2
        ):
            return []

    tier_cols: set[int] = set()
    tok_cols_list: list[list[int]] = []
    for b in rtoks:
        bw = max(1.0, b[2] - b[0])
        hits = [
            ci
            for ci, (cx0, cx1) in enumerate(col_intervals)
            if min(b[2], cx1) - max(b[0], cx0) >= max(2.0, min(0.15 * (cx1 - cx0), 0.25 * bw))
        ]
        if not hits:
            bcx = (b[0] + b[2]) * 0.5
            hits = [ci for ci, (cx0, cx1) in enumerate(col_intervals) if cx0 <= bcx <= cx1]
        tok_cols_list.append(hits)
        if len(hits) >= 2:
            tier_cols.update(hits)
    for ci in range(len(col_intervals)):
        if ci in tier_cols:
            continue
        ci_toks = [b for b, hits in zip(rtoks, tok_cols_list) if ci in hits]
        if len(ci_toks) >= 2 and len(_cluster_1d_tokens(ci_toks, axis=1, limit=ch)) >= 2:
            tier_cols.add(ci)

    tier_toks = [b for b, hits in zip(rtoks, tok_cols_list) if any(ci in tier_cols for ci in hits)]
    ycls = _cluster_1d_tokens(tier_toks or rtoks, axis=1, limit=ch)
    if len(ycls) < 2:
        return []

    split_ys: list[float] = []
    for k in range(len(ycls) - 1):
        split_y = (ycls[k][1] + ycls[k + 1][0]) * 0.5
        if split_y <= ry0 or split_y >= ry1:
            continue
        if header_bounds:
            # Both split bands must keep >= 50% overlap with the column header box so top-edge captions do not split off.
            if (
                min(split_y, hb_y1) - max(ry0, hb_y0) < 0.5 * max(1.0, split_y - ry0)
                or min(ry1, hb_y1) - max(split_y, hb_y0) < 0.5 * max(1.0, ry1 - split_y)
            ):
                continue
        band_lo_y = ycls[k][0] - 0.2 * line_h
        band_hi_y = ycls[k + 1][1] + 0.2 * line_h
        upper_toks = [b for b in rtoks if band_lo_y <= (b[1] + b[3]) * 0.5 <= split_y]
        lower_toks = [b for b in rtoks if split_y < (b[1] + b[3]) * 0.5 <= band_hi_y]
        if not upper_toks or not lower_toks:
            continue
        lower_cols = {
            ci
            for b in lower_toks
            for ci, (cx0, cx1) in enumerate(col_intervals)
            if cx0 <= (b[0] + b[2]) * 0.5 <= cx1
        }
        if len(lower_cols) < 2:
            continue
        uxs = _horizontal_x_clusters(upper_toks, line_h)
        if any(
            _upper_xcluster_spanned_cols(ux0, ux1, col_intervals, lower_cols, [ox for j, ox in enumerate(uxs) if j != idx])
            for idx, (ux0, ux1) in enumerate(uxs)
        ):
            split_ys.append(split_y)
    return split_ys


def _split_undersegmented_rows(
    row_intervals: list[tuple[float, float]],
    row_scores: list[float],
    col_intervals: Sequence[tuple[float, float]],
    tok_px: Sequence[BBox],
    ch: float,
    hdr_rows: set[int],
    crop_gray: np.ndarray | None = None,
    *,
    tok_texts: Sequence[str] | None = None,
    span_boxes: Sequence[BBox] = (),
    col_headers: Sequence[BBox] = (),
) -> tuple[list[tuple[float, float]], list[float], bool]:
    if not tok_px or len(col_intervals) < 2:
        return row_intervals, row_scores, False
    heights = sorted(max(4.0, b[3] - b[1]) for b in tok_px)
    line_h = heights[len(heights) // 2] if heights else 12.0

    # Measure inter-row whitespace from adjacent single-band rows after interval snapping.
    row_tok_idx: list[list[int]] = []
    row_single_bounds: list[tuple[float, float] | None] = []
    for ry0, ry1 in row_intervals:
        idxs = [i for i, b in enumerate(tok_px) if ry0 <= (b[1] + b[3]) * 0.5 <= ry1]
        row_tok_idx.append(idxs)
        if idxs:
            rtoks = [tok_px[i] for i in idxs]
            if len(_cluster_1d_tokens(rtoks, axis=1, limit=ch)) == 1:
                row_single_bounds.append((min(b[1] for b in rtoks), max(b[3] for b in rtoks)))
            else:
                row_single_bounds.append(None)
        else:
            row_single_bounds.append(None)

    inter_gaps = [
        row_single_bounds[r + 1][0] - row_single_bounds[r][1]
        for r in range(len(row_intervals) - 1)
        if row_single_bounds[r] is not None
        and row_single_bounds[r + 1] is not None
        and row_single_bounds[r + 1][0] > row_single_bounds[r][1]
    ]
    inter_gaps.sort()
    typical_inter_row_gap = inter_gaps[len(inter_gaps) // 2] if inter_gaps else 0.0

    body_horizontally_ruled = False
    if crop_gray is not None and len(row_intervals) >= 3:
        crop_w = float(crop_gray.shape[1])
        body_bounds = [
            row_intervals[r][1]
            for r in range(len(row_intervals) - 1)
            if r not in hdr_rows and (r + 1) not in hdr_rows
        ]
        if body_bounds:
            ruled = sum(
                1
                for by in body_bounds
                if _has_internal_rule(crop_gray, 0.0, max(0.0, by - 8.0), crop_w, min(ch, by + 8.0), (), [by])
            )
            body_horizontally_ruled = ruled >= max(1, (len(body_bounds) + 1) // 2)

    out_rows: list[tuple[float, float]] = []
    out_scores: list[float] = []
    did_split = False
    row_heights = [ry1 - ry0 for ry0, ry1 in row_intervals]
    for r_idx, ((ry0, ry1), score) in enumerate(zip(row_intervals, row_scores)):
        idxs = row_tok_idx[r_idx]
        rtoks = [tok_px[i] for i in idxs]
        if r_idx in hdr_rows:
            split_ys = _find_multitier_header_splits(
                rtoks, ry0, ry1, col_intervals, span_boxes, ch, line_h, header_bounds=col_headers
            )
            if split_ys:
                pts = [ry0, *split_ys, ry1]
                did_split = True
                sub_score = min(score, 0.85)
                for k in range(len(pts) - 1):
                    out_rows.append((pts[k], pts[k + 1]))
                    out_scores.append(sub_score)
                continue
            out_rows.append((ry0, ry1))
            out_scores.append(score)
            continue

        if len(rtoks) < 4:
            out_rows.append((ry0, ry1))
            out_scores.append(score)
            continue
        ycls = _cluster_1d_tokens(rtoks, axis=1, limit=ch)
        if len(ycls) >= 2:
            active_cols = len(
                {ci for b in rtoks for ci, (c0, c1) in enumerate(col_intervals) if c0 <= (b[0] + b[2]) * 0.5 <= c1}
            )
            req_cols = max(2, int(0.6 * active_cols))
            other_hs = sorted(h for j, h in enumerate(row_heights) if j != r_idx and h > 0.0)
            ref_row_h = other_hs[len(other_hs) // 2] if other_hs else 0.0
            row_h = ry1 - ry0
            min_gap_thr = max(0.75 * typical_inter_row_gap, 0.55 * line_h)

            groups: list[list[float]] = [[ycls[0][0], ycls[0][1]]]
            has_rule_split = False
            crop_w = float(crop_gray.shape[1]) if crop_gray is not None else 0.0
            for k in range(len(ycls) - 1):
                cur_y0, cur_y1 = groups[-1][0], groups[-1][1]
                nxt_y0, nxt_y1 = ycls[k + 1][0], ycls[k + 1][1]
                mid_y = (cur_y1 + nxt_y0) * 0.5
                has_rule = crop_gray is not None and _has_internal_rule(
                    crop_gray, 0.0, ry0, crop_w, ry1, (), [mid_y]
                )
                can_split = has_rule
                if has_rule:
                    has_rule_split = True
                if not can_split and not body_horizontally_ruled:
                    sub_gap = nxt_y0 - cur_y1
                    tall_enough = ref_row_h > 0.0 and row_h >= 1.75 * ref_row_h
                    if tall_enough and sub_gap >= min_gap_thr:
                        is_prose = False
                        if tok_texts is not None:
                            u_txts = [
                                tok_texts[i]
                                for i in idxs
                                if cur_y0 <= (tok_px[i][1] + tok_px[i][3]) * 0.5 <= cur_y1
                            ]
                            l_txts = [
                                tok_texts[i]
                                for i in idxs
                                if nxt_y0 <= (tok_px[i][1] + tok_px[i][3]) * 0.5 <= nxt_y1
                            ]
                            col_prose_hits = 0
                            col_split_cols = 0
                            for c0, c1 in col_intervals:
                                cu = [
                                    tok_texts[i]
                                    for i in idxs
                                    if c0 <= (tok_px[i][0] + tok_px[i][2]) * 0.5 <= c1
                                    and cur_y0 <= (tok_px[i][1] + tok_px[i][3]) * 0.5 <= cur_y1
                                ]
                                cl = [
                                    tok_texts[i]
                                    for i in idxs
                                    if c0 <= (tok_px[i][0] + tok_px[i][2]) * 0.5 <= c1
                                    and nxt_y0 <= (tok_px[i][1] + tok_px[i][3]) * 0.5 <= nxt_y1
                                ]
                                if cu and cl:
                                    col_split_cols += 1
                                    if _is_wrapped_prose_pair(cu, cl):
                                        col_prose_hits += 1
                            if col_split_cols > 0 and col_prose_hits == col_split_cols and (u_txts and l_txts):
                                is_prose = True
                        can_split = not is_prose
                if can_split:
                    groups.append([nxt_y0, nxt_y1])
                else:
                    groups[-1][1] = max(cur_y1, nxt_y1)

            eff_req_cols = 1 if has_rule_split else req_cols
            if len(groups) >= 2 and active_cols >= 2 and all(
                len(
                    {
                        ci
                        for b in rtoks
                        if y0 <= (b[1] + b[3]) * 0.5 <= y1
                        for ci, (c0, c1) in enumerate(col_intervals)
                        if c0 <= (b[0] + b[2]) * 0.5 <= c1
                    }
                )
                >= eff_req_cols
                for y0, y1 in groups
            ):
                sub = _snap_intervals(
                    [(y0 - ry0, y1 - ry0, 0.7) for y0, y1 in groups],
                    ry1 - ry0,
                    [(b[1] - ry0, b[3] - ry0) for b in rtoks],
                )
                if len(sub) == len(groups):
                    did_split = True
                    sub_score = min(score, 0.85)
                    for s0, s1 in sub:
                        out_rows.append((ry0 + s0, ry0 + s1))
                        out_scores.append(sub_score)
                    continue
        out_rows.append((ry0, ry1))
        out_scores.append(score)
    return out_rows, out_scores, did_split


def _synthesize_multitier_header_spans(
    row_intervals: Sequence[tuple[float, float]],
    row_scores: Sequence[float],
    col_intervals: Sequence[tuple[float, float]],
    header_rows: set[int],
    tok_px: Sequence[BBox],
    claimed: set[tuple[int, int]],
) -> list[tuple[int, int, int, int, bool, float]]:
    num_rows = len(row_intervals)
    num_cols = len(col_intervals)
    if num_rows < 2 or num_cols < 2 or not tok_px:
        return []
    hdr_end = 0
    while hdr_end < num_rows and hdr_end in header_rows:
        hdr_end += 1
    if hdr_end < 2:
        return []

    heights = sorted(max(4.0, b[3] - b[1]) for b in tok_px)
    line_h = heights[len(heights) // 2] if heights else 12.0
    extra_specs: list[tuple[int, int, int, int, bool, float]] = []

    for r_up in range(hdr_end - 1):
        r_lo = r_up + 1
        ry0, ry1 = row_intervals[r_up]
        ly0, ly1 = row_intervals[r_lo]
        upper_toks = [b for b in tok_px if ry0 <= (b[1] + b[3]) * 0.5 <= ry1]
        lower_toks = [b for b in tok_px if ly0 <= (b[1] + b[3]) * 0.5 <= ly1]
        lower_cols = {
            ci
            for b in lower_toks
            for ci, (cx0, cx1) in enumerate(col_intervals)
            if cx0 <= (b[0] + b[2]) * 0.5 <= cx1
        }
        if len(lower_cols) < 2:
            continue
        uxs = _horizontal_x_clusters(upper_toks, line_h)
        for idx, (ux0, ux1) in enumerate(uxs):
            hit = _upper_xcluster_spanned_cols(
                ux0, ux1, col_intervals, lower_cols, [ox for j, ox in enumerate(uxs) if j != idx]
            )
            if not hit:
                continue
            c0, c1 = hit[0], hit[-1]
            coords = [(r_up, c) for c in range(c0, c1 + 1)]
            if any(rc in claimed for rc in coords):
                continue
            claimed.update(coords)
            extra_specs.append((r_up, c0, 1, c1 - c0 + 1, False, min(row_scores[r_up], 0.85)))

    hdr_y0, hdr_y1 = row_intervals[0][0], row_intervals[hdr_end - 1][1]
    split_y0 = row_intervals[0][1]
    for c in range(num_cols):
        coords = [(r, c) for r in range(hdr_end)]
        if any(rc in claimed for rc in coords):
            continue
        cx0, cx1 = col_intervals[c]
        col_toks = [
            b for b in tok_px if cx0 <= (b[0] + b[2]) * 0.5 <= cx1 and hdr_y0 <= (b[1] + b[3]) * 0.5 <= hdr_y1
        ]
        if not col_toks:
            continue
        if len(_cluster_1d_tokens(col_toks, axis=1, limit=hdr_y1)) >= 2 or min(b[1] for b in col_toks) > split_y0:
            continue
        claimed.update(coords)
        extra_specs.append((0, c, hdr_end, 1, False, min(row_scores[0], 0.85)))
    return extra_specs


def _filter_projected_headers(
    proj_headers: Sequence[tuple[BBox, float]],
    tok_px: Sequence[BBox],
    col_intervals: Sequence[tuple[float, float]],
    num_rows: int,
    header_rows: set[int],
    cw: float,
) -> list[tuple[BBox, float]]:
    if not proj_headers:
        return []
    gap_tol = max(8.0, 0.025 * cw)
    kept: list[tuple[BBox, float]] = []
    for pb, score in proj_headers:
        rtoks = sorted((b for b in tok_px if pb[1] <= (b[1] + b[3]) * 0.5 <= pb[3]), key=lambda b: b[0])
        if rtoks:
            hclusters: list[list[float]] = []
            for b in rtoks:
                if not hclusters or b[0] > hclusters[-1][1] + gap_tol:
                    hclusters.append([b[0], b[2]])
                else:
                    hclusters[-1][1] = max(hclusters[-1][1], b[2])
            hit_cols = {
                ci
                for cl in hclusters
                for ci, (c0, c1) in enumerate(col_intervals)
                if c0 <= (cl[0] + cl[1]) * 0.5 <= c1
            }
            if len(hclusters) >= 2 and len(hit_cols) >= 2:
                continue
        kept.append((pb, score))

    num_cols = len(col_intervals)
    if num_cols >= 3 and kept and tok_px:
        body_rows = max(1, num_rows - len(header_rows))
        if len(kept) > 0.5 * body_rows:
            c0_hi = col_intervals[0][1]
            all_in_col0 = all(
                all(b[2] <= c0_hi + 2.0 for b in tok_px if pb[1] <= (b[1] + b[3]) * 0.5 <= pb[3])
                for pb, _ in kept
            )
            if all_in_col0:
                return []
    return kept


def _collapse_blocked_by_rules_or_content(
    block_cells: Sequence[TableCell],
    unspanned_cols: Sequence[int],
    r_start: int,
    r_end: int,
    crop_gray: np.ndarray | None,
    table_bbox_pt: BBox | None,
) -> bool:
    if crop_gray is not None and table_bbox_pt is not None:
        ch, cw = crop_gray.shape[:2]
        tx0, ty0, tx1, ty1 = table_bbox_pt
        sx = (tx1 - tx0) / cw if cw > 0 else 0.0
        sy = (ty1 - ty0) / ch if ch > 0 else 0.0
        if sx > 0.0 and sy > 0.0:
            for col_i in unspanned_cols:
                col_cells = sorted(
                    (c for c in block_cells if c.col <= col_i < c.col + c.colspan and c.bbox is not None),
                    key=lambda c: c.row,
                )
                if len(col_cells) < 2:
                    continue
                x_lo = min(c.bbox[0] for c in col_cells if c.bbox is not None)
                x_hi = max(c.bbox[2] for c in col_cells if c.bbox is not None)
                y_lo = min(c.bbox[1] for c in col_cells if c.bbox is not None)
                y_hi = max(c.bbox[3] for c in col_cells if c.bbox is not None)
                internal_ys = [
                    ((col_cells[m].bbox[3] + col_cells[m + 1].bbox[1]) * 0.5 - ty0) / sy
                    for m in range(len(col_cells) - 1)
                    if col_cells[m].bbox is not None and col_cells[m + 1].bbox is not None
                ]
                if internal_ys and _has_internal_rule(
                    crop_gray,
                    (x_lo - tx0) / sx,
                    (y_lo - ty0) / sy,
                    (x_hi - tx0) / sx,
                    (y_hi - ty0) / sy,
                    (),
                    internal_ys,
                ):
                    return True

    for col_i in unspanned_cols:
        col_cells = [c for c in block_cells if c.col <= col_i < c.col + c.colspan and c.text.strip()]
        if sum(1 for c in col_cells if _is_discrete_value_text(c.text)) >= 2:
            return True

    if len(unspanned_cols) >= 2:
        col_is_prose = {}
        for col_i in unspanned_cols:
            c_cells = sorted(
                (c for c in block_cells if c.col <= col_i < c.col + c.colspan and c.text.strip()),
                key=lambda c: c.row,
            )
            col_is_prose[col_i] = len(c_cells) >= 2 and any(
                _is_wrapped_prose_pair([c_cells[m].text], [c_cells[m + 1].text])
                for m in range(len(c_cells) - 1)
            )
        if not all(col_is_prose.values()):
            rows_with_short_items = 0
            for rr in range(r_start, r_end + 1):
                row_vals = [
                    c.text.strip()
                    for c in block_cells
                    if c.row == rr and c.col in unspanned_cols and c.text.strip()
                ]
                if len(row_vals) >= 2 and all(len(v.split()) <= 2 for v in row_vals):
                    rows_with_short_items += 1
            if rows_with_short_items >= 2:
                return True
    return False


def _collapse_multiline_rows(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
    *,
    crop_gray: np.ndarray | None = None,
    table_bbox_pt: BBox | None = None,
) -> tuple[tuple[TableCell, ...], int]:
    # Collapse row blocks where spanning columns anchor unruled wrapped sub-rows.
    if num_rows < 2 or num_cols < 2:
        return tuple(cells), num_rows
    by_row: dict[int, list[TableCell]] = {}
    for c in cells:
        by_row.setdefault(c.row, []).append(c)

    out: list[TableCell] = []
    r = 0
    new_r = 0
    while r < num_rows:
        row_cells = by_row.get(r, ())
        cand_spans = [c.rowspan for c in row_cells if c.rowspan >= 2]
        collapsed = False
        if cand_spans:
            k = max(cand_spans)
            r_end = r + k - 1
            if r_end < num_rows:
                block_cells = [c for rr in range(r, r_end + 1) for c in by_row.get(rr, ())]
                hdr_vals = {c.is_header for c in block_cells}
                crosses = any(
                    (c.row < r and c.row + c.rowspan > r) or (r <= c.row <= r_end and c.row + c.rowspan - 1 > r_end)
                    for c in cells
                )
                if len(hdr_vals) == 1 and not next(iter(hdr_vals)) and not crosses:
                    n_spanning = 0
                    unspanned_multi_cols: list[int] = []
                    for col_i in range(num_cols):
                        col_cells = [c for c in block_cells if c.col <= col_i < c.col + c.colspan]
                        if len(col_cells) == 1 and col_cells[0].row == r and col_cells[0].rowspan == k:
                            n_spanning += 1
                        elif sum(1 for c in col_cells if c.text.strip()) >= 2:
                            unspanned_multi_cols.append(col_i)
                    if (
                        unspanned_multi_cols
                        and n_spanning >= max(2, (num_cols + 1) // 2)
                        and not _collapse_blocked_by_rules_or_content(
                            block_cells, unspanned_multi_cols, r, r_end, crop_gray, table_bbox_pt
                        )
                    ):
                        for base in sorted(row_cells, key=lambda c: c.col):
                            if base.rowspan == k:
                                merged_txt = base.text
                            else:
                                parts = []
                                for rr in range(r, r_end + 1):
                                    sub = sorted(
                                        (
                                            c
                                            for c in by_row.get(rr, ())
                                            if base.col <= c.col < base.col + base.colspan and c.text.strip()
                                        ),
                                        key=lambda c: c.col,
                                    )
                                    if sub:
                                        parts.append(" ".join(c.text for c in sub))
                                merged_txt = " ".join(parts)
                            merged_box = base.bbox
                            if base.bbox is not None and base.rowspan < k:
                                ys = [
                                    c.bbox[3]
                                    for c in block_cells
                                    if c.bbox is not None and base.col <= c.col < base.col + base.colspan
                                ]
                                if ys:
                                    merged_box = (base.bbox[0], base.bbox[1], base.bbox[2], max(base.bbox[3], max(ys)))
                            out.append(
                                TableCell(
                                    row=new_r,
                                    col=base.col,
                                    rowspan=1,
                                    colspan=base.colspan,
                                    text=merged_txt,
                                    bbox=merged_box,
                                    is_header=base.is_header,
                                    confidence=base.confidence,
                                )
                            )
                        r = r_end + 1
                        new_r += 1
                        collapsed = True
        if not collapsed:
            for c in sorted(row_cells, key=lambda x: x.col):
                out.append(
                    TableCell(
                        row=new_r,
                        col=c.col,
                        rowspan=c.rowspan,
                        colspan=c.colspan,
                        text=c.text,
                        bbox=c.bbox,
                        is_header=c.is_header,
                        confidence=c.confidence,
                    )
                )
            r += 1
            new_r += 1
    return tuple(out), new_r


def _is_thin_continuous_vert_rule(ink_v: np.ndarray, sx0: int, sx1: int, height: int) -> bool:
    if sx1 <= sx0 or height < 18:
        return False
    sub = ink_v[:, sx0:sx1]
    col_fracs = sub.mean(axis=0)
    min_run = int(0.85 * height)
    for k in range(sx1 - sx0):
        if float(col_fracs[k]) < 0.85:
            continue
        left = k
        while left > 0 and float(col_fracs[left - 1]) >= 0.50:
            left -= 1
        right = k
        while right + 1 < sx1 - sx0 and float(col_fracs[right + 1]) >= 0.50:
            right += 1
        if right - left + 1 > 6:
            continue
        # Allow a 1-px anti-aliasing break inside a continuous vertical rule stroke.
        col = sub[:, k]
        run = 0
        best_run = 0
        gap = 0
        for px in col.tolist():
            if px:
                run += 1 + gap
                gap = 0
                if run > best_run:
                    best_run = run
            elif run > 0 and gap == 0:
                gap = 1
            else:
                run = 0
                gap = 0
        if best_run >= min_run:
            return True
    return False


def _has_internal_rule(
    crop_gray: np.ndarray,
    x_lo: float,
    y_lo: float,
    x_hi: float,
    y_hi: float,
    internal_xs: Sequence[float],
    internal_ys: Sequence[float],
) -> bool:
    ch, cw = crop_gray.shape[:2]
    ix0 = max(0, min(cw, int(round(x_lo))))
    ix1 = max(0, min(cw, int(round(x_hi))))
    iy0 = max(0, min(ch, int(round(y_lo))))
    iy1 = max(0, min(ch, int(round(y_hi))))
    if ix1 - ix0 < 4 or iy1 - iy0 < 4:
        return False
    tol_x = max(3, min(6, int(round(0.015 * cw))))
    tol_y = max(4, min(12, int(round(0.02 * ch))))
    if internal_xs and (iy1 - iy0) >= 18:
        patch = crop_gray[iy0:iy1, ix0:ix1].astype(np.int16)
        row_bg = np.median(patch, axis=1, keepdims=True)
        ink_v = np.abs(patch - row_bg) >= 35
        for bx in internal_xs:
            cx = int(round(bx))
            sx0 = max(0, cx - tol_x - ix0)
            sx1 = min(ix1 - ix0, cx + tol_x + 1 - ix0)
            if _is_thin_continuous_vert_rule(ink_v, sx0, sx1, iy1 - iy0):
                return True
    if internal_ys:
        for by in internal_ys:
            cy = int(round(by))
            sy0 = max(0, cy - tol_y)
            sy1 = min(ch, cy + tol_y + 1)
            if sy1 <= sy0:
                continue
            above_y0, above_y1 = max(0, sy0 - 6), max(0, sy0 - 1)
            below_y0, below_y1 = min(ch, sy1 + 1), min(ch, sy1 + 6)
            strip = crop_gray[sy0:sy1, ix0:ix1].astype(np.int16)
            if above_y1 > above_y0 and below_y1 > below_y0:
                bg_above = int(round(float(np.median(crop_gray[above_y0:above_y1, ix0:ix1]))))
                bg_below = int(round(float(np.median(crop_gray[below_y0:below_y1, ix0:ix1]))))
                ink_h = (np.abs(strip - bg_above) >= 35) & (np.abs(strip - bg_below) >= 35)
            else:
                bg = int(round(float(np.median(crop_gray[iy0:iy1, ix0:ix1]))))
                ink_h = np.abs(strip - bg) >= 35
            if float(ink_h.mean(axis=1).max()) >= 0.70:
                return True
    return False


def _passes_low_conf_span_guard(
    crop_gray: np.ndarray | None,
    row_intervals: Sequence[tuple[float, float]],
    col_intervals: Sequence[tuple[float, float]],
    tok_px: Sequence[BBox],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    *,
    tok_texts: Sequence[str] | None = None,
) -> bool:
    y_lo, y_hi = row_intervals[r0][0], row_intervals[r1][1]
    x_lo, x_hi = col_intervals[c0][0], col_intervals[c1][1]
    if crop_gray is not None:
        internal_xs = [col_intervals[c][1] for c in range(c0, c1)]
        internal_ys = [row_intervals[r][1] for r in range(r0, r1)]
        if _has_internal_rule(crop_gray, x_lo, y_lo, x_hi, y_hi, internal_xs, internal_ys):
            return False
    span_idx = [
        i
        for i, b in enumerate(tok_px)
        if x_lo <= (b[0] + b[2]) * 0.5 <= x_hi and y_lo <= (b[1] + b[3]) * 0.5 <= y_hi
    ]
    if len(span_idx) < 2:
        return True
    span_toks = [tok_px[i] for i in span_idx]
    heights = sorted(max(4.0, b[3] - b[1]) for b in span_toks)
    line_h = heights[len(heights) // 2]
    if c1 > c0:
        ordered_y = sorted(span_toks, key=lambda b: ((b[1] + b[3]) * 0.5, b[0]))
        y_lines: list[list[BBox]] = []
        for b in ordered_y:
            if not y_lines or not any(
                min(b[3], lb[3]) - max(b[1], lb[1]) >= 0.45 * min(max(1.0, b[3] - b[1]), max(1.0, lb[3] - lb[1]))
                for lb in y_lines[-1]
            ):
                y_lines.append([b])
            else:
                y_lines[-1].append(b)
        min_gap = 1.2 * line_h
        for line in y_lines:
            line.sort(key=lambda b: b[0])
            x_clusters: list[list[float]] = []
            for b in line:
                if not x_clusters or b[0] - x_clusters[-1][1] >= min_gap:
                    x_clusters.append([b[0], b[2]])
                else:
                    x_clusters[-1][1] = max(x_clusters[-1][1], b[2])
            if len(x_clusters) >= 2:
                for c in range(c0, c1):
                    bx = col_intervals[c][1]
                    left_cl = [cl for cl in x_clusters if (cl[0] + cl[1]) * 0.5 <= bx and cl[1] <= bx + 0.25 * line_h]
                    right_cl = [cl for cl in x_clusters if (cl[0] + cl[1]) * 0.5 >= bx and cl[0] >= bx - 0.25 * line_h]
                    if left_cl and right_cl and (right_cl[0][0] - left_cl[-1][1]) >= min_gap:
                        return False
    if r1 > r0:
        active = [
            [i for i in span_idx if row_intervals[r][0] <= (tok_px[i][1] + tok_px[i][3]) * 0.5 <= row_intervals[r][1]]
            for r in range(r0, r1 + 1)
        ]
        active = [grp for grp in active if grp]
        if len(active) >= 2:
            gaps = [
                max(0.0, min(tok_px[i][1] for i in active[k + 1]) - max(tok_px[i][3] for i in active[k]))
                for k in range(len(active) - 1)
            ]
            max_gap = max(gaps)
            if max_gap >= 0.95 * line_h:
                return False
            if max_gap >= 0.70 * line_h:
                if c1 > c0:
                    multi_col_rows = sum(
                        1
                        for grp in active
                        if len(
                            {
                                ci
                                for i in grp
                                for ci in range(c0, c1 + 1)
                                if col_intervals[ci][0] <= (tok_px[i][0] + tok_px[i][2]) * 0.5 <= col_intervals[ci][1]
                            }
                        )
                        >= 2
                    )
                    if multi_col_rows >= 2:
                        return False
                if tok_texts is not None:
                    discrete_rows = sum(
                        1
                        for grp in active
                        if _is_discrete_value_text(" ".join(tok_texts[i] for i in grp))
                    )
                    if discrete_rows >= 2:
                        return False
    return True


def _build_raw_cells(
    merged_specs: Sequence[tuple[int, int, int, int, bool, float]],
    claimed: set[tuple[int, int]],
    row_intervals: Sequence[tuple[float, float]],
    col_intervals: Sequence[tuple[float, float]],
    row_scores: Sequence[float],
    col_scores: Sequence[float],
    header_rows: set[int],
    proj_header_rows: set[int],
    tokens: Sequence[tuple[str, BBox]],
    crop_gray: np.ndarray | None,
    table_bbox_pt: BBox,
    tx0: float,
    ty0: float,
    sx: float,
    sy: float,
) -> tuple[list[TableCell], int, int]:
    num_rows = len(row_intervals)
    num_cols = len(col_intervals)
    cell_defs: list[tuple[int, int, int, int, BBox, bool, float]] = []
    for r0, c0, rspan, cspan, is_proj, span_score in merged_specs:
        y0, y1 = row_intervals[r0][0], row_intervals[r0 + rspan - 1][1]
        x0, x1 = col_intervals[c0][0], col_intervals[c0 + cspan - 1][1]
        pt_box = (
            round(tx0 + x0 * sx, 3),
            round(ty0 + y0 * sy, 3),
            round(tx0 + x1 * sx, 3),
            round(ty0 + y1 * sy, 3),
        )
        is_hdr = is_proj or any(r in header_rows or r in proj_header_rows for r in range(r0, r0 + rspan))
        r_avg = sum(row_scores[r0 : r0 + rspan]) / rspan
        c_avg = sum(col_scores[c0 : c0 + cspan]) / cspan
        cell_conf = round(max(0.0, min(_MAX_TATR_CONF, (r_avg + c_avg + span_score) / 3.0)), 4)
        cell_defs.append((r0, c0, rspan, cspan, pt_box, is_hdr, cell_conf))

    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) in claimed:
                continue
            y0, y1 = row_intervals[r]
            x0, x1 = col_intervals[c]
            pt_box = (
                round(tx0 + x0 * sx, 3),
                round(ty0 + y0 * sy, 3),
                round(tx0 + x1 * sx, 3),
                round(ty0 + y1 * sy, 3),
            )
            is_hdr = r in header_rows or r in proj_header_rows
            cell_conf = round(max(0.0, min(_MAX_TATR_CONF, (row_scores[r] + col_scores[c]) * 0.5)), 4)
            cell_defs.append((r, c, 1, 1, pt_box, is_hdr, cell_conf))

    cell_defs.sort(key=lambda item: (item[0], item[1]))
    expanded_tokens, split_pieces = _split_multicell_tokens(
        tokens, cell_defs, crop_gray=crop_gray, table_bbox_pt=table_bbox_pt
    )
    cell_tokens: list[list[tuple[str, BBox]]] = [[] for _ in cell_defs]
    assigned_nonempty = 0
    total_nonempty = sum(1 for txt, _ in expanded_tokens if txt.strip())
    for text, tb in expanded_tokens:
        tcx = (tb[0] + tb[2]) * 0.5
        tcy = (tb[1] + tb[3]) * 0.5
        best_idx: int | None = None
        best_overlap = 0.0
        for idx, (_, _, _, _, cb, _, _) in enumerate(cell_defs):
            if cb[0] <= tcx <= cb[2] and cb[1] <= tcy <= cb[3]:
                best_idx = idx
                break
            ix = max(0.0, min(tb[2], cb[2]) - max(tb[0], cb[0]))
            iy = max(0.0, min(tb[3], cb[3]) - max(tb[1], cb[1]))
            ov = ix * iy
            if ov > best_overlap:
                best_overlap = ov
                best_idx = idx
        if best_idx is not None:
            cell_tokens[best_idx].append((text, tb))
            if text.strip():
                assigned_nonempty += 1

    raw_cells: list[TableCell] = []
    for idx, (r, c, rspan, cspan, cb, is_hdr, cell_conf) in enumerate(cell_defs):
        toks = cell_tokens[idx]
        if any(tok in split_pieces for tok in toks):
            cell_conf = round(max(0.0, min(_MAX_TATR_CONF, cell_conf * 0.90)), 4)
        raw_cells.append(
            TableCell(
                row=r,
                col=c,
                rowspan=rspan,
                colspan=cspan,
                text=_sort_cell_tokens(toks),
                bbox=cb,
                is_header=is_hdr,
                confidence=cell_conf,
            )
        )
    return raw_cells, assigned_nonempty, total_nonempty


def objects_to_table(
    objects: Sequence[tuple[str, BBox, float]],
    crop_size: tuple[int, int],
    table_bbox_pt: BBox,
    tokens: Sequence[tuple[str, BBox]] = (),
    *,
    crop: Image.Image | None = None,
) -> ExtractedTable:
    cw, ch = float(crop_size[0]), float(crop_size[1])
    if cw <= 0.0 or ch <= 0.0:
        return ExtractedTable(
            bbox=table_bbox_pt,
            num_rows=0,
            num_cols=0,
            cells=(),
            source="tatr_onnx",
            confidence=0.0,
        )
    crop_gray: np.ndarray | None = None
    if crop is not None and crop.size == (int(crop_size[0]), int(crop_size[1])):
        crop_gray = _crop_to_gray(crop)

    tx0, ty0, tx1, ty1 = table_bbox_pt
    sx = (tx1 - tx0) / cw
    sy = (ty1 - ty0) / ch
    tok_px: list[BBox] = (
        [
            (
                (tb[0] - tx0) / max(1e-6, sx),
                (tb[1] - ty0) / max(1e-6, sy),
                (tb[2] - tx0) / max(1e-6, sx),
                (tb[3] - ty0) / max(1e-6, sy),
            )
            for _, tb in tokens
        ]
        if sx > 0.0 and sy > 0.0
        else []
    )
    tok_texts = [txt for txt, _ in tokens] if tok_px else None

    raw_rows = [(b, s) for lbl, b, s in objects if lbl == "table row"]
    raw_cols = [(b, s) for lbl, b, s in objects if lbl == "table column"]
    row_kept = [
        (max(0.0, a0), min(ch, a1), s)
        for a0, a1, s in _dedupe_1d_intervals(raw_rows, axis=1)
        if min(ch, a1) > max(0.0, a0)
    ]
    col_kept = [
        (max(0.0, a0), min(cw, a1), s)
        for a0, a1, s in _dedupe_1d_intervals(raw_cols, axis=0)
        if min(cw, a1) > max(0.0, a0)
    ]
    used_cluster_fallback = False
    if tok_px:
        y_clusters = _cluster_1d_tokens(tok_px, axis=1, limit=ch)
        if not row_kept or (len(row_kept) == 1 and len(y_clusters) >= 3):
            row_kept = y_clusters
            used_cluster_fallback = True
        if not col_kept and row_kept:
            col_kept = [(0.0, cw, 0.5)]
            used_cluster_fallback = True

    col_headers = [b for lbl, b, s in objects if lbl == "table column header" and s >= DEFAULT_THRESHOLD]
    hdr_y_max = max((hb[3] for hb in col_headers), default=-1.0)
    body_x_spans = [(b[0], b[2]) for b in tok_px if (b[1] + b[3]) * 0.5 > hdr_y_max]
    col_snap_spans = body_x_spans if body_x_spans else [(b[0], b[2]) for b in tok_px]

    row_intervals = _snap_intervals(row_kept, ch, [(b[1], b[3]) for b in tok_px])
    col_intervals = _snap_intervals(col_kept, cw, col_snap_spans, ignore_cross_spans=True)
    row_scores = [item[2] for item in row_kept]
    col_scores = [item[2] for item in col_kept]
    span_boxes = [
        b
        for lbl, b, s in objects
        if (lbl == "table spanning cell" and s >= _SPAN_THRESHOLD) or lbl == "table projected row header"
    ]
    raw_span_boxes = [
        b
        for lbl, b, s in objects
        if (lbl == "table spanning cell" and s >= _SPAN_GUARD_MIN) or lbl == "table projected row header"
    ]
    col_intervals, col_scores, did_split_cols = _split_undersegmented_cols(
        col_intervals, col_scores, tok_px, cw, col_headers, span_boxes
    )
    init_header_rows: set[int] = set()
    for hb in col_headers:
        init_header_rows.update(_overlap_indices(row_intervals, hb[1], hb[3]))
    row_intervals, row_scores, did_split_rows = _split_undersegmented_rows(
        row_intervals,
        row_scores,
        col_intervals,
        tok_px,
        ch,
        init_header_rows,
        crop_gray=crop_gray,
        tok_texts=tok_texts,
        span_boxes=raw_span_boxes,
        col_headers=col_headers,
    )

    num_rows = len(row_intervals)
    num_cols = len(col_intervals)
    if num_rows == 0 or num_cols == 0:
        return ExtractedTable(
            bbox=table_bbox_pt,
            num_rows=0,
            num_cols=0,
            cells=(),
            source="tatr_onnx",
            confidence=0.0,
        )

    header_rows: set[int] = set()
    for hb in col_headers:
        header_rows.update(_overlap_indices(row_intervals, hb[1], hb[3]))

    raw_proj_headers = sorted(
        [(b, s) for lbl, b, s in objects if lbl == "table projected row header"],
        key=lambda item: item[1],
        reverse=True,
    )
    proj_headers = _filter_projected_headers(raw_proj_headers, tok_px, col_intervals, num_rows, header_rows, cw)
    proj_header_rows: set[int] = set()
    for pb, _ in proj_headers:
        proj_header_rows.update(_overlap_indices(row_intervals, pb[1], pb[3]))

    # Claim merged regions in descending confidence order so higher-scoring spans win overlaps.
    spanning = sorted(
        [(b, s, False) for lbl, b, s in objects if lbl == "table spanning cell" and s >= _SPAN_GUARD_MIN]
        + [(b, s, True) for b, s in proj_headers],
        key=lambda item: item[1],
        reverse=True,
    )
    claimed: set[tuple[int, int]] = set()
    merged_specs: list[tuple[int, int, int, int, bool, float]] = []
    for box, score, is_proj in spanning:
        min_ov = 0.5 if is_proj else _SPAN_OVERLAP_FRAC
        rs = _overlap_indices(row_intervals, box[1], box[3], min_ov)
        cs = _overlap_indices(col_intervals, box[0], box[2], min_ov)
        if is_proj and len(rs) == 1 and len(cs) < num_cols and (box[2] - box[0]) >= 0.7 * cw:
            cs = list(range(num_cols))
        if len(rs) * len(cs) < 2:
            continue
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if any(rc in claimed for rc in coords):
            continue
        if not is_proj and score < 0.90 and crop_gray is not None:
            y_lo, y_hi = row_intervals[r0][0], row_intervals[r1][1]
            x_lo, x_hi = col_intervals[c0][0], col_intervals[c1][1]
            internal_xs = [col_intervals[c][1] for c in range(c0, c1)]
            internal_ys = [row_intervals[r][1] for r in range(r0, r1)]
            if _has_internal_rule(crop_gray, x_lo, y_lo, x_hi, y_hi, internal_xs, internal_ys):
                continue
        if not is_proj and score < _SPAN_THRESHOLD:
            if not _passes_low_conf_span_guard(
                crop_gray, row_intervals, col_intervals, tok_px, r0, r1, c0, c1, tok_texts=tok_texts
            ):
                continue
        claimed.update(coords)
        merged_specs.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1, is_proj, float(score)))

    base_claimed = set(claimed)
    base_specs = list(merged_specs)
    hdr_specs = _synthesize_multitier_header_spans(
        row_intervals, row_scores, col_intervals, header_rows, tok_px, claimed
    )
    if hdr_specs:
        merged_specs.extend(hdr_specs)

    raw_cells, assigned_nonempty, total_nonempty = _build_raw_cells(
        merged_specs,
        claimed,
        row_intervals,
        col_intervals,
        row_scores,
        col_scores,
        header_rows,
        proj_header_rows,
        tokens,
        crop_gray,
        table_bbox_pt,
        tx0,
        ty0,
        sx,
        sy,
    )
    if hdr_specs and not _cells_grid_valid(raw_cells, num_rows, num_cols, allow_empty=True):
        raw_cells, assigned_nonempty, total_nonempty = _build_raw_cells(
            base_specs,
            base_claimed,
            row_intervals,
            col_intervals,
            row_scores,
            col_scores,
            header_rows,
            proj_header_rows,
            tokens,
            crop_gray,
            table_bbox_pt,
            tx0,
            ty0,
            sx,
            sy,
        )
    cells, num_rows = _collapse_multiline_rows(
        raw_cells, num_rows, num_cols, crop_gray=crop_gray, table_bbox_pt=table_bbox_pt
    )

    table_scores = [s for lbl, _, s in objects if lbl == "table"]
    raw_conf = max(table_scores) if table_scores else sum(s for _, _, s in objects) / max(1, len(objects))
    conf = min(_MAX_TATR_CONF, float(raw_conf))
    rc_scores = row_scores + col_scores
    mean_rc = sum(rc_scores) / max(1, len(rc_scores))
    if mean_rc < 0.85:
        conf *= mean_rc / 0.85
    if used_cluster_fallback or did_split_cols or did_split_rows:
        conf *= 0.92
    if total_nonempty > 0:
        assigned_frac = assigned_nonempty / total_nonempty
        if assigned_frac < 0.95:
            conf *= 0.85 + 0.15 * assigned_frac
    return ExtractedTable(
        bbox=table_bbox_pt,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        source="tatr_onnx",
        confidence=round(max(0.0, min(_MAX_TATR_CONF, conf)), 4),
    )


def table_to_html(table: ExtractedTable) -> str:
    if table.num_rows == 0 or table.num_cols == 0 or not table.cells:
        return "<table></table>"
    rows: dict[int, list[TableCell]] = {}
    covered_rows: set[int] = set()
    for cell in table.cells:
        rows.setdefault(cell.row, []).append(cell)
        for covered_r in range(cell.row + 1, min(table.num_rows, cell.row + cell.rowspan)):
            covered_rows.add(covered_r)

    hdr_end = 0
    while hdr_end < table.num_rows:
        rcells = rows.get(hdr_end, ())
        if rcells:
            if all(c.is_header for c in rcells):
                hdr_end += 1
            else:
                break
        elif hdr_end in covered_rows and hdr_end > 0:
            hdr_end += 1
        else:
            break

    while hdr_end > 0:
        crossing = [c for c in table.cells if c.row < hdr_end < min(table.num_rows, c.row + c.rowspan)]
        if not crossing:
            break
        hdr_end = min(c.row for c in crossing)

    def _append_rows(out: list[str], r_start: int, r_stop: int) -> None:
        for r in range(r_start, r_stop):
            row_cells = sorted(rows.get(r, ()), key=lambda c: c.col)
            if not row_cells and r not in covered_rows:
                continue
            out.append("<tr>")
            for cell in row_cells:
                tag = "th" if cell.is_header else "td"
                attrs = ""
                if cell.rowspan > 1:
                    attrs += f' rowspan="{cell.rowspan}"'
                if cell.colspan > 1:
                    attrs += f' colspan="{cell.colspan}"'
                out.append(f"<{tag}{attrs}>{html.escape(cell.text)}</{tag}>")
            out.append("</tr>")

    out = ["<table>"]
    if hdr_end > 0:
        out.append("<thead>")
        _append_rows(out, 0, hdr_end)
        out.append("</thead>")
        if hdr_end < table.num_rows:
            out.append("<tbody>")
            _append_rows(out, hdr_end, table.num_rows)
            out.append("</tbody>")
    else:
        _append_rows(out, 0, table.num_rows)
    out.append("</table>")
    return "".join(out)


class TatrOnnxTableExtractor:
    name = "tatr-onnx"

    def __init__(
        self,
        *,
        device: str = "auto",
        score_threshold: float = DEFAULT_THRESHOLD,
        repo_id: str = TATR_ONNX_REPO_ID,
        revision: str = TATR_ONNX_REVISION,
        model_path: str | Path | None = None,
    ) -> None:
        self.device_name = device
        self.score_threshold = score_threshold
        self.repo_id = repo_id
        self.revision = revision
        self.model_path = model_path
        self._session: Any = None
        self._id2label: dict[int, str] = {}
        self.last_stats: RunStats | None = None

    def ensure_loaded(self) -> bool:
        if self._session is not None:
            return False
        import onnxruntime as ort

        folder = _resolve_model_folder(self.model_path, self.repo_id, self.revision)
        if self.device_name == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            try:
                ort.preload_dlls()
            except Exception:
                pass
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(folder / "onnx" / "model.onnx"), providers=providers)
        self._id2label = _load_id2label(folder)
        return True

    def detect_objects(self, crop: Image.Image) -> list[tuple[str, BBox, float]]:
        self.ensure_loaded()
        rgb = crop.convert("RGB")
        w, h = rgb.size
        if w <= 0 or h <= 0:
            return []
        pad = max(8, min(48, round(0.04 * max(w, h))))
        pw, ph = w + 2 * pad, h + 2 * pad
        padded = Image.new("RGB", (pw, ph), (255, 255, 255))
        padded.paste(rgb, (pad, pad))
        scale = _LONGEST_EDGE / max(pw, ph)
        nw, nh = max(1, round(pw * scale)), max(1, round(ph * scale))
        resized = padded if (pw, ph) == (nw, nh) else padded.resize((nw, nh), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = ((arr.transpose(2, 0, 1) - _MEAN) / _STD)[None]

        logits, pred_boxes = self._session.run(None, {"pixel_values": tensor})
        raw_logits = logits[0]
        ex = np.exp(raw_logits - raw_logits.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        class_probs = probs[:, :-1]
        cls_ids = class_probs.argmax(axis=-1)
        scores = class_probs.max(axis=-1)

        out: list[tuple[str, BBox, float]] = []
        rescue_rows: list[tuple[str, BBox, float]] = []
        rescue_cols: list[tuple[str, BBox, float]] = []
        min_rescue = min(self.score_threshold, _RESCUE_THRESHOLD)
        for cls_id, score, box in zip(cls_ids.tolist(), scores.tolist(), pred_boxes[0].tolist()):
            fscore = float(score)
            if fscore < min_rescue:
                continue
            label = self._id2label.get(int(cls_id))
            if label is None:
                continue
            cx, cy, bw, bh = box
            x0 = max(0.0, min(float(w), (cx - bw * 0.5) * pw - pad))
            y0 = max(0.0, min(float(h), (cy - bh * 0.5) * ph - pad))
            x1 = max(0.0, min(float(w), (cx + bw * 0.5) * pw - pad))
            y1 = max(0.0, min(float(h), (cy + bh * 0.5) * ph - pad))
            if x1 <= x0 or y1 <= y0:
                continue
            item = (label, (x0, y0, x1, y1), fscore)
            if (
                fscore >= self.score_threshold
                or (
                    label == "table spanning cell"
                    and fscore >= _SPAN_GUARD_MIN
                    and self.score_threshold <= _SPAN_THRESHOLD
                )
                or (
                    label == "table column header"
                    and fscore >= _HEADER_RESCUE_MIN
                    and self.score_threshold <= DEFAULT_THRESHOLD
                )
            ):
                out.append(item)
            elif label == "table row":
                rescue_rows.append(item)
            elif label == "table column":
                rescue_cols.append(item)
        if not any(lbl == "table row" for lbl, _, _ in out) and rescue_rows:
            out.extend(rescue_rows)
        if not any(lbl == "table column" for lbl, _, _ in out) and rescue_cols:
            out.extend(rescue_cols)
        return out

    def _extract_crop(
        self,
        image: Image.Image,
        table_bbox_pt: BBox,
        tokens: Sequence[tuple[str, BBox]],
    ) -> ExtractedTable:
        w, h = image.size
        tx0, ty0, tx1, ty1 = table_bbox_pt
        sx = (tx1 - tx0) / w if w > 0 else 0.0
        sy = (ty1 - ty0) / h if h > 0 else 0.0
        if sx > 0.0 and sy > 0.0 and tokens:
            elig_px = [
                (
                    (b[0] - tx0) / sx,
                    (b[1] - ty0) / sy,
                    (b[2] - tx0) / sx,
                    (b[3] - ty0) / sy,
                )
                for text, b in tokens
                if len(text.strip()) >= 3 and (b[2] - b[0]) > 0.0 and (b[3] - b[1]) > 0.0
            ]
            vert = sum(1 for pb in elig_px if (pb[3] - pb[1]) > 1.3 * max(1.0, pb[2] - pb[0]))
            if len(elig_px) >= 5 and vert >= 0.60 * len(elig_px):
                rot_img = image.transpose(Image.Transpose.ROTATE_270)
                fh = float(h)
                rot_tokens = [
                    (
                        text,
                        (
                            fh - (b[3] - ty0) / sy,
                            (b[0] - tx0) / sx,
                            fh - (b[1] - ty0) / sy,
                            (b[2] - tx0) / sx,
                        ),
                    )
                    for text, b in tokens
                ]
                rot_tbl = objects_to_table(
                    self.detect_objects(rot_img),
                    rot_img.size,
                    (0.0, 0.0, float(rot_img.width), float(rot_img.height)),
                    rot_tokens,
                    crop=rot_img,
                )
                mapped_cells = tuple(
                    TableCell(
                        row=c.row,
                        col=c.col,
                        rowspan=c.rowspan,
                        colspan=c.colspan,
                        text=c.text,
                        bbox=(
                            None
                            if c.bbox is None
                            else (
                                round(tx0 + c.bbox[1] * sx, 3),
                                round(ty0 + (fh - c.bbox[2]) * sy, 3),
                                round(tx0 + c.bbox[3] * sx, 3),
                                round(ty0 + (fh - c.bbox[0]) * sy, 3),
                            )
                        ),
                        is_header=c.is_header,
                        confidence=c.confidence,
                    )
                    for c in rot_tbl.cells
                )
                return ExtractedTable(
                    bbox=table_bbox_pt,
                    num_rows=rot_tbl.num_rows,
                    num_cols=rot_tbl.num_cols,
                    cells=mapped_cells,
                    source=rot_tbl.source,
                    confidence=rot_tbl.confidence,
                )
        objects = self.detect_objects(image)
        return objects_to_table(objects, image.size, table_bbox_pt, tokens, crop=image)

    def extract_from_page(
        self,
        page: Any,
        table_boxes_pt: Sequence[BBox],
        *,
        words: Sequence[Word] | None = None,
        is_raster: bool = False,
        ocr_backend: str = "rapidocr-onnx",
        ocr_lang: str | None = None,
        device: str | None = None,
        dpi: int = 200,
    ) -> list[ExtractedTable]:
        if not table_boxes_pt:
            self.last_stats = RunStats(
                load_sec=0.0,
                infer_sec=0.0,
                pages=1,
                peak_rss_mb=peak_rss_mb(),
                in_process=True,
            )
            return []

        t_load0 = time.perf_counter()
        loaded = self.ensure_loaded()
        load_sec = time.perf_counter() - t_load0 if loaded else 0.0

        from ..tables import crop_page_region, ocr_crop_tokens, tokens_in_box

        t0 = time.perf_counter()
        ocr_rec: Any = None
        native_tokens: list[tuple[str, BBox]] = []
        if is_raster:
            from .registry import get as get_backend

            ocr_rec = get_backend(
                "ocr",
                ocr_backend,
                device=self.device_name if device is None else device,
                **({"lang": ocr_lang} if ocr_lang is not None else {}),
            )
        elif words is None:
            from ..text import extract_words

            native_tokens = [(w.text, w.bbox) for w in extract_words(page)]
        else:
            native_tokens = [(w.text, w.bbox) for w in words]

        out: list[ExtractedTable] = []
        for box in table_boxes_pt:
            crop, crop_box = crop_page_region(page, box, dpi)
            if is_raster and ocr_rec is not None:
                box_tokens = ocr_crop_tokens(ocr_rec, crop, crop_box)
            else:
                box_tokens = tokens_in_box(native_tokens, crop_box)
            out.append(self._extract_crop(crop, crop_box, box_tokens))

        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=1,
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return out

    def extract(
        self,
        image: Image.Image,
        table_bbox_pt: BBox,
        tokens: Sequence[tuple[str, BBox]] = (),
    ) -> ExtractedTable:
        t_load0 = time.perf_counter()
        loaded = self.ensure_loaded()
        load_sec = time.perf_counter() - t_load0 if loaded else 0.0

        t0 = time.perf_counter()
        table = self._extract_crop(image, table_bbox_pt, tokens)
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=1,
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return table
