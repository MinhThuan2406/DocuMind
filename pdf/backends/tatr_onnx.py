from __future__ import annotations

import html
import json
import re
import time
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
_RESCUE_THRESHOLD = 0.30
_SPAN_OVERLAP_FRAC = 0.55
_LONGEST_EDGE = 800
_MAX_TATR_CONF = 0.96
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
_MULTICELL_SPLIT_RE = re.compile(r"\s+|(?<=\.\.)(?=[^.\s])|(?<=…)(?=\S)")


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


def _split_multicell_tokens(
    tokens: Sequence[tuple[str, BBox]],
    cell_defs: Sequence[tuple[int, int, int, int, BBox, bool, float]],
) -> tuple[list[tuple[str, BBox]], set[tuple[str, BBox]]]:
    out: list[tuple[str, BBox]] = []
    split_pieces: set[tuple[str, BBox]] = set()
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
                total_len = max(1, len(text))
                tw = tb[2] - tb[0]
                pos = 0
                for piece in pieces:
                    start = text.find(piece, pos)
                    end = start + len(piece)
                    pos = end
                    wx0 = tb[0] + (start / total_len) * tw
                    wx1 = tb[0] + (end / total_len) * tw
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


def _split_undersegmented_rows(
    row_intervals: list[tuple[float, float]],
    row_scores: list[float],
    col_intervals: Sequence[tuple[float, float]],
    tok_px: Sequence[BBox],
    ch: float,
    hdr_rows: set[int],
) -> tuple[list[tuple[float, float]], list[float], bool]:
    if not tok_px or len(col_intervals) < 2:
        return row_intervals, row_scores, False
    out_rows: list[tuple[float, float]] = []
    out_scores: list[float] = []
    did_split = False
    for r_idx, ((ry0, ry1), score) in enumerate(zip(row_intervals, row_scores)):
        if r_idx in hdr_rows:
            out_rows.append((ry0, ry1))
            out_scores.append(score)
            continue
        rtoks = [b for b in tok_px if ry0 <= (b[1] + b[3]) * 0.5 <= ry1]
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
            if active_cols >= 2 and all(
                len(
                    {
                        ci
                        for b in rtoks
                        if y0 <= (b[1] + b[3]) * 0.5 <= y1
                        for ci, (c0, c1) in enumerate(col_intervals)
                        if c0 <= (b[0] + b[2]) * 0.5 <= c1
                    }
                )
                >= req_cols
                for y0, y1, _ in ycls
            ):
                sub = _snap_intervals(
                    [(y0 - ry0, y1 - ry0, 0.7) for y0, y1, _ in ycls],
                    ry1 - ry0,
                    [(b[1] - ry0, b[3] - ry0) for b in rtoks],
                )
                if len(sub) == len(ycls):
                    did_split = True
                    sub_score = min(score, 0.85)
                    for s0, s1 in sub:
                        out_rows.append((ry0 + s0, ry0 + s1))
                        out_scores.append(sub_score)
                    continue
        out_rows.append((ry0, ry1))
        out_scores.append(score)
    return out_rows, out_scores, did_split


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


def _collapse_multiline_rows(
    cells: list[TableCell],
    num_rows: int,
    num_cols: int,
) -> tuple[tuple[TableCell, ...], int]:
    # Collapse row blocks where most columns share a rowspan and at most two wrap across sub-rows.
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
                    n_multi = 0
                    for col_i in range(num_cols):
                        col_cells = [c for c in block_cells if c.col <= col_i < c.col + c.colspan]
                        if len(col_cells) == 1 and col_cells[0].row == r and col_cells[0].rowspan == k:
                            n_spanning += 1
                        elif sum(1 for c in col_cells if c.text.strip()) >= 2:
                            n_multi += 1
                    if (
                        n_multi >= 1
                        and n_spanning >= max(2, (num_cols + 1) // 2)
                        and (n_multi == 1 or n_spanning > 2 * n_multi)
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


def objects_to_table(
    objects: Sequence[tuple[str, BBox, float]],
    crop_size: tuple[int, int],
    table_bbox_pt: BBox,
    tokens: Sequence[tuple[str, BBox]] = (),
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

    row_intervals = _snap_intervals(row_kept, ch, [(b[1], b[3]) for b in tok_px])
    col_intervals = _snap_intervals(col_kept, cw, [(b[0], b[2]) for b in tok_px], ignore_cross_spans=True)
    row_scores = [item[2] for item in row_kept]
    col_scores = [item[2] for item in col_kept]

    col_headers = [b for lbl, b, _ in objects if lbl == "table column header"]
    span_boxes = [b for lbl, b, _ in objects if lbl in ("table spanning cell", "table projected row header")]
    col_intervals, col_scores, did_split_cols = _split_undersegmented_cols(
        col_intervals, col_scores, tok_px, cw, col_headers, span_boxes
    )
    init_header_rows: set[int] = set()
    for hb in col_headers:
        init_header_rows.update(_overlap_indices(row_intervals, hb[1], hb[3]))
    row_intervals, row_scores, did_split_rows = _split_undersegmented_rows(
        row_intervals, row_scores, col_intervals, tok_px, ch, init_header_rows
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
        [(b, s, False) for lbl, b, s in objects if lbl == "table spanning cell" and s >= _SPAN_THRESHOLD]
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
        claimed.update(coords)
        merged_specs.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1, is_proj, float(score)))

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
    expanded_tokens, split_pieces = _split_multicell_tokens(tokens, cell_defs)
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
    cells, num_rows = _collapse_multiline_rows(raw_cells, num_rows, num_cols)

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
        scale = _LONGEST_EDGE / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = rgb if (w, h) == (nw, nh) else rgb.resize((nw, nh), resample=Image.Resampling.BILINEAR)
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
            x0 = max(0.0, min(float(w), (cx - bw * 0.5) * w))
            y0 = max(0.0, min(float(h), (cy - bh * 0.5) * h))
            x1 = max(0.0, min(float(w), (cx + bw * 0.5) * w))
            y1 = max(0.0, min(float(h), (cy + bh * 0.5) * h))
            if x1 <= x0 or y1 <= y0:
                continue
            item = (label, (x0, y0, x1, y1), fscore)
            if fscore >= self.score_threshold:
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
        return objects_to_table(objects, image.size, table_bbox_pt, tokens)

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
