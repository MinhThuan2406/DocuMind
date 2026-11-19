from __future__ import annotations

import html
import json
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
_SPAN_THRESHOLD = 0.68
_RESCUE_THRESHOLD = 0.30
_SPAN_OVERLAP_FRAC = 0.55
_LONGEST_EDGE = 800
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


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
    items = sorted((b[axis], b[axis + 2]) for b in tok_px if b[axis + 2] > b[axis])
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
    return [(max(0.0, c[0]), min(limit, c[1]), 0.6) for c in clusters]


def _snap_intervals(
    intervals: list[tuple[float, float, float]],
    limit: float,
    tok_spans: Sequence[tuple[float, float]] = (),
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    bounds = [0.0]
    for idx in range(len(intervals) - 1):
        mid = (intervals[idx][1] + intervals[idx + 1][0]) * 0.5
        if tok_spans and any(t0 + 1.0 < mid < t1 - 1.0 for t0, t1 in tok_spans):
            w_prev = max(1.0, intervals[idx][1] - intervals[idx][0])
            w_next = max(1.0, intervals[idx + 1][1] - intervals[idx + 1][0])
            lo_win = max(bounds[-1] + 1.0, intervals[idx][1] - 0.35 * w_prev, mid - 0.45 * min(w_prev, w_next))
            hi_win = min(limit - 1.0, intervals[idx + 1][0] + 0.35 * w_next, mid + 0.45 * min(w_prev, w_next))
            if hi_win > lo_win:
                pts = sorted(
                    {mid, lo_win, hi_win}
                    | {edge for t0, t1 in tok_spans for edge in (t0, t1) if lo_win <= edge <= hi_win}
                )
                cands = list(pts) + [(pts[k] + pts[k + 1]) * 0.5 for k in range(len(pts) - 1)]
                best_x = mid
                best_key = (sum(1 for t0, t1 in tok_spans if t0 + 1.0 < mid < t1 - 1.0), 0.0)
                for x in cands:
                    cuts = sum(1 for t0, t1 in tok_spans if t0 + 1.0 < x < t1 - 1.0)
                    key = (cuts, abs(x - mid))
                    if key < best_key:
                        best_key = key
                        best_x = x
                mid = best_x
        bounds.append(mid)
    bounds.append(limit)
    out: list[tuple[float, float]] = []
    for idx in range(len(intervals)):
        lo = bounds[idx]
        hi = max(lo + 1.0, bounds[idx + 1])
        out.append((lo, min(limit, hi) if limit > lo + 1.0 else hi))
    out[-1] = (out[-1][0], limit)
    return out


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
) -> list[tuple[str, BBox]]:
    out: list[tuple[str, BBox]] = []
    for text, tb in tokens:
        words = text.split()
        if len(words) >= 2 and (tb[2] - tb[0]) > 0.0:
            tcy = (tb[1] + tb[3]) * 0.5
            row_hits = sum(
                1
                for _, _, _, _, cb, _, _ in cell_defs
                if cb[1] <= tcy <= cb[3] and min(tb[2], cb[2]) - max(tb[0], cb[0]) >= 0.12 * max(1.0, cb[2] - cb[0])
            )
            if row_hits >= 2:
                total_len = max(1, len(text))
                tw = tb[2] - tb[0]
                pos = 0
                for word in words:
                    start = text.find(word, pos)
                    end = start + len(word)
                    pos = end
                    wx0 = tb[0] + (start / total_len) * tw
                    wx1 = tb[0] + (end / total_len) * tw
                    out.append((word, (wx0, tb[1], max(wx0 + 0.5, wx1), tb[3])))
                continue
        out.append((text, tb))
    return out


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
    row_kept = _dedupe_1d_intervals(raw_rows, axis=1)
    col_kept = _dedupe_1d_intervals(raw_cols, axis=0)
    if tok_px:
        y_clusters = _cluster_1d_tokens(tok_px, axis=1, limit=ch)
        if not row_kept or (len(row_kept) == 1 and len(y_clusters) >= 3):
            row_kept = y_clusters
        if not col_kept and row_kept:
            col_kept = [(0.0, cw, 0.5)]

    row_intervals = _snap_intervals(row_kept, ch, [(b[1], b[3]) for b in tok_px])
    col_intervals = _snap_intervals(col_kept, cw, [(b[0], b[2]) for b in tok_px])
    row_scores = [item[2] for item in row_kept]
    col_scores = [item[2] for item in col_kept]

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

    col_headers = [b for lbl, b, _ in objects if lbl == "table column header"]
    header_rows: set[int] = set()
    for hb in col_headers:
        header_rows.update(_overlap_indices(row_intervals, hb[1], hb[3]))

    proj_headers = sorted(
        [(b, s) for lbl, b, s in objects if lbl == "table projected row header"],
        key=lambda item: item[1],
        reverse=True,
    )
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
        cell_conf = round(max(0.0, min(1.0, (r_avg + c_avg + span_score) / 3.0)), 4)
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
            cell_conf = round(max(0.0, min(1.0, (row_scores[r] + col_scores[c]) * 0.5)), 4)
            cell_defs.append((r, c, 1, 1, pt_box, is_hdr, cell_conf))

    cell_defs.sort(key=lambda item: (item[0], item[1]))
    expanded_tokens = _split_multicell_tokens(tokens, cell_defs)
    cell_tokens: list[list[tuple[str, BBox]]] = [[] for _ in cell_defs]
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

    cells = tuple(
        TableCell(
            row=r,
            col=c,
            rowspan=rspan,
            colspan=cspan,
            text=_sort_cell_tokens(cell_tokens[idx]),
            bbox=cb,
            is_header=is_hdr,
            confidence=cell_conf,
        )
        for idx, (r, c, rspan, cspan, cb, is_hdr, cell_conf) in enumerate(cell_defs)
    )
    table_scores = [s for lbl, _, s in objects if lbl == "table"]
    conf = round(max(table_scores) if table_scores else sum(s for _, _, s in objects) / max(1, len(objects)), 4)
    return ExtractedTable(
        bbox=table_bbox_pt,
        num_rows=num_rows,
        num_cols=num_cols,
        cells=cells,
        source="tatr_onnx",
        confidence=conf,
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

    def extract_from_page(
        self,
        page: Any,
        table_boxes_pt: Sequence[BBox],
        *,
        words: Sequence[Word] | None = None,
        is_raster: bool = False,
        ocr_backend: str = "rapidocr-onnx",
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

            ocr_rec = get_backend("ocr", ocr_backend, device=self.device_name if device is None else device)
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
            objects = self.detect_objects(crop)
            out.append(objects_to_table(objects, crop.size, crop_box, box_tokens))

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
        objects = self.detect_objects(image)
        table = objects_to_table(objects, image.size, table_bbox_pt, tokens)
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=1,
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return table
