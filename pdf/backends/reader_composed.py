from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

from .._pdf_io import (
    bbox_area,
    bbox_inter_area,
    page_box_to_pdfium_box,
    pdfium_box_to_page_box,
    pdfium_page_box,
    round_bbox,
)
from ..layout import Cluster, postprocess_layout
from ..models import BBox, FigureBox, RasterPage, TextBlock, TextSource
from ..ocr.types import OcrLine
from ..text import cluster_by_overlap, xy_cut_order
from .base import LayoutDetector, RunStats, TextRecognizer
from .rss import peak_rss_mb

DEFAULT_RASTER_DPI = 150
DEFAULT_HIGH_DPI = 275
_PASS2_CONF_THRESHOLD = 0.75
_PASS2_PAD_PT = 4.0
_LIST_OR_HEADING_RE = re.compile(
    r"^(?:[•●○◦▪▫‣⁃]|[-–—*+]\s|\(\d{1,3}\)\s|\d{1,3}(?:\.\d+)*[.)]?\s+\S|[a-zA-Z]\)\s|\[[A-Za-z0-9_.:-]{1,40}\]\s|"
    r"(?:chapter|section|appendix|part)\s+(?:\d+|[A-Z]|[IVXLCDM]+)\b)",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(
    r"^(?:\d{1,3}(?:\.\d+)+\.?\s+\S|(?:chapter|section|appendix|part)\s+(?:\d+|[A-Z]|[IVXLCDM]+)\b)",
    re.IGNORECASE,
)


def _scale_box(box: BBox, sx: float, sy: float) -> BBox:
    return round_bbox((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy))


def _join_ocr_lines(lines: Sequence[OcrLine]) -> str:
    non_empty = [ln for ln in lines if ln.text and ln.text.strip()]
    if not non_empty:
        return ""
    rows = cluster_by_overlap(
        non_empty,
        lambda ln: (ln.bbox[1], ln.bbox[3]),
        lambda ln: ln.bbox[0],
    )
    ordered: list[OcrLine] = []
    for row in sorted(rows, key=lambda r: min(item.bbox[1] for item in r)):
        row.sort(key=lambda item: item.bbox[0])
        ordered.extend(row)
    return " ".join(ln.text.strip() for ln in ordered).strip()


def _estimate_font_size(lines: Sequence[OcrLine], sy: float) -> float:
    heights = [(ln.bbox[3] - ln.bbox[1]) * sy for ln in lines if ln.bbox[3] > ln.bbox[1] and sy > 0.0]
    if not heights:
        return 0.0
    raw = (sum(heights) / len(heights)) * 0.85
    est = round(raw, 2)
    return est if est > 0.0 else max(0.01, round(raw, 4))


def _lines_to_block(
    lines: Sequence[OcrLine],
    sx: float,
    sy: float,
    *,
    label: str,
    source: TextSource = "ocr",
) -> TextBlock | None:
    if not lines:
        return None
    text = _join_ocr_lines(lines)
    if not text:
        return None
    mean_conf = round(sum(ln.confidence for ln in lines) / len(lines), 4)
    ux0 = min(ln.bbox[0] for ln in lines)
    uy0 = min(ln.bbox[1] for ln in lines)
    ux1 = max(ln.bbox[2] for ln in lines)
    uy1 = max(ln.bbox[3] for ln in lines)
    return TextBlock(
        text=text,
        bbox=_scale_box((ux0, uy0, ux1, uy1), sx, sy),
        font_size=_estimate_font_size(lines, sy),
        bold=False,
        source=source,
        confidence=mean_conf,
        words=(),
        label=label,
    )


def _is_heading_or_list_line(text: str) -> bool:
    return bool(_LIST_OR_HEADING_RE.match(text.strip()))


def _overlaps_excluded(box: BBox, excluded_boxes_pt: Sequence[BBox]) -> bool:
    if not excluded_boxes_pt:
        return False
    cand_area = bbox_area(box)
    for ex in excluded_boxes_pt:
        ex_area = bbox_area(ex)
        if ex_area > 0.0 and bbox_inter_area(box, ex) > 0.25 * min(ex_area, cand_area):
            return True
    return False


def _inline_pair_connected(a: tuple[OcrLine, BBox], b: tuple[OcrLine, BBox]) -> bool:
    la, ba = a
    lb, bb = b
    if _SECTION_HEADING_RE.match(la.text.strip()) or _SECTION_HEADING_RE.match(lb.text.strip()):
        return False
    ah = max(1.0, ba[3] - ba[1])
    bh = max(1.0, bb[3] - bb[1])
    ratio = max(ah, bh) / min(ah, bh)
    x_ov = min(ba[2], bb[2]) - max(ba[0], bb[0])
    y_ov = min(ba[3], bb[3]) - max(ba[1], bb[1])
    same_row = y_ov >= 0.40 * min(ah, bh)
    if not same_row and (_is_heading_or_list_line(la.text) or _is_heading_or_list_line(lb.text)):
        return False
    if x_ov > -2.0 and y_ov > -2.0 and ratio <= 3.0:
        return True
    x_gap = max(ba[0] - bb[2], bb[0] - ba[2])
    return same_row and ratio <= 2.2 and x_gap <= min(0.65 * max(ah, bh), 18.0)


def _merge_inline_orphan_items(
    items: list[tuple[OcrLine, BBox]],
    excluded_boxes_pt: Sequence[BBox],
) -> list[tuple[list[tuple[OcrLine, BBox]], BBox]]:
    n = len(items)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    comp_boxes = [it[1] for it in items]
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = _find(i), _find(j)
            if ri == rj or not _inline_pair_connected(items[i], items[j]):
                continue
            bi, bj = comp_boxes[ri], comp_boxes[rj]
            merged_box = (min(bi[0], bj[0]), min(bi[1], bj[1]), max(bi[2], bj[2]), max(bi[3], bj[3]))
            if _overlaps_excluded(merged_box, excluded_boxes_pt):
                continue
            parent[rj] = ri
            comp_boxes[ri] = merged_box

    grouped: dict[int, list[tuple[OcrLine, BBox]]] = defaultdict(list)
    for i, it in enumerate(items):
        grouped[_find(i)].append(it)
    return [(members, comp_boxes[root]) for root, members in grouped.items()]


def _can_merge_orphan(
    group: list[tuple[OcrLine, BBox]],
    grp_box: BBox,
    cur_line: OcrLine,
    cur_box: BBox,
    excluded_boxes_pt: Sequence[BBox],
) -> bool:
    last_line, last_box = group[-1]
    lh = max(1.0, cur_box[3] - cur_box[1])
    last_lh = max(1.0, last_box[3] - last_box[1])
    if max(lh, last_lh) / min(lh, last_lh) > 1.25:
        return False
    if _is_heading_or_list_line(cur_line.text):
        return False

    y_overlap = min(cur_box[3], last_box[3]) - max(cur_box[1], last_box[1])
    if y_overlap >= 0.45 * min(lh, last_lh):
        if cur_box[0] < last_box[0] - 0.5 * lh:
            return False
        if cur_box[0] - last_box[2] > min(1.2 * min(lh, last_lh), 12.0):
            return False
    else:
        last_text = last_line.text.strip()
        cur_text = cur_line.text.strip()
        if _SECTION_HEADING_RE.match(last_text) and (last_lh > lh * 1.1 or cur_text.endswith((".", "!", "?"))):
            return False
        if len(group) == 1 and last_lh > lh * 1.15 and not last_text.endswith((",", ";", "-")):
            return False
        gap_y = cur_box[1] - grp_box[3]
        max_gap = max(0.95 * max(lh, last_lh), 6.0)
        if not (-0.35 * min(lh, last_lh) <= gap_y <= max_gap):
            return False
        if min(cur_box[2], grp_box[2]) <= max(cur_box[0], grp_box[0]):
            return False

    cand_box = (
        min(grp_box[0], cur_box[0]),
        min(grp_box[1], cur_box[1]),
        max(grp_box[2], cur_box[2]),
        max(grp_box[3], cur_box[3]),
    )
    return not _overlaps_excluded(cand_box, excluded_boxes_pt)


def _cluster_orphan_lines(
    orphans: Sequence[OcrLine],
    sx: float,
    sy: float,
    *,
    excluded_boxes_pt: Sequence[BBox] = (),
    source: TextSource = "ocr",
) -> list[TextBlock]:
    items = [(ln, _scale_box(ln.bbox, sx, sy)) for ln in orphans if ln.text and ln.text.strip()]
    if not items:
        return []
    units = _merge_inline_orphan_items(items, excluded_boxes_pt)
    ordered_units = xy_cut_order(units, lambda unit: unit[1])
    groups: list[list[tuple[OcrLine, BBox]]] = []
    group_boxes: list[BBox] = []
    for members, ubox in ordered_units:
        if (
            len(members) == 1
            and groups
            and _can_merge_orphan(groups[-1], group_boxes[-1], members[0][0], ubox, excluded_boxes_pt)
        ):
            groups[-1].append(members[0])
            gb = group_boxes[-1]
            group_boxes[-1] = (min(gb[0], ubox[0]), min(gb[1], ubox[1]), max(gb[2], ubox[2]), max(gb[3], ubox[3]))
        else:
            groups.append(list(members))
            group_boxes.append(ubox)

    blocks: list[TextBlock] = []
    for grp in groups:
        blk = _lines_to_block([ln for ln, _ in grp], sx, sy, label="text", source=source)
        if blk is not None:
            blocks.append(blk)
    return blocks


def _assign_lines_to_clusters(
    lines: list[OcrLine],
    clusters: list[Cluster],
) -> tuple[dict[int, list[OcrLine]], list[OcrLine]]:
    picture_boxes = [c.bbox for c in clusters if c.label == "picture"]
    text_targets = [c for c in clusters if c.label != "picture"]
    by_cluster: dict[int, list[OcrLine]] = defaultdict(list)
    orphans: list[OcrLine] = []

    for line in lines:
        l_area = bbox_area(line.bbox)
        best_cid: int | None = None
        best_frac = 0.2
        if l_area > 0:
            for c in text_targets:
                frac = bbox_inter_area(line.bbox, c.bbox) / l_area
                if frac > best_frac:
                    best_frac = frac
                    best_cid = c.id
        if best_cid is not None:
            by_cluster[best_cid].append(line)
            continue

        cx = (line.bbox[0] + line.bbox[2]) * 0.5
        cy = (line.bbox[1] + line.bbox[3]) * 0.5
        in_picture = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in picture_boxes)
        if not in_picture:
            orphans.append(line)

    return by_cluster, orphans


def _line_matches_cluster_pt(line_pt: BBox, cluster_pt: BBox) -> bool:
    l_area = bbox_area(line_pt)
    if l_area > 0.0 and bbox_inter_area(line_pt, cluster_pt) / l_area > 0.2:
        return True
    cx = (line_pt[0] + line_pt[2]) * 0.5
    cy = (line_pt[1] + line_pt[3]) * 0.5
    return (
        cluster_pt[0] - 2.0 <= cx <= cluster_pt[2] + 2.0
        and cluster_pt[1] - 2.0 <= cy <= cluster_pt[3] + 2.0
    )


class ComposedReader:
    name = "composed"
    isolated = False

    def __init__(
        self,
        detector: LayoutDetector,
        ocr: TextRecognizer,
        *,
        dpi: int = DEFAULT_RASTER_DPI,
        high_dpi: int = DEFAULT_HIGH_DPI,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.dpi = dpi
        self.high_dpi = high_dpi
        self.last_stats: RunStats | None = None

    def _refine_cluster_pass2(
        self,
        pdf_page: pdfium.PdfPage,
        hi_state: dict[str, Any],
        cluster_pt: BBox,
        pass1_lines: list[OcrLine],
        pass1_text: str,
        pass1_conf: float,
        page_w_pt: float,
        page_h_pt: float,
    ) -> list[OcrLine] | None:
        hi_image: Image.Image | None = hi_state.get("image")
        if hi_image is None:
            hi_zoom = self.high_dpi / 72.0
            hi_image = pdf_page.render(scale=hi_zoom).to_pil().convert("RGB")
            hi_w, hi_h = hi_image.size
            hi_state["image"] = hi_image
            hi_state["w"] = hi_w
            hi_state["h"] = hi_h
            hi_state["sx"] = page_w_pt / hi_w
            hi_state["sy"] = page_h_pt / hi_h

        hi_w = int(hi_state["w"])
        hi_h = int(hi_state["h"])
        hi_sx = float(hi_state["sx"])
        hi_sy = float(hi_state["sy"])

        px0 = max(0.0, cluster_pt[0] - _PASS2_PAD_PT)
        py0 = max(0.0, cluster_pt[1] - _PASS2_PAD_PT)
        px1 = min(page_w_pt, cluster_pt[2] + _PASS2_PAD_PT)
        py1 = min(page_h_pt, cluster_pt[3] + _PASS2_PAD_PT)

        hx0 = max(0, min(hi_w, int(round(px0 / hi_sx))))
        hy0 = max(0, min(hi_h, int(round(py0 / hi_sy))))
        hx1 = max(0, min(hi_w, int(round(px1 / hi_sx))))
        hy1 = max(0, min(hi_h, int(round(py1 / hi_sy))))
        if hx1 <= hx0 or hy1 <= hy0:
            return None

        crop = hi_image.crop((hx0, hy0, hx1, hy1))
        crop_lines = self.ocr.recognize(crop)
        if not crop_lines:
            return None

        pass2_lines_pt: list[OcrLine] = []
        for ln in crop_lines:
            if not ln.text or not ln.text.strip():
                continue
            line_pt = (
                round((ln.bbox[0] + hx0) * hi_sx, 3),
                round((ln.bbox[1] + hy0) * hi_sy, 3),
                round((ln.bbox[2] + hx0) * hi_sx, 3),
                round((ln.bbox[3] + hy0) * hi_sy, 3),
            )
            if _line_matches_cluster_pt(line_pt, cluster_pt):
                pass2_lines_pt.append(OcrLine(ln.text, line_pt, ln.confidence))

        if not pass2_lines_pt:
            return None
        pass2_text = _join_ocr_lines(pass2_lines_pt)
        if not pass2_text:
            return None
        pass2_conf = sum(ln.confidence for ln in pass2_lines_pt) / len(pass2_lines_pt)
        if not pass1_text or pass2_conf > pass1_conf:
            return pass2_lines_pt
        return None

    def _read_single_page(
        self,
        pdf_page: pdfium.PdfPage,
        pno: int = 1,
        page_box: BBox | None = None,
    ) -> tuple[RasterPage, list[dict[str, Any]]]:
        target_box = page_box if page_box is not None else pdfium_page_box(pdf_page)
        page_w_pt, page_h_pt = pdf_page.get_size()
        page_w_pt, page_h_pt = float(page_w_pt), float(page_h_pt)
        zoom = self.dpi / 72.0
        image = pdf_page.render(scale=zoom).to_pil().convert("RGB")

        iw, ih = image.size
        sx = page_w_pt / iw
        sy = page_h_pt / ih

        raw_items = self.detector.detect(image)
        clusters = postprocess_layout(raw_items, (float(iw), float(ih)), keep_empty_clusters=True)
        ocr_lines = self.ocr.recognize(image)

        by_cluster, orphans = _assign_lines_to_clusters(ocr_lines, clusters)
        blocks: list[TextBlock] = []
        figures: list[FigureBox] = []
        hi_state: dict[str, Any] = {}

        for c in clusters:
            if c.label == "picture":
                figures.append(
                    FigureBox(
                        bbox=_scale_box(c.bbox, sx, sy),
                        kind="figure",
                        source="layout_picture",
                        confidence=round(float(c.confidence), 4),
                    )
                )
                continue
            if c.label == "table":
                figures.append(
                    FigureBox(
                        bbox=_scale_box(c.bbox, sx, sy),
                        kind="table",
                        source="layout_table",
                        confidence=round(float(c.confidence), 4),
                    )
                )
            cluster_lines = by_cluster.get(c.id, [])
            pass1_text = _join_ocr_lines(cluster_lines) if cluster_lines else ""
            pass1_conf = (sum(ln.confidence for ln in cluster_lines) / len(cluster_lines)) if cluster_lines else 0.0

            used_pass2 = False
            if self.high_dpi > self.dpi and (not pass1_text or pass1_conf < _PASS2_CONF_THRESHOLD):
                cluster_pt = _scale_box(c.bbox, sx, sy)
                refined = self._refine_cluster_pass2(
                    pdf_page, hi_state, cluster_pt, cluster_lines, pass1_text, pass1_conf, page_w_pt, page_h_pt
                )
                if refined is not None:
                    blk = _lines_to_block(refined, 1.0, 1.0, label=c.label, source="ocr_high_dpi")
                    if blk is not None:
                        blocks.append(blk)
                    used_pass2 = True

            if not used_pass2 and cluster_lines:
                blk = _lines_to_block(cluster_lines, sx, sy, label=c.label, source="ocr")
                if blk is not None:
                    blocks.append(blk)

        excluded_boxes_pt = [_scale_box(c.bbox, sx, sy) for c in clusters if c.label in ("picture", "table")]
        blocks.extend(_cluster_orphan_lines(orphans, sx, sy, excluded_boxes_pt=excluded_boxes_pt, source="ocr"))

        if target_box[0] != 0.0 or target_box[1] != 0.0:
            blocks = [replace(b, bbox=round_bbox(pdfium_box_to_page_box(b.bbox, target_box))) for b in blocks]
            figures = [replace(f, bbox=round_bbox(pdfium_box_to_page_box(f.bbox, target_box))) for f in figures]

        page_warnings: list[dict[str, Any]] = []
        if not blocks and any(c.label not in ("picture", "table") for c in clusters):
            page_warnings.append({
                "page": pno,
                "code": "ocr_empty",
                "severity": "warning",
                "reason": "ocr_returned_no_text_for_detected_clusters",
                "fallback": None,
            })

        sizes = sorted(b.font_size for b in blocks if b.font_size > 0.0)
        med_fs = sizes[len(sizes) // 2] if sizes else 0.0
        ordered_blocks = tuple(xy_cut_order(blocks, lambda b: b.bbox, page_rect=target_box, font_size=med_fs))
        return (
            RasterPage(
                blocks=ordered_blocks,
                figures=tuple(figures),
                reading_order=tuple(range(len(ordered_blocks))),
            ),
            page_warnings,
        )

    def read(
        self,
        pdf_path: str | Path,
        pages: Sequence[int],
        *,
        pdfium_doc: Any = None,
        page_boxes: Mapping[int, BBox] | None = None,
    ) -> dict[int, RasterPage]:
        page_list = list(pages)
        if not page_list:
            return {}
        load_sec = 0.0
        for comp in (self.detector, self.ocr):
            ensure = getattr(comp, "ensure_loaded", None)
            if callable(ensure):
                t_load0 = time.perf_counter()
                if ensure():
                    load_sec += time.perf_counter() - t_load0

        t0 = time.perf_counter()
        owns_doc = pdfium_doc is None
        doc_pdf = pdfium.PdfDocument(str(pdf_path)) if owns_doc else pdfium_doc
        out: dict[int, RasterPage] = {}
        warnings: list[dict[str, Any]] = []
        try:
            for pno in page_list:
                if not (1 <= pno <= len(doc_pdf)):
                    raise ValueError(f"page {pno} out of range 1..{len(doc_pdf)}")
                rp, page_warns = self._read_single_page(
                    doc_pdf[pno - 1],
                    pno=pno,
                    page_box=page_boxes.get(pno) if page_boxes else None,
                )
                out[pno] = rp
                warnings.extend(page_warns)
        finally:
            if owns_doc:
                doc_pdf.close()
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=len(page_list),
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
            warnings=tuple(warnings),
        )
        return out

    def read_raster_regions(
        self,
        pdf_path: str | Path,
        regions_by_page: Mapping[int, Sequence[BBox]],
        *,
        pdfium_doc: Any = None,
        dpi: int = DEFAULT_HIGH_DPI,
        page_boxes: Mapping[int, BBox] | None = None,
    ) -> dict[int, list[TextBlock]]:
        if not regions_by_page:
            return {}
        load_sec = 0.0
        ensure = getattr(self.ocr, "ensure_loaded", None)
        if callable(ensure):
            t_load0 = time.perf_counter()
            if ensure():
                load_sec += time.perf_counter() - t_load0

        t0 = time.perf_counter()
        owns_doc = pdfium_doc is None
        doc_pdf = pdfium.PdfDocument(str(pdf_path)) if owns_doc else pdfium_doc
        out: dict[int, list[TextBlock]] = {}
        zoom = dpi / 72.0
        region_source: TextSource = "ocr_high_dpi" if dpi > DEFAULT_RASTER_DPI else "ocr"
        try:
            for pno, regions in regions_by_page.items():
                if not (1 <= pno <= len(doc_pdf)):
                    raise ValueError(f"page {pno} out of range 1..{len(doc_pdf)}")
                region_list = list(regions)
                if not region_list:
                    out[pno] = []
                    continue

                pdf_page = doc_pdf[pno - 1]
                target_box = page_boxes[pno] if (page_boxes and pno in page_boxes) else pdfium_page_box(pdf_page)
                page_w_pt, page_h_pt = pdf_page.get_size()
                page_w_pt, page_h_pt = float(page_w_pt), float(page_h_pt)
                image = pdf_page.render(scale=zoom).to_pil().convert("RGB")
                iw, ih = image.size
                sx = page_w_pt / iw
                sy = page_h_pt / ih

                page_blocks: list[TextBlock] = []
                for region in region_list:
                    fx0, fy0, fx1, fy1 = page_box_to_pdfium_box(region, target_box)
                    rx0 = max(0.0, min(page_w_pt, fx0))
                    ry0 = max(0.0, min(page_h_pt, fy0))
                    rx1 = max(0.0, min(page_w_pt, fx1))
                    ry1 = max(0.0, min(page_h_pt, fy1))
                    if rx1 <= rx0 or ry1 <= ry0:
                        continue
                    hx0 = max(0, min(iw, int(round(rx0 / sx))))
                    hy0 = max(0, min(ih, int(round(ry0 / sy))))
                    hx1 = max(0, min(iw, int(round(rx1 / sx))))
                    hy1 = max(0, min(ih, int(round(ry1 / sy))))
                    if hx1 <= hx0 or hy1 <= hy0:
                        continue

                    crop = image.crop((hx0, hy0, hx1, hy1))
                    crop_lines = self.ocr.recognize(crop)
                    pt_lines = [
                        OcrLine(
                            ln.text,
                            round_bbox(
                                pdfium_box_to_page_box(
                                    (
                                        (ln.bbox[0] + hx0) * sx,
                                        (ln.bbox[1] + hy0) * sy,
                                        (ln.bbox[2] + hx0) * sx,
                                        (ln.bbox[3] + hy0) * sy,
                                    ),
                                    target_box,
                                )
                            ),
                            ln.confidence,
                        )
                        for ln in crop_lines
                        if ln.text and ln.text.strip()
                    ]
                    page_blocks.extend(_cluster_orphan_lines(pt_lines, 1.0, 1.0, source=region_source))

                out[pno] = list(xy_cut_order(page_blocks, lambda b: b.bbox))
        finally:
            if owns_doc:
                doc_pdf.close()

        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=len(regions_by_page),
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return out
