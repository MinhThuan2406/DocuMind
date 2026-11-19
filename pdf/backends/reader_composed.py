from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ..layout import Cluster, postprocess_layout
from ..models import BBox, FigureBox, RasterPage, TextBlock
from ..ocr.types import OcrLine
from ..text import xy_cut_order
from .base import LayoutDetector, RunStats, TextRecognizer
from .rss import peak_rss_mb

DEFAULT_RASTER_DPI = 150
LAYOUT_BOX_CONFIDENCE = 0.95


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter_area(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _scale_box(box: BBox, sx: float, sy: float) -> BBox:
    return (
        round(box[0] * sx, 3),
        round(box[1] * sy, 3),
        round(box[2] * sx, 3),
        round(box[3] * sy, 3),
    )


def _join_ocr_lines(lines: list[OcrLine]) -> str:
    ordered = sorted(lines, key=lambda ln: (round((ln.bbox[1] + ln.bbox[3]) * 0.5 / 8.0), ln.bbox[0]))
    return " ".join(ln.text for ln in ordered if ln.text).strip()


def _assign_lines_to_clusters(
    lines: list[OcrLine],
    clusters: list[Cluster],
) -> tuple[dict[int, list[OcrLine]], list[OcrLine]]:
    picture_boxes = [c.bbox for c in clusters if c.label == "picture"]
    text_targets = [c for c in clusters if c.label != "picture"]
    by_cluster: dict[int, list[OcrLine]] = defaultdict(list)
    orphans: list[OcrLine] = []

    for line in lines:
        l_area = _area(line.bbox)
        best_cid: int | None = None
        best_frac = 0.2
        if l_area > 0:
            for c in text_targets:
                frac = _inter_area(line.bbox, c.bbox) / l_area
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


class ComposedReader:
    name = "composed"
    isolated = False

    def __init__(
        self,
        detector: LayoutDetector,
        ocr: TextRecognizer,
        *,
        dpi: int = DEFAULT_RASTER_DPI,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.dpi = dpi
        self.last_stats: RunStats | None = None

    def _read_single_page(self, pdf_page: pdfium.PdfPage) -> RasterPage:
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

        for c in clusters:
            if c.label == "picture":
                figures.append(
                    FigureBox(
                        bbox=_scale_box(c.bbox, sx, sy),
                        kind="figure",
                        source="layout_picture",
                        confidence=LAYOUT_BOX_CONFIDENCE,
                    )
                )
                continue
            if c.label == "table":
                figures.append(
                    FigureBox(
                        bbox=_scale_box(c.bbox, sx, sy),
                        kind="table",
                        source="layout_table",
                        confidence=LAYOUT_BOX_CONFIDENCE,
                    )
                )
            cluster_lines = by_cluster.get(c.id)
            if not cluster_lines:
                continue
            text = _join_ocr_lines(cluster_lines)
            if not text:
                continue
            mean_conf = round(sum(ln.confidence for ln in cluster_lines) / len(cluster_lines), 4)
            ux0 = min(ln.bbox[0] for ln in cluster_lines)
            uy0 = min(ln.bbox[1] for ln in cluster_lines)
            ux1 = max(ln.bbox[2] for ln in cluster_lines)
            uy1 = max(ln.bbox[3] for ln in cluster_lines)
            blocks.append(
                TextBlock(
                    text=text,
                    bbox=_scale_box((ux0, uy0, ux1, uy1), sx, sy),
                    font_size=0.0,
                    bold=False,
                    source="ocr",
                    confidence=mean_conf,
                    words=(),
                    label=c.label,
                )
            )

        for ln in orphans:
            blocks.append(
                TextBlock(
                    text=ln.text,
                    bbox=_scale_box(ln.bbox, sx, sy),
                    font_size=0.0,
                    bold=False,
                    source="ocr",
                    confidence=ln.confidence,
                    words=(),
                    label="text",
                )
            )

        ordered_blocks = tuple(xy_cut_order(blocks, lambda b: b.bbox))
        return RasterPage(
            blocks=ordered_blocks,
            figures=tuple(figures),
            reading_order=tuple(range(len(ordered_blocks))),
        )

    def read(self, pdf_path: str | Path, pages: Sequence[int]) -> dict[int, RasterPage]:
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
        doc_pdf = pdfium.PdfDocument(str(pdf_path))
        out: dict[int, RasterPage] = {}
        try:
            for pno in page_list:
                if not (1 <= pno <= len(doc_pdf)):
                    raise ValueError(f"page {pno} out of range 1..{len(doc_pdf)}")
                out[pno] = self._read_single_page(doc_pdf[pno - 1])
        finally:
            doc_pdf.close()
        self.last_stats = RunStats(
            load_sec=round(load_sec, 4),
            infer_sec=round(time.perf_counter() - t0, 4),
            pages=len(page_list),
            peak_rss_mb=peak_rss_mb(),
            in_process=True,
        )
        return out
