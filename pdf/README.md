# pdf

Lightweight multi-page PDF parser for DocuMind Problem 3 with zero `docling` dependency. Pages with a usable text layer run on PyMuPDF alone; the vendored Heron layout detector (`transformers` + `torch`) and RapidOCR (`onnxruntime`) are imported lazily only when `figures=True` or `ocr=True` is needed.

## 1. Layout and usage

```
pdf/
  __init__.py         public exports (models + extract_ir)
  models.py           frozen dataclasses in PDF points (top-left origin)
  router.py           classify_page(page) -> PageProfile (includes figure_candidate pre-filter)
  text.py             PyMuPDF word extraction + recursive XY-cut block ordering
  figures.py          lazy layout backend + v1 OpenCV grouping with anchor rule + layout_table
  raster.py           lazy layout + RapidOCR fallback for raster pages
  assemble.py         overlap filter, innermost-box word assignment, atomic XY-cut assembly
  ir.py               extract_ir(path, *, figures=True, ocr=True, layout="heron", ocr_backend="rapidocr", device="auto") -> dict
  benchmark.py        isolated timing/RAM benchmark and 20-page OmniDocBench recall
  layout/
    types.py          LayoutLabel, LayoutItem
    detector.py       LayoutDetector protocol, get_layout_detector(), clamp_and_rescale()
    heron_transformers.py  RT-DETR v2 Heron detector (pinned revision)
    postprocess.py    cluster thresholds, overlap resolution, container filtering, item build
    cache.py          cache keyed by backend, model revision, DPI, and PDF SHA-256
    NOTICE            MIT license and source paths from Docling 2.150.0
  ocr/
    types.py          OcrLine
    recognizer.py     TextRecognizer protocol
    rapidocr.py       direct RapidOCR wrapper
  fixtures/layout_dumps/  12 recorded AUTOSAR pages (raw and post-processed layout clusters, boxes only, captured once from Docling 2.150.0) used by test_layout.py for post-processing parity
  test_*.py           M1-M5 and layout verification scripts
```

```python
from pdf import extract_ir

# Pure PyMuPDF CPU path (imports neither torch, transformers, nor rapidocr):
ir_text = extract_ir("datasets/AUTOSAR/AUTOSAR_CP_EXP_VFB.pdf", figures=False, ocr=False)

# Full path (lazy Heron layout for candidate figure pages and RapidOCR for raster pages):
ir_full = extract_ir(
    "datasets/AUTOSAR/AUTOSAR_CP_EXP_VFB.pdf",
    figures=True,
    ocr=True,
    layout="heron",
    ocr_backend="rapidocr",
    device="auto",
)
```

Run tests from the repository root (PowerShell / cmd):

```powershell
.\.venv\Scripts\python -B pdf\test_router.py
.\.venv\Scripts\python -B pdf\test_text.py
.\.venv\Scripts\python -B pdf\test_layout.py
.\.venv\Scripts\python -B pdf\test_figures.py
.\.venv\Scripts\python -B pdf\test_assemble.py
.\.venv\Scripts\python -B pdf\test_raster.py
```

Tests that depend on local untracked datasets (`datasets/AUTOSAR/`, `datasets/OmniDocBench/`) print `SKIP` when those files are absent.

## 2. Pipeline rules

- **Router (`router.py`):** classifies each page as `native`, `raster`, or `hybrid` for text-source selection (`raster` when `word_count < 5` and a single embedded image covers `>= 0.96` of the page; `native` when `word_count >= 5`, `text_area_ratio >= 0.002`, `max_image_cover < 0.96`, and `not suspect_ocr_layer`; blank pages with zero text and zero images are `native`; all others are `hybrid`), and sets `figure_candidate` from non-background drawings/embedded images.
- **Text pipeline (`text.py`):** uses `pymupdf` `get_text("words")` as the word baseline and maps `font_size` and `bold` from `get_text("dict")` spans. Groups words to line segments (splitting across wide column gaps) and blocks, then orders blocks with a recursive XY-cut ported from `ir_postprocessor.py` (scaled by `72/150`).
- **Image pipeline (`figures.py`, `pdf/layout/`):** renders pages via PyMuPDF Pixmap in memory (`150 DPI`), uses `LayoutCache` keyed by `(backend, revision, dpi, pdf_sha256)`, and lazily loads `HeronDetector` (`docling-project/docling-layout-heron`, revision `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`) only on cache misses. Runs `postprocess_layout(..., keep_empty_clusters=False)` and `clusters_to_items()`, then OpenCV/numpy `v1` grouping with the `anchor` rule (`min_area = 0.015`, `footer_frac = 0.88`, `refine=False`, `add_missing=False`) and collects `layout_table` boxes. All `figures.py` boxes carry a fixed confidence (`0.95`).
- **Assembler & IR (`assemble.py`, `ir.py`):** drops any `table` box overlapped `>= 50%` by a `figure` box; assigns each word whose center lies in a box to the smallest-area enclosing box (`(area, box_index)` tie-break) before running body XY-cut; orders `[body TextBlocks + atomic figures/tables]` via page-level XY-cut; emits `figure` inner words (via local XY-cut) and `table` inner `order_blocks` as `level=2` child `text` elements sequentially after their parent (`reading_order` restarts at `0` per page).
- **Raster path (`raster.py`, `pdf/ocr/`):** runs `HeronDetector` + `postprocess_layout(..., keep_empty_clusters=True)` and `RapidOcrRecognizer` directly on single `raster` pages, assigns OCR lines to text-like clusters (preserving `metadata.layout_label`) and keeps orphan lines outside `picture` boxes, then orders blocks via XY-cut. When `ocr=False`, `raster` pages are listed in `metadata.unreadable_pages`.

## 3. IR differences

- **From `parsers/pptx_ir.py`:** uses the same 7 top-level keys (`doc_id`, `source_file`, `doc_type="pdf"`, `metadata`, `elements`, `tables=[]`, `diagrams=[]`) and `coord_unit="pt"`, `reading_order` 0-based per page. Differences: PDF elements emit `type` in `{"text", "figure", "table"}`, include top-level `source` and `confidence`, preserve `font_size` and `bold` in `elements[].metadata` (and `layout_label` for OCR blocks), omit `table_id` while `tables == []`, and use `source_ref: {file, page, bbox}` (no `shape_id`).
- **From `docs/01_architecture_and_ir_schema.md` (`feat/structure-and-diagram-recovery`):** coordinates are PDF `pt` instead of page pixels; `reading_order` is per-page 0-based instead of document-wide; `diagrams` is `[]` and `tables` is `[]` (no HTML or cell structure yet); `type` is normalized to `text`/`figure`/`table`; every element carries `source` (`text_layer` / `ocr` / `v1_group` / `layout_picture` / `layout_table`).

## 4. Measured results (2026-11-16, Windows, GTX 1650 + `.venv` Python 3.12, zero Docling)

- **Router:** AUTOSAR `Layered` (`{'native': 185}`), `VFB` (`{'native': 147}`), `ECUStateManager` (`{'native': 245}`), `OS` (`{'native': 407}`); 30 OmniDocBench PNGs wrapped as PDFs (`{'raster': 30}`).
- **Text pipeline:** 185-page `Layered` PDF ran in `3.032 s` (`16.39 ms/page`) with exact word parity against `get_text("words")` and zero `torch`/`transformers` imports.
- **Figure boxes (`detect_figures`, `include_tables=False`, `sys.modules["docling"] = None`, fresh cache):**
  - `test` (108 pages across 4 AUTOSAR PDFs): strict F1 **`0.959`** (`58/2/3`, `P=0.967, R=0.951`), lenient F1 **`0.926`** (`56/4/5`, `P=0.933, R=0.918`), negative-page FP **`0`** (matches the Docling baseline to three decimal places).
- **Raster OCR recall (20 English OmniDocBench pages via `pdf/benchmark.py --omnidoc-recall`, `HeronDetector` + `RapidOcrRecognizer`):**
  - `pooled_char_recall = 0.9020`, `mean = 0.8872`, `median = 0.9507`, `min = 0.2298` (Docling OCR baseline on the same 20 pages: `0.8866 / 0.8658 / 0.9532 / 0.2190`).
- **Isolated subprocess benchmark (5 pages per mode via `pdf/benchmark.py`):**

| Mode | Import (s) | First call / page (s) | Steady per page (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|
| `native` (`figures=False, ocr=False`) | 0.0178 | 0.0632 (5 pages) | 0.0126 | 51.88 |
| `figures_cached` (5 pages, `_layout_cache` hit) | 0.0179 | 0.0881 | 0.0536 | 94.02 |
| `figures_live` (5 pages, `HeronDetector`) | 0.0169 | 22.4569 | 0.3338 | 1524.91 |
| `raster` (5 pages, `HeronDetector` + `RapidOCR`) | 0.0389 | 37.0684 | 6.1676 | 2215.66 |

## 5. Known limits

- `HeronDetector` still uses PyTorch and `transformers` (no ONNX export of Heron in `docling-project/docling-layout-heron`); only `figures=False, ocr=False` avoids loading `torch`.
- No PDF table cell extraction yet (`tables` is `[]`; words inside table boxes appear as `level=2` child `text` elements).
- No heading/hierarchy tree detection yet (body text, figures, and tables are `level=1`; M7 deferred).
- No diagram node-edge reconstruction (`diagrams` is `[]`).
- Repeating running headers/footers and sub-visible watermarks (e.g., 2 pt text) are not filtered by default (`min_font_size=0.0`) so word parity with `get_text("words")` holds.
- If `v1_group` absorbs an adjacent `table` (`<= 45 px` gap) into a figure box, the `>= 50%` overlap filter drops the standalone `table` box.
- `hybrid` pages with `suspect_ocr_layer=True` still use `text.py` and emit a `metadata.warnings` entry rather than falling back to `raster.py`.
