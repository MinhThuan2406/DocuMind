# pdf

Lightweight multi-page PDF parser for DocuMind Problem 3 with zero `docling` or `pymupdf` dependency. Pages with a usable text layer run on `pdfplumber` (+ `pypdfium2` for page rendering and bookmarks); the swappable layout detectors (`heron-onnx` by default via `onnxruntime`, `heron` via `transformers` + `torch`, or `mineru-pipeline` in an isolated venv), table extractors (`auto` by default, or `tatr-onnx` / `pdfplumber`), and OCR recognizers (`rapidocr-onnx` by default, or `rapidocr-torch`) are imported lazily only when `figures=True`, `tables=True`, or `ocr=True` is needed.

## 1. Layout and usage

```
pdf/
  __init__.py         public exports (models + extract_ir)
  _pdf_io.py          shared pdfplumber / pypdfium2 helpers (bbox, word extraction, ToC, page/crop rendering)
  models.py           frozen dataclasses in PDF points (top-left origin)
  router.py           classify_page(page) -> PageProfile (includes figure_candidate pre-filter)
  text.py             pdfplumber word extraction (x_tolerance_ratio=0.15) + recursive XY-cut block ordering
  grouping.py         FigureGrouper implementations: v1_anchor (OpenCV + DSU) and raw_picture
  figures.py          lazy layout backend + cache + grouper + layout_table boxes
  tables.py           lazy table extractor dispatch (crop render + native tokens / OCR fallback)
  raster.py           lazy PageReader dispatch for raster pages
  assemble.py         overlap filter, innermost-box word assignment, atomic XY-cut assembly, running header/footer tagging
  hierarchy.py        heading detection, section tree builder (apply_structure, build_section_tree), caption linking
  ir.py               extract_ir(path, *, figures=True, ocr=True, tables=False, ..., structure=True) -> dict
  benchmark.py        isolated timing/RAM benchmark (native, figures_cached, figures_live, raster)
  backends/
    base.py           Protocols (LayoutDetector, LayoutBackend, TextRecognizer, PageReader, FigureGrouper, TableExtractor), PageLayout, RunStats
    registry.py       register(), get(kind, name, **opts), registered(), lazy built-in factories
    heron_onnx.py     RT-DETR v2 Heron ONNX detector (default layout/detector backend)
    heron.py          RT-DETR v2 Heron PyTorch/transformers detector
    rapidocr.py       RapidOCR recognizer (onnxruntime or torch engine)
    tatr_onnx.py      Table Transformer v1.1 ONNX structure extractor
    table_pdfplumber.py PdfPlumber TableFinder native table extractor (table="pdfplumber")
    table_auto.py     Hybrid table extractor (default table="auto": pdfplumber grid + TATR header refinement / fallback)
    mineru.py         MinerULayoutBackend and MinerUReader (isolated venv subprocess)
    mineru_child.py   child script run inside the MinerU venv
    layout_image.py   ImageLayoutBackend wrapping per-image LayoutDetector at 96.012 DPI PNG scale
    layout_cache.py   CachedLayoutBackend & LayoutCache keyed by backend, revision, DPI, and PDF SHA-256
    reader_composed.py ComposedReader combining a LayoutDetector and TextRecognizer for raster pages
    runtime.py        shared subprocess runner and venv lookup for isolated backends
    rss.py            stdlib-only peak RSS helper shared by backends and benchmark
  layout/
    types.py          LAYOUT_LABELS, LayoutLabel, LayoutItem
    detector.py       clamp_and_rescale()
    postprocess.py    cluster thresholds, overlap resolution, container filtering, item build
    NOTICE            MIT license and source paths from Docling 2.150.0
  ocr/
    types.py          OcrLine
  eval/
    gt.py             AUTOSAR figure and OmniDocBench character GT loaders
    metrics.py        strict/lenient figure matching, PRF, character counts, markup stripping
    compare.py        CLI comparing N backends with --split dev|test (--final guard on test)
    compare_hierarchy.py CLI comparing heading hierarchy modes (no_toc vs with_toc) against PDF bookmarks
  fixtures/
    layout_dumps/     12 recorded AUTOSAR pages (raw and post-processed layout clusters, boxes only, captured once from Docling 2.150.0) used by test_layout.py
    mineru_dumps/     stripped MinerU middle_json fixture for unit tests
  test_*.py           router, text, layout, figures, assemble, raster, eval, and hierarchy test suites
```

```python
from pdf import extract_ir

# Pure pdfplumber CPU path (imports neither pymupdf, onnxruntime, torch, transformers, nor rapidocr):
ir_text = extract_ir("datasets/AUTOSAR/AUTOSAR_CP_EXP_VFB.pdf", figures=False, ocr=False)

# Full path (default ONNX backends for Heron layout and RapidOCR):
ir_full = extract_ir(
    "datasets/AUTOSAR/AUTOSAR_CP_EXP_VFB.pdf",
    figures=True,
    ocr=True,
    layout="heron-onnx",
    grouper="v1_anchor",
    reader="composed",
    ocr_backend="rapidocr-onnx",
    device="auto",
)
```

Run tests and comparisons from the repository root (PowerShell / cmd):

```powershell
.\.venv\Scripts\python -B pdf\test_router.py
.\.venv\Scripts\python -B pdf\test_text.py
.\.venv\Scripts\python -B pdf\test_layout.py
.\.venv\Scripts\python -B pdf\test_figures.py
.\.venv\Scripts\python -B pdf\test_assemble.py
.\.venv\Scripts\python -B pdf\test_raster.py
.\.venv\Scripts\python -B pdf\test_eval.py
.\.venv\Scripts\python -B pdf\test_hierarchy.py

# Compare backends on dev split or OmniDocBench raster:
.\.venv\Scripts\python -B -m pdf.eval.compare --task figures --backends heron,heron-onnx --split dev --no-cache
.\.venv\Scripts\python -B -m pdf.eval.compare --task raster --backends heron-onnx+rapidocr-onnx,heron-onnx+rapidocr-torch
```

Tests that depend on local untracked datasets (`datasets/AUTOSAR/`, `datasets/OmniDocBench/`) or an isolated MinerU venv print `SKIP` when those files are absent.

## 2. Pipeline rules

- **Router (`router.py`):** classifies each page as `native`, `raster`, or `hybrid` for text-source selection (`raster` when `word_count < 5` and a single embedded image covers `>= 0.96` of the page; `native` when `word_count >= 5`, `text_area_ratio >= 0.002`, `max_image_cover < 0.96`, and `not suspect_ocr_layer`; blank pages with zero text and zero images are `native`; all others are `hybrid`), and sets `figure_candidate` from non-background drawings/embedded images.
- **Text pipeline (`text.py`):** uses `pdfplumber` `extract_words(x_tolerance_ratio=0.15, extra_attrs=["fontname", "size"])` as the word baseline with font size and bold detection from `fontname`. Groups words to line segments (splitting across wide column gaps) and blocks, then orders blocks with a recursive XY-cut ported from `ir_postprocessor.py` (scaled by `72/150`).
- **Backends & registry (`pdf/backends/registry.py`):** resolves backends lazily by `(kind, name)`:
  - `detector` / `layout`: `heron-onnx` (default, `docling-project/docling-layout-heron-onnx` revision `40bde044036bb181c130ddf6c51792187268748f`), `heron` (`docling-project/docling-layout-heron` revision `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`), `mineru-pipeline` (`layout` only, runs via `DOCUMIND_MINERU_PYTHON` or `~/.venvs/mineru-4.2.3`).
  - `ocr`: `rapidocr-onnx` (default), `rapidocr-torch`, `rapidocr` (CPU ONNX).
  - `reader`: `composed` (default, combines `layout` detector + `ocr_backend`), `heron-onnx+rapidocr-onnx`, `heron-onnx+rapidocr-torch`, `mineru-pipeline`.
  - `grouper`: `v1_anchor` (default, `source="v1_group"`), `raw_picture` (`source="layout_picture"`).
  - `table`: `auto` (default, hybrid `pdfplumber` + `tatr-onnx` with TATR header refinement and borderless/raster fallback), `tatr-onnx` (`Xenova/table-transformer-structure-recognition-v1.1-all`), `pdfplumber` (native bordered tables via `page.find_tables()`).
- **Image pipeline (`figures.py`, `grouping.py`, `pdf/layout/`):** renders pages via `pypdfium2` (`150 DPI`), uses `CachedLayoutBackend` / `LayoutCache` keyed by `(backend, revision, dpi, pdf_sha256)`, and runs the layout backend lazily on cache misses (`ImageLayoutBackend` rescales to `96.012 DPI`, runs `postprocess_layout(..., keep_empty_clusters=False)` and `clusters_to_items()`). Applies the selected `FigureGrouper` (`v1_anchor` uses OpenCV/numpy + DSU with `min_area = 0.015`, `footer_frac = 0.88`) and collects `layout_table` boxes (`include_tables=True`). All `figures.py` boxes carry a fixed confidence (`0.95`).
- **Assembler & IR (`assemble.py`, `hierarchy.py`, `ir.py`):** drops any `table` box overlapped `>= 50%` by a `figure` box; assigns each word whose center lies in a box to the smallest-area enclosing box (`(area, box_index)` tie-break) before running body XY-cut; orders `[body TextBlocks + atomic figures/tables]` via page-level XY-cut; emits `figure` inner words (via local XY-cut) and `table` inner blocks as `level=2` child `text` elements sequentially after their parent (`reading_order` restarts at `0` per page). On `raster` pages, `TextBlock`s inside `table` boxes also nest as `level=2` children. When `tables=True`, `extract_tables_on_page()` parses table structure into `ir["tables"]` and links `metadata.table_id`. `tag_running_headers_footers()` sets `metadata.role` (`page_header` / `page_footer`) on repeating top/bottom bands (or matching OCR labels), and `apply_structure()` (enabled via `structure=True`, passing ToC bookmarks via `pypdfium2`) links `caption` elements to `figure`/`table` boxes (`caption_of`, `caption_id`, `caption_text`), detects headings via bookmarks (ignoring slide `Folie`/`Slide`/`Page` bookmarks), numbering, and 5%-tolerance style clustering (`heading_level`, `section_parent_id`), and populates `metadata.outline`.
- **Raster & hybrid OCR fallback (`raster.py`, `pdf/backends/reader_composed.py`):** when `ocr=True`, `raster` pages and `hybrid` pages with `suspect_ocr_layer=True` route through `PageReader` (`ComposedReader` runs `LayoutDetector` + `postprocess_layout(..., keep_empty_clusters=True)` and `TextRecognizer`, assigns OCR lines to non-picture clusters preserving `metadata.layout_label`, keeps orphan lines outside `picture` boxes, and orders blocks via XY-cut; `MinerUReader` maps `middle_json`). When `ocr=False`, `raster` pages are listed in `metadata.unreadable_pages` and `hybrid` `suspect_ocr_layer` pages stay on `text_layer` with a `metadata.warnings` entry.

## 3. IR differences

- **From `parsers/pptx_ir.py`:** uses the same 7 top-level keys (`doc_id`, `source_file`, `doc_type="pdf"`, `metadata`, `elements`, `tables=[]`, `diagrams=[]`) and `coord_unit="pt"`, `reading_order` 0-based per page. Differences: PDF elements emit `type` in `{"text", "figure", "table"}`, include top-level `source` and `confidence`, preserve `font_size` and `bold` in `elements[].metadata` (and `layout_label` for OCR blocks), omit `table_id` while `tables == []`, and use `source_ref: {file, page, bbox}` (no `shape_id`).
- **From `docs/01_architecture_and_ir_schema.md` (`feat/structure-and-diagram-recovery`):** coordinates are PDF `pt` instead of page pixels; `reading_order` is per-page 0-based instead of document-wide; `diagrams` is `[]` and `tables` is `[]` (no HTML or cell structure yet); `type` is normalized to `text`/`figure`/`table`; every element carries `source` (`text_layer` / `ocr` / `v1_group` / `layout_picture` / `vector_only` / `layout_table`).

## 4. Measured results (Windows, `.venv` Python 3.12, zero Docling)

### Backend comparisons via `pdf.eval.compare` (2026-11-17, `outputs/eval_runs.jsonl`)

- **AUTOSAR `dev` figures (`44` pages, `22` negative, `30` GT boxes, `grouper="v1_anchor"`):**

| Backend | Strict TP/FP/FN (F1) | Lenient TP/FP/FN (F1) | Neg FP | Load (s) | Steady (s/p) | Peak RAM (MB) |
|---|---|---|---:|---:|---:|---:|
| `heron` (live) | `27/3/3` (`0.900`) | `26/4/4` (`0.867`) | 2 | 10.029 | 0.1793 | 1521.5 |
| `heron-onnx` (live) | `27/3/3` (`0.900`) | `26/4/4` (`0.867`) | 2 | 2.043 | 0.1545 | 916.8 |
| `mineru-pipeline` (cached) | `29/3/1` (`0.935`) | `26/6/4` (`0.839`) | 3 | — | — | — |

- **OmniDocBench 20-page raster OCR (`pooled_R = 0.9014`, `pooled_P = 0.8789`, `mean_R = 0.8864`, `median_R = 0.9496`, `min_R = 0.2298` across all three):**

| Reader | Load (s) | Steady (s/p) | Wall (s) | Peak RAM (MB) |
|---|---:|---:|---:|---:|
| `composed` (`heron` + `rapidocr-onnx`) | 10.850 | 4.6658 | 104.49 | 2917.9 |
| `heron-onnx+rapidocr-torch` | 2.975 | 1.6799 | 36.93 | 2446.2 |
| `heron-onnx+rapidocr-onnx` | 3.503 | 1.5375 | 34.57 | 2177.0 |

### Baseline & parity measurements (2026-11-16, GTX 1650, `heron` PyTorch)

- **Router:** AUTOSAR `Layered` (`{'native': 185}`), `VFB` (`{'native': 147}`), `ECUStateManager` (`{'native': 245}`), `OS` (`{'native': 407}`); 30 OmniDocBench PNGs wrapped as PDFs (`{'raster': 30}`).
- **Text pipeline:** 185-page `Layered` PDF ran in `3.032 s` (`16.39 ms/page`) with exact word parity against `get_text("words")` and zero `torch`/`transformers` imports.
- **Figure boxes (`detect_figures`, `layout="heron"`, `include_tables=False`, `sys.modules["docling"] = None`, fresh cache):**
  - `test` (108 pages across 4 AUTOSAR PDFs): strict F1 **`0.959`** (`58/2/3`, `P=0.967, R=0.951`), lenient F1 **`0.926`** (`56/4/5`, `P=0.933, R=0.918`), negative-page FP **`0`** (matches the former Docling baseline to three decimal places).
- **Layout parity with the former Docling output (all 984 AUTOSAR pages, `HeronDetector` + `postprocess_layout` + `clusters_to_items`, compared with cached Docling 2.150.0 items):**
  - v1 figure boxes equal on `984/984` pages (same count, every box IoU `>= 0.95`); label multiset equal on `981/984` pages; item order equal on `980/984` pages.
  - Every Docling box has a match (IoU `>= 0.95`, same label); the PyTorch detector emits 3 extra boxes on 3 pages (2 `table`, 1 `picture`) and a different item order on 4 pages, none changing the v1 figure boxes. Totals Docling versus `heron`: `table` 894 versus 896, `picture` 1391 versus 1392, `document_index` 18 versus 18, `formula` 4 versus 4.
- **Isolated subprocess benchmark (5 pages per mode via `pdf/benchmark.py`, `heron` PyTorch):**

| Mode | Import (s) | First call / page (s) | Steady per page (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|
| `native` (`figures=False, ocr=False`) | 0.0178 | 0.0632 (5 pages) | 0.0126 | 51.88 |
| `figures_cached` (5 pages, `_layout_cache` hit) | 0.0179 | 0.0881 | 0.0536 | 94.02 |
| `figures_live` (5 pages, `HeronDetector`) | 0.0169 | 22.4569 | 0.3338 | 1524.91 |
| `raster` (5 pages, `HeronDetector` + `RapidOCR`) | 0.0389 | 37.0684 | 6.1676 | 2215.66 |

## 5. Known limits

- Default full-pipeline execution (`heron-onnx` + `rapidocr-onnx`) uses `onnxruntime`, avoiding `torch` and `transformers`; the PyTorch `heron` and `rapidocr-torch` backends remain available by name.
- No PDF table cell extraction yet (`tables` is `[]`; words inside table boxes appear as `level=2` child `text` elements).
- No diagram node-edge reconstruction (`diagrams` is `[]`).
- Repeating running headers/footers are tagged (`metadata.role`), not dropped, and sub-visible watermarks (e.g., 2 pt text) are not filtered by default (`min_font_size=0.0`) so word parity with `get_text("words")` holds.
- If `v1_group` absorbs an adjacent `table` (`<= 45 px` gap) into a figure box, the `>= 50%` overlap filter drops the standalone `table` box.
