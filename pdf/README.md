# pdf

Lightweight multi-page PDF parser for DocuMind Problem 3 with zero `docling` or `pymupdf` dependency. Pages with a usable text layer run on `pdfplumber` (+ `pypdfium2` for page rendering and bookmarks); the swappable layout detectors (`heron-onnx` by default via `onnxruntime`, `heron` via `transformers` + `torch`, or `mineru-pipeline` in an isolated venv), table extractors (`auto` by default, or `tatr-onnx` / `pdfplumber`), and OCR recognizers (`rapidocr-onnx` by default, or `rapidocr-torch`) are imported lazily only when `figures=True`, `tables=True`, or `ocr=True` is needed.

## 1. Layout and usage

```
pdf/
  __init__.py         public exports (models + extract_ir)
  _pdf_io.py          shared pdfplumber / pypdfium2 helpers (PdfContext session, bbox/CropBox/Rotate mapping, word extraction, ToC, page/crop rendering)
  models.py           frozen dataclasses in PDF points (top-left origin), including PageProfile extensions, Word.direction, GroupedFigure, TableCell.confidence, ExtractedTable, and ExtractedDiagram
  router.py           classify_page(page) -> PageProfile (suspect_reasons, partial_raster_regions, watermark_word_count, figure_candidate)
  text.py             pdfplumber word extraction + 90/270 rotated words (btt/ttb), scale-free line/block grouping, recursive XY-cut ordering
  grouping.py         FigureGrouper implementations: v1_anchor (OpenCV + DSU), v2_vector_vision (vector + layout + Table-Barrier), and raw_picture -> GroupedFigure
  figures.py          lazy layout backend + cache + grouper + semantic_regions_out + layout_table boxes
  tables.py           lazy table extractor dispatch, continuation linking (link_table_continuations), table_to_ir_dict (cells[].confidence, HTML)
  diagrams.py         vector-first + OpenCV visual-fallback diagram node-edge-group recovery (`VectorFirstDiagramExtractor`, `diagram_to_ir_dict`)
  raster.py           lazy PageReader dispatch for full raster pages and hybrid partial_raster_regions
  assemble.py         overlap filter, innermost-box word assignment, atomic XY-cut assembly, running header/footer tagging
  hierarchy.py        heading detection, section tree builder (apply_structure, build_section_tree), caption linking
  ir.py               extract_ir(path, *, figures=True, ocr=True, tables=False, diagrams=False, ..., structure=True) -> dict
  benchmark.py        isolated timing/RAM benchmark (native, figures_cached, figures_live, raster, tables_live, balanced_e2e)
  backends/
    base.py           Protocols (LayoutDetector, LayoutBackend, TextRecognizer, PageReader, FigureGrouper, TableExtractor, DiagramExtractor), PageLayout, RunStats
    registry.py       register(), get(kind, name, **opts), registered(), lazy built-in factories
    heron_onnx.py     RT-DETR v2 Heron ONNX detector (default layout/detector backend)
    heron.py          RT-DETR v2 Heron PyTorch/transformers detector
    rapidocr.py       RapidOCR recognizer (onnxruntime or torch engine)
    tatr_onnx.py      Table Transformer v1.1 ONNX structure extractor
    table_pdfplumber.py PdfPlumber TableFinder native table extractor (table="pdfplumber")
    table_auto.py     Hybrid table extractor (default table="auto": pdfplumber grid + TATR header refinement / fallback)
    mineru.py         MinerULayoutBackend and MinerUReader (isolated venv subprocess)
    mineru_child.py   child script run inside the MinerU venv
    layout_image.py   ImageLayoutBackend wrapping per-image LayoutDetector at 96.012 DPI PNG scale (keep_empty_clusters=True)
    layout_cache.py   CachedLayoutBackend & LayoutCache keyed by backend, revision, DPI, postprocess_version ("v3"), and PDF SHA-256
    reader_composed.py ComposedReader combining a LayoutDetector and TextRecognizer (Pass 1 150 DPI + Pass 2 275 DPI ocr_high_dpi + region OCR)
    runtime.py        shared subprocess runner (with timeout_sec) and venv lookup for isolated backends
    rss.py            stdlib-only peak RSS helper shared by backends and benchmark
  layout/
    types.py          LAYOUT_LABELS, LayoutLabel, LayoutItem
    detector.py       clamp_and_rescale()
    postprocess.py    cluster thresholds, overlap resolution, container filtering, item build (preserves score)
    NOTICE            MIT license and source paths from Docling 2.150.0
  ocr/
    types.py          OcrLine
  eval/
    gt.py             AUTOSAR figure & OmniDocBench GT loaders, shared split/logging helpers
    metrics.py        strict/lenient figure matching, PRF, ECE/calibration summaries, character counts, markup stripping
    calibration.py    confidence calibration & monotonicity invariants evaluator (`figures`, `tables`, `monotonicity`)
    compare.py        CLI comparing N backends with --split dev|test (--final guard on test)
    compare_hierarchy.py CLI comparing heading hierarchy modes (no_toc vs with_toc) against PDF bookmarks
    tables.py         table TEDS / merged-cell span F1 evaluator (oracle and e2e modes)
    diagrams.py       diagram node/edge/graph F1, integrity checks, and synthetic vector/visual diagram suites
    reading_order.py  reading order Kendall's tau evaluator (synthetic and OmniDocBench splits)
  fixtures/
    layout_dumps/     12 recorded AUTOSAR pages (raw and post-processed layout clusters, boxes only, captured once from Docling 2.150.0) used by test_layout.py
    mineru_dumps/     stripped MinerU middle_json fixture for unit tests
  test_*.py           router, text, layout, figures, assemble, raster, tables, diagrams, hierarchy, and eval test suites
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
.\.venv\Scripts\python -B pdf\test_tables.py
.\.venv\Scripts\python -B pdf\test_diagrams.py
.\.venv\Scripts\python -B pdf\test_eval.py
.\.venv\Scripts\python -B pdf\test_hierarchy.py

# Compare backends on dev split or run table/diagram/reading-order evaluations:
.\.venv\Scripts\python -B -m pdf.eval.compare --task figures --backends heron,heron-onnx --split dev --no-cache
.\.venv\Scripts\python -B -m pdf.eval.compare --task raster --backends heron-onnx+rapidocr-onnx,heron-onnx+rapidocr-torch
.\.venv\Scripts\python -B -m pdf.eval.tables --task omnidoc --mode e2e --split dev
.\.venv\Scripts\python -B -m pdf.eval.diagrams --task all --split dev
.\.venv\Scripts\python -B -m pdf.eval.reading_order --task all --mode e2e --split dev
.\.venv\Scripts\python -B -m pdf.eval.calibration --split dev
```

Tests that depend on local untracked datasets (`datasets/AUTOSAR/`, `datasets/OmniDocBench/`) or an isolated MinerU venv print `SKIP` when those files are absent.

## 2. Pipeline rules

- **Router (`router.py`):** classifies each page as `native`, `raster`, or `hybrid` (`raster` when `word_count < 5` and a single embedded image covers `>= 0.96` of the page; `native` when `word_count >= 5`, `text_area_ratio >= 0.002`, `max_image_cover < 0.96`, and `not suspect_ocr_layer`; blank pages are `native`; all others are `hybrid`), populates `suspect_reasons` (including `(cid:)`/PUA detection), filters diagonal/micro watermarks from counts, collects `partial_raster_regions` on `hybrid` pages (excluding `< 10%` top/bottom header/footer logos), and sets `figure_candidate`.
- **Text pipeline (`text.py`):** extracts horizontal words via `pdfplumber` (`x_tolerance_ratio=0.15`) alongside 90° (`"btt"`) and 270° (`"ttb"`) rotated words directly from rotated character matrices; clusters lines and vertical columns via `cluster_by_overlap()` using scale-free font/page tolerances; and orders blocks via recursive `xy_cut_order()` with spanning-banner checks.
- **Backends & registry (`pdf/backends/registry.py`):** resolves backends lazily by `(kind, name)`:
  - `detector` / `layout`: `heron-onnx` (default, `docling-project/docling-layout-heron-onnx` revision `40bde044036bb181c130ddf6c51792187268748f`), `heron` (`docling-project/docling-layout-heron` revision `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`), `mineru-pipeline` (`layout` only, runs via `DOCUMIND_MINERU_PYTHON` or `~/.venvs/mineru-4.2.3`).
  - `ocr`: `rapidocr-onnx` (default), `rapidocr-torch`, `rapidocr` (CPU ONNX).
  - `reader`: `composed` (default, combines `layout` detector + `ocr_backend`), `heron-onnx+rapidocr-onnx`, `heron-onnx+rapidocr-torch`, `mineru-pipeline`.
  - `grouper`: `v1_anchor` (default, `source="v1_group"`), `v2_vector_vision` (`source="v2_vector_vision"`), `raw_picture` (`source="layout_picture"`).
  - `table`: `auto` (default, hybrid `pdfplumber` + `tatr-onnx` with IoU & token-coverage gate, overlapping-cell tolerance, TATR header refinement, and `<thead>`/`<tbody>` HTML), `tatr-onnx` (`Xenova/table-transformer-structure-recognition-v1.1-all`), `pdfplumber` (native bordered tables via `page.find_tables()`).
  - `diagram`: `vector_first` (default, vector-primitive node/edge/group recovery via `pdfplumber`).
- **Image pipeline (`figures.py`, `grouping.py`, `pdf/layout/`):** renders pages via `pypdfium2` (`150 DPI`, mapping coordinates accurately across `CropBox` offsets and `/Rotate` `0/90/180/270`), caches layout items under `(backend, revision, dpi, postprocess_version="v3", pdf_sha256)`, and runs `ImageLayoutBackend` lazily on cache misses (`96.012 DPI`, `postprocess_layout(..., keep_empty_clusters=True)`). `v1_anchor` and `v2_vector_vision` group figures returning `GroupedFigure` boxes with area-weighted scores while emitting `semantic_regions_out` for downstream hierarchy labeling and collecting `layout_table` boxes (`include_tables=True`).
- **Assembler & IR (`assemble.py`, `hierarchy.py`, `ir.py`):** opens one shared `PdfContext` per `extract_ir()`, resolves figure/table overlaps (clipping grouper figure boxes against bordered native tables rather than discarding the table), assigns words by enclosing `IoA >= 0.50`, merges native words with `region_ocr_blocks` on `hybrid` pages, and runs atomic page-level XY-cut (`reading_order` contiguous `0..N-1` per page). When `tables=True`, `extract_tables_on_page()` emits `ir["tables"]` (with `cells[].confidence`, valid `<thead>`/`<tbody>` HTML preserving `rowspan`, `table_parse_empty` warnings) and `link_table_continuations()` chains split tables while stopping at headings. When `diagrams=True`, `extract_diagrams_on_page()` emits `ir["diagrams"]` and `diagram_parse_empty` warnings. `tag_running_headers_footers()` handles repeating/even-odd banners and Roman page numbers (preserving `"Appendix C/D"` and `toc_titles`), and `apply_structure()` projects `semantic_regions` onto native text `layout_label`s, merges multiline headings/captions cleanly, validates numbering sequences, links captions bidirectionally (`caption_of` / `caption_id` / `caption_text` across elements, tables, and diagrams), and builds `metadata.outline`.
- **Raster & hybrid OCR fallback (`raster.py`, `pdf/backends/reader_composed.py`):** when `ocr=True`, `raster` pages and `hybrid` `suspect_ocr_layer` pages route through `PageReader.read_pages()`, while `hybrid` pages with `partial_raster_regions` also run `read_raster_regions()` across aligned `CropBox`/`/Rotate` page coordinates. `ComposedReader` runs Pass 1 (`150 DPI`), performs Pass 2 cropped `275 DPI` re-OCR (`source="ocr_high_dpi"`) on low-confidence/empty regions when score improves, tightens orphan line merging, and estimates `font_size > 0` in points.

## 3. IR differences

- **From `parsers/pptx_ir.py`:** uses the same 7 top-level keys (`doc_id`, `source_file`, `doc_type="pdf"`, `metadata`, `elements`, `tables`, `diagrams`) and `coord_unit="pt"`, `reading_order` 0-based per page. Differences: PDF elements emit `type` in `{"text", "figure", "table"}`, include top-level `source` and `confidence`, preserve `font_size`, `bold`, `layout_label`, and `caption_id`/`caption_text` in `elements[].metadata`, keep `table_id` / `diagram_id` in `elements[].metadata`, emit `source`/`confidence`/`continuation_of`/`cells[].confidence`/`html` on `tables[]` when `tables=True`, populate `diagrams[]` when `diagrams=True`, and use `source_ref: {file, page, bbox}` (no `shape_id`).
- **From `docs/01_architecture_and_ir_schema.md` (`feat/structure-and-diagram-recovery`):** coordinates are PDF `pt` instead of page pixels; `reading_order` is per-page 0-based instead of document-wide; `diagrams` and `tables` are populated only when `diagrams=True` / `tables=True` (both default `False`); `type` is normalized to `text`/`figure`/`table`; every element carries `source` (`text_layer` / `ocr` / `ocr_high_dpi` / `v1_group` / `v2_vector_vision` / `layout_picture` / `layout_table`).

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

### Step-1 dev baselines (2026-11-19, commit `ac53d66`, `outputs/eval_runs.jsonl`)

- **Figures `dev` (`pdf.eval.compare --task figures --split dev --backends heron-onnx --no-cache`, `44` pages / `22` negative / `30` GT boxes):** strict `28/3/2` (`F1 = 0.918`), lenient `27/4/3` (`F1 = 0.885`), `neg_fp = 2`, `load = 4.082 s`, `0.1897 s/p`, `987.0 MB` peak RSS.
- **Reading order (`pdf.eval.reading_order`, `dev`):**
  - `synthetic` (`3` pages, `e2e_native`): `mean_tau_matched = 1.0`, `mean_tau_penalized = 1.0`, `0.0465 s`.
  - `omnidoc` `oracle` (`368` pages): `mean_tau_matched = 0.9473`, `mean_tau_penalized = 0.9473`, `median = 1.0`, `0.1187 s`, `402.05 MB` RSS.
  - `omnidoc` `e2e` (`--limit 25`, `25` pages): `mean_tau_matched = 0.9744`, `mean_tau_penalized = 0.0249` (median `1.0` / `0.0826`), `37.5076 s`, `1557.99 MB` RSS.
- **Tables (`pdf.eval.tables`, `dev`):**
  - `omnidoc` `oracle` (`124` tables, `40` merged, `119` non-empty): `teds_struct = 0.8422` (`merged = 0.8599`), `teds_full = 0.758`, `merged_cell_span_f1 = 0.3742`, `load = 2.0123 s`, `76.46 ms/table`, `1935.35 MB` RSS.
  - `omnidoc` `e2e` (`--limit 25`, `20` pages, `25` GT tables, `24` matched): `det_P = 1.0`, `det_R = 0.96`, `det_F1 = 0.9796`, `teds_struct = 0.7931` (`matched = 0.8261`, `merged = 0.7384`), `teds_full = 0.7083`, `merged_cell_span_f1 = 0.2609`, `87.5222 s` (`3500.89 ms/table`), `1752.89 MB` RSS.
  - `autosar` `dev` (`4` tables): `pdfplumber` `4/4` non-empty (`3` merged, `91` cells, `15.43 ms/table`, `914.73 MB`); `tatr-onnx` `4/4` (`88` cells, `141.72 ms/table`, `1009.43 MB`); `auto` `4/4` (`91` cells, `21` header, `148.17 ms/table`, `1041.59 MB`).
- **Hierarchy `dev` (`pdf.eval.compare_hierarchy --split dev`, `with_toc`):** `VFB` (`147p`, `toc=113`) `toc_rec=0.965, lvl_acc=1.000` (`164.7 ms`); `SWS_OS` (`407p`, `toc=380`) `0.976 / 1.000` (`2943.8 ms`); `SWS_ECUStateManager` (`245p`, `toc=256`) `0.996 / 0.996` (`505.9 ms`); `Layered` (`185p`, `toc=0`) `out=388` (`700.2 ms`).
- **End-to-end 5-page benchmark (`pdf/benchmark.py`):**
  - `tables_live`: `tables_extracted=1`, `elements=68`, `import_sec=0.0`, `total_sec=6.1168` (`1.2234 s/p`), `1060.38 MB` peak RSS.
  - `balanced_e2e` (`4` native + `1` raster): `tables_extracted=1`, `elements=62`, `outline_len=3`, `import_sec=0.0`, `total_sec=7.67` (`1.534 s/p`), `1284.98 MB` peak RSS.

### Step-2 vector diagram checks (2026-11-19)

- **Synthetic vector diagram suite (`pdf.eval.diagrams --task synthetic --split dev`, `6` fixtures):** `node_f1 = 1.0`, `edge_f1 = 1.0`, `graph_f1 = 1.0`, `text_to_shape_acc = 1.0`, `integrity_violations = 0` (`0.0641 s`, `58.58 MB` RSS).
- **AUTOSAR `LayeredSoftwareArchitecture` full run (`185` pages, `figures=True, ocr=False, tables=False, diagrams=True`):** `138` diagrams (`layered: 93`, `block_diagram: 38`, `flowchart: 7`), `1` self-loop across all diagrams (`0` on `p28`, `p65`, `p109`), `0` integrity violations.

### Step-3 pipeline upgrades & bug-fix dev checks (2026-11-19)

- **Unit & integration test suite (`pdf/test_*.py`, `10` files):** **`144/144` passed** (`1` internal skip in `test_figures.py` when `docling` is absent).
- **Figures `dev` (`pdf.eval.compare --task figures --split dev --backends heron-onnx --no-cache`, `44` pages / `22` negative / `30` GT boxes):**
  - `grouper="v1_anchor"`: strict `28/3/2` (`F1 = 0.918`), lenient `27/4/3` (`F1 = 0.885`), `neg_fp = 2` (`0.1769 s/p`, `987.84 MB`).
  - `grouper="v2_vector_vision"`: strict **`29/2/1`** (**`F1 = 0.9508`**), lenient **`28/3/2`** (**`F1 = 0.918`**), `neg_fp = 2` (`0.1803 s/p`, `988.74 MB`).
- **Raster OCR (`pdf.eval.compare --task raster --split dev --pages 15`, `ComposedReader` `heron-onnx` + `rapidocr-onnx`, report-only OmniDoc sample subset):** `pooled_R = 0.9335`, `pooled_P = 0.9355`, `mean_R = 0.9279`, `mean_P = 0.9041` (`1.9063 s/p`, `1991.45 MB`).
- **Tables (`dev`, post-fix `#105`):**
  - `oracle` (`--limit 0`, `124` tables, `40` merged): `teds_struct = 0.8811` (`merged = 0.8517`), `teds_full = 0.8062`, `merged_cell_span_f1 = 0.4108` (`121.89 ms/table`, `1930.02 MB`).
  - `e2e` (`--limit 25`, `20` pages, `25` GT tables, `24` matched): `det_F1 = 0.9796` (`P=1.0, R=0.96`), `teds_struct = 0.7979` (`matched = 0.8312`), `teds_full = 0.7093`, `merged_cell_span_f1 = 0.2981` (`102.70 s`, `1781.34 MB`).
- **Hierarchy `dev` (`pdf.eval.compare_hierarchy --split dev`, `with_toc`):** `VFB` `119/118 (0.965/1.000)`, `SWS_OS` `415/414 (0.976/1.000)`, `SWS_ECUStateManager` `819/813 (0.996/0.996)`, `Layered` `597` outline items.
- **Reading order (`pdf.eval.reading_order`, `dev`):**
  - `oracle` (`--limit 25`): `synthetic` (`3` pages) `1.0 / 1.0`; `OmniDocBench` (`25` pages) `mean_tau_matched = 0.9794`, `mean_tau_penalized = 0.9794` (`0.0048 s`).
  - `e2e` (`--limit 25`): `synthetic` (`3` pages) `1.0 / 1.0`; `OmniDocBench` (`25` pages) `mean_tau_matched = 0.9744`, `mean_tau_penalized = 0.0251` (`median_matched = 1.0`, `45.48 s`, `1600.77 MB`).

### Step-4 conditional expansion, calibration & final checks (2026-11-20, Google Colab Pro Tesla T4 / Python 3.13)

- **Unit & integration test suite (`pdf/test_*.py`, `10` files):** **`163/163` passed** (`2` internal skips logged in `test_figures.py` for missing `docling`/MinerU venv).
- **Reading order (`pdf.eval.reading_order`, `--split dev`):**
  - `oracle` (`--limit 25`): `synthetic` `1.0 / 1.0`; `OmniDocBench` (`25` pages) `mean_tau_matched = 0.9918`, `mean_tau_penalized = 0.9918` (`median = 1.0 / 1.0`).
  - `e2e` (`--limit 25`): `synthetic` `1.0 / 1.0`; `OmniDocBench` (`25` pages) `mean_tau_matched = 0.9914`, **`mean_tau_penalized = 0.9548`** (`median = 1.0 / 1.0`).
- **Diagrams (`pdf.eval.diagrams --task all --split dev`):** `synthetic_diagrams` (`6` fixtures) `1.0 / 1.0 / 1.0`, `visual` (`6` fallback fixtures, `extraction_mode="visual_fallback"`, `confidence=0.82`) `1.0 / 1.0 / 1.0`, `integrity_violations = 0`.
- **Tables (`pdf.eval.tables --task omnidoc --split dev`):**
  - `oracle` (`--limit 0`, `124` tables, `40` merged): `teds_struct = 0.9012`, `teds_struct_merged = 0.8759`, `teds_full = 0.8294`, `merged_cell_span_f1 = 0.4931` (`87.29 ms/table`, `2891.38 MB`).
  - `e2e` (`--limit 25`, `20` pages, `25` GT / `24` matched): `det_F1 = 0.9796`, `teds_struct = 0.8492` (`matched = 0.8845`), `teds_struct_merged = 0.7462`, `teds_full = 0.7672`, `merged_cell_span_f1 = 0.3226`.
- **Calibration (`pdf.eval.calibration --split dev`):** `monotonicity_invariants` **`invariant_ok = true`** (`native = 1.0`, `suspect_ocr = 0.85 <= 0.85`); `tables_calibration` `ece_10 = 0.073017`, `paired_violations = 0`, `source_mean_monotonicity_ok = true`, `fp_ge_095 = 0`; `figures_calibration` (`44p`, `30` GT, `heron-onnx`/`v1_anchor`) `tp/fp/fn = 27/3/3` (`F1 = 0.9`), `ece_10 = 0.11952`, `mean_conf_tp = 0.8688`, `mean_conf_fp = 0.8744`.
- **Final held-out evaluation (`--split test --final`, 2026-11-20):**
  - **Figures `test` (`108` pages, `57` neg, `61` GT):** `heron-onnx` & `heron` (`v1_anchor` and `v2_vector_vision`) `strict = 55/5/6` (**`sF1 = 0.9091`**), `lenient = 53/7/8` (`lF1 = 0.8760`), **`neg_fp = 0`**; `figures_calibration` `ece_10 = 0.06405`, `mean_conf_tp = 0.9458 > mean_conf_fp = 0.9243`.
  - **Reading order `test` (`OmniDocBench_en_reading_order`):** `oracle` (`354` pages) `mean_tau_matched = 0.9508`, `mean_tau_penalized = 0.9508` (`median = 1.0`); `e2e` (`--limit 50`, `50` pages) `mean_tau_matched = 0.9464`, `mean_tau_penalized = 0.7913` (`median = 1.0 / 1.0`).
  - **Tables `test` (`OmniDocBench_en_tables`):** `oracle` (`119` tables, `47` merged, `119/119` non-empty) `teds_struct = 0.8731`, `teds_struct_merged = 0.8322`, `teds_full = 0.8071`, `merged_cell_span_f1 = 0.3474`; `e2e` (`--limit 50`, `38` pages, `50` GT / `48` matched) `det_F1 = 0.9697`, `teds_struct = 0.8483` (`matched = 0.8837`), `teds_struct_merged = 0.7980`, `teds_full = 0.7888`, `merged_cell_span_f1 = 0.3002`.
  - **Calibration `test` (`pdf.eval.calibration --split test --final`):** `monotonicity_invariants` **`invariant_ok = true`** (`native = 1.0`, `suspect_ocr = 0.85`); `tables_calibration` (`22p`, `29` GT / `28` pred, `det_F1 = 0.9474`) `ece_10 = 0.051714`, `paired_violations = 0`, `source_mean_monotonicity_ok = true`.

## 5. Known limits

- **Default full-pipeline execution (`heron-onnx` + `rapidocr-onnx`)** uses `onnxruntime`, avoiding `torch` and `transformers`; the PyTorch `heron` and `rapidocr-torch` backends remain available by name. (`pdf/_pdf_io.py` still imports `cv2` at top level.)
- **Table cell extraction** requires `tables=True` (default `False` on `extract_ir()`) and relies on `table` regions detected via `figures=True` or `ocr=True` (with `figures=False, ocr=False, tables=True`, `tables` stays `[]`). While Step 4 raised `oracle` `teds_struct` to `0.9012` (`dev`) / `0.8731` (`test`), OmniDocBench `e2e` `teds_full` (`0.7672` `dev` / `0.7888` `test`) and `merged_cell_span_f1` (`0.3226` `dev` / `0.3002` `test`) still trail `0.88`.
- **Diagram extraction** requires `diagrams=True` (default `False`) and `figures=True`; OpenCV contour/line/OCR `visual_fallback` handles bitmap diagrams when vector primitives are absent (`confidence=0.82`), while figures that yield no recovered graph emit `diagram_parse_empty` warnings (`confidence=0.0`), and real `AUTOSAR` golden diagram GT remains unannotated beyond synthetic fixtures.
- **Running furniture & watermarks:** repeating running headers/footers are tagged (`metadata.role`), not dropped, and sub-visible watermarks (e.g., 2 pt text) are not filtered by default (`min_font_size=0.0`) so word parity with `get_text("words")` holds.
- **Figure vs. table overlap:** with `grouper="v1_anchor"`, an adjacent `table` within `<= 45 px` can still be partially overlapped, though bordered native tables now clip the grouper figure box rather than being dropped. Use `grouper="v2_vector_vision"` for strict Table-Barrier separation.
