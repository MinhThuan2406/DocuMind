# pdf

Lightweight multi-page PDF parser for DocuMind Problem 3. Pages with a usable text layer run on PyMuPDF alone; Docling and PyTorch are imported lazily only when figure detection (`figures=True`, and only on pages the router marks as figure candidates) or raster OCR (`ocr=True`, and only on raster pages) is invoked.

## 1. Layout and usage

```
pdf/
  __init__.py       public exports (models + extract_ir)
  models.py         frozen dataclasses in PDF points (top-left origin)
  router.py         classify_page(page) -> PageProfile
  text.py           PyMuPDF word extraction + recursive XY-cut block ordering
  figures.py        lazy Docling (OCR off) + v1 grouping with anchor rule + docling_table
  raster.py         lazy single-page Docling (OCR on) fallback
  assemble.py       overlap filter, innermost-box word assignment, atomic XY-cut assembly
  ir.py             extract_ir(path, *, figures=True, ocr=True, device="auto") -> dict
  benchmark.py      isolated M6 timing/RAM benchmark and M5 20-page OmniDocBench recall
  test_*.py         M1-M5 verification scripts
```

```python
from pdf import extract_ir

# Pure PyMuPDF CPU path (imports neither docling nor torch):
ir_text = extract_ir("datasets/AUTOSAR/AUTOSAR_CP_EXP_VFB.pdf", figures=False, ocr=False)

# Full path (lazy Docling for figure/table boxes and raster-page OCR fallback):
ir_full = extract_ir("datasets/AUTOSAR/AUTOSAR_CP_EXP_VFB.pdf", figures=True, ocr=True, device="auto")
```

Run tests from the repository root (PowerShell / cmd):

```powershell
.\.venv\Scripts\python -B pdf\test_router.py
.\.venv\Scripts\python -B pdf\test_text.py
.\.venv\Scripts\python -B pdf\test_figures.py
.\.venv\Scripts\python -B pdf\test_assemble.py
.\.venv\Scripts\python -B pdf\test_raster.py
```

Tests that depend on local untracked datasets (`datasets/AUTOSAR/`, `datasets/OmniDocBench/`, `outputs/08_pdf_figure_crops/`) print `SKIP` when those files are absent.

## 2. Pipeline rules

- **M1 Router (`router.py`):** classifies each page as `native`, `raster`, or `hybrid` for text-source selection only (`raster` when `word_count < 5` and a single embedded image covers `>= 0.96` of the page; `native` when `word_count >= 5`, `text_area_ratio >= 0.002`, `max_image_cover < 0.96`, and `not suspect_ocr_layer`; blank pages with zero text and zero images are `native`; all others are `hybrid`). `suspect_ocr_layer` checks invalid/replacement character ratio (`>= 0.05`) and large-image/tiny-text ratio (`image >= 0.80` with `text_area_ratio < 0.01`) without running OCR. `figure_candidate` gates figure detection per page: true when an embedded image of at least 1.5% of the page lies above the footer band (top of the footer band at 0.88 of the page height), or the page holds at least 8 drawing primitives or at least 3 drawings with both sides at least 1.5 pt, excluding full-page backgrounds and the footer band. `extract_ir` sends only candidate pages to `detect_figures`, so a document with no candidate page never imports Docling or torch even with `figures=True`.
- **M2 Text pipeline (`text.py`):** uses `pymupdf` `get_text("words")` as the word baseline and maps `font_size` and `bold` from `get_text("dict")` spans. Groups words to line segments (splitting across wide column gaps) and blocks, then orders blocks with a recursive XY-cut ported from `ir_postprocessor.py` (scaled by `72/150`).
- **M3 Image pipeline (`figures.py`):** renders pages via PyMuPDF Pixmap in memory (`150 DPI`), reads `outputs/08_pdf_figure_crops/<stem>/_docling_cache_dpi150/pNNN.json` when present, and imports `docling` only when a cache entry is missing. Runs OpenCV/numpy `v1` grouping with the `anchor` rule (`min_area = 0.015`, `footer_frac = 0.88`, `refine=False`, `add_missing=False`) and collects `docling_table` boxes. All `figures.py` boxes carry a fixed confidence (`0.95`) regardless of cache vs live execution because Docling's `iterate_items()` and `ProvenanceItem` do not return per-item confidence.
- **M4 Assembler & IR (`assemble.py`, `ir.py`):** drops any `table` box overlapped `>= 50%` by a `figure` box; assigns each word whose center lies in a box to the smallest-area enclosing box (`(area, box_index)` tie-break) before running body XY-cut; orders `[body TextBlocks + atomic figures/tables]` via page-level XY-cut; emits `figure` inner words (via local XY-cut) and `table` inner `order_blocks` as `level=2` child `text` elements sequentially after their parent (`reading_order` restarts at `0` per page).
- **M5 Raster path (`raster.py`):** runs Docling with `do_ocr=True` on single `raster` pages only (reusing the converter per process), maps coordinates to PDF points, preserves Docling item labels in `metadata.docling_label`, and reads `ConversionResult.confidence.pages[1].ocr_score` with fallback `0.85`. When `ocr=False`, `raster` pages are listed in `metadata.unreadable_pages`.

## 3. IR differences

- **From `parsers/pptx_ir.py`:** uses the same 7 top-level keys (`doc_id`, `source_file`, `doc_type="pdf"`, `metadata`, `elements`, `tables=[]`, `diagrams=[]`) and `coord_unit="pt"`, `reading_order` 0-based per page. Differences: PDF elements emit `type` in `{"text", "figure", "table"}`, include top-level `source` and `confidence`, preserve `font_size` and `bold` in `elements[].metadata` (and `docling_label` for OCR blocks), omit `table_id` while `tables == []`, and use `source_ref: {file, page, bbox}` (no `shape_id`).
- **From `docs/01_architecture_and_ir_schema.md` (`feat/structure-and-diagram-recovery`):** coordinates are PDF `pt` instead of page pixels; `reading_order` is per-page 0-based instead of document-wide; `diagrams` is `[]` and `tables` is `[]` (no HTML or cell structure yet); `type` is normalized to `text`/`figure`/`table` instead of raw Docling labels (`section_header`, `caption`, etc.); every element carries `source` (`text_layer` / `ocr` / `v1_group` / `docling_table`).

## 4. Measured results (2026-11-16, Windows, GTX 1650 + `.venv` Python 3.12)

- **M1 Router:** AUTOSAR `Layered` (`{'native': 185}`), `VFB` (`{'native': 147}`), `ECUStateManager` (`{'native': 245}`), `OS` (`{'native': 407}`); 30 OmniDocBench PNGs wrapped as PDFs (`{'raster': 30}`).
- **M2 Text pipeline:** 185-page `Layered` PDF ran in `1.793 s` (`9.69 ms/page`) with exact word parity against `get_text("words")` and zero `docling`/`torch` imports.
- **M3 Figure boxes (`detect_figures`, `include_tables=False`, from `_docling_cache_dpi150`):**
  - `dev` (44 Layered pages): strict F1 `0.900` (TP/FP/FN `27/3/3`), lenient F1 `0.867` (`26/4/4`), negative-page FP `2`.
  - `test` (108 pages across 4 AUTOSAR PDFs): strict F1 `0.959` (`58/2/3`), lenient F1 `0.926` (`56/4/5`), negative-page FP `0`.
- **M5 Raster OCR recall (20 English OmniDocBench pages in `pdf/benchmark.py`):** `pooled_char_recall = 0.8866`, `mean = 0.8658`, `median = 0.9532`, `min = 0.2190` (note: RapidOCR fell back to CPU `AzureExecutionProvider` in this environment).
- **M6 Isolated subprocess benchmark (5 pages per mode via `pdf/benchmark.py`):**

| Mode | Import (s) | First call / page (s) | Steady per page (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|
| `native` (`figures=False, ocr=False`) | 0.0170 | 0.0601 (5 pages) | 0.0120 | 51.84 |
| `figures_cached` (5 pages, cache hit) | 0.0184 | 0.0699 | 0.0483 | 92.83 |
| `figures_live` (5 pages, Docling OCR off) | 0.0275 | 28.4406 | 1.0616 | 1948.45 |
| `raster` (5 pages, Docling OCR on) | 0.0154 | 30.1626 | 6.5785 | 2402.88 |

## 5. Known limits

- No PDF table cell extraction yet (`tables` is `[]`; words inside table boxes appear as `level=2` child `text` elements).
- No heading/hierarchy tree detection yet (body text, figures, and tables are `level=1`; M7 deferred).
- No diagram node-edge reconstruction (`diagrams` is `[]`).
- Repeating running headers/footers and sub-visible watermarks (e.g., 2 pt text) are not filtered by default (`min_font_size=0.0`) so word parity with `get_text("words")` holds.
- If `v1_group` absorbs an adjacent `table` (`<= 45 px` gap) into a figure box, the `>= 50%` overlap filter drops the standalone `table` box.
- The `figure_candidate` thresholds were chosen by looking at all 984 AUTOSAR pages (including the test split), so they are not held out. Measured with an equivalent standalone script against the cached Docling boxes, the rule gated 689 of 984 AUTOSAR pages (about 30% skipped) with no figure page missed; one table page (ECUStateManager page 242) is skipped, and a table drawn without ruling lines can be missed the same way, in which case its words stay in body text.
- `hybrid` pages with `suspect_ocr_layer=True` still use `text.py` and emit a `metadata.warnings` entry rather than falling back to `raster.py`.
