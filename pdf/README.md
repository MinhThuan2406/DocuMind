# `pdf` — Hướng dẫn sử dụng PDF Parser (`DocuMind`)

Parser PDF đa trang cho **DocuMind** (không phụ thuộc `docling` hay `pymupdf`):
- **Trang text số (`native` / `hybrid`):** trích xuất từ và tọa độ bằng `pdfplumber` + `pypdfium2`, chạy **Heron (`heron-onnx`)** mặc định để chia khối bố cục (`text`, `section_header`, `list_item`, `formula`, `code`, `caption`, `footnote`, `page_header`, `page_footer`, `figure`, `table`) và sắp thứ tự đọc đa cột bằng XY-cut.
- **Trang scan / ảnh (`raster`) & vùng ảnh trên trang `hybrid`:** dùng `heron-onnx` + `rapidocr-onnx` (có bước Pass 2 DPI cao cho vùng mờ).
- **Trích xuất có cấu trúc (bật theo cờ):** bảng (`auto` = `pdfplumber` + `tatr-onnx`), đồ thị sơ đồ kỹ thuật (`vector_first` ra `nodes`/`edges`/`groups`), công thức toán (`unimernet-onnx` ra LaTeX), cây mục lục (`ToC` / `heading`), và xuất thẳng **Unified JSON IR** hoặc **Markdown**.

---

## 1. Cài đặt môi trường

Từ thư mục gốc của repo `DocuMind`:

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux / macOS:        source .venv/bin/activate

pip install -r requirements.txt
```

Nếu máy có **GPU NVIDIA (CUDA)** và muốn tăng tốc ONNX (`heron-onnx`, `tatr-onnx`, `rapidocr-onnx`), thay `onnxruntime` bằng `onnxruntime-gpu`:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

> **Lưu ý lần chạy đầu:** Khi bật ML (`figures=True`, `ocr=True`, `tables=True`, hoặc `formulas=True`), parser sẽ tải trọng số ONNX từ Hugging Face Hub về cache local (lần sau dùng lại từ cache).

---

## 2. Cách sử dụng nhanh (Python API)

Luôn chạy Python từ **thư mục gốc của repo** để import `from pdf import ...`:

### 2.1. Trích xuất cơ bản (văn bản + layout Heron + OCR trang scan + mục lục)

```python
from pdf import extract_ir, to_markdown

ir = extract_ir("path/to/document.pdf")
md = to_markdown(ir)
```

### 2.2. Trích xuất đầy đủ (thêm Bảng + Đồ thị Sơ đồ + Công thức LaTeX)

Mặc định `tables=False`, `diagrams=False`, `formulas=False` để tiết kiệm tài nguyên. Khi cần phân tích tài liệu đầy đủ (báo cáo tài chính, tài liệu kỹ thuật có sơ đồ/bảng sát nhau), bật rõ các cờ:

```python
from pdf import extract_ir, to_markdown

ir = extract_ir(
    "path/to/document.pdf",
    figures=True,
    ocr=True,
    tables=True,
    diagrams=True,
    formulas=False,             # đổi thành True nếu muốn parse công thức ra LaTeX
    grouper="v2_vector_vision", # khuyên dùng khi bảng nằm sát hình/sơ đồ
    ocr_lang=None,              # ví dụ: "en", "ch", "japan" khi cần ép ngôn ngữ OCR
    device="auto",              # "auto" | "cpu" | "cuda"
)

md = to_markdown(
    ir,
    include_furniture=False,    # False: bỏ header/footer lặp lại
    include_diagrams=True,      # render diagram sang Mermaid khi có nodes/edges
    page_breaks=False,          # True: chèn "---" giữa các trang
)
```

> Lưu ý: Nếu cần chạy ở môi trường hoàn toàn thuần CPU / không muốn load bất kỳ model ML nào cho file PDF text số đơn giản, truyền **`figures=False, ocr=False`**:
> ```python
> ir_fast = extract_ir("path/to/digital.pdf", figures=False, ocr=False)
> ```

### 2.3. Lấy text thuần theo từng trang hoặc cây mục lục

```python
from pdf import build_section_tree, extract_ir, extract_plain_text

ir = extract_ir("path/to/document.pdf")

# Dict {page_no: text} theo thứ tự đọc chuẩn, mặc định bỏ running header/footer
pages_text = extract_plain_text(ir, include_furniture=False)

# Cây phân cấp section lồng nhau từ elements
section_tree = build_section_tree(ir["elements"])
```

---

## 3. Tham số chính của `extract_ir()`

Hàm [`extract_ir(path, ...)`](ir.py) nằm tại `pdf/ir.py` và được export ở `pdf/__init__.py`:

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `figures` | `True` | Bật/tắt phát hiện hình/vùng bố cục (`semantic_layout` + grouper). |
| `ocr` | `True` | Bật/tắt OCR cho trang `raster` và vùng ảnh trên trang `hybrid`. |
| `tables` | `False` | Bật/tắt trích xuất cấu trúc bảng (`cells`, `html`, `continuation_of`). |
| `diagrams` | `False` | Bật/tắt tái dựng đồ thị vector/visual (`nodes`, `edges`, `groups`). |
| `formulas` | `False` | Bật/tắt nhận diện công thức toán sang chuỗi `latex`. |
| `semantic_layout` | `"always"` | `"always"`: chạy layout trên mọi trang không phải raster (khi bật `figures`/`tables`/`formulas`) và dùng bbox layout để chia block.<br>`"auto"`: chỉ chạy trên trang nghi có hình/bảng.<br>`"off"`: tắt gán vùng semantic. |
| `structure` | `True` | Nhận diện heading, cây mục lục `outline`, gán `section_parent_id`, liên kết caption, tag running header/footer. |
| `layout` | `"heron-onnx"` | Backend layout: `"heron-onnx"` (ONNX), `"heron"` (PyTorch/transformers), `"mineru-pipeline"`. |
| `grouper` | `"v1_anchor"` | Grouper gom hình/sơ đồ: `"v1_anchor"`, `"v2_vector_vision"` (tách bảng–hình tốt hơn), `"raw_picture"`. |
| `reader` | `"composed"` | Reader cho trang scan/OCR: `"composed"`, `"heron-onnx+rapidocr-onnx"`, `"heron-onnx+rapidocr-torch"`, `"mineru-pipeline"`. |
| `ocr_backend` | `"rapidocr-onnx"` | Engine OCR: `"rapidocr-onnx"`, `"rapidocr-torch"`, `"rapidocr"` (CPU). |
| `ocr_lang` | `None` | Ép ngôn ngữ OCR nếu cần (ví dụ `"en"`, `"ch"`, `"japan"`). |
| `table_backend` | `"auto"` | Backend bảng: `"auto"` (`pdfplumber` + `tatr-onnx`), `"pdfplumber"`, `"tatr-onnx"`. |
| `diagram_backend` | `"vector_first"` | Backend đồ thị: `"vector_first"`. |
| `formula_backend` | `"auto"` | Backend công thức: `"auto"` (`"unimernet-onnx"`), `"unimernet-onnx"`, `"rapid-latex-ocr"`. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, hoặc `"cuda"`. |
| `min_font_size` | `0.0` | Lọc chữ siêu nhỏ (ví dụ watermark `2pt`) ngay khi đọc PDF nếu > `0.0`. |

---

## 4. Định dạng đầu ra Unified JSON IR

`extract_ir()` trả về `dict` chuẩn hóa JSON-serializable với tọa độ tính theo **PDF points (`pt`)**, gốc `(0, 0)` ở **góc trên–trái** (đã chuẩn hóa `CropBox` và `/Rotate`):

```json
{
  "doc_id": "document",
  "source_file": "document.pdf",
  "doc_type": "pdf",
  "metadata": {
    "coord_unit": "pt",
    "reading_order_status": "xy_cut",
    "pages": [{"number": 1, "kind": "native", "width_pt": 595.0, "height_pt": 842.0, "...": "..."}],
    "outline": [{"id": "elem_0001", "title": "1. Overview", "level": 1, "page": 1, "parent_id": null}],
    "unreadable_pages": [],
    "warnings": []
  },
  "elements": [],
  "tables": [],
  "diagrams": [],
  "formulas": []
}
```

* **`elements[]`:** mỗi phần tử gồm `id`, `type` (`"text"` | `"figure"` | `"table"`), `page` (1-based), `bbox` `[x0, y0, x1, y1]`, `reading_order` (0-based theo từng trang), `level` (`1` ở ngoài hoặc `2` cho phần tử con nằm trong hình/bảng), `parent_id`, `text`, `source` (`"text_layer"` | `"ocr"` | `"ocr_high_dpi"` | ...), `confidence`, `source_ref` (`{file, page, bbox}`), và `metadata`:
  * Với `text`: có thể chứa `font_size`, `bold`, `layout_label` (`"text"`, `"section_header"`, `"title"`, `"list_item"`, `"caption"`, `"footnote"`, `"formula"`, `"code"`, `"page_header"`, `"page_footer"`), `role` (`"heading"`, `"caption"`, `"page_header"`, `"page_footer"`), `heading_level`, `section_parent_id`, `caption_of`.
  * Với `figure` / `table` / `formula`: có thể kèm `diagram_id`, `table_id`, `formula_id`, `latex`, `caption_id`, `caption_text`.
* **`tables[]`:** bảng có `id`, `page`, `bbox`, `num_rows`, `num_cols`, `cells[]` (`row`, `col`, `rowspan`, `colspan`, `text`, `bbox`, `is_header`, `is_merged`, `confidence`), `html` (`<thead>`/`<tbody>`), `continuation_of`.
* **`diagrams[]`:** đồ thị có `id`, `diagram_type`, `bbox`, `nodes[]` (`shape_type`, `label`, `bbox`, `parent_group_id`, `text_element_ids`), `edges[]` (`source_node_id`, `target_node_id`, `direction`, `style`, `label`, `waypoints`), `groups[]`.
* **`formulas[]`:** công thức có `id`, `element_id`, `page`, `bbox`, `latex`, `backend`, `confidence`.

### Kiểm tra chất lượng & cảnh báo downstream

Trước khi đưa dữ liệu vào suy luận hoặc đối chiếu tài liệu, nên kiểm tra `ir["metadata"]["warnings"]` và `confidence`:
* `unreadable_pages`: trang `raster` khi `ocr=False`.
* `suspect_ocr_layer`: trang có text layer ẩn/hỏng (tự kích hoạt OCR fallback khi `ocr=True`).
* `table_parse_empty` / `diagram_parse_empty` / `formula_parse_empty`: vùng được detect nhưng trích xuất chi tiết rỗng.
* `low_confidence`: có phần tử mang `confidence < 0.70`.

---

## 5. Cấu trúc thư mục

```text
pdf/
  __init__.py         API công khai (extract_ir, to_markdown, extract_markdown, extract_plain_text, ...)
  ir.py               Điều phối toàn pipeline extract_ir()
  router.py           Phân loại trang native / raster / hybrid & phát hiện candidate hình/bảng
  text.py             Trích xuất word, xử lý chữ xoay 90/270, gom block theo vùng Heron, sắp xếp XY-cut
  figures.py          Chạy layout backend (lazy), cache _layout_cache và group figure + layout_table
  grouping.py         V1AnchorGrouper, V2VectorVisionGrouper (Table-Barrier), RawPictureGrouper
  tables.py           Trích xuất bảng & nối bảng đa trang (link_table_continuations)
  diagrams.py         Tái dựng đồ thị node / edge / group cho sơ đồ vector & bitmap
  formulas.py         Crop công thức và gọi backend sinh LaTeX
  raster.py           Xử lý trang raster và vùng hybrid bằng PageReader
  assemble.py         Lọc chồng lấn figure/table, gán word vào box, gắn running header/footer, extract_plain_text
  hierarchy.py        Nhận diện heading, cây mục lục (apply_structure, build_section_tree), nối caption
  markdown.py         Xuất Markdown / Mermaid / LaTeX / HTML table (to_markdown, extract_markdown)
  models.py           Dataclasses dùng chung (đơn vị PDF point)
  backends/           Cài đặt các backend ONNX/PyTorch/native và registry.py
  layout/             Taxonomy 17 nhãn Heron và postprocess_layout()
  eval/               Bộ công cụ chấm điểm & benchmark nội bộ
  test_*.py           Unit & integration test scripts
```

> Lưu ý cache: `detect_figures()` mặc định lưu JSON cache layout của mỗi PDF tại `outputs/08_pdf_figure_crops/<tên_file>/_layout_cache/` (khóa theo `backend`, `revision`, `dpi`, `postprocess_version`, `pdf_sha256`).

---

## 6. Kiểm thử & Kết quả Benchmark

Chạy toàn bộ unit test (các script phụ thuộc dataset local sẽ tự động `SKIP` nếu thiếu file):

```powershell
foreach ($f in @("test_text.py","test_layout.py","test_figures.py","test_assemble.py","test_hierarchy.py","test_formulas.py","test_markdown.py","test_tables.py","test_raster.py","test_router.py")) {
  .venv\Scripts\python -B "pdf/$f"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

### Số đo kiểm chứng (Colab Tesla T4 GPU, so với `Docling`)

* **`DocLayNet` (113 trang, 6 miền EN):** `layout_f1 = 0.7849`, `layout_macro_f1 = 0.7996`, `bbox_agnostic_f1 = 0.8685` (vượt `Docling` `0.8552`), `formula F1 = 0.8571`, `0.736 s/trang`, `2,816 MB` RAM (`Docling`: `1.082 s/trang`, `4,008 MB`).
* **`BizDocBench` (15 PDF GT, 493 trang EN):** `heading_recall = 0.8906` / `precision = 0.8472` (`Docling`: `0.8018` / `0.8096`), `parent_chain_accuracy = 0.3914` (`Docling`: `0.0000`), `nid = 0.9430`, `table_record_match = 0.8528` (`Docling`: `0.8065`).
* **`Japanese Insurance` (113 PDF, 1.437 trang JP):** `mean_cer = 0.0115` (`Docling`: `0.0976`), `tau_matched = 0.9956`, `tau_penalized = +0.3191`, `outline_f1 = 0.8412`, `fig_strict_f1 = 0.9676`, `diag_node_f1 = 0.8742` (`Docling`: `0.0000`), `merged_span_f1 = 0.6451`, `1,853 MB` RAM.
* **`PubTables-1M` (80 bảng Val, Wave C):** `tbl_det_f1 = 0.9744` (`Docling`: `0.6667`), `teds_struct = 0.8716` (`Docling`: `0.4582`).
* **`OmniDocBench` (60 trang EN/ZH, Wave C):** `mean_cer = 0.1426`, `tau_matched = 0.9832`, `tbl_det_f1 = 0.9636`, `teds_struct = 0.8260`.
