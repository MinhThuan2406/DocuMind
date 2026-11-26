# Tổng hợp Benchmark End-to-End: `default_docling` vs. `plugged_docling` vs. `custom_parser`

* **Dataset:**
  * **OmniDocBench EN** (`755` trang tiếng Anh; `747` trang OCR, `719` trang Reading Order, `243` bảng).
  * **OmniDocBench JASyn** (`518` trang tiếng Nhật gồm `442` trang viết ngang & `76` trang viết dọc; `406` bảng).
* **Môi trường chạy:** Kaggle `2× NVIDIA Tesla T4` (15 GB VRAM/GPU), chạy song song 2 worker (1 GPU/worker).
* **Chi tiết thực nghiệm:** [OmniDocBench EN/evaluation.ipynb](OmniDocBench%20EN/evaluation.ipynb) | [OmniDocBench JASyn/evaluation.ipynb](OmniDocBench%20JASyn/evaluation.ipynb)

---

## 1. Định nghĩa các Metric

1. **Layout Detection:**
   * **`F1@0.50`:** Trung bình điều hòa giữa Precision và Recall tại ngưỡng `IoU ≥ 0.50` *(càng cao càng tốt)*.
   * **`mAP@0.50:0.95`:** Độ chính xác trung bình trên tất cả các lớp nhãn qua 10 ngưỡng `IoU` từ `0.50` đến `0.95` *(càng cao càng tốt)*.
2. **Text / OCR:**
   * **`NED` *(Normalized Edit Distance)*:** `Levenshtein(pred, gt) / max(|pred|, |gt|)` *(càng thấp càng tốt)*.
   * **`Similarity` *(`1 - NED`)*:** Độ tương đồng ký tự theo phần trăm `(1 - NED) × 100%` *(càng cao càng tốt)*.
   * **`Micro CER`:** Tỷ lệ lỗi ký tự tổng trên toàn tập `∑ Levenshtein / ∑ |gt| × 100%` *(càng thấp càng tốt)*.
3. **Reading Order:**
   * **`Block Match %`:** Tỷ lệ khối văn bản đáp án khớp được 1-1 với dự đoán (`IoU ≥ 0.10`) để tính thứ tự đọc *(càng cao càng tốt)*.
   * **`Kendall's Tau (τ)`:** Hệ số tương quan thứ tự trong `[-1, 1]` tính trên các khối khớp được *(càng gần `1.0` càng đúng thứ tự)*.
   * **`NID` *(Normalized Indel Distance)*:** `1 - LIS(ranks) / k` trong `[0, 1]` *(càng gần `0.0` càng tốt)*.
4. **Table Structure:**
   * **`Match %`:** Tỷ lệ bảng đáp án được khoanh trúng (`IoU ≥ 0.10`).
   * **`TEDS` / `TEDS-Struct`:** Độ tương đồng cây HTML bảng (exact 3-level Zhang-Shasha) trên cả nội dung + cấu trúc (`TEDS`) và thuần cấu trúc hàng/cột/gộp ô (`TEDS-Struct`). `All` tính toàn bộ bảng (bảng bỏ sót = `0%`); `Matched` chỉ tính trên các bảng khớp được.
5. **Hiệu năng hệ thống:**
   * **`Avg s/trang`:** Thời gian xử lý trung bình mỗi trang trên 1 GPU worker *(càng thấp càng tốt)*.
   * **`Wall s/trang`:** Thời gian thực tế mỗi trang khi chạy song song 2 GPU *(càng thấp càng tốt)*.
   * **`Peak RAM` / `Peak VRAM`:** Mức bộ nhớ RAM đỉnh trên mỗi worker (`MB`) và VRAM đỉnh trên mỗi GPU (`MB`).

---

## 2. Model & Cách tiếp cận của từng Pipeline

| Thành phần | `default_docling` | `plugged_docling` | `custom_parser` |
| :--- | :--- | :--- | :--- |
| **Cách tiếp cận** | Pipeline chuẩn của Docling (hybrid text + OCR `OcrMode.DEFAULT`, fixed `scale=2.0`) | Docling gắn 3 plugin đã chọn qua benchmark (`OcrMode.DEFAULT` + Dynamic Image Scaling) | Pipeline độc lập ưu tiên `pdfplumber` (CPU) cho digital PDF, gọi ONNX cho layout/OCR/bảng không kẻ |
| **Layout Detection** | `docling-layout-heron` | `DocLayout-YOLO` | `heron-onnx` + heuristic gộp/lọc (`V1AnchorGrouper`, `semantic_layout="always"`) |
| **Text / OCR** | `pypdfium2` + `EasyOCR` (`lang=["ja","en"]`) | `pypdfium2` + `RapidOCR PP-OCRv6_medium` (`lang=["ch"]`, ONNX GPU) | `pdfplumber` + `RapidOCR` (ONNX GPU, mặc định đa ngôn ngữ) |
| **Reading Order** | Bộ `ReadingOrderModel` mặc định của Docling | Bộ `ReadingOrderModel` mặc định của Docling | Thuật toán hình học `xy_cut_order` |
| **Table Structure** | `TableFormer ACCURATE` | `RapidTable — UniTable` (PyTorch GPU, `use_ocr=False`) | `table_auto` (`pdfplumber` cho bảng kẻ + `TATR ONNX` cho bảng không kẻ) |

---

## 3. Kết quả so sánh

### 3.1. OmniDocBench EN (755 trang tiếng Anh)

| Khía cạnh | Metric | `default_docling` | `plugged_docling` | `custom_parser` |
| :--- | :--- | :---: | :---: | :---: |
| **Layout** | `F1@0.50` (↑) | 58.12% | 60.72% | **72.13%** |
| | `mAP@0.50:0.95` (↑) | 26.68% | **36.57%** | 33.14% |
| **Text / OCR** | `NED` (↓) | 0.4324 | 0.2448 | **0.2305** |
| | `Similarity %` (↑) | 56.76% | 75.52% | **76.95%** |
| | `Micro CER %` (↓) | 41.56% | 24.79% | **19.86%** |
| **Reading Order** | `Block Match %` (↑) | 78.68% | 85.35% | **92.33%** |
| | `Kendall's Tau` (↑) | 0.9387 | 0.9433 | **0.9724** |
| | `NID` (↓) | 0.0491 | 0.0438 | **0.0249** |
| **Table Structure** | `Table Match %` (↑) | 80.66% | **84.77%** | 80.66% |
| | `TEDS Matched %` (↑) | 58.40% | 74.44% | **78.86%** |
| | `TEDS All %` (↑) | 47.10% | 63.11% | **63.60%** |
| | `TEDS-Struct All %` (↑) | 54.02% | 70.28% | **70.70%** |
| **Hiệu năng** | `Passed / Failed` | 755 / 0 | 755 / 0 | 755 / 0 |
| | `Avg s/trang` (1 GPU) (↓) | 6.55s | **4.68s** | 33.26s |
| | `Wall s/trang` (2 GPU) (↓) | 3.32s | **2.36s** | 16.64s |
| | `Peak RAM` (MB/worker) (↓) | 4,030 | 4,607 | **2,754** |
| | `Peak VRAM` (MB/GPU) (↓) | **4,615** | 12,247 | 5,397 |

### 3.2. OmniDocBench JASyn (518 trang tiếng Nhật)

| Khía cạnh | Metric | `default_docling` | `plugged_docling` | `custom_parser` |
| :--- | :--- | :---: | :---: | :---: |
| **Layout** | `F1@0.50` (↑) | 32.03% | **51.72%** | 45.81% |
| | `mAP@0.50:0.95` (↑) | 12.29% | **35.45%** | 21.57% |
| **Text / OCR** | `NED` (↓) | 0.9003 | **0.3476** | 0.4486 |
| | `Similarity %` (↑) | 9.97% | **65.24%** | 55.14% |
| | `Similarity %` — Horizontal (`442` trang) (↑) | 11.48% | **73.05%** | 61.83% |
| | `Micro CER %` (↓) | 89.59% | **32.08%** | 43.55% |
| **Reading Order** | `Block Match %` (↑) | 17.47% | **81.46%** | 66.38% |
| | `Kendall's Tau` (↑) | 0.7865\* | 0.6446 | **0.7227** |
| | `NID` (↓) | 0.0839\* | 0.2020 | **0.1575** |
| **Table Structure** | `Table Match %` (↑) | 30.79% | **95.32%** | 67.49% |
| | `TEDS Matched %` (↑) | 35.65% | 80.95% | **81.82%** |
| | `TEDS All %` (↑) | 10.98% | **77.16%** | 55.22% |
| | `TEDS-Struct All %` (↑) | 12.66% | **85.93%** | 61.08% |
| **Hiệu năng** | `Passed / Failed` | 518 / 0 | 518 / 0 | 517 / 1 |
| | `Avg s/trang` (1 GPU) (↓) | **1.91s** | 3.07s | 30.73s |
| | `Wall s/trang` (2 GPU) (↓) | **0.99s** | 1.57s | 15.47s |
| | `Peak RAM` (MB/worker) (↓) | **2,315** | 4,032 | 2,450 |
| | `Peak VRAM` (MB/GPU) (↓) | **2,625** | 10,509 | 4,323 |

*\*Ghi chú JASyn Reading Order:* `default_docling` chỉ khớp được **17.47%** khối đáp án (316/518 trang có ≥2 khối khớp), nên `Tau`/`NID` cao là do bỏ sót phần lớn khối khó; khi xét độ phủ thực tế, `plugged_docling` đạt `Block Match = 81.46%` (516/518 trang) và `custom_parser` đạt `66.38%` (469/518 trang).

---

## 4. Kết luận

1. **Nhận xét:**
   * **`plugged_docling` vs `default_docling`:** Thay thế 3 plugin (`DocLayout-YOLO`, `PP-OCRv6_medium`, `UniTable`) kèm *Dynamic Image Scaling* cải thiện mạnh trên cả hai ngôn ngữ — đặc biệt trên tiếng Nhật, kéo `OCR Sim` từ `9.97% → 65.24%` và `TEDS All` từ `10.98% → 77.16%`, đồng thời chạy nhanh hơn trên tiếng Anh (`6.55s → 4.68s/trang`).
   * **`plugged_docling` vs `custom_parser`:**
     * Trên **tiếng Anh (EN)**: `custom_parser` nhỉnh hơn về bố cục (`F1@50` `72.13%` vs `60.72%`), thứ tự đọc (`Tau` `0.9724` vs `0.9433`) và `Micro CER` (`19.86%` vs `24.79%`), gần như ngang bằng về bảng (`TEDS All` `63.60%` vs `63.11%`) và nhẹ RAM/VRAM hơn, nhưng **chậm hơn ~7×** (`33.26s` vs `4.68s/trang`).
     * Trên **tiếng Nhật (JASyn)**: `plugged_docling` **vượt trội toàn diện** từ layout (`mAP` `35.45%` vs `21.57%`), OCR (`65.24%` vs `55.14%`) đến bảng (`TEDS All` `77.16%` vs `55.22%`), **nhanh hơn ~10×** (`3.07s` vs `30.73s/trang`) và không lỗi file nào (`0` vs `1` failed).
   * **Lý do `custom_parser` chậm và kém hiệu quả trên tiếng Nhật:**
     * *Về tốc độ:* Kiến trúc của `custom_parser` tối ưu cho PDF digital (tận dụng text/đường kẻ sẵn có qua `pdfplumber`), nhưng khi gặp PDF ảnh (`raster` — 100% tập OmniDocBench) và tắt cache layout, pipeline phải chạy nhiều bước xử lý ảnh, OCR theo từng vùng (*table tiling*) và hậu xử lý bảng tuần tự trên CPU Python → bị nghẽn cổ chai CPU dù có GPU.
     * *Về tiếng Nhật:* Bộ layout `heron-onnx` tổng quát kém hơn `DocLayout-YOLO` trên văn bản tiếng Nhật (khiến `Table Match` chỉ đạt `67.49%` vs `95.32%`), đồng thời thiếu cơ chế scale ảnh động nên OCR kém chính xác hơn trên trang chữ nhỏ, mật độ dày.
   * **Ảnh hưởng của chữ dọc trên `JASyn`:** Nhóm tài liệu chữ dọc (`book_vertical`, `newspaper_vertical` — 76 trang) là nguyên nhân chính kéo điểm trung bình OCR và Reading Order (`Tau` / `NID`) của cả 3 pipeline xuống thấp; khi xét riêng 442 trang viết ngang, `plugged_docling` đạt `OCR Sim = 73.05%`.

2. **Quyết định:**
   * **Chọn `plugged_docling`** làm pipeline PDF chính thức cho bài toán song ngữ **Anh – Nhật** nhờ độ chính xác cân bằng cao trên cả hai ngôn ngữ, vượt trội toàn diện trên tiếng Nhật và tốc độ nhanh hơn `custom_parser` **7×–10×** (đánh đổi mức `Peak VRAM` `~10.5–12.2 GB/GPU`, vẫn nằm gọn trong giới hạn 15 GB của Tesla T4).
