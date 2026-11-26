# Docling Plugins

## Cấu hình Plugin đã chốt

| Thành phần | Default (`default_docling`) | Đã chọn (`plugged_docling`) | Nguồn tích hợp |
| :--- | :--- | :--- | :--- |
| **Layout Detection** | `docling-layout-heron` | **`DocLayout-YOLO`** | Plugin `_doclayout_yolo` |
| **OCR** | `EasyOCR` (`OcrAutoOptions`) | **`PP-OCRv6_medium`** (`RapidOCR`) | Built-in `RapidOcrOptions` (ONNX GPU) |
| **Table Structure** | `TableFormer ACCURATE` | **`RapidTable — UniTable`** | Plugin `_rapid_table` (PyTorch GPU) |
| **Image Scaling** | Fixed `scale=2.0` | **Dynamic Image Scaling** | Runtime stage scaling |

> **Cách đánh giá:** Layout và OCR được đánh giá độc lập nhau; sau khi chọn xong cả hai mới cố định làm đầu vào để đánh giá Table Structure, rồi so sánh tổng thể End-to-End.

---

## Tóm tắt kết quả Benchmark

### 1. Layout Detection
*Chi tiết:* [layout/SUMMARY.md](../../benchmark/layout/SUMMARY.md)

| Tập dữ liệu | `docling-layout-heron` | `DocLayout-YOLO` | `PP-DocLayoutV3` |
| :--- | :---: | :---: | :---: |
| **DocLayNet** (`4,999` trang — `F1@50` / `mAP`) | **88.25%** / **55.75%** | 69.11% / 26.13% | 77.88% / 28.37% |
| **OmniDocBench EN** (`755` trang — `F1@50` / `mAP`) | 73.50% / 32.68% | 76.40% / **41.16%** | **76.77%** / 39.98% |
| **OmniDocBench JASyn** (`518` trang — `F1@50` / `mAP`) | 38.21% / 14.00% | **44.93%** / **29.84%** | 36.88% / 22.77% |

* **Chọn `DocLayout-YOLO`:** tổng quát tốt nhất trên dữ liệu mới ở cả tiếng Anh và tiếng Nhật (dẫn đầu `mAP` trên cả EN lẫn JASyn và đứng đầu các lớp quan trọng như `table`, `picture`). *(Trade-off: giữ 11 nhãn cốt lõi của Docling, mất 6 nhãn chuyên biệt của `heron` như `checkbox`, `form`, `section_header` — trong đó `section_header` có thể bù bằng `HeadingHierarchyOptions`).*

---

### 2. OCR
*Chi tiết:* [ocr/SUMMARY.md](../../benchmark/ocr/SUMMARY.md) *(cố định Layout = `DocLayout-YOLO`, `OcrMode.FULL_PAGE`)*

| Tập dữ liệu | `EasyOCR` | `PP-OCRv6_medium` | `Surya OCR` |
| :--- | :---: | :---: | :---: |
| **OmniDocBench EN** (`747` trang — `Sim` / `CER`) | 54.10% / 48.36% | 71.16% / 34.11% | **72.07%** / **31.86%** |
| **OmniDocBench JASyn** (`518` trang — `Sim` / `CER`) | 14.45% / 85.06% | **55.18%** / **50.16%** | 50.79% / 67.63% |

* **Chọn `PP-OCRv6_medium` (`RapidOCR`):** bám sát `Surya OCR` trên tiếng Anh và dẫn đầu trên **13/14 loại tài liệu** tiếng Nhật (`Micro CER` thấp hơn `Surya` tới ~17.5 điểm phần trăm).

---

### 3. Table Structure
*Chi tiết:* [table structure/SUMMARY.md](../../benchmark/table%20structure/SUMMARY.md) *(trên các bảng khớp được — `TEDS` / `TEDS-Struct`)*

| Tập dữ liệu | `TableFormer ACCURATE` | `RapidTable — SLANet_plus` | `RapidTable — UniTable` |
| :--- | :---: | :---: | :---: |
| **PubTables-1M** (`1,000` bảng) | 87.58% / 91.01% | 87.02% / **93.20%** | **88.31%** / 92.46% |
| **OmniDocBench EN** (`243` bảng) | 72.09% / 79.26% | **75.71%** / **87.02%** | 74.62% / 83.22% |
| **OmniDocBench JASyn** (`406` bảng) | 79.90% / 85.98% | 75.44% / 85.66% | **81.84%** / **90.22%** |

* **Chọn `RapidTable — UniTable`:** dẫn đầu `TEDS` trên cả `PubTables-1M` lẫn `OmniDocBench JASyn` (đặc biệt mạnh trên bảng gộp ô phức tạp tiếng Nhật `75.45%`), đồng thời ổn định trên `OmniDocBench EN`.

---

### 4. End-to-End (`default_docling` vs. `plugged_docling` vs. `custom_parser`)
*Chi tiết:* [default_docling vs. plugged_docling vs. custom_parser/SUMMARY.md](../../benchmark/default_docling%20vs.%20plugged_docling%20vs.%20custom_parser/SUMMARY.md)

| Tập dữ liệu — Hệ thống | Layout `mAP` | OCR `Sim` | RO `Tau` | Table `TEDS All` | `Avg s/trang` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EN** — `default_docling` | 26.68% | 56.76% | 0.9387 | 47.10% | 6.55s |
| **EN** — `plugged_docling` | **36.57%** | 75.52% | 0.9433 | 63.11% | **4.68s** |
| **EN** — `custom_parser` | 33.14% | **76.95%** | **0.9724** | **63.60%** | 33.26s |
| **JASyn** — `default_docling` | 12.29% | 9.97% | 0.7865\* | 10.98% | **1.91s** |
| **JASyn** — `plugged_docling` | **35.45%** | **65.24%** | 0.6446 | **77.16%** | 3.07s |
| **JASyn** — `custom_parser` | 21.57% | 55.14% | **0.7227** | 55.22% | 30.73s |

*\*`default_docling` trên JASyn chỉ khớp được `17.47%` khối text đáp án (`plugged_docling`: `81.46%`).*

* **Kết luận tổng thể:** **`plugged_docling`** vượt xa `default_docling` trên cả 2 ngôn ngữ, cân bằng chất lượng tốt nhất giữa tiếng Anh và tiếng Nhật (vượt trội `custom_parser` trên JASyn), đồng thời **nhanh hơn `custom_parser` `7×–10×`**.
