# OCR Benchmark Summary

Đánh giá 3 mô hình OCR trong pipeline **Docling** (cố định Layout = `DocLayout-YOLO`, `OcrMode.FULL_PAGE`, Dynamic Image Scaling) trên **1.265 trang PDF**:
* **Model:**
  * **`EasyOCR`** *(Docling built-in — PyTorch)*
  * **`PP-OCRv6_medium`** *(Docling built-in `RapidOCR` — ONNXRuntime GPU)*
  * **`Surya OCR`** *(plugin `docling-surya` — PyTorch)*
* **Dataset:**
  * **`OmniDocBench EN`** *(747 trang — tiếng Anh)*
  * **`OmniDocBench JASyn`** *(518 trang — tiếng Nhật)*

---

## 1. Metric

* **`NED` *(Normalized Edit Distance)*:** Khoảng cách chỉnh sửa chuẩn hóa (`Levenshtein(pred, gt) / max(|pred|, |gt|)`), nằm trong `[0, 1]`. Giá trị càng nhỏ nghĩa là văn bản OCR càng sát đáp án.
* **`Similarity` *(`1 - NED`)*:** Độ tương đồng ký tự tính theo phần trăm (`(1 - NED) × 100%`), giúp so sánh trực quan mức độ khớp chữ giữa các model.
* **`Micro CER` *(Character Error Rate)*:** Tỷ lệ lỗi ký tự tổng trên toàn tập (`∑ Levenshtein / ∑ |gt| × 100%`), nhạy với cả lỗi đọc sót lẫn lỗi sinh thừa chữ.

---

## 2. Kết quả tổng hợp (`NED ↓` / `Similarity ↑` / `Micro CER ↓`)

| Tập dữ liệu | Số trang | `EasyOCR` | `PP-OCRv6_medium` | `Surya OCR` | Model dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OmniDocBench EN** | 747 | `0.4590` / 54.10% / 48.36% | `0.2884` / 71.16% / 34.11% | **`0.2793`** / **72.07%** / **31.86%** | **`Surya` ≈ `PP-v6`** |
| **OmniDocBench JASyn** | 518 | `0.8555` / 14.45% / 85.06% | **`0.4482`** / **55.18%** / **50.16%** | `0.4921` / 50.79% / 67.63% | **`PP-OCRv6_medium`** |

---

## 3. Phân tích chi tiết

### 3.1. Theo loại tài liệu tiếng Anh — `OmniDocBench EN` (`Similarity 1-NED % ↑`)

| Loại tài liệu | Số trang | `EasyOCR` | `PP-OCRv6_medium` | `Surya OCR` |
| :--- | :---: | :---: | :---: | :---: |
| `academic_literature` | 205 | 47.93% | 63.40% | **64.19%** |
| `book` | 130 | 51.26% | 71.23% | **71.44%** |
| `PPT2PDF` | 118 | 63.19% | **77.16%** | 75.11% |
| `magazine` | 84 | 61.05% | 81.04% | **83.64%** |
| `exam_paper` | 81 | 52.84% | **77.41%** | 73.09% |
| `newspaper` | 77 | 57.18% | 65.25% | **79.15%** |
| `colorful_textbook` | 51 | 52.11% | **72.35%** | 68.38% |

### 3.2. Theo loại tài liệu tiếng Nhật — `OmniDocBench JASyn` (`Similarity 1-NED % ↑`)

| Loại tài liệu | Số trang | `EasyOCR` | `PP-OCRv6_medium` | `Surya OCR` |
| :--- | :---: | :---: | :---: | :---: |
| `book_vertical` | 55 | 1.15% | 14.40% | **14.47%** |
| `text_book` | 53 | 11.87% | **61.94%** | 56.34% |
| `report` | 42 | 16.05% | **58.45%** | 55.53% |
| `slide` | 38 | 17.71% | **46.74%** | 40.32% |
| `press_release` | 38 | 19.22% | **64.61%** | 59.21% |
| `magazine` | 37 | 17.59% | **65.07%** | 54.01% |
| `user_manual` | 37 | 21.28% | **59.40%** | 53.92% |
| `two_column_simple` | 36 | 29.16% | **76.55%** | 75.06% |
| `academic_paper` | 36 | 6.42% | **66.01%** | 64.26% |
| `handwritten_note` | 35 | 12.82% | **52.11%** | 48.70% |
| `lab_notebook` | 33 | 13.40% | **55.85%** | 52.01% |
| `exam_paper` | 29 | 19.60% | **68.45%** | 61.71% |
| `newspaper` | 28 | 15.43% | **65.80%** | 59.32% |
| `newspaper_vertical` | 21 | 5.02% | **28.00%** | 25.38% |

---

## 4. Kết luận

1. **Nhận xét:**
   * **`PP-OCRv6_medium`** cân bằng nhất trên cả 2 ngôn ngữ: bám sát `Surya OCR` trên `OmniDocBench EN` (`71.16%` so với `72.07%` Similarity) và dẫn đầu trên **13/14 loại tài liệu** của `OmniDocBench JASyn` (`Micro CER` `50.16%` so với `67.63%` của `Surya OCR`).
   * **`EasyOCR`** kém hơn rõ trên cả 2 tập, đặc biệt giảm rất mạnh trên tiếng Nhật (`14.45%` Similarity).
   * Trên `OmniDocBench JASyn`, nhóm chữ dọc (`book_vertical`, `newspaper_vertical`) là nguyên nhân chính kéo điểm trung bình của cả 3 model xuống thấp.

2. **Quyết định:**
   * **Chọn `PP-OCRv6_medium` (`RapidOCR`)** làm model chính cho tác vụ OCR vì đạt hiệu quả cao và ổn định nhất trên cả tiếng Anh lẫn tiếng Nhật.
