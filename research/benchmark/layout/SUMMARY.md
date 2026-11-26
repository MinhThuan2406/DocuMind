# Layout Detection Benchmark Summary

Đánh giá 3 mô hình Layout trong pipeline **Docling** trên **6.272 trang PDF**:
* **Model:**
  * **`docling-layout-heron`** *(IBM — default)*
  * **`DocLayout-YOLO`** *(OpenDataLab / MinerU — plugin)*
  * **`PP-DocLayoutV3`** *(Baidu / PaddleOCR — plugin)*
* **Dataset:**
  * **`DocLayNet`** *(4.999 trang — tiếng Anh)*
  * **`OmniDocBench EN`** *(755 trang — tiếng Anh)*
  * **`OmniDocBench JASyn`** *(518 trang — tiếng Nhật)*

---

## 1. Metric

* **`F1@0.50`:** Trung bình điều hòa giữa độ chính xác (*Precision*) và độ phủ (*Recall*) khi khung dự đoán chồng lấp khung đáp án từ `50%` trở lên (`IoU ≥ 0.50`). Phản ánh trung thực nhất chất lượng đầu ra JSON của Docling (vốn không lưu `confidence score` thô).
* **`mAP@0.50:0.95`:** Độ chính xác trung bình trên tất cả các lớp và 10 ngưỡng chồng lấp từ `0.50` đến `0.95` — đánh giá khắt khe cả độ đúng nhãn lẫn độ sát viền khung.
* **`IoU` *(Intersection over Union)*:** Tỷ lệ diện tích phần giao trên phần hợp giữa khung dự đoán và khung đáp án (từ `0` đến `1`), đo mức độ khớp vị trí và làm ngưỡng xét đúng/sai cho `F1` và `mAP`.

---

## 2. Kết quả tổng hợp (`F1@0.50 ↑` / `mAP@0.50:0.95 ↑`)

| Tập dữ liệu | Số trang | `docling-layout-heron` | `DocLayout-YOLO` | `PP-DocLayoutV3` | Model dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DocLayNet** | 4,999 | **88.25%** / **55.75%** | 69.11% / 26.13% | 77.88% / 28.37% | **`heron`** |
| **OmniDocBench EN** | 755 | 73.50% / 32.68% | 76.40% / **41.16%** | **76.77%** / 39.98% | **`YOLO` ≈ `V3`** |
| **OmniDocBench JASyn** | 518 | 38.21% / 14.00% | **44.93%** / **29.84%** | 36.88% / 22.77% | **`DocLayout-YOLO`** |

---

## 3. Phân tích chi tiết — Theo từng lớp nhãn (`F1@0.50 ↑`)

### 3.1. `DocLayNet` (9 lớp nhãn)

| Lớp nhãn | Số GT Boxes | `docling-layout-heron` | `DocLayout-YOLO` | `PP-DocLayoutV3` |
| :--- | :---: | :---: | :---: | :---: |
| `Text` | 40,462 | **89.77%** | 70.16% | 85.56% |
| `Section-header` | 8,885 | **86.40%** | 80.35% | 81.26% |
| `Page-footer` | 3,994 | **85.37%** | 61.27% | 63.65% |
| `Picture` | 3,534 | **80.77%** | 55.42% | 59.34% |
| `Page-header` | 3,366 | **88.47%** | 46.83% | 61.48% |
| `Table` | 2,394 | **86.98%** | 85.08% | 85.44% |
| `Formula` | 1,966 | **91.81%** | 78.45% | 42.33% |
| `Caption` | 1,543 | **83.24%** | 55.89% | 58.18% |
| `Footnote` | 387 | **80.45%** | 21.78% | 37.66% |

### 3.2. `OmniDocBench EN` (9 lớp nhãn)

| Lớp nhãn | Số GT Boxes | `docling-layout-heron` | `DocLayout-YOLO` | `PP-DocLayoutV3` |
| :--- | :---: | :---: | :---: | :---: |
| `text` | 7,348 | 79.52% | 80.77% | **82.66%** |
| `formula` | 1,802 | 66.64% | 66.98% | **68.39%** |
| `section_header` | 1,569 | 71.49% | **71.95%** | 68.13% |
| `page_footer` | 807 | 61.45% | 79.76% | **88.57%** |
| `caption` | 727 | **50.24%** | 49.71% | 44.38% |
| `page_header` | 713 | 67.46% | 78.87% | **80.72%** |
| `picture` | 488 | 64.81% | **75.45%** | 62.46% |
| `table` | 243 | 81.55% | **87.93%** | 85.25% |
| `footnote` | 225 | 22.51% | **43.48%** | 43.09% |

### 3.3. `OmniDocBench JASyn` (5 lớp nhãn)

| Lớp nhãn | Số GT Boxes | `docling-layout-heron` | `DocLayout-YOLO` | `PP-DocLayoutV3` |
| :--- | :---: | :---: | :---: | :---: |
| `text` | 7,280 | 32.47% | **35.26%** | 27.76% |
| `section_header` | 1,543 | 59.38% | **60.12%** | 52.20% |
| `picture` | 532 | 34.64% | **69.93%** | 60.75% |
| `table` | 406 | 46.70% | **84.51%** | 78.08% |
| `formula` | 107 | 8.33% | **29.93%** | 26.09% |

---

## 4. Kết luận

1. **Nhận xét:**
   * **`docling-layout-heron`** dẫn đầu trên `DocLayNet` (`88.25% F1`), nhưng giảm mạnh khi sang `OmniDocBench EN` và `JASyn` (`14.00% mAP`).
   * **`DocLayout-YOLO`** ổn định nhất trên dữ liệu mới: dẫn đầu `mAP` trên cả `OmniDocBench EN` (`41.16%`) lẫn `JASyn` (`29.84%`), đồng thời đứng đầu ở các lớp quan trọng như `table` (`87.93%`), `picture` (`75.45%`) và `section_header`.
   * Cả 3 model đều đạt điểm thấp hơn rõ trên `OmniDocBench JASyn` so với hai tập tiếng Anh, trong đó nhóm tài liệu chữ dọc (`book_vertical`, `newspaper_vertical`) là một trong những nguyên nhân chính kéo điểm trung bình xuống thấp.

2. **Quyết định:**
   * **Chọn `DocLayout-YOLO`** làm model chính cho tác vụ detect layout nhờ khả năng giữ hiệu năng tốt nhất trên cả tiếng Anh và tiếng Nhật.
   * **Áp dụng dynamic image scaling** và giới hạn **phạm vi hoạt động (*Scope*)** ở tài liệu bố cục ngang tiêu chuẩn (tiếng Anh và tiếng Nhật).

3. **Trade-off:**
   * `DocLayout-YOLO` giữ đủ **11 nhãn cốt lõi** của Docling (`text`, `title`, `table`, `picture`, `caption`, `footnote`, `formula`, `list_item`, `code`, `page_header`, `page_footer`, không làm mất nội dung chữ), nhưng **mất 6 nhãn chuyên biệt của `heron`**: `section_header` (gộp về `title`, có thể bù lại bằng `HeadingHierarchyOptions` để phân cấp H1–H6), `document_index` (mục lục), `checkbox_selected` / `checkbox_unselected`, `form` và `key_value_region`.
   * Đánh đổi này là chấp nhận được cho tài liệu tổng quát Anh–Nhật để đổi lấy độ chính xác khung (`mAP`) vượt trội ở các thành phần trọng yếu (`table`, `picture`, `footnote`).
