# Table Structure Benchmark Summary

Đánh giá 3 mô hình Table Structure trong pipeline **Docling** (cố định Layout = `DocLayout-YOLO`, OCR = `PP-OCRv6_medium`, Dynamic Image Scaling) trên **1.649 bảng PDF**:
* **Model:**
  * **`TableFormer ACCURATE`** *(IBM — Docling default)*
  * **`RapidTable — SLANet_plus`** *(Baidu — plugin `rapid-table`)*
  * **`RapidTable — UniTable`** *(Transformer — plugin `rapid-table`)*
* **Dataset:**
  * **`PubTables-1M`** *(1.000 bảng ngẫu nhiên, `seed=42` — tiếng Anh)*
  * **`OmniDocBench EN`** *(243 bảng — tiếng Anh)*
  * **`OmniDocBench JASyn`** *(406 bảng — tiếng Nhật)*

---

## 1. Metric

* **`TEDS` *(Tree-Edit-Distance-based Similarity)*:** Đo độ tương đồng giữa cây HTML bảng dự đoán và cây đáp án trên cả **cấu trúc bảng** (`<tr>`, `<td>`, `rowspan`, `colspan`) lẫn **nội dung chữ trong từng ô**.
* **`TEDS-Struct` *(Structure-only TEDS)*:** Chỉ đo độ chính xác **thuần cấu trúc hàng/cột/gộp ô**, giúp tách riêng năng lực dựng khung bảng khỏi sai số OCR.

---

## 2. Kết quả tổng hợp — Trên các bảng bắt trúng (`TEDS ↑` / `TEDS-Struct ↑`)

| Tập dữ liệu | Bảng khớp / Tổng | Tỷ lệ khớp (Layout) | `TableFormer ACCURATE` | `RapidTable — SLANet_plus` | `RapidTable — UniTable` | Model dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PubTables-1M** | 980 / 1,000 | 98.00% | 87.58% / 91.01% | 87.02% / **93.20%** | **88.31%** / 92.46% | **`UniTable`** |
| **OmniDocBench EN** | 205 / 243 | 84.36% | 72.09% / 79.26% | **75.71%** / **87.02%** | 74.62% / 83.22% | **`SLANet_plus`** |
| **OmniDocBench JASyn** | 385 / 406 | 94.83% | 79.90% / 85.98% | 75.44% / 85.66% | **81.84%** / **90.22%** | **`UniTable`** |

> *Ghi chú: **Tỷ lệ khớp** do bước Layout (`DocLayout-YOLO`) đảm nhận việc khoanh vùng bảng trước khi đưa vào model Table.*

---

## 3. Phân tích chi tiết — Bảng đơn giản vs Bảng có gộp ô (`TEDS % ↑`)

| Tập dữ liệu — Loại bảng | Số bảng | `TableFormer ACCURATE` | `RapidTable — SLANet_plus` | `RapidTable — UniTable` |
| :--- | :---: | :---: | :---: | :---: |
| **PubTables-1M** — `Simple (Grid)` | 482 | 88.52% | 88.87% | **90.20%** |
| **PubTables-1M** — `Complex (Spanning)` | 518 | **83.15%** | 81.94% | **83.15%** |
| **OmniDocBench EN** — `Simple (Grid)` | 156 | 59.04% | **65.19%** | 63.42% |
| **OmniDocBench EN** — `Complex (Spanning)` | 87 | **64.00%** | 61.51% | 62.11% |
| **OmniDocBench JASyn** — `Simple (Grid)` | 129 | **85.50%** | 80.33% | 82.23% |
| **OmniDocBench JASyn** — `Complex (Spanning)` | 277 | 71.23% | 67.44% | **75.45%** |

---

## 4. Kết luận

1. **Nhận xét:**
   * **`RapidTable — UniTable`** mạnh nhất tổng thể: dẫn đầu `TEDS` trên cả `PubTables-1M` (`88.31%`) lẫn `OmniDocBench JASyn` (`81.84% TEDS` / `90.22% TEDS-Struct`), đặc biệt vượt trội trên bảng gộp ô tiếng Nhật (`75.45% TEDS`).
   * **`RapidTable — SLANet_plus`** dẫn đầu trên `OmniDocBench EN` (`75.71% TEDS`), nhưng giảm rõ trên bảng tiếng Nhật (`75.44%` so với `81.84%` của `UniTable`).
   * **`TableFormer ACCURATE`** không dẫn đầu trên tập dữ liệu nào ở bảng tổng hợp.

2. **Quyết định:**
   * **Chọn `RapidTable — UniTable`** làm model chính cho tác vụ nhận diện cấu trúc bảng nhờ hiệu năng ổn định nhất trên cả bảng tiếng Anh lẫn bảng tiếng Nhật phức tạp.
