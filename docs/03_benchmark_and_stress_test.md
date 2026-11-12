# BÁO CÁO THỰC NGHIỆM ĐỊNH LƯỢNG & KIỂM THỬ ÁP LỰC TRÊN OMNIDOCBENCH

Tài liệu này tổng hợp toàn bộ kết quả thực nghiệm định lượng, kiểm thử độ ổn định và phân tích lỗi biên (Edge Cases Analysis) của hệ thống **DocuMind** trên bộ dữ liệu chuẩn quốc tế **OmniDocBench v1.6** (1.651 trang tài liệu đa dạng).

> **Cách đọc tài liệu này (cập nhật):**
> - Số Kendall's Tau của **Tầng 1** được tính bằng cách đưa **bbox ground truth** vào thuật toán sắp xếp, không chạy Docling. Đó là **trần oracle** (0,9369 trên 1.602 trang hợp lệ, 49 trang bị loại vì dưới 2 phần tử), không phải điểm end-to-end. Điểm end-to-end đo sau này trên 300 trang phân tầng là Tau phạt 0,64-0,67 (Tau khớp 0,93-0,95); xem `docs/04_session_handoff_and_next_steps.md` mục 4.2.
> - Bước gom sơ đồ trong Tầng 2 nay **tắt mặc định** (giữ box `picture` thô, bật bằng `--group-diagrams`), vì đo định lượng cho thấy gom làm hại (`docs/02` mục 5).
> - Phần kiểm tra over-merge của Tầng 1 đã bị xóa (đo sai nhãn, xem mục 3).

---

## 1. Môi Trường Thực Nghiệm & Thiết Kế Kiểm Thử

- **Bộ dữ liệu:** OmniDocBench v1.6 (1.651 trang thuộc 10 nguồn tài liệu và 5 kiểu bố cục).
- **Môi trường phần cứng:** CPU Intel/AMD x86_64, RAM 32GB, **CUDA: False** (Môi trường chạy CPU thuần).
- **Công cụ (môi trường hiện tại `.venv`):** Python 3.12, IBM Docling 2.150.0, OpenCV 5.0.0.93 (`opencv-python-headless`), PyMuPDF 2.1, SciPy 1.18, NumPy 2.5. Các đánh giá end-to-end về sau chạy trên NVIDIA GTX 1650 (4 GB) với torch CUDA (xem `docs/04` mục 3).
- **Thiết kế kiểm thử 2 tầng (2-Tier Stress Test):**
  - **Tầng 1 (toàn dataset, 1.602 trang hợp lệ trên 1.651):** Đánh giá thuật toán phân tích cột và sắp xếp thứ tự đọc `reading_order` thông qua hệ số tương quan hạng **Kendall's Tau** so sánh với Ground Truth. Đầu vào là **bbox GT**, nên đây là trần oracle của thuật toán sắp xếp.
  - **Tầng 2 (Stratified Hard Subset - 20 trang khó nhất):** Chạy trọn vẹn pipeline End-to-End: `Docling -> (box picture thô, hoặc group_whole_diagrams với --group-diagrams) -> cắt ảnh từng diagram -> postprocess_ir_structure` trên 20 mẫu đại diện cho 10 loại tài liệu và các dị thường phức tạp.

---

## 2. Kết Quả Thử Nghiệm Baseline 4 Mẫu Cơ Sở (`outputs/01_docling_benchmark/`)

Trước khi chạy diện rộng, hệ thống đã kiểm thử và chuẩn hóa trên 4 mẫu tài liệu tiêu biểu:

| STT | Mã Mẫu | Loại Tài Liệu | Đặc Điểm Bố Cục | Kết Quả Phân Tích & Chuẩn Hóa |
|---|---|---|---|---|
| 01 | `01_academic_2col` | Academic Literature | Báo cáo khoa học 2 cột, 2 bảng | Bóc tách 16 blocks. Thuật toán `postprocess_ir_structure` giải quyết triệt để lỗi nhảy thứ tự: `Table 2` được đưa về `order: 9` ngay trước `Caption Table 2` (`order: 10`). |
| 02 | `02_ppt_slide` | PPT Presentation | Slide trình chiếu, text đè nhau | Bóc tách 11 blocks, nhận diện chính xác danh sách mục (bullet points) và cấu trúc card. |
| 03 | `03_merged_table` | Merged Table | Bảng khoa học phức tạp ma trận lớn | Bóc tách toàn vẹn bảng ma trận **41 hàng x 12 cột = 492 tế bào (cells)** kèm đầy đủ thuộc tính `rowspan`, `colspan`. |
| 04 | `04_financial_report` | Financial Report | Báo cáo tài chính doanh nghiệp | Bóc tách 17 blocks, 2 bảng tài chính (8x4 và 4x4) với 48 cells chi tiết. |

---

## 3. Kết Quả Tầng 1: Đánh Giá Toàn Bộ Dataset (Trần Oracle Kendall's Tau = 0.9369)

Sau khi nâng cấp thuật toán sắp xếp thứ tự đọc lên **True Recursive XY-Cut** (ưu tiên phát hiện cắt dọc X-Cut song song trước khi cắt ngang theo dải, giải quyết triệt để lỗi ngắt quãng giữa trang của tài liệu 2 cột và 3 cột) và chuẩn hóa bộ lọc Ground Truth (loại bỏ các block mask/ignore không tham gia tính thứ tự đọc):

- **Tổng số trang đánh giá:** 1.602 trang nội dung hợp lệ (các trang bìa/trắng không có block text được bỏ qua).
- **Thời gian xử lý Tầng 1:** `0.72 giây` cho 1.602 trang.
- **Điểm Kendall's Tau trung bình toàn bộ Dataset (trên box GT):** **`0.9369`** (Điểm trung vị: **`1.0000`**).
- **Về chỉ số "0 over-merging" từng được báo cáo ở đây:** đã bị **loại bỏ**. Kiểm tra cũ đưa phần tử GT (khóa `type`) vào hàm gom đọc khóa `label`, nên nhánh xử lý `picture` không bao giờ chạy, và lỗi bị nuốt bởi `except: pass`; con số 0 không mang thông tin. Over-merge được đo đúng cách ở `scripts/run_diagram_crop_eval_v3.py` / `v4.py` (xem `docs/02` mục 5).

### 3.1. Phân Tích Theo Kiểu Bố Cục (Layout Breakdown)

| Kiểu Bố Cục (Layout) | Số Trang | Mean Kendall Tau | Median | Min | Max | Tỷ lệ Đạt Chuẩn ($\ge 0.70$) | Tỷ lệ Kém ($< 0.50$) |
|---|---|---|---|---|---|---|---|
| `single_column` | 838 | **0.9699** | **1.0000** | -0.8000 | 1.0 | **96.4%** | 1.1% |
| `three_column` | 53 | **0.9552** | **1.0000** | 0.3306 | 1.0 | **92.5%** | 5.7% |
| `double_column` | 184 | **0.9521** | **1.0000** | -0.2000 | 1.0 | **95.1%** | 0.5% |
| `other_layout` | 372 | **0.8805** | **0.9569** | -0.9556 | 1.0 | **89.2%** | 4.0% |
| `1andmore_column` | 155 | **0.8701** | **0.9957** | 0.2991 | 1.0 | **80.0%** | 2.6% |

> **Mức cải thiện của thuật toán trên box GT (trần oracle, không phải điểm end-to-end; khi chạy sau Docling, B so với thứ tự gốc của Docling chỉ khác biệt nhỏ và `double_column` -0,016, `three_column` -0,001, xem `docs/04` §4.2):**
> - **Bố cục 2 cột (`double_column`):** Tăng vọt từ `0.6839` lên **`0.9521`** (Trung vị đạt tuyệt đối `1.0000`, $95.1\%$ số trang đạt chuẩn $\ge 0.70$, chỉ còn $0.5\%$ trang dưới $0.50$).
> - **Bố cục 3 cột (`three_column`):** Tăng từ `0.6489` lên **`0.9552`** (Trung vị đạt tuyệt đối `1.0000`), giải quyết hoàn hảo luồng đọc 3 cột.
> - **Bố cục hỗn hợp 1 và nhiều cột (`1andmore_column`):** Tăng từ `0.5361` lên **`0.8701`** (Trung vị đạt `0.9957`), loại bỏ hoàn toàn hiện tượng dính biên giữa tiêu đề trên và 2 cột dưới.
> - **Bố cục phức tạp (`other_layout`):** Tăng từ `0.4864` lên **`0.8805`**, tỷ lệ đạt chuẩn tăng từ $16.9\%$ lên **$89.2\%$**.

### 3.2. Phân Tích Theo Nguồn Tài Liệu (Data Source Breakdown)

| Nguồn Tài Liệu (Data Source) | Số Trang | Mean Kendall Tau | Median | Min | Max | Đánh Giá Mức Độ Thích Ứng |
|---|---|---|---|---|---|---|
| `book` | 266 | **0.9677** | 1.0000 | 0.0769 | 1.0 | **Xuất sắc** (Sách văn học & kỹ thuật) |
| `PPT2PDF` | 234 | **0.9664** | 1.0000 | -0.3333 | 1.0 | **Xuất sắc** (Slide khối độc lập) |
| `note` | 105 | **0.9585** | 1.0000 | 0.1667 | 1.0 | **Xuất sắc** (Ghi chú & biểu mẫu) |
| `academic_literature` | 215 | **0.9567** | 1.0000 | 0.3091 | 1.0 | **Xuất sắc** (Báo cáo khoa học đa cột) |
| `research_report` | 131 | **0.9428** | 1.0000 | -0.0714 | 1.0 | **Xuất sắc** (Báo cáo nghiên cứu thị trường) |
| `colorful_textbook` | 154 | **0.9326** | 1.0000 | 0.4032 | 1.0 | **Xuất sắc** (Sách giáo khoa màu sắc) |
| `exam_paper` | 192 | **0.9160** | 1.0000 | 0.3987 | 1.0 | **Xuất sắc** (Cấu trúc đề thi) |
| `newspaper` | 151 | **0.9130** | 0.9740 | 0.4220 | 1.0 | **Xuất sắc** (Báo chí cột zíc-zắc) |
| `magazine` | 149 | **0.9012** | 1.0000 | -0.2000 | 1.0 | **Xuất sắc** (Tạp chí thời trang/kinh tế) |
| `historical_document` | 5 | **-0.8012** | -0.7610 | -0.9556 | -0.7311 | **Dị thường** (Chữ cổ viết dọc từ phải qua trái) |

### 3.3. Tác Động Trên Các Dị Thường Phức Tạp (Special Issues)

| Hiện Tượng Đặc Biệt (Special Issue) | Số Lượng | Mean Kendall Tau | Mức Độ An Toàn |
|---|---|---|---|
| `table_span` (Bảng merged cells) | 107 | **0.9477** | Cực kỳ an toàn |
| `table_omission_line` (Bảng khuyết viền) | 43 | **0.9373** | Cực kỳ an toàn |
| `fuzzy_scan` (Bản quét mờ) | 29 | **0.9341** | Cực kỳ an toàn |
| `colorful_backgroud` (Nền màu trang trí) | 247 | **0.9275** | Cực kỳ an toàn |
| `table_horizontal` (Bảng nằm ngang) | 319 | **0.9206** | Cực kỳ an toàn |
| `table_wireless_line` (Bảng không viền) | 8 | **0.9141** | Cực kỳ an toàn (Tăng từ 0.3571) |
| `text_L-shaped_wrap` (Văn bản bọc chữ L) | 4 | **0.9116** | Cực kỳ an toàn (Tăng từ 0.3816) |
| `table_fewer_line` (Bảng ít đường kẻ) | 128 | **0.9033** | Cực kỳ an toàn |
| `table_veritical` (Bảng xoay dọc) | 7 | **0.7619** | An toàn |

---

## 4. Kết Quả Tầng 2: 20 Trang Khó Nhất Chạy End-to-End (Mean Tau Tăng Lên 0.7687)

Điểm tương quan Kendall's Tau trên tập 20 trang khó nhất tăng vọt từ `0.4798` lên **`0.7687`**:

| ID | Short Name | Nguồn Tài Liệu | Layout | Dị Thường (Issues) | Blocks | Diagrams | Kendall Tau Cũ | Kendall Tau Mới |
|---|---|---|---|---|---|---|---|---|
| 01 | `sample_01_PPT2PDF_1andmo` | `PPT2PDF` | `1andmore_column` | `colorful_backgroud` | 19 | 3 | 0.8190 | **0.9341** |
| 02 | `sample_02_PPT2PDF_other_` | `PPT2PDF` | `other_layout` | `colorful_backgroud` | 13 | 4 | 0.4722 | **0.8611** |
| 03 | `sample_03_academic_1andmo` | `academic_literature` | `1andmore_column` | `table_wireless_line` | 13 | 2 | 0.8182 | **0.9394** |
| 04 | `sample_04_academic_double` | `academic_literature` | `double_column` | `table_omission_line, table_veritical` | 11 | 3 | 1.0000 | **1.0000** |
| 05 | `sample_05_book_other_` | `book` | `other_layout` | `text_L-shaped_wrap` | 23 | 4 | 0.4978 | **0.9619** |
| 06 | `sample_06_book_single` | `book` | `single_column` | `table_span, table_veritical` | 4 | 1 | 1.0000 | **1.0000** |
| 07 | `sample_07_colorful_other_` | `colorful_textbook` | `other_layout` | `colorful_backgroud, table_span` | 22 | 3 | 0.3333 | **0.3333** |
| 08 | `sample_08_colorful_1andmo` | `colorful_textbook` | `1andmore_column` | `colorful_backgroud` | 33 | 2 | 1.0000 | **1.0000** |
| 09 | `sample_09_exam_pap_1andmo` | `exam_paper` | `1andmore_column` | `table_wireless_line` | 26 | 3 | 0.9905 | **0.9905** |
| 10 | `sample_10_exam_pap_other_` | `exam_paper` | `other_layout` | `colorful_backgroud` | 32 | 3 | 0.4725 | **0.4725** |
| 11 | `sample_11_historic_other_` | `historical_document` | `other_layout` | `geometric_deformation` | 58 | 4 | -0.2645 | **-0.2645** |
| 12 | `sample_12_historic_single` | `historical_document` | `single_column` | `None` | 5 | 1 | -0.6667 | **-0.6667** |
| 13 | `sample_13_magazine_three_` | `magazine` | `three_column` | `colorful_backgroud` | 41 | 9 | 0.5783 | **0.9938** |
| 14 | `sample_14_magazine_other_` | `magazine` | `other_layout` | `table_omission_line` | 10 | 2 | 0.2889 | **1.0000** |
| 15 | `sample_15_newspape_other_` | `newspaper` | `other_layout` | `table_wireless_line` | 41 | 6 | 0.4073 | **0.9825** |
| 16 | `sample_16_newspape_three_` | `newspaper` | `three_column` | `None` | 36 | 4 | 0.8413 | **1.0000** |
| 17 | `sample_17_note_other_` | `note` | `other_layout` | `None` | 18 | 1 | 0.4248 | **1.0000** |
| 18 | `sample_18_note_single` | `note` | `single_column` | `None` | 1 | 1 | 1.0000 | **1.0000** |
| 19 | `sample_19_research_1andmo` | `research_report` | `1andmore_column` | `table_wireless_line` | 33 | 4 | 0.5833 | **0.8718** |
| 20 | `sample_20_research_single` | `research_report` | `single_column` | `table_veritical, colorful_backgroud` | 3 | 0 | -1.0000 | **1.0000** |

*Điểm Kendall's Tau trung bình tập khó trước khi sửa thuật toán:* **`0.4798`** (sau sửa: **`0.7687`**).

> Cột `Diagrams` trong bảng trên là số nhóm do gom v1 tạo ra ở lần chạy đó. Với cấu hình mặc định hiện tại (không gom), mỗi `picture`/`figure` của Docling là một mục `diagrams[]`: 45 picture, 45 file crop trên 20 trang.

---

## 5. Phân Tích Chuyên Sâu 6 Vấn Đề Biên (Edge Cases) & Kế Hoạch Xử Lý

Hệ thống đã tự động kết xuất 6 ảnh đối chứng trực quan side-by-side (Ground Truth màu xanh vs DocuMind màu đỏ kèm chân trang mô tả) lưu tại `outputs/03_dataset_stress_test/`:

### 5.1. Vấn Đề 01: Vỡ Luồng Đọc Bố Cục 3 Cột (three_column Collapse)
- **File so sánh:** [`issue_01_3column_order_compare.png`](../outputs/03_dataset_stress_test/issue_01_3column_order_compare.png)
- **Triệu chứng:** Cột ở giữa bị cắt đôi hoặc bị gộp nhầm sang cột trái.
- **Nguyên nhân:** Thuật toán hiện tại chỉ có 1 ngưỡng phân tách tại `mid_x = W / 2`.
- **Kế hoạch Tuần 3:** Tự động phát hiện số cột $k$ bằng X-projection histogram, chia trang thành $k$ dải cột độc lập trước khi sắp xếp thứ tự đọc.
- **Trạng thái:** đã triển khai (`sort_elements_reading_order`, Recursive XY-Cut với $k$ cột động). Mô tả "Nguyên nhân" ở trên là của phiên bản cũ.

### 5.2. Vấn Đề 02: Dính Biên Tiêu Đề Trong Bố Cục 1+2 Cột (1andmore_column Bleed)
- **File so sánh:** [`issue_02_mixed_1and2col_compare.png`](../outputs/03_dataset_stress_test/issue_02_mixed_1and2col_compare.png)
- **Triệu chứng:** Khối Tiêu đề/Abstract (1 cột phía trên) bị hút nhầm vào thứ tự đọc của cột trái phía dưới.
- **Nguyên nhân:** Hai cột phía dưới bắt đầu ở độ cao không bằng nhau, khiến tiêu đề bị lọt vào vùng bounding box của cột trái.
- **Kế hoạch Tuần 3:** Áp dụng thuật toán **Recursive XY-Cut** để cắt ngang trang tại khoảng trắng phân cách trước khi phân cột.
- **Trạng thái:** đã triển khai cùng mục 5.1.

### 5.3. Vấn Đề 03: Bảng Không Viền Kẻ Bị Xé Thành Text Rời Rạc (Wireless Table Missed)
- **File so sánh:** [`issue_03_wireless_table_compare.png`](../outputs/03_dataset_stress_test/issue_03_wireless_table_compare.png)
- **Triệu chứng:** Docling bỏ sót cấu trúc bảng, xé các hàng/cột thành hàng chục đoạn văn bản rời rạc.
- **Nguyên nhân:** Mô hình TableFormer phụ thuộc nhiều vào các đường lưới viền (grid lines).
- **Kế hoạch Tuần 4:** Bổ sung bộ Heuristic Table Recovery dựa trên căn lề khoảng trắng (Whitespace Column Clustering) tại tầng hậu xử lý.
- **Trạng thái:** chưa làm.

### 5.4. Vấn Đề 04: Gom Cụm Sơ Đồ Quá Tay Trên Nền Màu (Diagram Over-merging)
- **File so sánh:** [`issue_04_colorful_overmerge_compare.png`](../outputs/03_dataset_stress_test/issue_04_colorful_overmerge_compare.png)
- **Triệu chứng:** Mảng màu nền trang trí lớn nuốt trọn toàn bộ chữ, công thức và bảng bên trong thành 1 Diagram khổng lồ.
- **Nguyên nhân:** Thuật toán OpenCV coi viền nền màu là contour ngoài cùng và gom toàn bộ nội dung bên trong.
- **Kế hoạch Tuần 3:** Bổ sung điều kiện chặn Semantic Guard: Bỏ qua các container chiếm diện tích $> 35\%$ hoặc chứa $\ge 2$ tiêu đề phân mục.
- **Trạng thái:** Semantic Guard đã được thêm, nhưng đo định lượng trên 613 trang cho thấy gom v1 vẫn làm hỏng 166/1.057 hình đang đúng. Vì vậy gom ảnh phẳng đã được **tắt mặc định** thay vì tinh chỉnh thêm (`docs/02` mục 5).

### 5.5. Vấn Đề 05: Bảng Xoay Dọc 90 Độ (Vertical Rotated Table Split)
- **File so sánh:** [`issue_05_vertical_table_compare.png`](../outputs/03_dataset_stress_test/issue_05_vertical_table_compare.png)
- **Triệu chứng:** Bảng xoay ngang trong trang dọc bị đọc theo chiều ngang của trang giấy, làm đảo lộn hoàn toàn thứ tự hàng và cột.
- **Nguyên nhân:** Không có metadata xoay trang (`/Rotate 90`).
- **Kế hoạch Tuần 4:** Tích hợp kiểm tra hướng văn bản (Text Orientation Detection) bằng OpenCV Moments / Tesseract OSD để tự động xoay 90 độ vùng ảnh trước khi đưa vào Table Engine.
- **Trạng thái:** chưa làm.

### 5.6. Vấn Đề 06: Văn Bản Bọc Chữ L Quanh Ảnh (L-shaped Newspaper Wrap)
- **File so sánh:** [`issue_06_l_shaped_wrap_compare.png`](../outputs/03_dataset_stress_test/issue_06_l_shaped_wrap_compare.png)
- **Triệu chứng:** Đoạn văn uốn lượn quanh ảnh quảng cáo bị cắt ngang giữa chừng, làm luồng đọc của 2 bài báo bị trộn lẫn.
- **Nguyên nhân:** Bounding Box hình chữ nhật không mô tả được đa giác lõm (concave polygon).
- **Kế hoạch Tuần 4:** Phân rã đa giác lõm thành 2 bounding box lồi con và liên kết thứ tự đọc tuần tự giữa chúng.
- **Trạng thái:** chưa làm. Hiện tượng tương tự (hình bị chữ bao quanh nên crop lẫn chữ) vẫn còn ở bước cắt ảnh.

---

## 6. Kết Luận & Định Hướng Phát Triển Tiếp Theo

Thực nghiệm trên `OmniDocBench` cho thấy:
1. **Độ ổn định:** pipeline chạy hết 1.602 trang hợp lệ của Tầng 1 trong 0,72 giây (đầu vào là box GT) và 20 trang Tầng 2 end-to-end không crash.
2. **Giá trị của thuật toán sắp xếp:** trên box GT thuật toán đạt trần 0,9369; sau Docling (300 trang) cải thiện so với thứ tự gốc là có thật nhưng nhỏ (Tau phạt 0,6441 so với 0,6047, Δ +0,0394, CI95 [+0,022, +0,059]). Hai con số này không được thay thế cho nhau.
3. **Gom sơ đồ:** không mang lại lợi ích trên ảnh phẳng (`docs/02` mục 5); mặc định dùng box `picture` thô.
4. **Mục tiêu tiếp theo:** mục tiêu Tuần 3 của Topic3 là Kendall $\ge 0{,}85$. Số liệu end-to-end hiện là 0,64-0,67 (Tau phạt) và 0,93-0,95 (Tau khớp), nên cần nói rõ độ đo nào khi so với mốc. Việc kế tiếp theo thứ tự ưu tiên của `docs/04` mục 6: kiểm chứng trên PDF số kiểu AUTOSAR (dữ liệu đích), sửa các lỗi hierarchy còn lại, rồi bộ phục hồi bảng không viền.
