# BÁO CÁO BÀN GIAO SESSION & PHƯƠNG HƯỚNG TIẾP THEO

> Ngày: 2026-11-11 → 2026-11-12 · Nhánh: `feat/structure-and-diagram-recovery` (nội dung từ mục 1 đến 9 viết tại commit `37d0ab2`; **các thay đổi sau đó nằm ở mục 10 và 11**) · Người đọc: session mới tiếp quản dự án.
> File này tóm tắt **những gì đã làm/đo được, trạng thái repo và môi trường, các lỗi/rủi ro đã biết, và việc cần làm tiếp**. Mọi con số lấy từ các lần chạy thật trong session; chỗ nào là suy luận chưa kiểm chứng đều được đánh dấu **[CHƯA KIỂM CHỨNG]**.
> **Kết luận quan trọng nhất (đọc mục 4.4 và 6):** phép đo "gom sơ đồ" trên OmniDocBench **không thể** kiểm chứng mục tiêu "chụp tổng thể cả cụm/diagram", vì nhãn `figure` của dataset không nhất quán về độ chi tiết. Đã đề xuất **dừng tinh chỉnh nhánh ảnh phẳng** và chuyển sang kiểm chứng trên PDF số kiểu AUTOSAR. Mục 10 ghi lại việc đã làm tiếp theo kết luận này (tắt gom mặc định, thử thêm hai veto, sửa lỗi IR, cập nhật docs). **Mục 11 ghi lại bước chuyển sang PDF AUTOSAR** (tải 4 file, pipeline PDF → crop hình, đo trên một file có ground truth gán tay, luật neo cho v1).

---

## 1. Mục tiêu hiện tại của người dùng

- Hướng chung: DocuMind (Topic3, 8 tuần) — khôi phục thứ tự đọc, hierarchy, bảng, và **sơ đồ khối**.
- Mục tiêu **hiện tại, đã được người dùng làm rõ**: chỉ cần *trích xuất ảnh sao cho đúng*; nếu là **một cụm ảnh liên quan nhau hoặc một diagram** thì thuật toán phải **chụp tổng thể (một box bao trọn cả cụm)**, không chụp từng ảnh/mảnh riêng. **Việc xem trong ảnh có gì (OCR, nhãn, node–edge) tính sau.** Không cần OCR cho mục tiêu này.
- Thiết kế người dùng chốt: **chữ → parse bằng code thuần** (text layer của file số); **ảnh → chỉ cắt (extract), chưa cần biết nội dung**.
- Dữ liệu đích thật: tài liệu **kiểu AUTOSAR** (PDF số có text layer, nhiều diagram vector/block diagram). **Hiện tại cứ dùng OmniDocBench để test trước** (người dùng nói vậy). **[CẬP NHẬT: sau đó người dùng chuyển sang test trên AUTOSAR, xem mục 11.]** OmniDocBench là ảnh phẳng, nên các kết luận ở đây chỉ chuyển sang AUTOSAR một phần.
- Quyết định về header/footer: người dùng muốn **đưa header (lớp FURNITURE của Docling) vào** khi đánh giá thứ tự đọc. Đã thực hiện bằng `--include-furniture` (mặc định của script vẫn tắt). **Chưa quyết** có áp dụng cho pipeline sản xuất (sinh IR/Markdown) hay không (xem mục 9).
- Phong cách làm việc của người dùng trong session này (session mới nên hỏi lại): nói tiếng Việt; yêu cầu **giao toàn bộ tác vụ code cho một subagent** (model Haiku, effort xhigh), agent chính điều phối, kiểm tra và báo cáo. ID subagent chỉ có hiệu lực trong session cũ.

---

## 2. Trạng thái repo (đã chia commit nguyên tử và push; xem `git log`)

Các thay đổi dưới đây đã được commit theo thứ tự phụ thuộc và push lên `origin/feat/structure-and-diagram-recovery` theo yêu cầu của người dùng. Bảng giữ nguyên để biết từng file làm gì.

| Trạng thái | File | Ghi chú |
|---|---|---|
| Sửa | `scripts/run_dataset_stress_test.py` | Import `compute_kendall_tau` từ `eval_utils`; xử lý `None`; báo số trang bị loại |
| **Sửa (file đã track, có ảnh hưởng thuật toán gom)** | `scripts/test_diagram_grouping.py` | Thêm `read_image_bgr` (np.fromfile + imdecode, **an toàn đường dẫn Unicode**, ném `ValueError` nếu ảnh hỏng/rỗng); `detect_visual_containers_opencv` dùng hàm này thay `cv2.imread`. **Logic gom không đổi.** Xem lỗi ở 4.5 |
| Mới | `scripts/eval_utils.py` | `compute_kendall_tau` (trả `None` khi n<2), lọc GT, đổi toạ độ Docling→pixel, matching Hungarian, `detection_prf`, containment, over-merge, split, phân loại vỡ/nguyên vẹn/bỏ sót… |
| Mới | `scripts/test_eval_utils.py` | **35/35 pass** (chạy bằng `python`, pytest chưa cài trong venv) |
| Mới | `scripts/run_e2e_reading_order_eval.py` | Đánh giá thứ tự đọc end-to-end 4 phương pháp A/B/C/D; `--ocr-backend {auto,onnxruntime,torch,off}`, `--device`, `--pages-file`, **`--include-furniture`** (mặc định tắt); cache theo cấu hình; mô tả Docling trong báo cáo lấy từ config thật |
| Mới | `scripts/check_furniture_layer.py` | Kiểm tra lớp FURNITURE của Docling trên một số trang |
| Mới | `scripts/run_diagram_crop_eval.py`, `scripts/run_diagram_crop_eval_v2.py` | Đánh giá crop hình v1 (100 trang), v2 (613 trang, whole/split/merge, phân loại GT) |
| Mới | `scripts/diagram_grouping_v2.py` | `group_whole_diagrams_image_v2` (gom thận trọng, ngưỡng tương đối) — **không đạt mục tiêu, xem 4.4**; hàm gốc không bị sửa |
| Mới | `scripts/run_diagram_crop_eval_v3.py` | Harness leave-source-out 5 fold, lưới 384 cấu hình, OOF |
| Mới | `scripts/analyze_pair_features.py` | Phân tích đặc trưng cặp mảnh (AUC, đường đánh đổi) |
| Mới | `docs/04_session_handoff_and_next_steps.md` | File này |

- Commit chỉ được tạo khi người dùng yêu cầu (đã yêu cầu cuối session). `master` không bị đụng tới.
- Các thư mục kết quả nằm trong `outputs/` (bị `.gitignore`): `04_e2e_eval/`, `04_e2e_eval_gpu_test/`, `05_diagram_crop_eval/`.
- Tại thời điểm viết mục 1-9: không sửa logic `sort_elements_reading_order`, `postprocess_ir_structure` và `group_whole_diagrams_*`; chỉ có thay đổi đọc ảnh Unicode-safe ở `test_diagram_grouping.py`. **Điều này không còn đúng** sau các commit ở mục 10 (`postprocess_ir_structure` đã sửa; `group_whole_diagrams_image` vẫn nguyên thân hàm).

---

## 3. Môi trường (đã thay đổi trong session)

- `.venv` Python 3.12. **Đã cài torch `2.14.1+cu126` và torchvision `0.29.1+cu126`** (trước đó là `2.15.0` bản CPU và `0.29.2+cpu`). Kênh CUDA chưa có torch 2.15 cho Python 3.12 nên phải hạ xuống 2.14.1.
- Lệnh cài đã dùng: `uv pip install --python .venv/Scripts/python.exe "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126` và tương tự `torchvision==0.29.1+cu126`. `pip check` báo tương thích.
- Bản `freeze` trước khi đổi: `C:\Users\ADMIN\AppData\Local\Temp\claude\bee8685ac587660f\753cfb07-d7bf-460d-bbd2-24e59fc9cc1d\scratchpad\venv_freeze_before.txt` (file tạm, có thể đã mất). README hiện chỉ ghi `uv pip install docling ...` nên **không tái tạo đúng môi trường này**.
- GPU: NVIDIA GTX 1650, **4 GB VRAM**, driver 621.01. Docling 2.150.0, rapidocr 3.10.0, onnxruntime 1.31.1 (**chỉ CPUExecutionProvider**, không có `onnxruntime-gpu`).
- Docling mặc định dùng OCR "auto" → RapidOCR backend onnxruntime **trên CPU**. Backend `torch` của RapidOCR chạy được trên GPU (model `.pth` PP-OCRv6 có sẵn trong `rapidocr/models`; lần gọi đầu mất ~115 s do tải thêm model từ ModelScope).
- Tốc độ (8 trang, trung vị bỏ trang đầu): onnxruntime CPU 18,05 s/trang; layout GPU + OCR CPU 13,88 s; **torch GPU 4,68 s** (≈3–4 lần nhanh hơn, **kết quả giống hệt**). VRAM đỉnh thường ~1,1 GB, nhưng đã thấy ~3,1 GB khi trang báo có 70+ box.
- Docling **tắt OCR** (`do_ocr=False`): ~0,22 s/trang (layout 0,14 s).
- **Windows: `cv2.imread` không đọc được đường dẫn có ký tự không-ASCII** (đã xác nhận; xem 4.5).

---

## 4. Kết quả đo được

### 4.1. Con số 0,9369 (Tầng 1 cũ) KHÔNG phải điểm end-to-end

- Tầng 1 (`run_dataset_stress_test.py`) đưa **bbox ground truth** vào `sort_elements_reading_order`, không chạy Docling. Phương pháp D (`oracle_xycut_gt`) trên toàn bộ dataset tái tạo đúng **0,9369 trên 1.602 trang** (49 trang bị loại vì <2 phần tử GT; README ghi "1.651 trang" là không chính xác).
- Hàm Tau cũ trả 1,0 khi n<2 (thổi phồng các trang ít block); đã sửa thành `None`.

### 4.2. Thứ tự đọc end-to-end (300 trang phân tầng layout × data_source, seed 42; Docling torch + CUDA; IoU khớp 0,3)

Cùng 300 trang, hai cách thu box:

| | Tau khớp / Tau phạt / Coverage — **chỉ lớp BODY (mặc định)** | Tau khớp / Tau phạt / Coverage — **gồm FURNITURE** |
|---|---|---|
| A `docling_native` | 0,8769 / 0,6047 / 0,7805 | 0,8755 / 0,6327 / 0,8133 |
| **B `documind_xycut`** | 0,9471 / 0,6441 / 0,7805 | 0,9349 / **0,6709** / 0,8133 |
| C `naive_yx` | 0,7915 / 0,5190 / 0,7805 | 0,7854 / 0,5480 / 0,8133 |
| D `oracle_xycut_gt` (mốc, box GT) | 0,9250 / 0,9250 / 1,0000 | 0,9250 / 0,9250 / 1,0000 |

- **B so với A không đổi nhiều** khi thêm FURNITURE: Δ Tau phạt **+0,0382** (CI95 bootstrap [+0,0209, +0,0582]; thắng/hoà/thua 89/176/24 trên 289 trang); Δ Tau khớp +0,0594 ([+0,0346, +0,0834]; 274 trang). (BODY-only: +0,0394 và +0,0702.) Cải thiện **có thật nhưng nhỏ**.
- BODY-only, theo layout (Δ B−A, Tau phạt): `other_layout` +0,084, `single_column` +0,038, `1andmore_column` +0,023, **`double_column` −0,016, `three_column` −0,001**; theo nguồn: `note` +0,174, `newspaper` +0,111.
- Không nên so Tau-khớp của B với D: khác tập trang và tập phần tử.
- **Giới hạn:** không có tập giữ riêng (thuật toán đã chỉnh trên chính dataset này → tune-on-test); GT không khớp bị phạt theo *quy ước* (xếp cuối) nên điểm phạt phụ thuộc nặng vào coverage.

**FURNITURE (đã kiểm chứng):**
- Trong luồng ảnh/PDF của Docling chỉ có hai nhãn được xếp vào lớp FURNITURE: **`page_header` và `page_footer`** (`docling/models/stages/reading_order/readingorder_model.py` ~dòng 138 và ~1040). Số trang thường được gán `page_footer`. Mọi nhãn khác là BODY. Enum còn có `BACKGROUND`, `INVISIBLE`, `NOTES`; mặc định `iterate_items()` chỉ lấy `BODY` (`DEFAULT_CONTENT_LAYERS`), nên các lớp kia cũng bị loại. Các backend DOCX/HTML/PPTX có quy tắc riêng.
- Trên 10 trang có header/page_number/footer: 19 item FURNITURE (13 `page_header`, 6 `page_footer`); duyệt mặc định lấy **0/19**, duyệt đủ lớp 19/19. Coverage trên 300 trang: `header` **25,8% → 76,3%**, `page_number` **1,8% → 80,7%**, tổng 0,7805 → 0,8133.
- **`equation_caption` 0% KHÔNG do FURNITURE:** 19 GT loại này nằm gần hết (containment 0,90–1,00) trong một box `formula` ở lớp BODY, IoU chỉ 0,006–0,039 → Docling gộp số công thức vào box công thức.
- **Chưa lý giải:** trong nhóm `equation_caption` có 1 item FURNITURE không xuất hiện ngay cả khi duyệt đủ lớp. Cần điều tra trước khi coi "duyệt đủ lớp" là chuẩn.
- Ghi chú sai "trên CPU, OCR bật" trong `report.md` đã được sửa ở script sinh báo cáo (báo cáo cũ dựng lại từ `results.json`, số liệu không đổi).
- Quy trình: lần chạy `--include-furniture` từng dừng ở 94/300 rồi được tiếp tục từ cache; kết quả cuối: 300 trang, 0 lỗi, `results_furniture.json`, `report_furniture.md`.

**Thời gian xử lý (lần chạy 300 trang gồm FURNITURE):** trung vị 3,09 s/trang, trung bình 4,93 s, tối đa 48,8 s; 34/300 trang (11%) > 10 s (ngưỡng IQR 10,00 s và MAD 10,16 s trùng nhau). Phân loại:
- 27/34 do **OCR** (trang báo dày chữ, 60–207 box, OCR 15–28 s; tương quan thời gian–OCR +0,83); tái lập trong cả hai lần chạy → bình thường.
- Trang WSJ 48,8 s: `table_structure` 34,0 s dù **0 item bảng** được giữ; log có "Discarding the … structure … hit the 1024-step decode limit" (**[CHƯA KIỂM CHỨNG]** là cùng trang).
- Trang 95: 34,4 s so với 0,8 s ở lần chạy khác; 31,7 s không thuộc giai đoạn nào → **khởi động nguội** (trang đầu sau khi tiến trình khởi động lại).
- Không thấy suy giảm theo thời gian. Có/không FURNITURE trung bình 4,93 s vs 4,50 s: Docling tính toán y hệt, chênh lệch là dao động giữa các lần chạy.
- Script phân tích: `C:\Users\ADMIN\AppData\Local\Temp\claude\bee8685ac587660f\753cfb07-d7bf-460d-bbd2-24e59fc9cc1d\scratchpad\timing_spikes.py` (file tạm).

### 4.3. Tắt OCR có an toàn để cắt ảnh không?

- Với box `picture`: **có**. Trên 100/100 trang, box thô khi tắt OCR **giống hệt** khi bật OCR; pipeline nhanh hơn ~17 lần (3,70 → 0,22 s/trang).
- Tác dụng phụ: **mất toàn bộ box loại chữ** (`text` 1511→0, `section_header` 412→0, `list_item` 392→0, `caption` 81→0, `footnote` 8→0 trên 100 trang); `picture` 284→284, `table` 17→17, `formula` 53→69.
- **Cơ chế (đã đọc source):** `docling/utils/layout_postprocessor.py` ~dòng 238–245 loại mọi cluster không có ô chữ (`keep_empty_clusters=False` mặc định), trừ `FORMULA`; `picture`/`table` thuộc nhóm xử lý riêng nên không bị loại. Tức Docling bỏ box chữ khi không có chữ, không phải model không phát hiện. **[CHƯA THỬ]** `keep_empty_clusters=True` + OCR tắt: có thể giữ lại box chữ (vị trí, nhãn, không có nội dung) → làm manh mối gom/tách hình (vd box `caption`).
- Hệ quả: `group_whole_diagrams_image` dùng nhãn chữ làm hàng rào (chặn gộp qua `section_header`/`list_item`, bỏ container chứa ≥2 `section_header`/`title`) nên các hàng rào đó **không còn tác dụng** khi tắt OCR; các demo gom "thẻ PPT"/infographic từng đẹp thực chất dựa vào box chữ do OCR sinh ra (trang `01_ppt_cards` với OCR tắt: Docling không tạo box `picture` nào).
- Thứ tự đọc chỉ dùng box + nhãn, **không dùng nội dung chữ**, nhưng **cần các box chữ**, nên đo thứ tự đọc **phải bật OCR**.
- Với **PDF số có text layer**, Docling lấy chữ bằng code (`docling-parse`), không cần OCR, và box chữ vẫn có (theo thiết kế của Docling; **[CHƯA KIỂM CHỨNG]** trong repo vì dữ liệu đo toàn ảnh phẳng).

### 4.4. Gom sơ đồ trên ảnh phẳng — kết quả và **kết luận: dừng nhánh này**

Đánh giá trên 613 trang OmniDocBench có GT `figure` (1.385 hình), Docling **OCR tắt**. R1 = box `picture` thô, R3 = `group_whole_diagrams_image` cũ, R5 = `group_whole_diagrams_image_v2` (đánh giá ngoài mẫu theo `data_source`, 5 fold).

Phân loại GT theo box thô: **nguyên vẹn** 1.057 (76,3%) · **bị vỡ** (≥2 mảnh chứa ≥0,7 trong hình, không box đơn nào IoU≥0,5) **38 (2,7%)** · **bị bỏ sót** (không box nào bao ≥0,7) **275 (19,9%)** · khác 15 (1,1%).

| Tập | n | whole@0,5 R1 / R3 / **R5 (OOF)** | Hình đúng bị hỏng (so R1) |
|---|---|---|---|
| bị vỡ | 38 | 0,0% / 28,9% (sửa 11) / **0,0% (sửa 0)** | — |
| nguyên vẹn | 1.057 | 100% / 84,4% / 99,2% | R3: **166** (15,7%); R5: **8** (0,76%) |
| tất cả | 1.385 | 76,3% / 65,3% / 75,7% | |

- **R3 (bản cũ) làm hại** nhiều hơn là giúp (sửa 11/38, hỏng 166/1.057); hình riêng gần nhau bị gộp nhầm 27,2% (R3) so với 10,7% (R1). Nguyên nhân hỏng: gộp theo khoảng cách (pad 45 px, gap ≤160 px, **ngưỡng pixel tuyệt đối**) ~107, `container_snap` ~37, hút `formula`/`table` ~20.
- **R5 (v2) "không hại" chỉ vì hầu như không gom gì:** cấu hình được chọn dùng bằng chứng container, sửa **0/38**; hại 8/1.057 (đạt ≤1%) nhưng nuốt nhầm hình riêng **+1,2 điểm %** so với R1 (11,9% vs 10,7%; ràng buộc +1,0 → **không đạt**). Trong 384 cấu hình **không cấu hình nào** thoả cả hai ràng buộc. Cấu hình "tight" tốt nhất sửa 18/38 nhưng hỏng ~320 hình đúng.
- **Phân tích đặc trưng cặp mảnh** (919 cặp ứng viên: 140 dương, 779 âm; dương = hai mảnh cùng thuộc một GT bị vỡ): AUC đơn biến cao nhất ~0,80 (`max_area_rel`), 0,79 (`union_area_rel`), 0,73 (`n_pieces_on_page`), `gap_rel` 0,56; cây độ sâu 2 OOF AUC 0,53, logistic 0,65. Trần oracle (gộp đúng mọi cặp dương) sửa 31/38. **Điểm vận hành tốt nhất với hại ≤1% chỉ sửa 3/38; không điểm nào sửa ≥10/38.** Tức đặc trưng hình học/khoảng cách **không phân biệt được** "các mảnh của một hình" với "hai hình riêng nằm gần nhau". 138/278 cặp dương có khoảng hở > 1× đường chéo mảnh nhỏ → chúng không nằm trong tập ứng viên.
- **Đã xem trực tiếp ảnh minh hoạ** (agent điều phối xem `viz_v3/broken_still_wrong_R5_01..04` và `intact_harmed_by_R5_03`, `_06`; **các ảnh còn lại chưa ai xem**). Điều thấy được:
  - Các ca "bị vỡ" phần lớn là **nhóm panel anh em do người chú thích gộp thành một `figure`**: 4 cặp biểu đồ vòng (61%/67%…) mỗi cặp một GT; hàng 3 hình clip-art (mì/bánh sandwich/hamburger); hai sơ đồ hình học đặt cạnh nhau; dải sản phẩm (máy giặt + bảng giá + tủ lạnh). Chúng không phải "một sơ đồ bị Docling xé vụn".
  - **Nhãn GT không nhất quán:** trang con dấu có lưới 2×2 bốn hình nhỏ nhưng GT tách thành **4 figure riêng** (R5 gộp thành 1); trong khi các hàng clip-art tương tự lại được gộp thành 1 GT.
  - Ca R5 hại thật: gộp **hai biểu đồ riêng** (图14 + 图15) thành một box, và nở theo khung container bao cả tiêu đề, chú thích nguồn (cho mục tiêu "chụp tổng thể" việc kèm tiêu đề có thể chấp nhận được, nhưng GT không tính vậy). Cũng có một box dương giả ở logo/hình trang trí đầu trang.
- **Kết luận:** nhóm 38 ca "vỡ" phần lớn đo **quy ước gán nhãn**, không đo mục tiêu "chụp cả diagram". OmniDocBench **không phải bài kiểm thử phù hợp** cho mục tiêu này. Lỗi lớn nhất của nhánh ảnh phẳng là **bỏ sót hoàn toàn** (19,9%, gấp ~7 lần số vỡ): 43,3% trong số đó có diện tích <1% trang, 73,1% (201/275) nằm trong một container OpenCV nhưng chỉ 26,5% (73/275) có container khớp hình (IoU ≥0,5); 21 hình (7,6%) bị một box Docling khác phủ ≥0,5 (số liệu từ `results_v3.json` sau khi sửa lỗi Unicode; bản trước đó ghi 68,4% và 25,1% là số **trước** khi sửa) → phát hiện từ pixel có triển vọng một phần.
- **Với ảnh phẳng, OCR tắt: mặc định nên dùng box `picture` thô của Docling** (F1@0,5 = 0,766; recall 0,825; IoU khớp TB 0,929).
- Giới hạn: định nghĩa "bị vỡ" dựa vào box Docling; chỉ 38 ca (CI rộng: 0/38 = [0; 9,2%], 3/38 = [2,7; 20,8%], 31/38 = [66,6; 90,8%]); ngưỡng cạnh 80 px của định nghĩa "hình gần nhau" là định nghĩa đánh giá cũ; chưa đo trên PDF vector.
- Báo cáo đầy đủ: `outputs/05_diagram_crop_eval/{report_v2.md,report_v3.md,results_v3.json,pair_features.json}`; bản trước khi sửa lỗi Unicode: `*_before_unicode_fix.*`.

### 4.5. Lỗi đọc ảnh Unicode (đã sửa)

- `cv2.imread` trả `None` với đường dẫn không-ASCII trên Windows (vd `book_en_搬书匠-...png`); `cv2.imdecode(np.fromfile(...))` đọc được. `detect_visual_containers_opencv` trước đây thất bại **im lặng** (không có container), chỉ in WARN. Ảnh hưởng 35/613 trang (62 GT `colorful_textbook`, 7 GT `book`; 8/38 ca vỡ).
- Đã sửa bằng `read_image_bgr`; sau sửa 35/35 trang có container (tổng 318); trang ASCII cho mảng ảnh giống hệt `cv2.imread`. Chạy lại v2/v3: **kết luận không đổi** (R3 sửa 11/38 như cũ, nguyên vẹn bị hỏng 165→166; R5 OOF vẫn sửa 0/38). Giả thuyết "lỗi container là nguyên nhân các ca vỡ không-ASCII thất bại" đã bị bác (6/8 ca là `no_covering_box`).
- Cũng ảnh hưởng pipeline sản xuất nếu đường dẫn file có ký tự không-ASCII (vd tên tiếng Việt có dấu).

---

## 5. Vấn đề kiến trúc/code đã biết và CHƯA sửa

Từ lần review kiến trúc đầu session (đã xác minh bằng đọc code `ir_postprocessor.py` và `run_dataset_stress_test.py`):

1. **`ir_postprocessor.py:372`** — `header`/`page_header` bị xử lý như `section_header` trong hierarchy: running header đầu trang trở thành `current_section_id`, toàn bộ nội dung sau thành con của nó. **Lưu ý:** nếu đưa FURNITURE vào pipeline sản xuất thì lỗi này sẽ bộc lộ rõ.
2. **`ir_postprocessor.py:321-323`** — khi IoU liên kết bảng = 0, fallback gán `table_elements[0]` → liên kết sai im lặng, không có confidence/cảnh báo. **[ĐÃ SỬA, mục 10]**: không liên kết, ghi `metadata.link_warning`.
3. Hierarchy phẳng: `level = 1 if no title else 2`; docs nói nhận diện "1.1", "A." nhưng code **không** làm.
4. Ngưỡng **pixel tuyệt đối** (8, 10, 12, 80, 300 px, hệ số 0,55…) trên ảnh có độ phân giải khác nhau; cần chuyển sang đơn vị tương đối.
5. Chữ dọc / RTL: `historical_document` cho Tau âm (trên 300 trang chỉ 2 trang; toàn dataset 5 trang −0,80). Chưa có phát hiện hướng viết.
6. Chỉ xử lý **một trang đơn lẻ**; chưa có schema đa trang, bảng ngắt trang, hierarchy xuyên trang.
7. Topic3 cần DOCX/XLSX/PPTX, đối chiếu mâu thuẫn liên tài liệu có `source_ref` + confidence, và các độ đo TEDS / Graph F1 — **chưa có**.
8. **[ĐÃ XÓA, mục 10]** Chỉ số "0 over-merging" của Tầng 1 cũ gần như tautology (guard chặn container >35%, bộ phát hiện chỉ báo khi >45%; khối bọc `except Exception: pass`) → **không có giá trị**; đo v2/v3 cho thấy over-merge thật.
9. Cấu trúc repo: không có package/`pyproject`, không có CI; logic lõi nằm trong `scripts/`; `compute_iou` lặp 3 file; các file `test_*.py` thực chất là script chạy pipeline (ngoại trừ `test_eval_utils.py`).
10. **[ĐÃ CẬP NHẬT, mục 10]** **Docs lệch code/số liệu:** `docs/03_benchmark_and_stress_test.md` (mục 3–6 còn số cũ, mục 5–6 mô tả vấn đề đã sửa như chưa sửa, "Kendall Tau tập khó: 0.4798" cũ); `docs/01` schema khác code (`table_id`/`caption_id` ở top-level vs nằm trong `metadata`; khoá `id`); docs ghi PyMuPDF 1.25 nhưng `.venv` có 2.1; README nói "1.651 trang" cho Tầng 1; docs ca ngợi gom sơ đồ nhưng đo cho thấy làm hại.
11. **[ĐÃ ĐO, mục 11]** Nhánh vector `group_whole_diagrams_vector` (đo trên PDF AUTOSAR: F1 chặt 0,416, hút chữ và khoanh bảng); dilation cố định 12–14 px, Union-Find O(n²), nguy cơ gộp cả khung bảng/đường kẻ trang.

---

## 6. Phương hướng tiếp theo (ưu tiên đề xuất)

### P1 — Dừng tinh chỉnh gom ảnh phẳng trên OmniDocBench

- Kết luận ở 4.4: phép đo không kiểm chứng được mục tiêu, và đặc trưng khoảng cách không phân biệt được. Không giao thêm việc tinh chỉnh `group_whole_diagrams_image` cho tới khi có dữ liệu phù hợp.
- Mặc định cho ảnh phẳng (OCR tắt): **box `picture` thô của Docling**.
- **Đã thử thêm ở mục 10** (veto caption oracle và veto khoảng trắng trên nền v2): không đạt tiêu chí. Ý tưởng raster dưới đây vẫn **chưa thử** ở dạng Docling thật, chỉ làm nếu người dùng muốn tiếp tục nhánh ảnh: OCR tắt + `keep_empty_clusters=True` để giữ box `caption`/chữ làm tín hiệu "hai hình riêng" (caption riêng là manh mối mạnh). Cũng có thể xử lý phần "bỏ sót" 19,9% bằng phát hiện từ pixel (OpenCV container), nhưng đó là bài toán khác.

### P2 — Chuyển sang PDF số kiểu AUTOSAR (dữ liệu đích thật) **[ĐÃ BẮT ĐẦU, mục 11]**

- Cần ~20–50 trang AUTOSAR-like có diagram, **tự gán nhãn box "cả diagram"** (hoặc người dùng cung cấp file). **Hỏi người dùng** gửi file, hoặc **xin phép** trước khi tải các specification AUTOSAR công khai. **Không tự tải khi chưa được cho phép.** **[Đã giải quyết: người dùng yêu cầu tải; 4 PDF ở `datasets/AUTOSAR/`, mục 11.]**
- Chia nguồn thông tin: chữ + toạ độ bằng code (PyMuPDF `get_text("words"/"blocks")` hoặc `docling-parse`), hình khối/mũi tên bằng `page.get_drawings()`, ảnh nhúng bằng toạ độ từ PDF rồi cắt. Nhãn chữ trong sơ đồ có sẵn miễn phí → đây là lợi thế lớn so với nhánh ảnh.
- Đo `group_whole_diagrams_vector` (và biến thể cải tiến): khung bao cả diagram, không gộp nhầm khung bảng/đường kẻ trang, dilation tương đối thay vì 12–14 px cố định.
- Có thể bổ sung dữ liệu có ground truth: **nhóm shape trong PPTX** xuất sang PDF/ảnh, hoặc dữ liệu tổng hợp có kiểm soát (`generate_synthetic_flowchart_pdf`).

### P3 — FURNITURE: hoàn tất phần còn lại

- Điều tra item FURNITURE không xuất hiện khi duyệt đủ lớp (nhóm `equation_caption`).
- Quyết định có đưa header/footer vào pipeline sản xuất không (xem mục 9); nếu có, sửa lỗi hierarchy ở mục 5.1 trước.
- Cân nhắc đánh giá toàn bộ 1.651 trang (ước tính ~2,2 giờ với torch GPU) và **thiết lập tập giữ riêng** (chia theo `data_source`) khi so sánh XY-Cut để xử lý tune-on-test.

### P4 — Cập nhật docs/README theo số liệu thật **[ĐÃ LÀM ở mục 10; xem các việc còn lại ở đó]**

Cần sửa: "Kendall's Tau 0,9369 là thành tích pipeline" (thực chất oracle trên box GT, 1.602 trang; end-to-end ~0,64–0,67); "đột phá" 2/3 cột (end-to-end không cải thiện so với Docling); "0 over-merging"; gom sơ đồ "hoàn hảo" (đo cho thấy làm hại trên ảnh phẳng); mục 5–6 `docs/03`; schema `docs/01` vs code; phiên bản thư viện; hướng dẫn cài đặt (torch CUDA, `uv` index); lưu ý Unicode path trên Windows.

### P5 — Sửa các lỗi code ở mục 5 (ưu tiên 1, 2, 4), rồi hierarchy nhiều cấp. **[Một phần: mục 5.2 và 5.8 đã xử lý ở mục 10; 5.1 (header) và 5.4 (ngưỡng pixel) còn lại]**

### P6 — Đóng gói và kiểm thử

- Chuyển lõi sang package có test (pytest, `pyproject`), loại `compute_iou` trùng lặp, thêm CI, ghim phiên bản; `onnxruntime-gpu` **không cần** (đã có backend torch).

### Thứ tự khuyến nghị

P1 (dừng) → P2 (đã bắt đầu, mục 11) → P4 → P5 → P3 → P6. Sau đó mới quay lại phạm vi Topic3 (đa định dạng, đối chiếu liên tài liệu).

---

## 7. Cách chạy lại (từ thư mục gốc repo, Windows)

```powershell
# Kiểm thử các hàm đánh giá (35 test)
.venv\Scripts\python scripts\test_eval_utils.py

# Thứ tự đọc end-to-end, 300 trang phân tầng, torch trên GPU (~20-23 phút, có resume theo cache)
.venv\Scripts\python scripts\run_e2e_reading_order_eval.py --stratified 300 --seed 42 --ocr-backend torch --device cuda
#   thêm --include-furniture để lấy cả header/footer (cache riêng *_furniture; ghi results_furniture.json / report_furniture.md)
#   --ocr-backend off : Docling OCR tắt (nhanh); mặc định cũ: --ocr-backend auto --device cpu (khớp cache cũ)

# Đánh giá crop hình ảnh phẳng (613 trang, OCR tắt, dùng cache)
.venv\Scripts\python scripts\run_diagram_crop_eval_v2.py
.venv\Scripts\python scripts\run_diagram_crop_eval_v3.py        # leave-source-out, ~4-5 phút
.venv\Scripts\python scripts\analyze_pair_features.py           # đặc trưng cặp mảnh

# Tier 1/2 (mặc định: box picture thô; thêm --group-diagrams để chạy gom v1). GHI ĐÈ outputs/03_dataset_stress_test/
.venv\Scripts\python scripts\run_dataset_stress_test.py

# Veto over-merge (mục 10.2), ~3-4 phút
.venv\Scripts\python scripts\run_diagram_crop_eval_v4.py

# Test hậu xử lý IR (6 test)
.venv\Scripts\python scripts\test_ir_postprocessor.py

# PDF AUTOSAR -> crop hình (mục 11), ~0,3 s/trang với Docling OCR tắt trên GPU; cache Docling theo trang
.venv\Scripts\python scripts\run_pdf_figure_crops.py datasets\AUTOSAR\AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf --add-vector-missing
.venv\Scripts\python scripts\eval_pdf_figure_variants.py datasets\AUTOSAR\AUTOSAR_CP_EXP_LayeredSoftwareArchitecture.pdf   # chấm với 30 box gán tay
.venv\Scripts\python scripts\run_figure_variants_all.py datasets\AUTOSAR\*.pdf   # 9 biến thể x 4 PDF, ảnh ghép cho trang bất đồng
```

- Đầu ra: `outputs/04_e2e_eval/{report.md,results.json,report_furniture.md,results_furniture.json,run*.log,cache_ocr-*}`, `outputs/05_diagram_crop_eval/{report_v2.md,report_v3.md,results_v*.json,pair_features.json,viz*/}`.
- Cache Docling theo cấu hình: `cache/` (CPU, 59 trang cũ), `cache_ocr-torch_dev-cuda/` (300 trang, BODY), `cache_ocr-torch_dev-cuda_furniture/` (300 trang, gồm FURNITURE; mỗi item có `content_layer`), `cache_ocr-off_dev-cuda/` (613 trang, OCR tắt). Cache **không** trộn giữa các cấu hình.

---

## 8. Cạm bẫy và lưu ý vận hành

- **GPU 4 GB**: không chạy hai tiến trình Docling GPU song song; đã thấy VRAM ~3,1 GB trên trang báo nhiều box.
- **Tiến trình nền:** chạy bằng tác vụ nền của harness (`run_in_background`) để nhận thông báo khi xong; tiến trình tự `nohup` thì không có thông báo. Trên Windows, python của venv sinh tiến trình con nên mỗi lần chạy thấy 2 PID — bình thường. `sleep` đứng trước lệnh khác bị chặn bởi harness.
- **Windows + OpenCV:** luôn đọc ảnh qua `read_image_bgr` (Unicode-safe); `cv2.imread` thất bại im lặng với tên file không-ASCII.
- Subagent trong session này **nhiều lần kết thúc lượt mà không gửi báo cáo cuối** (dù được dặn bắt buộc) và gửi lại bản rút gọn; luôn **tự kiểm tra file/log trên đĩa** thay vì chỉ tin báo cáo. Agent điều phối đã đối chiếu các bảng chính với `report*.md`/`results*.json` và xem lại 6 ảnh minh hoạ.
- `README.md` và `docs/01`-`docs/03` đã được đối chiếu lại với code ở mục 10; vẫn nên kiểm tra với code khi sửa thuật toán.
- Hiểu nhầm đã xảy ra: câu "cứ để furniture như cũ" từng bị hiểu là "không đưa FURNITURE vào", nhưng ý đúng là **đưa header vào**. Với các chỉ dẫn mơ hồ, hãy xác nhận lại.

---

## 9. Các câu hỏi còn mở cho người dùng

1. ~~Người dùng có thể **gửi mẫu PDF kiểu AUTOSAR** (hoặc cho phép tải specification công khai) để bắt đầu P2 không?~~ **[Đã trả lời: có, 4 PDF đã tải, mục 11.]**
2. (Đã xong cho mục 1-9) Các thay đổi đó đã commit và push. Các commit của mục 10 **chưa được push** khi viết. Có muốn push và mở PR vào `master` không? PR sẽ gồm toàn bộ các commit của nhánh so với `master` (`git rev-list --count master..HEAD`).
3. **Áp dụng `--include-furniture` (header/footer) cho pipeline sản xuất** (sinh IR/Markdown) hay chỉ cho phần đánh giá? Nếu có, cần sửa lỗi hierarchy 5.1.
4. Thống nhất cấu hình OCR cho sản xuất: cắt ảnh = OCR tắt; IR đầy đủ với ảnh phẳng = OCR bật (torch GPU); PDF số = đọc text layer bằng code?
5. Có muốn mở rộng đánh giá thứ tự đọc ra 1.651 trang (~2,2 giờ) và thiết lập tập giữ riêng cho XY-Cut không?
6. Có muốn thử ý tưởng raster còn lại (OCR tắt + `keep_empty_clusters=True`) không, hay bỏ hẳn nhánh ảnh phẳng?

---

## 10. Cập nhật sau handoff (2026-11-12, các commit sau `37d0ab2`)

### 10.1. Hai thay đổi hành vi
- **Gom sơ đồ ảnh phẳng tắt mặc định.** `scripts/test_diagram_grouping.py` có `scale_items_to_pixels` (cùng phép đổi tọa độ như bước đầu của `group_whole_diagrams_image`, giữ tọa độ pixel mà các ngưỡng 80/300 px của IR giả định) và `raw_picture_groups` (một mục cho mỗi `picture`/`figure`). Demo và Tầng 2 dùng chúng trừ khi truyền `--group-diagrams`. Thân hàm `group_whole_diagrams_image` **không đổi** (baseline R3 của v3/v4 và test dựa vào nó). Căn cứ: OCR bật, 100 trang, micro-F1 0,766 (box thô) so với 0,281 (sau gom v1) trong `results.json`.
- **Tầng 2 cắt ảnh thật:** mỗi diagram được cắt vào `<output>/crops/` và `cropped_image` được điền (trước đây luôn rỗng). Bbox bảng và ô được đổi sang pixel như các item: nếu không, việc bỏ fallback `table_elements[0]` làm lộ 3/20 trang có kích thước trang Docling khác ảnh (tỉ lệ ~2,78) bị mất liên kết bảng. **Chưa tìm nguyên nhân gốc** của việc kích thước khác nhau; cùng rủi ro còn ở `run_e2e_reading_order_eval.py:356,367` (dùng `page_info.width/height` cho box đã scale, **[CHƯA KIỂM CHỨNG]**).
- Kiểm chứng trên 20 trang Tầng 2 (grouping tắt): 45 picture Docling = 45 mục `diagrams[]` = 45 file crop; không có `cropped_image` rỗng; không bbox vượt kích thước ảnh. Chỉ kiểm tra số lượng và hình học, **chưa đánh giá chất lượng crop**.

### 10.2. Thử veto chống over-merge trên nền v2 (kết quả âm)
`scripts/run_diagram_crop_eval_v4.py` (`veto` là tham số tùy chọn mới của `group_whole_diagrams_image_v2`, mặc định `None`). Đặt trên v2 với `GroupParamsV2()` mặc định (cấu hình "deployed" chỉ tạo 7 nhóm nên không có gì để chặn), leave-source-out 5 fold, tiêu chí đăng ký trước: sửa ≥ 5/38 hình vỡ, hại ≤ 1% hình nguyên vẹn (≤ 10), near-merge ≤ R1 + 1 điểm %.

| Hệ thống | Sửa | Hại | near-merge (R1 = 10,7%) | whole@0,5 (R1 = 1057/1385) |
|---|---|---|---|---|
| v2 mặc định, không veto | 13/38 | 197/1.057 | 33,2% | 873 |
| + caption oracle (caption GT, cận trên) | 14/38 | 137 | 24,7% | 934 |
| + khoảng trắng tự chuẩn hóa (OOF) | 3/38 | 15 | 12,5% | 1045 |
| + cả hai (OOF) | 3/38 | 10 | 12,2% | 1050 |

**Không veto nào đạt tiêu chí và mọi cấu hình đều thấp hơn box thô.** Caption oracle chỉ chặn 56/200 cặp gộp sai (và 1/79 cặp gộp đúng); caption Docling chỉ có ở 107/613 trang (cache OCR bật). Veto khoảng trắng chặn lẫn lộn cặp đúng và sai. Trần lý thuyết: gom hoàn hảo cứu tối đa 31/1.385 hình (+2,2 điểm %), trong khi 275 hình (19,9%) bị bỏ sót hoàn toàn. Kết luận của mục 4.4 (dừng nhánh ảnh phẳng, dùng box thô) được củng cố. Mặc định `GroupParamsV2()` vẫn là cấu hình có hại (docstring "DO-NO-HARM" của `diagram_grouping_v2.py` không đúng với default); v2 hiện chỉ được gọi từ mã đánh giá và test, không có luồng sản xuất nào gọi nó. Lưu ý thước đo: `harm_intact` dùng chính nhãn GT không nhất quán, nên một phần "hại" là bất đồng quy ước nhãn.

### 10.3. Sửa lỗi code
- `postprocess_ir_structure` (`scripts/ir_postprocessor.py`): (a) caption liên kết với bảng/hình được xử lý ở **lượt thứ hai** để `level = target.level + 1` đúng cả khi caption đứng trước đích; (b) bỏ fallback `table_elements[0]`, ghi `tables[].metadata.link_warning` khi không có phần tử bảng giao nhau. Bảng như vậy không xuất hiện trong Markdown (Markdown duyệt theo `elements`); trên 20 trang Tầng 2 còn 3 bảng như thế (cùng một trang). Lỗi `header`/`page_header` bị coi là `section_header` **chưa sửa** (hoãn vì chỉ lộ ra khi bật FURNITURE).
- `scripts/test_ir_postprocessor.py` (mới, 6 test). Chạy trên bản cũ của `ir_postprocessor.py`: 2/6 fail đúng chỗ (caption trước hình, bảng không giao nhau), nên test phân biệt được lỗi. `test_eval_utils.py` vẫn 35/35.
- `scripts/run_dataset_stress_test.py`: xóa khối kiểm tra over-merge của Tầng 1 (đọc sai khóa nhãn và nuốt lỗi); thêm `--group-diagrams`.

### 10.4. Quan sát khi xem trực tiếp crop Tầng 2 (mới xem vài ảnh, chưa phải đánh giá hệ thống)
- Hình bị chữ bao quanh vẫn lẫn chữ (hình túi da ở `sample_03`: box chữ nhật của Docling chồng lên vài cột chữ).
- Hình trang trí nhỏ (ngôi sao, linh vật, icon, logo) thành crop riêng; chưa có bộ lọc.
- Chú thích nằm ngoài box bị mất (nhãn bên cạnh bảng lịch ở `sample_10`).
- Hồ sơ 275 hình bị bỏ sót (`results_v3.json`, `missed_summary`): <1% trang: 119; 1-5%: 80; 5-20%: 61; ≥20%: 15; 201 nằm trong container OpenCV; 21 bị một box Docling khác phủ; nhiều nhất `exam_paper` (77) và `colorful_textbook` (61).
- Đã bàn nhưng **chưa làm**, kèm đánh giá: (1) cờ hình trang trí chỉ có ý nghĩa nếu có đoạn mã đọc nó; không dùng luật theo tỉ lệ khung hình (banner rộng vẫn là nội dung); nên kiểm tra bộ phân loại picture của Docling (`do_picture_classification`) trước, và cần quyết định icon nhỏ có tính là "hình" hay không (OmniDocBench GT tính, mục tiêu AUTOSAR nhiều khả năng không). (2) co box khỏi box chữ chồng một phần (chỉ các box chữ nằm chủ yếu ngoài hình, vì chữ nằm trọn trong hình thường là nhãn sơ đồ), đo trực tiếp bằng "phần crop bị box chữ GT phủ giảm" và "phần hình GT nằm trong crop không giảm", **không** dùng IoU@0,5; box chữ Docling chỉ có ở cache OCR bật (107 trang có hình), nếu dùng box chữ GT thì phải ghi rõ là oracle. (3) cứu hình bị bỏ sót bằng container OpenCV: hoãn (xung đột với (1), cần thêm chỉ số precision).

### 10.5. Docs đã cập nhật
`README.md`, `docs/01`, `docs/02`, `docs/03` được sửa cho khớp với code và số liệu đo (không còn: `row_idx`/`col_idx`, `table_id`/`caption_id` ở top-level `tables`, `Δy <= 80 px` cho caption, ngưỡng IoU 0,8, reading_order bắt đầu từ 1, tên hàm `group_whole_diagrams_pdf_vector`, mô tả lọc nét < 3 px/dilation 12 px, "0 over-merging", mục tiêu "nâng 0,5881 lên 0,85" mâu thuẫn với 0,9369, "1.651 trang" cho Tầng 1; các liên kết `file:///C:/...` đã đổi sang tương đối). `outputs/README.md` nằm trong thư mục bị `.gitignore`, **chưa được sửa** và có thể còn mô tả thuật toán cũ.

### 10.6. Việc còn lại / câu hỏi mở thêm
- Quyết định: icon, logo nhỏ có tính là "hình" không (chặn hướng lọc trang trí và cứu hình bị bỏ sót).
- `outputs/README.md` (cục bộ) chưa cập nhật; đường dẫn `C:/Workspace/DocuMind` vẫn hard-code trong 4 hàm `main()`; ngưỡng pixel tuyệt đối trong `ir_postprocessor.py` (80/300 px) và gom v1; `compute_iou` trùng lặp; chưa có `pyproject`/CI/pytest.
- Đường `--group-diagrams` bật chưa chạy Tier 2 đầy đủ; demo `test_diagram_grouping.py` chưa chạy end-to-end sau thay đổi (sẽ ghi đè `outputs/02_diagram_grouping/`).
- ~~Nhánh vector vẫn chưa được kiểm chứng trên PDF thật~~ **[ĐÃ ĐO ở mục 11]**: các rủi ro suy ra từ code (khoanh bảng, hút chữ) được xác nhận trên PDF AUTOSAR.

---

## 11. Cập nhật: thử nghiệm trên PDF AUTOSAR (2026-11-12)

### 11.1. Bối cảnh và quyết định của người dùng
- Người dùng yêu cầu tải vài tài liệu AUTOSAR về repo và test trên đó. Đã tải 4 PDF Classic Platform R24-11 từ `autosar.org` vào `datasets/AUTOSAR/` (gitignore; spec có điều khoản sử dụng riêng, kiểm tra trước khi chia sẻ): `LayeredSoftwareArchitecture` (185 trang, slide), `VFB` (147), `SWS_ECUStateManager` (245), `SWS_OS` (407). Cả bốn file đều có text layer; sơ đồ ở dạng nét vẽ vector kèm vài ảnh nhúng nhỏ.
- **Mục tiêu hiện tại (người dùng nói rõ):** chỉ cần crop hình cho đúng; xử lý nội dung trong hình để sau. **Chữ dính thêm quanh hình chấp nhận được** (khi đưa hình cho agent có thể dặn bỏ qua chữ lẫn vào). Không cần phân biệt "ảnh thường" với "diagram": mỗi khối đồ họa là một crop. Vì vậy tiêu chí "dung sai" (hình trọn vẹn trong một box, box không quá 2 lần hình) được dùng song song với IoU.

### 11.2. Đã thêm vào repo
- `scripts/run_pdf_figure_crops.py`: pipeline PDF → crop hình (render trang, Docling OCR tắt, gom v1 + luật neo, tùy chọn `--refine-vector` và `--add-vector-missing`, lọc logo/chân trang, crop và `manifest.json`). Mô tả ở `docs/02` mục 6. Thân `group_whole_diagrams_image` không bị sửa (baseline R3 của v3/v4 và test dựa vào nó); luật neo đặt trong `v1_group_boxes`.
- `scripts/run_figure_variants_all.py`: chạy 9 biến thể trên nhiều PDF, ghi box, crop của 3 biến thể chính và ảnh ghép 4 biến thể cho các trang bất đồng.
- `scripts/eval_pdf_figure_variants.py` và `scripts/ground_truth/autosar_layered_figures.json`: chấm điểm với 30 box gán tay (44 trang, bỏ 3 trang mơ hồ).
- Test hiện có chạy lại sau thay đổi: `test_eval_utils.py` 35/35, `test_ir_postprocessor.py` 6/6.

### 11.3. Kết quả (chi tiết và bảng ở `docs/02` mục 5.3-5.4)
- Trên `LayeredSoftwareArchitecture`, Docling + v1 hơn hẳn nhánh vector (F1 chặt 0,885 so với 0,416; dung sai 0,852 so với 0,364). Vector đơn thuần hụt vì box quá lớn (hút chữ) và khoanh bảng.
- v1 **không** chặn việc khoanh nhầm bảng ở spec nhiều bảng: trên OS và ECUStateManager, mọi box chỉ có ở v1 (67 và 29) đều chứa bảng. Luật neo loại hết; sau đó trên 3 file spec mọi biến thể Docling cho kết quả giống hệt box `picture` thô.
- `refine` không giúp dưới tiêu chí dung sai; `add` thêm khoảng 1 hình. Cấu hình hiện được khuyến nghị: Docling (OCR tắt) + v1 + neo + `--add-vector-missing` (F1 0,918 chặt, 0,885 dung sai trên Layered). Đây là kết quả trên **một tài liệu, 30 box gán tay**, các ngưỡng chỉnh trên chính tập này; chênh lệch 1-2 hình giữa các cấu hình sát nhau không có ý nghĩa thống kê (tune-on-test như đã nêu ở mục 4.2).
- Quá trình có hỏi ý kiến phản biện độc lập: kết luận là không sửa thân hàm v1, luật "phải có picture" chưa đủ (đã thay bằng luật neo), không tinh chỉnh `refine`, và không gọi một cấu hình là "tốt nhất" khi chênh lệch chỉ 1-2 hình.

### 11.4. Việc còn lại và lưu ý
1. **Ground truth cho spec nhiều bảng (OS, ECUStateManager) chưa có**: chưa biết recall thật của box thô/anchor ở đó. Nên gán tay 15-20 trang, ưu tiên các trang có box lớn hơn nửa trang (ECUStateManager 25 box, mới xem 2 là sơ đồ thật; danh sách ở `outputs/12_autosar_all_variants_anchor/huge_boxes.md`, cục bộ).
2. Chưa có đường **PDF → IR**: pipeline PDF chỉ cắt hình. Còn thiếu: IR nhiều trang (`page` đang ghi cứng 1), thứ tự đọc, bảng, chữ lấy từ text layer thay vì OCR, nối crop vào đúng section. Phần IR hiện chỉ xử lý một trang ảnh.
3. Chưa tách đóng góp của từng luật A/B/C của v1 trên AUTOSAR; mọi ngưỡng v1 là pixel tuyệt đối, chỉ thử ở 150 DPI.
4. Các hộp "chỉ từ vector" của `add` (trang 83, 121, 137, 139 của Layered) mới có trang 121 trong mẫu có ground truth; các trang khác chưa kiểm tra.
5. Việc "phân biệt ảnh thường và diagram" không còn cần cho mục tiêu hiện tại; nếu cần sau này, PDF cho tín hiệu nguồn gốc (ảnh nhúng so với nét vẽ vector) rẻ hơn đoán từ pixel.
6. Một lần chạy Docling OCR bật (torch) trên các trang AUTOSAR bị dừng giữa chừng, nên mọi số liệu ở mục 11 là OCR tắt.
7. `outputs/` (cục bộ, gitignore) chứa kết quả: `07_autosar_branch_compare`, `08_pdf_figure_crops`, `09_refine_vector_compare`, `10_autosar_all_variants` (v1 chưa neo), `11_v1_rule_compare`, `12_autosar_all_variants_anchor` (v1 có neo).

---

## 12. Cập nhật: bộ đánh giá held-out cho crop hình (2026-11-13)

Người dùng yêu cầu dựng bộ đánh giá đáng tin sau khi mục 11 chỉ có 30 box gán tay trên một tài liệu.

- **Đã thêm:** `scripts/ground_truth/autosar_figures_v2.json` (108 trang test trên 4 PDF AUTOSAR, lấy mẫu có seed và phân tầng, gán nhãn mù, kèm 57 trang không có hình cần chấm; cộng 44 trang dev cũ của Layered), `scripts/eval_pdf_figures_v2.py` (chặt/dung sai, theo tài liệu và gộp, bootstrap theo trang, hiệu ghép cặp, dương tính giả trên trang trống). Chi tiết phương pháp và bảng số ở `docs/02` mục 5.5.
- **Kết quả test:** Docling + v1 + luật neo tốt nhất hoặc ngang tốt nhất (F1 dung sai 0,926, CI 0,854-0,984); bỏ luật neo giảm 0,07-0,08 (có ý nghĩa); `add`/`refine` không giúp trên test (trên dev thì khác chiều, trong nhiễu); nhánh vector đơn thuần 0,37-0,47. Ba file spec/VFB đạt 1,000 vì phần lớn là box `picture` thô và trang bảng; lỗi còn lại nằm ở slide Layered (5 trang: 19, 63, 75, 83, 95).
- **Thật thà về nhãn:** một nhãn (Layered trang 179) sai lúc gán mù và được sửa sau khi thấy dương tính giả; đã xem lại toàn bộ 23 trang âm tính bị biến thể nào đó dự đoán (đều là bảng/mục lục). Nhãn do một người gán (sai số ±5-10 pt), chưa có người thứ hai; hình nhỏ mà không biến thể nào bắt được có thể còn nằm trong "không hình".
- **Gợi ý bước kế tiếp:** (1) cấu hình mặc định nên là Docling + v1 + neo (đơn giản nhất không thua); `--add-vector-missing` giữ tùy chọn; (2) cải thiện slide: trang 19 (hộp có phần trống bị cắt), 83 (gộp với tiêu đề), 95 (tách đôi), 75 (hình full-width không box); (3) khi tune, chỉ dùng tập dev và để test cho đo cuối; mỗi lần chạy test để chọn cấu hình là làm ô nhiễm nó.
- Ảnh kiểm tra (cục bộ, gitignore): `outputs/13_eval_set/` (`selection.json`, `sheets/`, `grid/`, `gt_check/`, `report_test.md`, `report_dev.md`).
