# ĐẶC TẢ KIẾN TRÚC & UNIFIED JSON IR SCHEMA

Tài liệu này đặc tả chi tiết mô hình dữ liệu trung gian chuẩn hóa (**Unified JSON Intermediate Representation - IR**) theo **Mục 5.3 của `Topic3_Document_Understanding_8_Weeks.md`**, cùng các thuật toán hậu xử lý cấu trúc tài liệu (`scripts/ir_postprocessor.py`) trong hệ thống **DocuMind**.

---

## 1. Mục Tiêu Thiết Kế Của Unified JSON IR

Các thư viện bóc tách tài liệu hiện nay (như Docling, PyMuPDF, Unstructured) thường xuất dữ liệu ở dạng phẳng (flat list) hoặc định dạng Markdown thuần túy. Điều này tạo ra ba rào cản lớn đối với hệ thống AI Agent:
1. **Thiếu ngữ cảnh phân cấp (Flat Hierarchy):** Các đoạn văn bản, bảng biểu, ảnh minh họa không biết mình thuộc về chương/mục nào.
2. **Sai lệch thứ tự đọc trong bố cục đa cột (Reading Order Jumps):** Các thuật toán quét tọa độ từ trên xuống dưới theo trang giấy ($y$-axis sort) thường làm đảo lộn nội dung giữa cột trái và cột phải (ví dụ: một bảng ở đầu cột phải bị đọc xen vào giữa đoạn văn của cột trái).
3. **Mất liên kết thực thể (Entity Disconnect):** Khối bảng (Table) trong danh sách phần tử chỉ chứa text sơ sài, trong khi cấu trúc tế bào chi tiết (cells, merged headers) lại nằm tách biệt trong đối tượng khác mà không có khóa ngoại (foreign key) liên kết.

**Unified JSON IR** được thiết kế để giải quyết toàn diện ba thách thức trên, cung cấp định dạng chuẩn hóa, dễ dàng mở rộng và bổ sung metadata từ các mô hình AI/LLM.

---

## 2. Đặc Tả Chi Tiết Schema (Unified JSON IR Specification)

Mỗi tài liệu sau khi qua pipeline của DocuMind sẽ được xuất thành một file `.json` tuân thủ nghiêm ngặt cấu trúc dưới đây:

```json
{
  "$schema": "https://documind.ai/schemas/unified_ir_v1.json",
  "doc_id": "01_academic_2col",
  "source_file": "page-83587fc3-fe55-4e51-a3c5-f0664edba743.png",
  "doc_type": "academic_literature",
  "metadata": {
    "category": "academic_literature",
    "description": "Hai cot, co bang bieu va chu thich",
    "page_width": 1700.0,
    "page_height": 2200.0,
    "parser": "docling-2.15 + ir_postprocessor",
    "reading_order_status": "column_aware_sorted",
    "total_elements": 16,
    "total_tables": 2,
    "total_diagrams": 0
  },
  "elements": [
    {
      "id": "elem_0001",
      "type": "section_header",
      "level": 1,
      "parent_id": null,
      "reading_order": 1,
      "bbox": [152.0, 185.0, 780.0, 220.0],
      "text": "3. EXPERIMENTAL SETUP",
      "metadata": {
        "column": "left_col"
      }
    },
    {
      "id": "elem_0009",
      "type": "table",
      "level": 2,
      "parent_id": "elem_0001",
      "reading_order": 9,
      "bbox": [880.0, 180.0, 1550.0, 480.0],
      "text": "[Table tbl_0002: Table 2: Performance metrics on benchmark dataset]",
      "table_id": "tbl_0002",
      "metadata": {
        "column": "right_col"
      }
    },
    {
      "id": "elem_0010",
      "type": "caption",
      "level": 3,
      "parent_id": "elem_0009",
      "reading_order": 10,
      "bbox": [880.0, 490.0, 1550.0, 520.0],
      "text": "Table 2: Performance metrics on benchmark dataset",
      "metadata": {
        "column": "right_col",
        "target_id": "elem_0009"
      }
    }
  ],
  "tables": [
    {
      "table_id": "tbl_0002",
      "element_id": "elem_0009",
      "caption_id": "elem_0010",
      "caption_text": "Table 2: Performance metrics on benchmark dataset",
      "bbox": [880.0, 180.0, 1550.0, 480.0],
      "num_rows": 5,
      "num_cols": 4,
      "markdown": "| Metric | Baseline | Ours | Gain |\n|---|---|---|---|\n| Accuracy | 78.4% | 86.2% | +7.8% |",
      "cells": [
        {
          "row_idx": 0,
          "col_idx": 0,
          "rowspan": 1,
          "colspan": 1,
          "is_merged": false,
          "is_header": true,
          "text": "Metric"
        }
      ],
      "metadata": {
        "column": "right_col",
        "has_merged_cells": false
      }
    }
  ],
  "diagrams": []
}
```

---

## 3. Ba Thuật Toán Hậu Xử Lý Cốt Lõi (`ir_postprocessor.py`)

### 3.1. Thuật Toán Sắp Xếp Thứ Tự Đọc Recursive XY-Cut (Horizontal Banding & Dynamic k-Columns)

#### Bài toán:
Trong tài liệu đa cột (báo cáo khoa học 2 cột, báo chí 3 cột, tài liệu hỗn hợp 1 cột + 2 cột, slide thuyết trình), việc chia cột toàn cục cứng nhắc dễ gặp các lỗi nghiêm trọng:
1. **Lỗi vỡ luồng đọc 3 cột:** Cột giữa bị cắt đôi hoặc gom nhầm sang cột trái.
2. **Lỗi dính biên bố cục hỗn hợp (1+2 cột):** Phần tiêu đề / Abstract 1 cột phía trên bị nuốt vào luồng đọc của cột trái bên dưới.
3. **Lỗi nhảy cóc trong văn bản bọc chữ L hoặc công thức thụt lề:** Khi các đoạn văn bản nằm ở các dải độ cao khác nhau nhưng vẫn bị ép chia cột ngang trang.

#### Giải pháp True Recursive XY-Cut trong `postprocess_ir_structure()`:
1. **Phân tách Rìa Trang (Page Margin Separation):**
   - Header (`header`, `page_header` với $y_{\min} < 0.15 \cdot H$) được ưu tiên xếp đầu trang theo $(y_{\min}, x_{\min})$.
   - Footer / Page Number (`footer`, `page_number`, `page_footnote` với $y_{\max} > 0.85 \cdot H$) được đưa xuống cuối trang.
2. **Ưu Tiên Cắt Dọc Các Cột Song Song (Vertical X-Cut First):**
   - Phân tích các cột dọc $c_1, c_2, \dots, c_k$ dựa trên hình chiếu trục $X$ có co biên thích ứng.
   - Nếu tồn tại $\ge 2$ cột dọc rời rạc **và** có sự chồng lấn theo trục dọc $Y$ giữa các cột ($\Delta Y > 10\text{px}$, chứng tỏ đây là các cột bố cục song song ngang hàng nhau chứ không phải 2 đoạn văn trên-dưới):
     $\rightarrow$ **Ưu tiên cắt dọc trước!** Duyệt từng cột từ trái sang phải và đệ quy xử lý từng cột độc lập từ đỉnh xuống đáy.
     *(Nhờ đó, tài liệu 2 cột hoặc 3 cột không bao giờ bị cắt ngang giữa trang khi các cột có khe trắng ngẫu nhiên, và các cụm hình-caption song song được đọc chuẩn: hình trái -> chú thích trái, sau đó mới sang hình phải -> chú thích phải).*
3. **Cắt Ngang Khi Có Banner / Spanning Item (Horizontal Y-Cut & Spanning Split):**
   - Nếu không thể cắt dọc (do có tiêu đề/banner rộng trải dài $\ge 55\%$ chiều rộng dải), tách khối banner làm vách ngăn rồi đệ quy cho nửa trên và nửa dưới.
   - Gom các phần tử không chồng lấn $Y$ thành các dải ngang và xử lý tuần tự từ trên xuống dưới.
4. **Fallback Sorting:**
   - Khi không thể phân tách thêm theo cả trục $X$ và $Y$, sắp xếp các khối theo $(\text{round}(y_{\min} / 12), x_{\min})$.
5. **Hiệu quả thực nghiệm:**
   - **Kendall's Tau toàn dataset OmniDocBench (1.602 trang):** Đạt **`0.9369`** (Trung vị: **`1.0000`**), trong đó:
     + `single_column`: **0.9699** (Trung vị: 1.0000, 96.4% $\ge 0.70$)
     + `three_column`: **0.9552** (Trung vị: 1.0000, 92.5% $\ge 0.70$)
     + `double_column`: **0.9521** (Trung vị: 1.0000, 95.1% $\ge 0.70$)
     + `other_layout`: **0.8805** (Trung vị: 0.9569, 89.2% $\ge 0.70$)
     + `1andmore_column`: **0.8701** (Trung vị: 0.9957, 80.0% $\ge 0.70$)

---

### 3.2. Thuật Toán Xây Dựng Cây Phân Cấp Logic (Hierarchy Tree Construction)

Thuật toán tự động tái lập mối quan hệ cha-con (`parent_id`) và cấp bậc ngữ nghĩa (`level`) dựa trên luồng đọc đã được chuẩn hóa:

```mermaid
graph TD
    DOC["Tài liệu (doc_id)"] --> T1["title (level 1, parent: null)"]
    T1 --> S1["section_header 1 (level 1, parent: title)"]
    S1 --> P1["text paragraph (level 2, parent: S1)"]
    S1 --> L1["list_item (level 3, parent: P1)"]
    S1 --> S2["subsection_header 1.1 (level 2, parent: S1)"]
    S2 --> TBL["table (level 3, parent: S2)"]
    TBL --> CAP["caption (level 4, parent: TBL)"]
    S2 --> DIAG["diagram (level 3, parent: S2)"]
    DIAG --> DNODE["diagram node/member (level 4, parent: DIAG)"]
```

#### Quy tắc gán `level` và `parent_id`:
1. `title`: Gán `level = 1`, `parent_id = null`. Cập nhật `current_title_id = elem.id`.
2. `section_header`:
   - Nếu là tiêu đề chính: `level = 1`, `parent_id = current_title_id`.
   - Nếu là tiêu đề phụ (bắt đầu bằng `1.1`, `A.`,...): `level = 2`, `parent_id = current_section_id`.
   - Cập nhật `current_section_id = elem.id`, `current_section_level = level`.
3. `text`, `table`, `picture`, `formula`:
   - `level = current_section_level + 1`.
   - `parent_id = current_section_id` (nếu có) hoặc `current_title_id` hoặc `null`.
4. `list_item`:
   - Gán phụ thuộc vào block văn bản dẫn nhập phía trước: `level = last_text_level + 1`, `parent_id = last_text_id`.
5. `caption`:
   - Tự động dò tìm khối Bảng (`table`) hoặc Hình vẽ (`picture`) liền kề phía trước hoặc phía sau trong khoảng cách $\Delta y \le 80\text{px}$.
   - Gán: `parent_id = target_element.id`, `level = target_element.level + 1`.
6. Phần tử con trong sơ đồ khối (`diagram_member`):
   - Gán: `parent_id = diagram_group.id`, `level = 3`.

---

### 3.3. Thuật Toán Liên Kết Thực Thể Hai Chiều (Bidirectional Entity Linking)

Để bảo đảm tính đồng nhất dữ liệu giữa biểu diễn trực quan và dữ liệu ngữ nghĩa:
1. **Chiều Xuôi (`elements -> tables`):**
   - Với mỗi phần tử `type: "table"` trong danh sách `elements`, tìm kiếm bảng tương ứng trong mảng `tables` dựa trên chỉ số hoặc tọa độ BBox giao nhau (IoU $\ge 0.8$).
   - Gán trường `table_id = "tbl_XXXX"` vào đối tượng `element`.
   - Cập nhật trường `text` của element bằng chuỗi tóm tắt ngữ nghĩa:
     `"[Table tbl_XXXX: <Tên Caption Nếu Có>]"` thay vì để trống `""`.
2. **Chiều Ngược (`tables -> elements & captions`):**
   - Trong mỗi bảng ở mảng `tables`, bổ sung các trường khóa ngoại:
     - `element_id`: Trỏ ngược lại `id` của khối layout trong `elements`.
     - `caption_id`: Trỏ tới `id` của nhãn chú thích tương ứng.
     - `caption_text`: Lưu nội dung chú thích phục vụ trích xuất nhanh mà không cần join bảng.
3. **Chuẩn Hóa Tế Bào (Cell-Level Normalization):**
   - Trích xuất chi tiết từng tế bào: `row_idx`, `col_idx`, `rowspan`, `colspan`, `is_merged`, `text`, `is_header`.
   - Tự động phát hiện cờ `has_merged_cells: true` khi có bất kỳ ô nào có `rowspan > 1` hoặc `colspan > 1`.
   - Sinh bảng Markdown chuẩn (`markdown`) phản ánh đúng ma trận tế bào đã gộp.

---

## 4. Cơ Chế Sinh File Kép (JSON IR + GitHub-Flavored Markdown)

Hàm `generate_markdown_from_ir(processed_ir)` duyệt tuần tự theo mảng `elements` đã sắp xếp theo `reading_order` chuẩn:
- `title` -> Chuyển thành `# <text>`
- `section_header` -> Chuyển thành `## <text>` hoặc `### <text>` tùy theo `level`.
- `text` -> Chuyển thành đoạn văn `<text>\n\n`.
- `list_item` -> Chuyển thành `- <text>`.
- `table` -> Nhúng trực tiếp chuỗi `markdown` từ mảng `tables` qua khóa ngoại `table_id`.
- `diagram` -> Nhúng ảnh crop tương ứng hoặc mã nguồn **Mermaid Flowchart** nếu có dữ liệu quan hệ các nút (`nodes`, `edges`).

Nhờ đó, dữ liệu đầu ra phục vụ song song cả hai nhu cầu: **AI Agent đọc JSON có cấu trúc để reasoning** và **Con người / LLM đọc Markdown để duyệt nội dung tự nhiên**.
