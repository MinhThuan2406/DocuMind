# DocuMind – Complex Multi-Format Document Understanding & Structure Recovery Agent

**Kế hoạch R&D và Triển khai – Thời gian dự kiến: 8 tuần**

> Tài liệu này xác lập đặc tả yêu cầu, kiến trúc hệ thống và kế hoạch hành động 8 tuần cho dự án **DocuMind (Document Understanding & Structure Recovery Agent)**.
> 
> **Quy ước nhãn nguồn nội dung:**
> - **[SPEC]** – Đặc tả yêu cầu bài toán kỹ thuật cốt lõi.
> - **[IMPL]** – Đề xuất kiến trúc, giải pháp kỹ thuật và kế hoạch do nhóm tự thiết kế.

---

## 1. Tổng quan dự án

| Hạng mục | Nội dung | Phân loại |
|---|---|---|
| Lĩnh vực | Document AI, Multimodal Extraction, Computer Vision, Graph Reasoning | [SPEC] |
| Mức độ phức tạp | Nâng cao (Xử lý đa định dạng, nested table, diagram rã nhóm, cross-doc inconsistency) | [IMPL] |
| Thời gian | 8 tuần (R&D $\rightarrow$ Engine Prototype $\rightarrow$ Benchmark & Product Demo) | [IMPL] |
| Dữ liệu đầu vào | `PDF`, `DOCX`, `XLSX`, `PPTX` – từng file riêng lẻ hoặc theo bộ liên kết | [SPEC] |
| Dữ liệu đầu ra | Cấu trúc tài liệu máy đọc được (JSON IR); Đồ thị quan hệ Diagram (Node–Edge); Báo cáo sai khác kèm dẫn chứng nguồn (Source References) | [SPEC] |
| Hướng tiếp cận | Hybrid: Native XML Parsing + Document Layout Analysis + Graph Theory + Multimodal LLM | [IMPL] |
| Bối cảnh ứng dụng | Ngành Công nghiệp Ô tô (Automotive): Sổ tay kỹ thuật, hướng dẫn bảo dưỡng xe điện (EV), bảng thông số ECU/linh kiện, quy trình kiểm thử | [SPEC] |

---

## 2. Mô tả bài toán kỹ thuật [SPEC]

**Bối cảnh thực tế (Background & Context):**  
Tài liệu kỹ thuật và doanh nghiệp thường chứa mật độ thông tin dày đặc: bảng lồng nhau hoặc gộp ô (merged cells), floating shapes và diagram ghép từ nhiều hình khối rời rạc không được group. Khi lưu trữ, chuyển đổi hoặc làm phẳng (flatten/rasterize), toàn bộ cấu trúc logic bị biến mất, đồng thời các phiên bản tài liệu khác nhau dễ nảy sinh thông tin mâu thuẫn, sai lệch.

**Mục tiêu bài toán (Problem Description):**  
Xây dựng một Agent thông minh có khả năng hiểu, khôi phục cấu trúc phân cấp từ tài liệu phức tạp (gồm diagram chưa group và layout bị suy giảm), đồng thời tự động phát hiện các điểm mâu thuẫn/sai khác thông tin giữa các tài liệu trong cùng một bộ hồ sơ.

### 2.1 Các mục tiêu kỹ thuật trọng tâm (Core Objectives)

1. **Cấu trúc & Đọc hiểu:** Khôi phục chuẩn xác thứ tự đọc (**reading order**), phân cấp đề mục (**hierarchy**), ranh giới bảng (**table boundaries**) và ô gộp (**merged cells**).
2. **Liên kết ngữ nghĩa hình ảnh:** Gắn chuẩn xác văn bản nhãn (**text**) với hình khối (**shapes**) và biểu đồ (**charts**).
3. **Phục hồi sơ đồ (Diagram Recovery):** Tự động gom nhóm các khối hình vẽ rời rạc thành các diagram có nghĩa, tái dựng đồ thị quan hệ nút – cạnh (**node–edge relationships**).
4. **Đối chiếu liên tài liệu (Cross-Document Consistency):** Tự động phát hiện thông tin tương đồng, cập nhật hoặc **mâu thuẫn/xung đột** giữa các file trong cùng một bộ tài liệu kỹ thuật, kèm **dẫn chứng nguồn chính xác (with source references)**.

### 2.2 Phạm vi xử lý (Scope)

- **Định dạng file:** Hỗ trợ đầy đủ `PDF`, `DOCX`, `XLSX`, và `PPTX`.
- **Phương thức nạp:** Từng file đơn lẻ hoặc theo bộ hồ sơ liên quan (Folder/Related Document Set).
- **Đầu ra tiêu chuẩn:** Cấu trúc tài liệu máy đọc được (Machine-readable IR/JSON), đồ thị quan hệ diagram trực quan và danh sách cảnh báo mâu thuẫn kèm tọa độ/nguồn trích dẫn.

### 2.3 Tiêu chí đánh giá chất lượng hệ thống

| Mã | Tiêu chí | Mô tả chi tiết |
|---|---|---|
| **E1** | **Structure Recovery Accuracy** | Độ chính xác của cấu trúc được khôi phục (reading order, hierarchy tree, table matrix & merged cells) so với Ground Truth. |
| **E2** | **Diagram & Graph Correctness** | Độ đúng của việc gom cụm hình khối (diagram groups) và các cạnh quan hệ (node–edge relations). |
| **E3** | **Consistency Precision & Grounding** | Độ chính xác của các phát hiện mâu thuẫn và tính xác thực của các trích dẫn nguồn (Source References). |
| **E4** | **Generalization & Robustness** | Khả năng thích ứng ổn định trên các tài liệu mới lạ, layout phức tạp chưa từng huấn luyện/cấu hình từ trước. |

### 2.4 Các phạm vi nhóm chủ động thiết kế & làm chủ [IMPL]

Vì là dự án tự chủ hoàn toàn (End-to-End Project):
- **Chuẩn hóa IR (Intermediate Representation):** Tự xây dựng Schema chuẩn JSON đại diện cho mọi thành phần tài liệu.
- **Xây dựng Golden Test Dataset:** Thiết kế bộ tài liệu mẫu thực tế về lĩnh vực Ô tô (Automotive Spec Suite) có đủ 4 định dạng với các bẫy cấu trúc và mâu thuẫn gài sẵn.
- **Evaluation Engine:** Tự phát triển module đo lường định lượng các chỉ số (Kendall Tau, TEDS, Graph F1, Precision/Recall).

---

## 3. Nguyên tắc & Ràng buộc kỹ thuật cốt lõi

- **Tính minh bạch & Truy xuất nguồn gốc (Traceability & Groundedness):** Mọi thông tin trích xuất hoặc kết luận mâu thuẫn đều phải có bằng chứng cụ thể (`source_ref`: tên file, số trang/slide/sheet, tọa độ bbox hoặc cell ID) kèm độ tin cậy (`confidence level`). Tuyệt đối không sinh thông tin giả (hallucination).
- **Xử lý tài liệu thực tế:** Hệ thống cần xử lý được cả file native (có sẵn text layer, vector shapes) lẫn các trang tài liệu bị làm phẳng (flattened PDF, layout phức tạp).
- **Chống Hard-code (Anti-Overfitting):** Thuật toán và pipeline xử lý dựa trên nguyên lý cấu trúc học (layout heuristics), hình học không gian và ngữ nghĩa mô hình, không hard-code theo tên file, số trang hay vị trí tọa độ cố định.
- **Tối ưu hiệu năng:** Tách biệt rõ ràng luồng xử lý hàng loạt (Batch Processing cho cả kho tài liệu) và truy vấn thời gian thực (Interactive Query/Inference).

---

## 4. Phân rã bài toán kỹ thuật (Work Breakdown Structure)

| ID | Module chức năng | Mục tiêu tương ứng | Biểu hiện & Thách thức kỹ thuật |
|---|---|---|---|
| **P3-01** | Reading Order Recovery | Objective 1 | Văn bản đa cột, text box nổi, hình ảnh/hình khối xen giữa các đoạn văn |
| **P3-02** | Hierarchy Extraction | Objective 1 | Đề mục không theo chuẩn Style heading, đánh số đa cấp, bullet/list lồng nhau |
| **P3-03** | Table Boundary Detection | Objective 1 | Bảng không viền (borderless), bảng lồng trong bảng, bảng ngắt trang dài nhiều trang |
| **P3-04** | Merged Cell Resolution | Objective 1 | Span XML trong DOCX/PPTX, merged range trong XLSX, ô gộp bằng hình học trong PDF |
| **P3-05** | Shape/Chart Text Association | Objective 2 | Nhãn văn bản nằm ngoài khối hình, chú thích biểu đồ, text xoay góc |
| **P3-06** | Diagram Grouping | Objective 3 | Hàng chục hình hộp, đường vẽ rời rạc không được nhóm (ungrouped objects) |
| **P3-07** | Node–Edge Graph Builder | Objective 3 | Connector thông minh (PPTX), mũi tên/đường vector (PDF/DOCX) $\rightarrow$ Đồ thị có hướng |
| **P3-08** | Degraded Layout Handling | Robustness | Tài liệu PDF bị flatten, mất phân tầng vector |
| **P3-09** | Chart Information Parsing | Objective 2 | Trích xuất trục, legend, chuỗi dữ liệu (series) và tiêu đề biểu đồ |
| **P3-10** | Cross-Document Entity Linker | Objective 4 | Liên kết các thực thể chung (mã linh kiện, thông số kỹ thuật, phiên bản) qua các file |
| **P3-11** | Conflict & Inconsistency Detector | Objective 4 | Phát hiện sai lệch số liệu, mâu thuẫn thông số kỹ thuật giữa các tài liệu |
| **P3-12** | Model Generalization | Tiêu chí E4 | Hệ thống vận hành tốt trên bố cục mới lạ |
| **P3-13** | Groundedness & Provenance | Traceability | Mọi output đều có bằng chứng gốc (evidence), hạn chế (limitation) và confidence |

---

## 5. Kiến trúc hệ thống đề xuất [IMPL]

### 5.1 Sơ đồ Pipeline tổng thể

```
 Input Files (PDF / DOCX / XLSX / PPTX)
        │
        ▼
 [1] Format Detector & Loader
        │
        ├── DOCX / XLSX / PPTX ──► Native XML Parser (lxml, openpyxl, python-pptx)
        └── PDF ─────────────────► Layout & Geometry Parser (PyMuPDF, Docling)
        │
        ▼
 [2] Unified Intermediate Representation (IR Engine)
        ▼
 [3] Reading Order & Hierarchy Analyzer
        ▼
 [4] Table Reconstruction Engine (Boundaries, Rows/Cols, Merged Cells)
        ▼
 [5] Shape & Chart Associator
        ▼
 [6] Diagram Grouping & Node–Edge Graph Builder
        ▼
 [7] Cross-Document Entity Linker
        ▼
 [8] Consistency Reasoning Engine ──► Findings + Source References
        │
        ├──► Machine-Readable IR Output (JSON-LD / Structured Data)
        ├──► Interactive Knowledge Graph (Node-Edge Visualization)
        └──► Audit Report (Conflict Summary with Direct Source Anchors)
        │
        ▼
 [9] Evaluation & Benchmark Module ──► TEDS, Kendall Tau, Graph F1, Precision
```

### 5.2 Ngăn xếp công nghệ (Technology Stack)

| Thành phần | Công nghệ lựa chọn | Mục đích sử dụng |
|---|---|---|
| **DOCX Parsing** | `python-docx`, `lxml` | Trích xuất trực tiếp cây XML: paragraph style, table XML grid, drawing canvas |
| **XLSX Parsing** | `openpyxl` | Đọc bảng tính nguyên bản: sheet names, formulas, cell coordinates, merged ranges |
| **PPTX Parsing** | `python-pptx` | Lấy tọa độ shapes, connector properties (begin/end connections), layout slide |
| **PDF Parsing** | `PyMuPDF` (`fitz`), `Docling`, `pdfplumber` | Bóc tách text blocks, vector drawings, rect boundaries, phục hồi bảng phức tạp |
| **Graph Processing** | `networkx`, `Cytoscape.js` / `React Flow` | Dựng đồ thị quan hệ node–edge và trực quan hóa tương tác trên UI |
| **AI / LLM Reasoning** | Multimodal LLM (Gemini API / Qwen2.5-VL / Local LLM) | Hỗ trợ phân tích ngữ nghĩa, hiểu sơ đồ phức tạp và suy luận mâu thuẫn liên tài liệu |
| **UI & Demo** | `Streamlit` hoặc `FastAPI` + `React/Next.js` | Giao diện tương tác người dùng: xem tài liệu, xem sơ đồ quan hệ và danh sách cảnh báo mâu thuẫn |
| **Testing & Benchmark** | `pytest`, custom evaluation scripts | Tự động hóa đánh giá độ chính xác so với Golden Dataset |

### 5.3 Schema IR nội bộ (Unified Intermediate Representation)

```json
{
  "doc_id": "EV_Braking_System_Specs.pptx",
  "doc_type": "pptx",
  "metadata": {
    "title": "EV Braking Control Architecture",
    "pages_or_slides": 5
  },
  "elements": [
    {
      "id": "elem_0102",
      "type": "shape",
      "page": 2,
      "bbox": [120.5, 80.0, 320.0, 160.0],
      "text": "Brake Control Unit (BCU)",
      "parent_id": "group_diagram_01",
      "reading_order": 4,
      "level": 2,
      "source_ref": {
        "file": "EV_Braking_System_Specs.pptx",
        "slide": 2,
        "shape_name": "Rectangle 4"
      },
      "confidence": 0.98
    }
  ],
  "tables": [
    {
      "id": "tbl_001",
      "page": 3,
      "bbox": [50.0, 100.0, 500.0, 350.0],
      "num_rows": 5,
      "num_cols": 4,
      "cells": [
        {
          "row": 0, "col": 0, "rowspan": 2, "colspan": 1,
          "text": "Component Code",
          "is_merged": true
        }
      ]
    }
  ],
  "edges": [
    {
      "src": "elem_0102",
      "dst": "elem_0108",
      "label": "CAN Bus Signals",
      "source_ref": {
        "file": "EV_Braking_System_Specs.pptx",
        "slide": 2,
        "connector_id": "Conn_12"
      }
    }
  ],
  "findings": [
    {
      "finding_id": "conf_001",
      "type": "numerical_conflict",
      "description": "Mâu thuẫn thông số áp suất phanh định mức giữa tài liệu thiết kế và bảng kiểm thử.",
      "evidence": [
        {
          "file": "Braking_Service_Manual.pdf",
          "page": 14,
          "text": "Nominal braking pressure: 120 bar"
        },
        {
          "file": "Component_Specs.xlsx",
          "sheet": "Tolerances",
          "cell": "C12",
          "text": "140 bar"
        }
      ],
      "confidence": 0.95,
      "limitations": "Cần xác nhận lại xem phiên bản linh kiện có trùng mã revision không."
    }
  ]
}
```

---

## 6. Phương pháp kỹ thuật chi tiết theo bài toán con [IMPL]

### 6.1 Thứ tự đọc & Cây phân cấp (Reading Order & Hierarchy)
- **DOCX/PPTX:** Dùng thứ tự phần tử XML gốc kết hợp phân tích tọa độ không gian (Top-to-Bottom, Left-to-Right) để xử lý các text box nổi.
- **PDF:** Thuật toán phân cột dựa trên dải phân cách trắng (Whitespace projection analysis), sắp xếp luồng đọc theo khối cột trước khi sang cột tiếp theo.
- **Hierarchy:** Kết hợp đa tín hiệu: Heading Style XML, cỡ chữ (font size), độ đậm (font weight), mẫu đánh số đề mục (`1.`, `1.1`, `a)`) và khoảng thụt lề (indentation).

### 6.2 Xử lý Bảng và Ô gộp (Tables & Merged Cells)
- **Native Files (XLSX, DOCX):** Bóc tách trực tiếp thẻ XML `<w:gridSpan>`, `<w:vMerge>` hoặc thuộc tính `merged_cells` của `openpyxl`. Chuyển đổi thành ma trận 2D chuẩn hóa với thuộc tính `rowspan`, `colspan`.
- **PDF:** Nhận diện đường kẻ lưới vector (explicit lines). Đối với bảng không viền (borderless), sử dụng giải thuật căn lề khoảng trống chữ (whitespace column clustering). Tự động ghép nối các bảng nối dài qua nhiều trang dựa vào dòng tiêu đề lặp lại (repeated headers).

### 6.3 Nhóm sơ đồ & Đồ thị quan hệ (Diagrams & Node–Edge)
- **Gắn nhãn vào khối:** Phép toán chứa đựng hình học (geometric containment) hoặc thuật toán khoảng cách ngắn nhất (Euclidean distance matching).
- **Gom cụm sơ đồ:** Thuật toán phân cụm không gian (Spatial clustering / Connected components) gom các hình khối, nhãn chữ nằm trong cùng vùng ranh giới logic.
- **Nhận diện cạnh (Edges):** Trong PPTX sử dụng trực tiếp các điểm neo kết nối (`begin_connected_shape_id`, `end_connected_shape_id`). Với PDF/DOCX, sử dụng vector ray-casting từ hai đầu mút của mũi tên/đường thẳng đến hình khối gần nhất để tạo cạnh có hướng.

### 6.4 Liên kết & Kiểm tra mâu thuẫn (Cross-Doc Consistency)
- **Chuẩn hóa thực thể:** Trích xuất các mã kỹ thuật, số hiệu phụ tùng (Part Numbers), thông số vật lý (áp suất, điện áp, mô-men xoắn kèm đơn vị chuẩn SI).
- **Bộ máy đối chiếu:** Đối chiếu bảng chéo (cross-reference matrix) giữa các file. Chỉ kích hoạt cảnh báo khi có đủ ngữ cảnh tương đương (cùng thực thể, cùng đơn vị đo lường) để đảm bảo Precision cao (tiêu chí E3).
- **Trích dẫn minh chứng:** Mỗi phát hiện bắt buộc neo trực tiếp vào 2 hoặc nhiều vị trí gốc trong từng file tương ứng.

---

## 7. Kế hoạch đánh giá & Tiêu chuẩn chất lượng

### 7.1 Ma trận tiêu chí đánh giá dự án (Project Rubric)

| Tiêu chí | Trọng số | Yêu cầu đạt chuẩn |
|---|---|---|
| **Problem Fit & Domain Value** | 20% | Bám sát bài toán kỹ thuật ô tô thực tế; xác định rõ người dùng mục tiêu (Kỹ sư R&D, Kiểm định viên chất lượng) và giá trị tự động hóa. |
| **AI & Algorithm Quality** | 25% | Kết hợp hiệu quả giữa Native Parser, Heuristic và Multimodal AI; có bộ chỉ số đo lường định lượng rõ ràng. |
| **Software Architecture** | 20% | Kiến trúc module hóa sạch sẽ, dữ liệu đi qua chuẩn IR rõ ràng; có API và sơ đồ luồng dữ liệu minh bạch. |
| **Workload & Scalability** | 15% | Tối ưu hóa tài nguyên (CPU/RAM/GPU), xử lý tốt cả tài liệu dung lượng lớn; phân tách batch và streaming inference. |
| **Creativity & UX** | 10% | Trực quan hóa sơ đồ diagram sinh động; giao diện đối chiếu mâu thuẫn hai màn hình (side-by-side) dễ sử dụng. |
| **Presentation & Documentation** | 10% | Báo cáo kiến trúc rõ ràng, video/live demo mượt mà, tài liệu kỹ thuật đầy đủ. |

### 7.2 Bộ chỉ số đo lường định lượng (Evaluation Metrics)

| Thành phần | Chỉ số đo lường | Phương pháp tính |
|---|---|---|
| **Reading Order** | Kendall's Tau / Normalized Edit Distance | So sánh thứ tự đọc thực tế vs. Ground Truth (thang điểm [0, 1]) |
| **Hierarchy** | F1-Score | Độ chính xác của các cặp quan hệ nút cha – nút con (Parent-Child) |
| **Table Structure** | TEDS (Tree-Edit-Distance-based Similarity) | Đánh giá cả cấu trúc cây HTML/ma trận và nội dung ô (đặc biệt tính riêng cho ô gộp) |
| **Diagram Grouping** | Grouping F1 | Độ khớp giữa tập hợp các object trong nhóm thực tế so với nhóm chuẩn |
| **Node–Edge Graph** | Graph Edge Precision & Recall | Đo lường độ chính xác của bộ ba `(source, target, relation_label)` |
| **Consistency** | Precision & False Positive Rate | Tỷ lệ mâu thuẫn phát hiện là chính xác (ưu tiên Precision để tránh báo động giả) |
| **Grounding / Source Ref** | Source Validity Ratio | 100% các finding đều phải trỏ tới vị trí có thực trong tài liệu nguồn |

---

## 8. Kế hoạch hành động 8 tuần (Timeline & Milestones) [IMPL]

```
Tuần 1: Thiết lập nền tảng & Golden Dataset
Tuần 2: Bộ nạp đa định dạng (Native + PDF Layout)
Tuần 3: Thuật toán Reading Order & Hierarchy
Tuần 4: Khôi phục Bảng & Ô gộp phức tạp
Tuần 5: Gom cụm Diagram & Xây dựng Đồ thị quan hệ
Tuần 6: Liên kết thực thể & Bộ máy đối chiếu mâu thuẫn
Tuần 7: Đo lường Robustness, Tối ưu hóa & Freeze tính năng
Tuần 8: Hoàn thiện Product Demo & Bàn giao tài liệu
```

| Tuần | Trọng tâm | Nhiệm vụ kỹ thuật chính | Kết quả đầu ra (Milestones) |
|---|---|---|---|
| **Tuần 1** | **Khởi tạo & Đặc tả** | - Thống nhất kiến trúc tổng thể và Schema IR v0.1.<br>- Tạo bộ dữ liệu mẫu Automotive Golden Dataset (PDF, DOCX, XLSX, PPTX) kèm Ground Truth. | Repo khởi tạo, IR Schema v0.1, Golden Dataset v1.0 |
| **Tuần 2** | **Bộ nạp Parser** | - Xây dựng parser XML cho DOCX, XLSX, PPTX.<br>- Xây dựng parser hình học cho PDF (PyMuPDF / Docling). | Trích xuất thành công IR thô cho cả 4 định dạng |
| **Tuần 3** | **Reading Order & Cây phân cấp** | - Cài đặt thuật toán phân tích dải trắng & sắp xếp thứ tự đọc.<br>- Thuật toán phát hiện Heading và cấu trúc cây phân cấp. | Benchmark đạt Kendall Tau $\ge 0.85$, cây đề mục chuẩn hóa |
| **Tuần 4** | **Phục hồi Bảng** | - Xử lý merged cells, colspan, rowspan trên cả file native và PDF.<br>- Ghép bảng đa trang trong PDF. | Benchmark TEDS $\ge 0.88$ trên các bảng phức tạp |
| **Tuần 5** | **Diagram & Đồ thị** | - Thuật toán gắn text vào shape.<br>- Phân cụm các shape rời rạc thành Diagram.<br>- Trích xuất quan hệ Node-Edge (NetworkX). | Trích xuất đồ thị quan hệ sơ đồ, Graph F1 $\ge 0.80$ |
| **Tuần 6** | **Đối chiếu liên tài liệu** | - Chuẩn hóa thực thể kỹ thuật ô tô (Part IDs, specs, units).<br>- Engine phát hiện mâu thuẫn kèm source references. | Báo cáo mâu thuẫn tự động, Precision $\ge 0.90$ |
| **Tuần 7** | **Đánh giá & Tối ưu** | - **Feature Freeze** (ngừng thêm tính năng mới).<br>- Chạy kiểm thử trên tập dữ liệu layout mới lạ.<br>- Tinh chỉnh thuật toán, sửa lỗi failure cases. | Báo cáo kiểm thử chất lượng toàn diện (Benchmark Report) |
| **Tuần 8** | **Hoàn thiện Sản phẩm** | - Xây dựng Web UI Demo (Streamlit/Next.js) trực quan hóa sơ đồ và mâu thuẫn.<br>- Hoàn thiện Architecture Diagram, Technical Document & Video Demo. | Trọn bộ sản phẩm DocuMind hoàn chỉnh |

---

## 9. Danh mục bàn giao sản phẩm (Project Deliverables)

- [ ] **Mã nguồn hệ thống (DocuMind Repository):** Pipeline xử lý hoàn chỉnh, module hóa sạch sẽ, có hướng dẫn cài đặt và vận hành chi tiết.
- [ ] **Bộ nạp & Chuyển đổi IR:** Hỗ trợ đầy đủ 4 định dạng `PDF`, `DOCX`, `XLSX`, `PPTX` sang chuẩn JSON IR thống nhất.
- [ ] **Engine phân tích Diagram & Graph:** Xuất ra đồ thị nút - cạnh có cấu trúc, kèm trực quan hóa sơ đồ.
- [ ] **Engine phát hiện mâu thuẫn (Consistency Auditor):** Tự động phát hiện sai lệch số liệu/phiên bản liên tài liệu, 100% kèm `source_ref` chính xác.
- [ ] **Bộ dữ liệu chuẩn (Automotive Benchmark Suite):** Tập tài liệu mẫu và Ground Truth phục vụ kiểm định chất lượng.
- [ ] **Bộ công cụ đánh giá tự động (Evaluation Suite):** Script đo lường Kendall Tau, TEDS, F1-score và Precision.
- [ ] **Giao diện người dùng (Interactive Demo UI):** Web app cho phép upload bộ tài liệu, duyệt cấu trúc, tương tác sơ đồ và xem cảnh báo mâu thuẫn.
- [ ] **Tài liệu kiến trúc & Báo cáo kỹ thuật:** Sơ đồ kiến trúc phần mềm, luồng dữ liệu, phân tích năng lực chịu tải và hướng phát triển tương lai.

---

## 10. Quản trị rủi ro kỹ thuật & Giải pháp dự phòng [IMPL]

| Rủi ro kỹ thuật | Mức độ | Giải pháp kiểm soát |
|---|---|---|
| PDF layout quá phức tạp, mất cấu trúc | Cao | Kết hợp PyMuPDF trích xuất text/vector với model layout chuyên sâu (Docling); ghi rõ các trường hợp biên trong tài liệu limitations. |
| Mũi tên diagram bị đứt đoạn hoặc chồng chéo | Cao | Đặt ngưỡng khoảng cách (distance threshold) và góc phương vị (ray-angle) để liên kết; gán chỉ số `confidence` cho mỗi quan hệ cạnh được dựng. |
| Mâu thuẫn bị báo động giả (False Positives) | Cao | Áp dụng cơ chế lọc ngữ cảnh nghiêm ngặt: chỉ so sánh khi trùng khớp định danh thực thể và chuẩn hóa về cùng hệ đơn vị đo lường. |
| Định dạng tài liệu phong phú gây vỡ Schema | Trung bình | Tách lớp Adapter riêng biệt cho từng định dạng file trước khi đưa vào lõi IR thống nhất. |
| Quá tải thời gian xử lý với file nhiều trang | Trung bình | Cơ chế xử lý phân trang/phân luồng bất đồng bộ (Async/Parallel processing) và caching kết quả IR. |

---

## 11. Các quyết định thiết kế nhóm chủ động xác lập

1. **Chuẩn hóa Schema IR:** Dùng format JSON phẳng kèm quan hệ cha con (`parent_id`) và danh sách cạnh (`edges`) để vừa dễ biểu diễn dạng cây vừa dễ biểu diễn dạng đồ thị (Graph).
2. **Quy ước biểu diễn Source Reference:**
   - PDF: `{ "file": string, "page": int, "bbox": [x0, y0, x1, y1] }`
   - DOCX: `{ "file": string, "paragraph_idx": int, "style": string }`
   - XLSX: `{ "file": string, "sheet": string, "cell": string, "merged_range": string }`
   - PPTX: `{ "file": string, "slide": int, "shape_name": string, "bbox": [x0, y0, x1, y1] }`
3. **Phân loại mâu thuẫn kỹ thuật (Consistency Taxonomy):**
   - `numerical_conflict`: Sai lệch thông số định lượng (áp suất, mô-men, kích thước, điện áp).
   - `version_mismatch`: Sai lệch số hiệu phiên bản phần mềm/phần cứng hoặc linh kiện.
   - `procedural_conflict`: Mâu thuẫn thứ tự các bước trong quy trình kiểm định/bảo dưỡng.
4. **Phối hợp công nghệ:** Tối đa hóa việc đọc trực tiếp XML/vector (tốc độ nhanh, chính xác 100%, chi phí tính toán thấp), chỉ kích hoạt Multimodal LLM cho các tác vụ suy luận cấp cao (suy luận mâu thuẫn ngữ nghĩa và giải mã sơ đồ phức tạp).
