# Quy Ước Tọa Độ OOXML (DrawingML): EMU, Điểm và Biến Đổi Nhóm

Tài liệu này mô tả module `geometry/transform.py` và bộ test `geometry/test_transform.py`: cách chuyển tọa độ shape, connector và group trong PPTX (đơn vị EMU) về một hệ tọa độ slide duy nhất (đơn vị điểm, trục y hướng xuống).

---

## 1. Mục đích và vị trí trong dự án

- Đây là nền tảng cho **P3-07 Node–Edge Graph Builder** (đề cương Topic3 của dự án): connector trong PPTX cần điểm đầu và điểm cuối nằm trong cùng hệ tọa độ slide để nối với node.
- Module **chưa được gắn vào pipeline nào**. Nó không đọc file PPTX, không dựng đồ thị, và chưa module nào khác import nó.
- Chỉ dùng thư viện chuẩn: không dùng python-pptx hay lxml.

---

## 2. Đơn vị đầu vào và đầu ra

| Đại lượng | Đơn vị | Ghi chú |
|---|---|---|
| `off`, `ext`, `chOff`, `chExt` | EMU (số nguyên) | 12 700 EMU = 1 pt; 914 400 EMU = 1 inch |
| `rot` | 1/60 000 độ | 21 600 000 = 360 độ; dương = theo chiều kim đồng hồ (hệ y hướng xuống) |
| `flipH`, `flipV` | cờ | bool, hoặc `"1"`, `"true"`, `"0"`, `"false"`, `""`, `None` |
| Mọi tọa độ, kích thước đầu ra | pt (float) | làm tròn 6 chữ số thập phân |
| Góc đầu ra (`rotation_deg`) | độ, trong [0, 360) | |

Hằng số: `EMU_PER_POINT = 12_700`, `EMU_PER_INCH = 914_400`, `ROT_UNITS_PER_DEGREE = 60_000`, `FULL_TURN_UNITS = 21_600_000`.

Chuyển đổi: `emu_to_pt(v)` nhận int, float hoặc chuỗi số và trả về float. `pt_to_emu(v)` làm tròn về số nguyên gần nhất.

---

## 3. Chuẩn hóa đầu vào: `Xfrm`

`Xfrm` là dataclass đông cứng (`frozen`). `__post_init__` chuẩn hóa mọi trường để hai đầu vào tương đương so sánh bằng nhau:

- **EMU:** nhận int hoặc chuỗi số như `"12700"`. Số thực không nguyên (ví dụ `1.5`) gây `ValueError`. Giá trị `bool` bị từ chối.
- **rot:** đưa về `rot % 21_600_000`. Vì vậy `0`, `21600000` và `-21600000` đều thành 0. `-5400000` và `16200000` đều thành `16200000` (270 độ).
- **Cờ lật:** bool, `None`, hoặc các chuỗi kể trên. Giá trị khác gây `ValueError`.
- **chOff, chExt** thiếu thì mặc định 0.

Đọc từ XML: `Xfrm.from_element(el)` nhận phần tử `a:xfrm`, `p:xfrm`, hoặc `a:grpSpPr` (khi đó lấy `a:xfrm` con trực tiếp). Các thẻ được khớp theo tên cục bộ (bỏ phần namespace), nên mọi tiền tố namespace đều hoạt động.

---

## 4. Công thức

### 4.1 Ma trận affine

Ma trận 2x3 biểu diễn phép biến đổi:

```
x' = a·x + c·y + e
y' = b·x + d·y + f
```

Ký hiệu `A @ B` nghĩa là áp dụng `B` trước, rồi `A`.

Phép xoay thuận chiều kim đồng hồ trong hệ y hướng xuống có `a = cos θ`, `b = sin θ`, `c = −sin θ`, `d = cos θ`. Với bội số 90 độ, cos và sin được gán đúng giá trị (0 hoặc ±1), không có nhiễu kiểu 6e−17.

### 4.2 Khung của một shape

`frame(xfrm)` đưa điểm cục bộ `(0..w, 0..h)` (gốc tại góc trên trái chưa xoay) về không gian cha:

```
frame(xfrm) = T(tâm) @ R(rot) @ Flip @ T(−w/2, −h/2)
```

- `w = ext_cx / 12700`, `h = ext_cy / 12700` (pt).
- `tâm = off + ext/2`.
- `Flip = scale(−1 nếu flipH else 1, −1 nếu flipV else 1)`.

**Thứ tự trong một shape:** lật bên trong hộp trước, rồi xoay quanh tâm hộp. Xem mục 12(b) về trạng thái xác minh của thứ tự này.

### 4.3 Nhóm: ma trận thuần và quy tắc từng hộp

**Ma trận thuần (đường tham chiếu).** `group_matrix` và `compose_groups` giữ nguyên hành vi cũ:

```
group_matrix(xfrm) = frame(group) @ Scale(sx, sy) @ T(−chOff)
```

- `sx = ext_cx / chExt_cx`, `sy = ext_cy / chExt_cy`. Nếu `chExt` trên một trục bằng 0 thì hệ số trên trục đó là 1.0.
- Ma trận này chỉ cho cùng kết quả với quy tắc bên dưới khi tỷ lệ đều (`sx = sy`), hoặc con không xoay, hoặc con xoay đúng bội số 90 độ. Với tỷ lệ không đều và con xoay góc khác, tích ma trận làm con bị nghiêng (shear). Vì vậy `resolve_shape`, `resolve_connector` và `resolve_tree` không dùng tích này trong trường hợp đó.

**Quy tắc từng hộp (box-by-box).** `resolve_*` tính khung hiệu lực `E` của từng nút X, theo thứ tự từ ngoài vào trong, đơn vị pt:

1. **Nút cấp cao nhất** (không có nhóm cha): `E` là khung riêng của X, không đổi.
2. **Nút con của nhóm cha P.** Gọi `(Wp, Hp)` là kích thước hiệu lực của P (kích thước `E` của P, đã qua co giãn và đổi chiều). `chOff` và `chExt` của P được đổi sang pt.
   - `sx = Wp / chExt_x` nếu `chExt_x ≠ 0`, ngược lại `sx = 1.0`. `sy` làm tương tự với `Hp`.
   - Tâm của X trong không gian con (dùng giá trị thô của X): `c = (off_x + w/2, off_y + h/2)`.
   - Tâm trong hệ cục bộ của P, trước khi P xoay: `c_local = ((c_x − chOff_x)·sx, (c_y − chOff_y)·sy)`.
   - `deg = rot // 60000`, với `rot` đã chuẩn hóa vào [0, 21 600 000). Đây là **dải đổi chiều** nếu `45 ≤ deg < 135` hoặc `225 ≤ deg < 315`.
   - Trong dải: `w' = w·sy`, `h' = h·sx`. Ngoài dải: `w' = w·sx`, `h' = h·sy`.
   - `E` có `off = c_local − (w'/2, h'/2)`, `ext = (w', h')`, cùng `rot`, `flipH`, `flipV` với X.
3. **Ma trận đặt nút:** `M = M_cha @ frame(E)`, với `M` khởi đầu là đơn vị. `frame(E)` theo công thức mục 4.2 với kích thước `(w', h')`. Các con của X được tính với `E` của X làm cha.

Lưu ý:
- Cha dùng kích thước hiệu lực `(Wp, Hp)`, không dùng `ext` thô. Khi cha nằm trong một nhóm co giãn không đều, hai giá trị này khác nhau.
- Dải chỉ xét `rot` của chính X, không cộng `rot` của tổ tiên. Co giãn diễn ra trong hệ cục bộ của P trước khi P xoay, nên góc cần xét là góc của X so với P.
- Các đầu ra `corners`, `bbox`, `center`, `width`, `height` được tính từ `M` với hộp cục bộ `(w', h')`, không từ `ext` thô.

### 4.4 Nhóm lồng nhau

Với tỷ lệ đều, kết quả của quy tắc từng hộp trùng với tích ma trận thuần, ghép theo thứ tự từ ngoài vào trong (xem test `test_uniform_scale_rotated_child_equals_pure_matrix`):

```
Total = G_outer @ ... @ G_inner @ frame(shape)
```

`compose_groups(ancestors)` trả về `G_outer @ ... @ G_inner` (tổ tiên truyền vào theo thứ tự ngoài trước). Hàm này là đường tham chiếu, không phải đường tính của `resolve_*`.

`resolve_tree` đi theo cây một lượt, giữ ma trận đang chạy và không gian con của nhóm gần nhất, thay vì tính lại cả chuỗi tổ tiên cho từng nút. Hộp của một nhóm là khung hiệu lực của nhóm đó (mục 4.3), đặt trong không gian của nhóm cha. Các con của nhóm được tính với khung hiệu lực đó. Kết quả giống hệt khi gọi `resolve_shape(xfrm, chuỗi tổ tiên)`.

### 4.5 Phân rã chính tắc

Từ phần tuyến tính `(a, b, c, d)` của ma trận tổng:

- Nếu `det = a·d − b·c < 0` (có lật): `flip_h = True` và `rotation = atan2(−c, d)`.
- Ngược lại: `flip_h = False` và `rotation = atan2(b, a)`.
- `flip_v` luôn là `False`. Lật dọc tương đương với lật ngang cộng xoay 180 độ.

Hệ quả:
- `flipH + flipV` với `rot = 0` và `rot = 180` (không lật) cho cùng một kết quả.
- Chỉ `flipV` cho `flip_h = True` và `rotation_deg = 180`.

---

## 5. Kiểu dữ liệu đầu ra

- **`ResolvedShape`** (frozen):
  - `corners`: 4 điểm theo thứ tự TL, TR, BR, BL (góc cục bộ đã biến đổi, trong hệ slide).
  - `bbox`: `(min_x, min_y, max_x, max_y)`, hộp bao theo trục.
  - `center`: tâm sau biến đổi.
  - `width`, `height`: độ dài ảnh của trục x và y cục bộ, nhân với kích thước hiệu lực `w'` và `h'` (mục 4.3). Công thức: `hypot(a, b)·w'` và `hypot(c, d)·h'`. Phép xoay và lật không co giãn, nên giá trị này bằng đúng `w'` và `h'`.
  - `rotation_deg` trong [0, 360).
  - `flip_h`, `flip_v` (`flip_v` luôn `False`).
- **`ResolvedConnector`** (frozen): `start`, `end`, `box` (một `ResolvedShape`).
- **`ShapeNode(xfrm, name="", is_connector=False)`** và **`GroupNode(xfrm, children=(), name="")`** (frozen). `children` được chuyển thành tuple.
- **`resolve_tree(nodes)`** trả về danh sách `(node, resolved)` theo thứ tự pre-order. Mỗi nhóm được phát ra cùng hộp hiệu lực của chính nó (`ResolvedShape`, đặt trong không gian của nhóm cha theo mục 4.3) trước các con của nó.

---

## 6. Làm tròn và tất định

- Mọi số đầu ra được làm tròn 6 chữ số thập phân. Số `-0.0` đổi thành `0.0` (cộng `+0.0` sau khi làm tròn).
- `rotation_deg` lấy modulo 360 trước khi làm tròn. Nếu kết quả làm tròn bằng 360.0 thì đổi thành 0.0, để giữ khoảng [0, 360).
- Không có phần nào phụ thuộc thứ tự của dict hay set. Các đầu vào tương đương (sau chuẩn hóa) cho kết quả giống hệt nhau.
- `corners`, `bbox` và `children` là tuple, để so sánh `==` hoạt động đúng với dataclass đông cứng.

---

## 7. Quy tắc cho connector

Connector là đoạn thẳng từ điểm cục bộ `(0, 0)` đến `(w, h)`, tức góc trên trái đến góc dưới phải của hộp `ext`. Lật và xoay đã nằm trong ma trận, nên:

- `start = M(0, 0)` và `end = M(w', h')`, với `(w', h')` là kích thước hiệu lực của hộp cục bộ (mục 4.3). Với shape không nằm trong nhóm, `(w', h') = (w, h)`.
- Connector có `cx = 0` hoặc `cy = 0` (đường ngang hoặc đường dọc thuần) vẫn hợp lệ.
- Luôn có `start == box.corners[0]` và `end == box.corners[2]`.

---

## 8. Trường hợp suy biến

- **chExt = 0 trên một trục:** hệ số co giãn trên trục đó là 1.0 (không co giãn). Slide-root `grpSpPr` có mọi giá trị bằng 0 là phép đồng nhất.
- **ext = 0 nhưng chExt ≠ 0:** hệ số co giãn là 0 và các con co về một điểm. Không có ngoại lệ, không chia cho 0, hộp có kích thước 0. Góc xoay và cờ lật của con vẫn được giữ, vì chúng không đi qua hệ số co giãn. Khi đó `rotation_deg` của con là góc riêng của con, không suy biến về 0 như trước. Đây là một giả định (xem mục 12(d)).
- **Nhóm có chOff = off, chExt = ext, không xoay, không lật:** là phép đồng nhất, cho kết quả giống hệt khi không có nhóm.

---

## 9. Ví dụ tính tay

Nhóm: `off(0,0)`, `ext(100,100)`, `chOff(0,0)`, `chExt(100,100)`, `rot = 90°`, `flipH`. Shape con: `off(0,0)`, `ext(20,10)`. Đơn vị là pt, trục y hướng xuống.

1. Hệ số co giãn của nhóm là 1 (chExt = ext) và chOff = 0, nên ma trận nhóm chỉ còn là `frame(nhóm)`.
2. Tâm nhóm là `(50, 50)`. Với điểm `p`, đặt `v = p − (50, 50)`. Lật ngang làm `v_x → −v_x`, rồi xoay 90 độ. Phần tuyến tính tổng là `[[0, −1], [−1, 0]]`, nên `p → (100 − p_y, 100 − p_x)`.
3. Bốn góc của con:
   - `(0,0) → (100, 100)`
   - `(20,0) → (100, 80)`
   - `(20,10) → (90, 80)`
   - `(0,10) → (90, 100)`
4. Kết quả: `bbox = (90, 80, 100, 100)`, `flip_h = True` (vì `det = −1`), `rotation_deg = atan2(1, 0) = 90`.

Nếu đảo thứ tự (xoay trước, lật sau), các góc con sẽ là `(0,0)`, `(0,20)`, `(10,20)`, `(10,0)`, tức `bbox = (0, 0, 10, 20)`. Test `test_flip_then_rotate_order` kiểm tra đúng trường hợp này.

**Ví dụ tương đương (tham chiếu nhanh):**
- `rot = 0`, `21600000`, `-21600000` là giống nhau.
- `-5400000`, `16200000`, `37800000` là giống nhau (đều là 270 độ).
- `flipH + flipV` với `rot = 0` giống `rot = 10800000` không lật.
- Lưu ý: `27000000 % 21600000 = 5400000` (90 độ), không phải 270 độ. Test ghim giá trị này.

---

## 10. Cách chạy test

```powershell
.venv\Scripts\python geometry\test_transform.py
```

Chạy thành công khi thấy `25/25 passed`. Test dùng `assert` thuần, không cần pytest. Các giá trị kỳ vọng được tính tay trong đơn vị pt. Chúng chứng minh tính nhất quán nội bộ với quy ước, không chứng minh khớp với PowerPoint (xem mục 12(a)).

---

## 11. Ngoài phạm vi

Các mục sau **không** có trong module này:

- Kế thừa `xfrm` từ placeholder và layout (vị trí ghi đè trên slide master và layout).
- Điểm nối `stCxn` và `endCxn` của connector: cần hình học preset (vị trí điểm nối theo từng dạng hình).
- Đường đi tùy biến `custGeom`.
- Đọc file PPTX (gói zip và các part XML).
- Dựng đồ thị node–edge (P3-07, công việc sau này).

---

## 12. Giới hạn và giả định

**(a) Fixture tính tay.** Các giá trị kỳ vọng trong `geometry/test_transform.py` được tính tay từ công thức. Chúng chứng minh tính nhất quán nội bộ, không chứng minh đầu ra khớp với PowerPoint thật. Repo hiện không có file PPTX thật nào để đối chiếu.

**(b) Thứ tự lật rồi xoay: có hai triển khai độc lập ủng hộ, nhưng chưa được đặc tả hay PowerPoint thật xác nhận.** Các điểm dưới đây rút ra từ việc đọc mã nguồn LibreOffice (nhánh `master`) và Apache POI (nhánh `trunk`) qua web vào ngày 2026-11-13, không ghim commit, nên có thể khác ở bản phát hành khác. Phần LibreOffice chỉ đọc nguyên văn khoảng 120.000 trên 158.000 ký tự đầu của `shape.cxx`; đoạn xử lý biến đổi nằm trong phần đã đọc nguyên văn. Cách hiểu thứ tự áp dụng của `Graphics2D` là suy luận từ hành vi chuẩn của Java2D, không được ghi trong chính mã nguồn POI.
- LibreOffice, `oox/source/drawingml/shape.cxx`: comment trong mã nói rằng OOXML lật shape trước khi xoay nó.
- Apache POI XSLF, `DrawShape.applyTransform`: gọi xoay trước, rồi gọi lật ngang và lật dọc trên `Graphics2D`. Do thứ tự áp dụng của `Graphics2D`, hình học được lật trước rồi mới xoay.
- Trang [MS-OI29500, Part 1 §20.1.7.6 (xfrm)](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-oi29500/c7555d2c-472a-4d15-b230-41046bc41819) xác nhận rằng trong Office, `rot` là xoay theo chiều kim đồng hồ, và `flipH`, `flipV`, `rot` bị bỏ qua khi áp cho graphicFrame. Trang này **không nói gì về thứ tự lật so với xoay**. Tài liệu Open XML SDK và python-pptx cũng không nói về thứ tự này.

Vì vậy thứ tự "lật trong hộp, rồi xoay quanh tâm" là **quy ước được hai triển khai ủng hộ, chưa được xác nhận bằng đặc tả**. Cần kiểm tra bằng ISO/IEC 29500-1 §20.1.7.6, hoặc so sánh kết quả render một shape có cả `flipH` và `rot` khác 0 trong PowerPoint và LibreOffice.

**(c) Nhóm co giãn không đều kèm con bị xoay: quy tắc từng hộp, là một quy ước đã chọn, chưa kiểm chứng với PowerPoint.** Nguồn đọc theo cùng cách và cùng giới hạn như mục (b) (LibreOffice `master`, Apache POI `trunk`, ngày 2026-11-13, không ghim commit).
- Module co giãn hộp của con khi chưa xoay, rồi mới xoay quanh tâm của hộp đã co giãn, nên không có shear. Comment trong LibreOffice (`oox/source/drawingml/shape.cxx`) ghi đúng ý này: "First scale object then rotate. Other way round would introduce shearing." Apache POI cũng co giãn hình chữ nhật neo của con rồi xoay quanh tâm.
- Dải đổi chiều (mục 4.3): khi góc `rot` chuẩn hóa của chính con, tính theo độ nguyên, nằm trong [45, 135) hoặc [225, 315), chiều rộng và chiều cao được đổi chỗ trước khi co giãn. Quy tắc này theo `lcl_doSpecialMSOWidthHeightToggle` của LibreOffice. Comment trong mã đó nói MSO hành xử bất thường ở chỗ này. Apache POI có một trường hợp đặc biệt khác, dựa trên thử và sai, cho cùng các dải góc.
- Đây là một **quy ước đã chọn**, không phải hành vi đã xác minh. Ranh giới dùng cắt cụt số độ nguyên của `rot` chuẩn hóa: ví dụ 44 độ 59 999 đơn vị vẫn thuộc 44 độ và không đổi chiều. Chưa kiểm chứng với PowerPoint thật.
- **Chưa mô hình hóa:**
  - LibreOffice còn xoay thêm 180 độ cho con trong dải khi nhóm cha có đúng một trục bị lật (`flipH` xor `flipV`). Module không làm điều này.
  - Lật của chính con khi nhóm có lật: theo mục 4, tức lật con trong hộp của nó, xoay, rồi áp dụng lật và xoay của nhóm ở bên ngoài.
  - Không phải thiếu sót mà là lựa chọn có chủ ý: xoay của tổ tiên không dùng để xét dải, vì co giãn diễn ra trước khi cha xoay (xem mục 4.3).

**(d) Nhóm có ext = 0 nhưng chExt ≠ 0.** Các con co về một điểm (hệ số 0). Hành vi này được giữ nguyên và có test không lỗi, nhưng chưa được đối chiếu với PowerPoint.

**(e) Test quy ước.** `test_convention_group_rot_and_child_rot` (nhóm xoay 60 độ, con xoay 30 độ, kết quả 90 độ) ghim quy ước cộng góc của module khi tỷ lệ đều. Các test tỷ lệ không đều (`test_nonuniform_group_rotated_child_not_sheared`, `test_nonuniform_group_right_angle_child_band_swap`, `test_band_boundaries_nonuniform_group`, `test_nested_nonuniform_groups_hand_computed`, `test_band_group_inside_nonuniform_group`) ghim quy tắc từng hộp và quy ước dải của mục 4.3 và 12(c). Các test `test_uniform_scale_rotated_child_equals_pure_matrix`, `test_rot_0_and_180_children_equal_pure_matrix`, `test_connector_nonuniform_group_endpoints` và `test_tree_matches_chain_nonuniform` kiểm tra tính nhất quán nội bộ. Tất cả đều **không phải** sự thật về PowerPoint.
