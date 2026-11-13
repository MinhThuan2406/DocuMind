# OOXML (DrawingML) Coordinate Conventions: EMU, Points and Group Transforms

This document describes the `geometry` package, made of the module `geometry/transform.py` and the test suite `geometry/test_transform.py`. It converts the coordinates of shapes, connectors and groups in PPTX files (EMU units) into a single slide coordinate system (points, y axis pointing down).

---

## 1. Purpose and place in the project

- It is the foundation for **P3-07 Node-Edge Graph Builder** (the project's Topic3 plan): a connector in PPTX needs its start and end points in the same slide coordinate system as the nodes it joins.
- The module is **not wired into any pipeline yet**. It does not read PPTX files, does not build graphs, and no other module imports it.
- Standard library only: no python-pptx and no lxml.

---

## 2. Input and output units

| Quantity | Unit | Notes |
|---|---|---|
| `off`, `ext`, `chOff`, `chExt` | EMU (integers) | 12,700 EMU = 1 pt; 914,400 EMU = 1 inch |
| `rot` | 1/60,000 degree | 21,600,000 = 360 degrees; positive = clockwise (y axis down) |
| `flipH`, `flipV` | flags | bool, or `"1"`, `"true"`, `"0"`, `"false"`, `""`, `None` |
| All output coordinates and sizes | pt (float) | rounded to 6 decimal places |
| Output angle (`rotation_deg`) | degrees, in [0, 360) | |

Public constant: `EMU_PER_POINT = 12_700`.

Conversion: `emu_to_pt(v)` takes a number and returns `v / 12_700` as a float, without rounding.

---

## 3. Input normalization: `Xfrm`

`Xfrm` is a frozen dataclass. `__post_init__` normalizes every field so that equivalent inputs compare equal:

- **EMU:** accepts an int, an integral float, or a numeric string such as `"12700"`. A non-integral float (for example `1.5`) or a string that is not an integer raises `ValueError`. A `bool` raises `TypeError`.
- **rot:** reduced to `rot % 21_600_000`. So `0`, `21600000` and `-21600000` all become 0, and `-5400000` and `16200000` both become `16200000` (270 degrees).
- **Flip flags:** bool, `None`, or the strings listed above. Anything else raises `ValueError`.
- **chOff, chExt** default to 0 when missing.

Reading from XML: `Xfrm.from_element(el)` accepts an `a:xfrm` element, a `p:xfrm` element, or an `a:grpSpPr` element (in which case its direct `a:xfrm` child is used). Tags are matched by local name (the namespace part is dropped), so any namespace prefix works.

---

## 4. Formulas

### 4.1 Affine matrices

A 2x3 matrix represents a transform:

```
x' = a·x + c·y + e
y' = b·x + d·y + f
```

`A @ B` means apply `B` first, then `A`.

A clockwise rotation in a y-down system has `a = cos θ`, `b = sin θ`, `c = −sin θ`, `d = cos θ`. Floating point noise of about 1e−16 (for example `cos 90° ≈ 6e−17`) is removed by the output rounding step (section 6).

### 4.2 The frame of a shape

`frame(xfrm)` maps local points `(0..w, 0..h)` (origin at the unrotated top-left corner) into the parent space:

```
frame(xfrm) = T(center) @ R(rot) @ Flip @ T(−w/2, −h/2)
```

- `w = ext_cx / 12700`, `h = ext_cy / 12700` (pt).
- `center = off + ext/2`.
- `Flip = scale(−1 if flipH else 1, −1 if flipV else 1)`.

**Order within one shape:** flip inside the box first, then rotate about the box center. See section 12(b) for how well this order is verified.

### 4.3 Groups: the box-by-box rule

The plain matrix product `frame(group) @ Scale(sx, sy) @ T(−chOff)` is only correct when the scale is uniform (`sx = sy`), or the child is not rotated, or the child is rotated by an exact multiple of 90 degrees. With a non-uniform scale and any other child rotation, the matrix product shears the child. The module therefore does not use that product and applies the box-by-box rule below instead. In the cases listed first, both approaches give the same result.

**Box-by-box rule.** `resolve_*` computes the effective frame `E` of every node X, from the outside in, in points:

1. **Top-level node** (no parent group): `E` is X's own frame, unchanged.
2. **Child of a parent group P.** Let `(Wp, Hp)` be the effective size of P (the size of P's `E`, after scaling and axis swapping). P's `chOff` and `chExt` are converted to pt.
   - `sx = Wp / chExt_x` if `chExt_x ≠ 0`, otherwise `sx = 1.0`. `sy` is computed the same way with `Hp`.
   - The center of X in child space (using X's raw values): `c = (off_x + w/2, off_y + h/2)`.
   - The center in P's local space, before P rotates: `c_local = ((c_x − chOff_x)·sx, (c_y − chOff_y)·sy)`.
   - `deg = rot // 60000`, with `rot` already normalized into [0, 21,600,000). The node is in the **axis-swap band** if `45 ≤ deg < 135` or `225 ≤ deg < 315`.
   - Inside the band: `w' = w·sy`, `h' = h·sx`. Outside the band: `w' = w·sx`, `h' = h·sy`.
   - `E` has `off = c_local − (w'/2, h'/2)`, `ext = (w', h')`, and the same `rot`, `flipH`, `flipV` as X.
3. **Placement matrix:** `M = M_parent @ frame(E)`, where `M` starts as the identity. `frame(E)` follows section 4.2 with size `(w', h')`. The children of X are computed with X's `E` as their parent.

Notes:
- The parent uses its effective size `(Wp, Hp)`, not its raw `ext`. When the parent itself sits in a non-uniformly scaled group, the two differ.
- The band only looks at X's own `rot`, not the sum of its ancestors' rotations. Scaling happens in P's local space before P rotates, so the relevant angle is X's angle relative to P.
- The outputs `corners`, `bbox`, `center`, `width` and `height` are computed from `M` with the local box `(w', h')`, not from the raw `ext`.
- A slide-root `grpSpPr` has every value equal to 0, which gives `sx = sy = 1` and `chOff = 0`, that is, the identity.

### 4.4 Nested groups

With a uniform scale, the box-by-box result equals the plain matrix product chained from the outside in (see the test `test_uniform_scale_matches_plain_matrix_composition`):

```
Total = G_outer @ ... @ G_inner @ frame(shape)
```

`resolve_shape(xfrm, ancestors)` takes the ancestor chain outermost first. `resolve_tree` walks the tree once, carrying the running matrix and the child space of the nearest group, instead of recomputing the whole ancestor chain for every node. A group's box is the effective frame of that group (section 4.3), placed in the space of its parent group, and the group's children are computed from that effective frame. The result is identical to calling `resolve_shape(xfrm, ancestor chain)`. The `ancestors` argument of `resolve_tree` lets it start from a subtree that already sits inside groups.

### 4.5 Canonical decomposition

From the linear part `(a, b, c, d)` of the total matrix:

- If `det = a·d − b·c < 0` (a mirror is present): `mirrored = True` and `rotation = atan2(−c, d)`.
- Otherwise: `mirrored = False` and `rotation = atan2(b, a)`.
- A vertical flip equals a horizontal flip plus a 180 degree rotation, so a single `mirrored` flag is enough.

Consequences:
- `flipH + flipV` with `rot = 0` and `rot = 180` (no flips) give the same result.
- `flipV` alone gives `mirrored = True` and `rotation_deg = 180`.

---

## 5. Output types

The public interface of the `geometry` package is the list `geometry.__all__`: `EMU_PER_POINT`, `emu_to_pt`, `Xfrm`, `ResolvedShape`, `ResolvedConnector`, `ShapeNode`, `GroupNode`, `resolve_shape`, `resolve_connector`, `resolve_tree`. Every other name is internal.

- **`ResolvedShape`** (frozen):
  - `corners`: 4 points in the order TL, TR, BR, BL (the transformed local corners, in slide space).
  - `bbox`: `(min_x, min_y, max_x, max_y)`, the axis-aligned bounding box.
  - `center`: the center after the transform.
  - `width`, `height`: the effective sizes `w'` and `h'` of the local box (section 4.3). The matrix only holds rotation, mirroring and translation, and scaling is already part of the box size, so these two values do not change under rotation and flipping.
  - `rotation_deg` in [0, 360).
  - `mirrored`: whether the shape is mirrored (section 4.5).
- **`ResolvedConnector`** (frozen): `start`, `end` and `box` (a `ResolvedShape`). `start` always equals `box.corners[0]` and `end` always equals `box.corners[2]`.
- **`ShapeNode(xfrm, name="", is_connector=False)`** and **`GroupNode(xfrm, children=(), name="")`** (frozen). `children` is converted to a tuple.
- **`resolve_tree(nodes)`** returns a list of `(node, resolved)` in pre-order. Every group is emitted with its own effective box (a `ResolvedShape`, placed in its parent group's space according to section 4.3) before its children.

---

## 6. Rounding and determinism

- Every output number is rounded to 6 decimal places. `-0.0` becomes `0.0` (by adding `+0.0` after rounding).
- `rotation_deg` is taken modulo 360 before rounding. If the rounded value equals 360.0 it becomes 0.0, so the range stays [0, 360).
- Nothing depends on the ordering of a dict or a set. Equivalent inputs (after normalization) give identical results.
- `corners`, `bbox` and `children` are tuples, so `==` works correctly on the frozen dataclasses.

---

## 7. Connector rules

A connector is a straight segment from the local point `(0, 0)` to `(w, h)`, that is, from the top-left to the bottom-right corner of its `ext` box. Flipping and rotation are already in the matrix, so:

- `start = M(0, 0)` and `end = M(w', h')`, where `(w', h')` is the effective size of the local box (section 4.3). For a shape that is not inside a group, `(w', h') = (w, h)`.
- A connector with `cx = 0` or `cy = 0` (a purely horizontal or vertical line) is valid.
- `start` and `end` always equal `box.corners[0]` and `box.corners[2]`.

---

## 8. Degenerate cases

- **chExt = 0 on one axis:** the scale factor on that axis is 1.0 (no scaling). A slide-root `grpSpPr` with all values 0 is the identity.
- **ext = 0 but chExt ≠ 0:** the scale factor is 0 and the children collapse to a point. There is no exception and no division by zero, and the box has size 0. The child's rotation and flip flags are kept, because they do not go through the scale factor, so the child's `rotation_deg` is its own angle. This is an assumption (see section 12(d)).
- **A group with chOff = off, chExt = ext, no rotation and no flip:** it is the identity and gives exactly the same result as no group.

---

## 9. Worked example

Group: `off(0,0)`, `ext(100,100)`, `chOff(0,0)`, `chExt(100,100)`, `rot = 90°`, `flipH`. Child shape: `off(0,0)`, `ext(20,10)`. Units are pt, y axis down.

1. The group's scale factor is 1 (chExt = ext) and chOff = 0, so the group matrix reduces to `frame(group)`.
2. The group center is `(50, 50)`. For a point `p`, let `v = p − (50, 50)`. The horizontal flip maps `v_x → −v_x`, then the 90 degree rotation applies. The total linear part is `[[0, −1], [−1, 0]]`, so `p → (100 − p_y, 100 − p_x)`.
3. The four corners of the child:
   - `(0,0) → (100, 100)`
   - `(20,0) → (100, 80)`
   - `(20,10) → (90, 80)`
   - `(0,10) → (90, 100)`
4. Result: `bbox = (90, 80, 100, 100)`, `mirrored = True` (because `det = −1`), `rotation_deg = atan2(1, 0) = 90`.

If the order were reversed (rotate first, flip second), the child corners would be `(0,0)`, `(0,20)`, `(10,20)`, `(10,0)`, that is, `bbox = (0, 0, 10, 20)`. The test `test_flip_is_applied_before_rotation` checks exactly this case.

**Equivalent inputs (quick reference):**
- `rot = 0`, `21600000` and `-21600000` are the same.
- `-5400000`, `16200000` and `37800000` are the same (all 270 degrees).
- `flipH + flipV` with `rot = 0` equals `rot = 10800000` with no flips.
- Note: `27000000 % 21600000 = 5400000` (90 degrees), not 270 degrees. A test pins this value.

---

## 10. Running the tests

From the repository root:

```powershell
.venv\Scripts\python -m unittest geometry.test_transform
```

The run succeeds when it ends with `OK` (41 tests). The tests use the standard library `unittest`, so pytest is not required. The expected values are computed by hand in pt units. They prove internal consistency with the conventions, not agreement with PowerPoint (see section 12(a)).

The tests that compare against the plain matrix product use a reference function written separately inside the test file (direct formulas, without reusing the module's code).

---

## 11. Out of scope

The following are **not** part of this module:

- Inheriting `xfrm` from placeholders and layouts (positions overridden on the slide master and layout).
- The connection sites `stCxn` and `endCxn` of connectors: they need preset geometry (the connection point positions of each shape type).
- Custom paths in `custGeom`.
- Reading PPTX files (the zip package and its XML parts).
- Building the node-edge graph (P3-07, later work).

---

## 12. Limits and assumptions

**(a) Hand-computed fixtures.** The expected values in `geometry/test_transform.py` are computed by hand from the formulas. They prove internal consistency, not that the output matches real PowerPoint. The repository currently holds no real PPTX file to compare against.

**(b) Flip-then-rotate order: supported by two independent implementations, but not confirmed by the specification or by real PowerPoint.** The points below come from reading the LibreOffice source (`master` branch) and Apache POI (`trunk` branch) on the web on 2026-11-13, with no commit pinned, so they may differ in other releases. Only about 120,000 of the first 158,000 characters of LibreOffice's `shape.cxx` were read verbatim; the transform handling is inside the part read verbatim. The reading of the `Graphics2D` application order is an inference from standard Java2D behavior and is not stated in the POI source itself.
- LibreOffice, `oox/source/drawingml/shape.cxx`: a code comment says that OOXML flips shapes before rotating them.
- Apache POI XSLF, `DrawShape.applyTransform`: calls rotate first, then flip horizontal and flip vertical on `Graphics2D`. Because of the order in which `Graphics2D` applies them, the geometry is flipped first and rotated afterwards.
- The page [MS-OI29500, Part 1 §20.1.7.6 (xfrm)](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-oi29500/c7555d2c-472a-4d15-b230-41046bc41819) confirms that in Office `rot` is a clockwise rotation, and that `flipH`, `flipV` and `rot` are ignored when applied to a graphicFrame. That page **says nothing about the order of flip relative to rotation**. The Open XML SDK documentation and python-pptx do not state this order either.

So the order "flip inside the box, then rotate about the center" is a **convention supported by two implementations and not confirmed by the specification**. It should be checked against ISO/IEC 29500-1 §20.1.7.6, or by comparing how PowerPoint and LibreOffice render a shape that has both `flipH` and a non-zero `rot`.

**(c) Non-uniform group scale with a rotated child: the box-by-box rule is a chosen convention, not verified against PowerPoint.** The sources were read the same way and with the same limits as in (b) (LibreOffice `master`, Apache POI `trunk`, 2026-11-13, no commit pinned).
- The module scales the child's box while it is still unrotated, then rotates about the center of the scaled box, so there is no shear. A comment in LibreOffice (`oox/source/drawingml/shape.cxx`) says exactly this: "First scale object then rotate. Other way round would introduce shearing." Apache POI also scales the child's anchor rectangle and then rotates about its center.
- The axis-swap band (section 4.3): when the child's own normalized `rot`, in whole degrees, falls in [45, 135) or [225, 315), the width and height are swapped before scaling. The rule follows `lcl_doSpecialMSOWidthHeightToggle` in LibreOffice. A comment in that code says MSO behaves strangely here. Apache POI has a different special case, based on trial and error, for the same angle bands.
- This is a **chosen convention**, not verified behavior. The boundaries use truncation to whole degrees of the normalized `rot`: for example 44 degrees plus 59,999 units still counts as 44 degrees and does not swap. It has not been checked against real PowerPoint.
- **Not modelled:**
  - LibreOffice additionally rotates a child inside the band by 180 degrees when the parent group has exactly one mirrored axis (`flipH` xor `flipV`). The module does not do this.
  - The child's own flip when the group is flipped: as in section 4, the child is flipped in its own box, rotated, and then the group's flip and rotation are applied outside.
  - This is a deliberate choice rather than an omission: the ancestors' rotation is not used to decide the band, because scaling happens before the parent rotates (see section 4.3).

**(d) A group with ext = 0 but chExt ≠ 0.** The children collapse to a point (factor 0). This behavior is kept and has a no-error test, but it has not been compared against PowerPoint.

**(e) Convention tests.** `test_rotations_add_across_a_uniform_group` pins the angle-addition convention for a uniform scale. The following tests in `NonUniformGroupTest` pin the box-by-box rule and the band convention of sections 4.3 and 12(c): `test_rotated_child_is_scaled_before_rotating`, `test_right_angle_child_swaps_width_and_height`, `test_axis_swap_boundaries`, `test_nested_groups_with_a_rotated_inner_group`, `test_rotated_group_inside_the_swap_band`. The tests `test_rotated_child_differs_from_plain_matrix_composition`, `test_unrotated_and_half_turn_children_match_plain_matrix_composition`, `test_uniform_scale_matches_plain_matrix_composition`, `test_tree_matches_the_explicit_ancestor_chain` and `test_tree_starting_inside_ancestors_matches_the_explicit_chain` check internal consistency. None of them is a statement of truth about PowerPoint.
