"""DrawingML (OOXML) coordinate normalization and recursive geometry transforms.

Turns native PPTX shape, connector and group transforms (integer EMU) into one
slide-level coordinate system in points (y down). Stdlib only; reads no files.
Conventions are documented in geometry/README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

EMU_PER_POINT = 12_700
EMU_PER_INCH = 914_400
ROT_UNITS_PER_DEGREE = 60_000
FULL_TURN_UNITS = 360 * ROT_UNITS_PER_DEGREE  # 21_600_000
QUARTER_TURN_UNITS = 90 * ROT_UNITS_PER_DEGREE  # 5_400_000

_DECIMALS = 6
_EMU_FIELDS = (
    "off_x", "off_y", "ext_cx", "ext_cy",
    "ch_off_x", "ch_off_y", "ch_ext_cx", "ch_ext_cy",
)
_TRUE_TEXT = frozenset({"1", "true"})
_FALSE_TEXT = frozenset({"0", "false", ""})
# Exact (cos, sin) for rot = 0, 90, 180, 270 degrees, so no 6e-17 noise appears.
_EXACT_COS_SIN = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

Point = tuple[float, float]
NumberLike = Union[int, float, str]


# ----------------------------------------------------------------------
# Unit conversion
# ----------------------------------------------------------------------
def _as_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not a number: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise ValueError(f"{name}: not a number: {value!r}") from None
    else:
        raise TypeError(f"{name}: unsupported type {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError(f"{name}: not finite: {value!r}")
    return number


def _as_int(value: object, name: str) -> int:
    """Strict integer for EMU and rot values: rejects non-integral floats."""
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not an integer EMU value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name}: non-integral value {value!r}; EMU values must be integers")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            pass
        number = _as_number(text, name)
        if not number.is_integer():
            raise ValueError(f"{name}: non-integral value {value!r}; EMU values must be integers")
        return int(number)
    raise TypeError(f"{name}: unsupported type {type(value).__name__}")


def _as_flag(value: object, name: str) -> bool:
    """Flip flags: bool, None, or the OOXML text forms '1', 'true', '0', 'false', ''."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        if value in _TRUE_TEXT:
            return True
        if value in _FALSE_TEXT:
            return False
    raise ValueError(f"{name}: expected bool, '1', 'true', '0', 'false', '' or None, got {value!r}")


def emu_to_pt(value: NumberLike) -> float:
    """Convert EMU to points (12,700 EMU = 1 pt). Accepts int, float or numeric str."""
    return _as_number(value, "emu") / EMU_PER_POINT + 0.0


def pt_to_emu(value: NumberLike) -> int:
    """Convert points to EMU, rounded to the nearest integer."""
    return int(round(_as_number(value, "pt") * EMU_PER_POINT))


def _clean(value: float) -> float:
    """Round to 6 decimals and turn -0.0 into 0.0 (adding +0.0 does this)."""
    return round(value, _DECIMALS) + 0.0


def _clean_point(point: tuple[float, float]) -> Point:
    return (_clean(point[0]), _clean(point[1]))


def _clean_angle(degrees: float) -> float:
    """Map to [0, 360) after rounding; a value that rounds up to 360 becomes 0."""
    value = round(degrees % 360.0, _DECIMALS)
    if value >= 360.0:
        value = 0.0
    return value + 0.0


# ----------------------------------------------------------------------
# Xfrm: one a:xfrm / p:xfrm / a:grpSpPr/a:xfrm, normalized
# ----------------------------------------------------------------------
def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""  # comments and processing instructions
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _direct_child(element: object, name: str):
    for child in element:  # type: ignore[attr-defined]
        if _local_name(child.tag) == name:
            return child
    return None


# Child element tag -> (field for x-or-cx, field for y-or-cy, x-or-cx attr, y-or-cy attr)
_XFRM_CHILDREN = {
    "off": ("off_x", "off_y", "x", "y"),
    "ext": ("ext_cx", "ext_cy", "cx", "cy"),
    "chOff": ("ch_off_x", "ch_off_y", "x", "y"),
    "chExt": ("ch_ext_cx", "ch_ext_cy", "cx", "cy"),
}


@dataclass(frozen=True)
class Xfrm:
    """Normalized 2D transform. EMU fields are ints; rot is in [0, 21600000)."""

    off_x: int = 0
    off_y: int = 0
    ext_cx: int = 0
    ext_cy: int = 0
    rot: int = 0
    flip_h: bool = False
    flip_v: bool = False
    ch_off_x: int = 0
    ch_off_y: int = 0
    ch_ext_cx: int = 0
    ch_ext_cy: int = 0

    def __post_init__(self) -> None:
        for name in _EMU_FIELDS:
            object.__setattr__(self, name, _as_int(getattr(self, name), name))
        object.__setattr__(self, "rot", _as_int(self.rot, "rot") % FULL_TURN_UNITS)
        object.__setattr__(self, "flip_h", _as_flag(self.flip_h, "flip_h"))
        object.__setattr__(self, "flip_v", _as_flag(self.flip_v, "flip_v"))

    @classmethod
    def from_element(cls, el) -> Xfrm:
        """Build from an xfrm element, or from a grpSpPr element holding one.

        Tags are matched by local name, so any namespace prefix works. A missing
        xfrm gives the all-zero default.
        """
        xfrm_el = el if _local_name(el.tag) == "xfrm" else _direct_child(el, "xfrm")
        if xfrm_el is None:
            return cls()
        values: dict[str, object] = {}
        for attr, field_name in (("rot", "rot"), ("flipH", "flip_h"), ("flipV", "flip_v")):
            raw = xfrm_el.get(attr)
            if raw is not None:
                values[field_name] = raw
        for child in xfrm_el:
            spec = _XFRM_CHILDREN.get(_local_name(child.tag))
            if spec is None:
                continue
            field_x, field_y, attr_x, attr_y = spec
            raw_x = child.get(attr_x)
            raw_y = child.get(attr_y)
            if raw_x is not None:
                values[field_x] = raw_x
            if raw_y is not None:
                values[field_y] = raw_y
        return cls(**values)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Affine 2x3 matrices: x' = a*x + c*y + e ; y' = b*x + d*y + f
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Affine:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def __matmul__(self, other: Affine) -> Affine:
        """self @ other applies other first, then self."""
        return Affine(
            a=self.a * other.a + self.c * other.b + 0.0,
            b=self.b * other.a + self.d * other.b + 0.0,
            c=self.a * other.c + self.c * other.d + 0.0,
            d=self.b * other.c + self.d * other.d + 0.0,
            e=self.a * other.e + self.c * other.f + self.e + 0.0,
            f=self.b * other.e + self.d * other.f + self.f + 0.0,
        )

    def apply(self, x: float, y: float) -> Point:
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)

    def determinant(self) -> float:
        return self.a * self.d - self.b * self.c


IDENTITY = Affine()


def translate(tx: float, ty: float) -> Affine:
    return Affine(a=1.0, b=0.0, c=0.0, d=1.0, e=float(tx) + 0.0, f=float(ty) + 0.0)


def scale(sx: float, sy: float) -> Affine:
    return Affine(a=float(sx) + 0.0, b=0.0, c=0.0, d=float(sy) + 0.0, e=0.0, f=0.0)


def rotation(rot: int) -> Affine:
    """Clockwise rotation in y-down space; rot in 1/60000 degree, exact at multiples of 90."""
    units = _as_int(rot, "rot") % FULL_TURN_UNITS
    if units % QUARTER_TURN_UNITS == 0:
        cos_v, sin_v = _EXACT_COS_SIN[(units // QUARTER_TURN_UNITS) % 4]
    else:
        radians = math.radians(units / ROT_UNITS_PER_DEGREE)
        cos_v, sin_v = math.cos(radians), math.sin(radians)
    # 0.0 - sin_v (not -sin_v) so that a zero sine never yields -0.0.
    return Affine(a=cos_v, b=sin_v, c=0.0 - sin_v, d=cos_v, e=0.0, f=0.0)


# ----------------------------------------------------------------------
# Frames and group matrices
# ----------------------------------------------------------------------
def _frame_matrix_pt(
    off_x: float, off_y: float, w: float, h: float,
    rot: int, flip_h: bool, flip_v: bool,
) -> Affine:
    """Frame in points: T(centre) @ R(rot) @ Flip @ T(-w/2, -h/2), centre = off + (w/2, h/2)."""
    flip = scale(-1.0 if flip_h else 1.0, -1.0 if flip_v else 1.0)
    return (
        translate(off_x + w / 2, off_y + h / 2)
        @ rotation(rot)
        @ flip
        @ translate(-w / 2, -h / 2)
    )


def frame_matrix(xfrm: Xfrm) -> Affine:
    """Shape-local points (0..w, 0..h, origin at unrotated top-left) to parent space.

    T(centre) @ R(rot) @ Flip @ T(-w/2, -h/2): flip inside the box first, then rotate
    about the box centre. The flip order is an assumption; see geometry/README.md.
    """
    return _frame_matrix_pt(
        xfrm.off_x / EMU_PER_POINT, xfrm.off_y / EMU_PER_POINT,
        xfrm.ext_cx / EMU_PER_POINT, xfrm.ext_cy / EMU_PER_POINT,
        xfrm.rot, xfrm.flip_h, xfrm.flip_v,
    )


def _child_scale(ext: int, ch_ext: int) -> float:
    # chExt = 0 means "no child scaling": use 1.0 (slide-root grpSpPr is all zeros).
    if ch_ext == 0:
        return 1.0
    return ext / ch_ext


def group_matrix(xfrm: Xfrm) -> Affine:
    """Pure-matrix path: child space to parent space, frame(group) @ Scale(ext / chExt) @ T(-chOff).

    Scale is applied before the group's own flip and rotation. This is the reference
    product. It equals the resolve_* result when the scale is uniform (sx == sy), or
    when the child is unrotated or rotated by a multiple of 90 degrees. With a
    non-uniform scale and a child rotated by any other angle, the product shears the
    child, so resolve_shape, resolve_connector and resolve_tree do not use it there;
    they apply the box-by-box rule (geometry/README.md, section 4.3).
    A zero ext with non-zero chExt gives scale 0 and collapses the children to a point.
    """
    sx = _child_scale(xfrm.ext_cx, xfrm.ch_ext_cx)
    sy = _child_scale(xfrm.ext_cy, xfrm.ch_ext_cy)
    unshift = translate(-xfrm.ch_off_x / EMU_PER_POINT, -xfrm.ch_off_y / EMU_PER_POINT)
    return frame_matrix(xfrm) @ scale(sx, sy) @ unshift


def compose_groups(ancestors: Sequence[Xfrm]) -> Affine:
    """Pure-matrix path: G_outer @ ... @ G_inner for ancestor groups given outermost first.

    Same caveat as group_matrix: resolve_* does not use this product for a non-uniform
    scale with a rotated child (it would shear the child).
    """
    result = IDENTITY
    for group in ancestors:
        result = result @ group_matrix(group)
    return result


# ----------------------------------------------------------------------
# Resolved outputs (all points, rounded to 6 decimals, no -0.0)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedShape:
    corners: tuple[Point, Point, Point, Point]  # local TL, TR, BR, BL in slide space
    bbox: tuple[float, float, float, float]     # min_x, min_y, max_x, max_y
    center: Point
    width: float          # |image of local x-axis| * w' (effective, scaled local size)
    height: float         # |image of local y-axis| * h' (effective, scaled local size)
    rotation_deg: float   # [0, 360)
    flip_h: bool
    flip_v: bool          # always False (canonical form)


@dataclass(frozen=True)
class ResolvedConnector:
    start: Point
    end: Point
    box: ResolvedShape


def _resolve_matrix(matrix: Affine, w: float, h: float) -> ResolvedShape:
    """Outputs for a placed local box of effective size (w, h) in points."""
    corners = tuple(
        _clean_point(matrix.apply(x, y))
        for x, y in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
    )
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    center = _clean_point(matrix.apply(w / 2, h / 2))
    width = _clean(math.hypot(matrix.a, matrix.b) * w)   # local x-axis maps to (a, b)
    height = _clean(math.hypot(matrix.c, matrix.d) * h)  # local y-axis maps to (c, d)

    # Canonical form: a mirrored linear part is flip_h + rotation atan2(-c, d).
    if matrix.determinant() < 0:
        flip_h = True
        angle = math.degrees(math.atan2(-matrix.c, matrix.d))
    else:
        flip_h = False
        angle = math.degrees(math.atan2(matrix.b, matrix.a))
    return ResolvedShape(
        corners=corners,  # type: ignore[arg-type]
        bbox=bbox,  # type: ignore[arg-type]
        center=center,
        width=width,
        height=height,
        rotation_deg=_clean_angle(angle),
        flip_h=flip_h,
        flip_v=False,
    )


# ----------------------------------------------------------------------
# Box-by-box resolution (geometry/README.md, section 4.3). All values in points.
#
# Each node gets an effective frame in its parent group's box-local space: the
# parent's scale maps the node's centre into that space, the node's own size is
# scaled (width and height swapped at angles near 90/270), and the frame is then
# composed onto the parent's matrix. Uniform scales give the pure-matrix product.
# Non-uniform scales with a rotated child do not shear.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Frame:
    """Effective frame of one node in points: box offset, box size, orientation."""

    off_x: float
    off_y: float
    w: float
    h: float
    rot: int
    flip_h: bool
    flip_v: bool

    def matrix(self) -> Affine:
        return _frame_matrix_pt(
            self.off_x, self.off_y, self.w, self.h, self.rot, self.flip_h, self.flip_v,
        )


@dataclass(frozen=True)
class _ChildSpace:
    """What an effective group frame gives its children: scale into its box, chOff shift."""

    sx: float
    sy: float
    ch_off_x: float
    ch_off_y: float


def _is_band(rot: int) -> bool:
    """True for whole-degree bands [45, 135) and [225, 315) of the normalized rot."""
    deg = rot // ROT_UNITS_PER_DEGREE
    return 45 <= deg < 135 or 225 <= deg < 315


def _child_space_of(xfrm: Xfrm, frame: _Frame) -> _ChildSpace:
    """Child space of a group, from its EFFECTIVE box size (frame.w, frame.h)."""
    sx = frame.w / (xfrm.ch_ext_cx / EMU_PER_POINT) if xfrm.ch_ext_cx != 0 else 1.0
    sy = frame.h / (xfrm.ch_ext_cy / EMU_PER_POINT) if xfrm.ch_ext_cy != 0 else 1.0
    return _ChildSpace(sx, sy, xfrm.ch_off_x / EMU_PER_POINT, xfrm.ch_off_y / EMU_PER_POINT)


def _effective_frame(xfrm: Xfrm, space: _ChildSpace | None) -> _Frame:
    """Own frame at top level; otherwise the box-by-box rule relative to the parent group."""
    off_x = xfrm.off_x / EMU_PER_POINT
    off_y = xfrm.off_y / EMU_PER_POINT
    w = xfrm.ext_cx / EMU_PER_POINT
    h = xfrm.ext_cy / EMU_PER_POINT
    if space is None:
        return _Frame(off_x, off_y, w, h, xfrm.rot, xfrm.flip_h, xfrm.flip_v)
    # Centre in child space -> parent's box-local space (before the parent rotates).
    c_local_x = (off_x + w / 2 - space.ch_off_x) * space.sx
    c_local_y = (off_y + h / 2 - space.ch_off_y) * space.sy
    if _is_band(xfrm.rot):
        new_w, new_h = w * space.sy, h * space.sx
    else:
        new_w, new_h = w * space.sx, h * space.sy
    return _Frame(
        c_local_x - new_w / 2, c_local_y - new_h / 2, new_w, new_h,
        xfrm.rot, xfrm.flip_h, xfrm.flip_v,
    )


def _step(matrix: Affine, space: _ChildSpace | None, xfrm: Xfrm) -> tuple[Affine, _Frame]:
    """Place one node. Returns its local-box-to-slide matrix and its effective frame.

    Every path (ancestor chain, single shape, tree) goes through this function, so the
    float operations and therefore the results are identical.
    """
    frame = _effective_frame(xfrm, space)
    return matrix @ frame.matrix(), frame


def _chain_state(ancestors: Sequence[Xfrm]) -> tuple[Affine, _ChildSpace | None]:
    """Matrix and child space after folding the ancestor groups, outermost first."""
    matrix: Affine = IDENTITY
    space: _ChildSpace | None = None
    for group in ancestors:
        matrix, frame = _step(matrix, space, group)
        space = _child_space_of(group, frame)
    return matrix, space


def _connector_result(placed: Affine, frame: _Frame) -> ResolvedConnector:
    return ResolvedConnector(
        start=_clean_point(placed.apply(0.0, 0.0)),
        end=_clean_point(placed.apply(frame.w, frame.h)),
        box=_resolve_matrix(placed, frame.w, frame.h),
    )


def resolve_shape(xfrm: Xfrm, ancestors: Sequence[Xfrm] = ()) -> ResolvedShape:
    """Resolve one shape's frame through its ancestor groups (outermost first).

    Box-by-box rule (geometry/README.md, section 4.3): each ancestor's effective frame is scaled
    inside its parent's box and the shape's own box is scaled the same way, so a
    rotated child is never sheared by a non-uniform group scale.
    """
    matrix, space = _chain_state(ancestors)
    placed, frame = _step(matrix, space, xfrm)
    return _resolve_matrix(placed, frame.w, frame.h)


def resolve_connector(xfrm: Xfrm, ancestors: Sequence[Xfrm] = ()) -> ResolvedConnector:
    """A straight line from local (0,0) to the effective local corner (w', h').

    start = M(0,0) and end = M(w', h'), where (w', h') is the effective local size from
    the box-by-box rule. So start == box.corners[0] and end == box.corners[2].
    """
    matrix, space = _chain_state(ancestors)
    placed, frame = _step(matrix, space, xfrm)
    return _connector_result(placed, frame)


# ----------------------------------------------------------------------
# Recursive tree
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ShapeNode:
    xfrm: Xfrm
    name: str = ""
    is_connector: bool = False


@dataclass(frozen=True)
class GroupNode:
    xfrm: Xfrm
    children: tuple[Union[ShapeNode, GroupNode], ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))


def resolve_tree(
    nodes: Sequence[Union[ShapeNode, GroupNode]],
    ancestors: Sequence[Xfrm] = (),
) -> list[tuple[Union[ShapeNode, GroupNode], Union[ResolvedShape, ResolvedConnector]]]:
    """Resolve a node tree in pre-order.

    Each group's own box is emitted (ResolvedShape, placed by the box-by-box rule
    relative to its parent group) before its children. Children are resolved with
    the group's effective frame, so the results equal resolve_shape / resolve_connector
    called with the explicit ancestor chain.
    """
    matrix, space = _chain_state(ancestors)
    out: list[tuple[Union[ShapeNode, GroupNode], Union[ResolvedShape, ResolvedConnector]]] = []
    _resolve_nodes(nodes, matrix, space, out)
    return out


def _resolve_nodes(
    nodes: Sequence[Union[ShapeNode, GroupNode]],
    matrix: Affine,
    space: _ChildSpace | None,
    out: list,
) -> None:
    """Recursive pre-order walk that carries the running matrix and child space."""
    for node in nodes:
        if isinstance(node, GroupNode):
            placed, frame = _step(matrix, space, node.xfrm)
            out.append((node, _resolve_matrix(placed, frame.w, frame.h)))
            _resolve_nodes(node.children, placed, _child_space_of(node.xfrm, frame), out)
        elif isinstance(node, ShapeNode):
            placed, frame = _step(matrix, space, node.xfrm)
            if node.is_connector:
                out.append((node, _connector_result(placed, frame)))
            else:
                out.append((node, _resolve_matrix(placed, frame.w, frame.h)))
        else:
            raise TypeError(f"unsupported node type: {type(node).__name__}")
