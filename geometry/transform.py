from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from xml.etree.ElementTree import Element

EMU_PER_POINT = 12_700

_UNITS_PER_DEGREE = 60_000
_FULL_TURN = 360 * _UNITS_PER_DEGREE
_DECIMALS = 6
_INTEGER_FIELDS = (
    "off_x", "off_y", "ext_cx", "ext_cy", "rot",
    "ch_off_x", "ch_off_y", "ch_ext_cx", "ch_ext_cy",
)
_FLAG_TEXT = {"1": True, "true": True, "0": False, "false": False, "": False}
_XFRM_CHILDREN = {
    "off": ("off_x", "off_y"),
    "ext": ("ext_cx", "ext_cy"),
    "chOff": ("ch_off_x", "ch_off_y"),
    "chExt": ("ch_ext_cx", "ch_ext_cy"),
}

Point = tuple[float, float]


def emu_to_pt(value: float) -> float:
    return value / EMU_PER_POINT


def _to_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name}: expected an integer, got bool")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name}: expected an integer, got {value!r}") from None


def _to_flag(name: str, value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in _FLAG_TEXT:
        return _FLAG_TEXT[value]
    raise ValueError(f"{name}: expected a bool or an OOXML flag, got {value!r}")


def _round(value: float) -> float:
    return round(value, _DECIMALS) + 0.0


def _round_point(point: Point) -> Point:
    return _round(point[0]), _round(point[1])


def _round_angle(degrees: float) -> float:
    angle = _round(degrees % 360.0)
    return 0.0 if angle >= 360.0 else angle


def _local_name(tag: object) -> str:
    return tag.rpartition("}")[2] if isinstance(tag, str) else ""


def _find_child(element: Element, name: str) -> Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


@dataclass(frozen=True)
class Xfrm:
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
        for name in _INTEGER_FIELDS:
            object.__setattr__(self, name, _to_int(name, getattr(self, name)))
        object.__setattr__(self, "rot", self.rot % _FULL_TURN)
        for name in ("flip_h", "flip_v"):
            object.__setattr__(self, name, _to_flag(name, getattr(self, name)))

    @classmethod
    def from_element(cls, element: Element) -> Xfrm:
        xfrm = element if _local_name(element.tag) == "xfrm" else _find_child(element, "xfrm")
        if xfrm is None:
            return cls()
        values = {
            field: xfrm.get(attribute)
            for attribute, field in (("rot", "rot"), ("flipH", "flip_h"), ("flipV", "flip_v"))
            if xfrm.get(attribute) is not None
        }
        for child in xfrm:
            for field in _XFRM_CHILDREN.get(_local_name(child.tag), ()):
                raw = child.get(field.rsplit("_", 1)[1])
                if raw is not None:
                    values[field] = raw
        return cls(**values)


@dataclass(frozen=True)
class _Affine:
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def __matmul__(self, other: _Affine) -> _Affine:
        return _Affine(
            self.a * other.a + self.c * other.b,
            self.b * other.a + self.d * other.b,
            self.a * other.c + self.c * other.d,
            self.b * other.c + self.d * other.d,
            self.a * other.e + self.c * other.f + self.e,
            self.b * other.e + self.d * other.f + self.f,
        )

    def apply(self, x: float, y: float) -> Point:
        return self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f

    @property
    def is_mirrored(self) -> bool:
        return self.a * self.d - self.b * self.c < 0


def _translation(x: float, y: float) -> _Affine:
    return _Affine(e=x, f=y)


def _scaling(x: float, y: float) -> _Affine:
    return _Affine(a=x, d=y)


def _rotation(units: int) -> _Affine:
    angle = math.radians(units / _UNITS_PER_DEGREE)
    cos, sin = math.cos(angle), math.sin(angle)
    return _Affine(cos, sin, -sin, cos)


@dataclass(frozen=True)
class _Frame:
    off_x: float
    off_y: float
    width: float
    height: float
    rot: int
    flip_h: bool
    flip_v: bool

    @classmethod
    def of(cls, xfrm: Xfrm) -> _Frame:
        return cls(
            xfrm.off_x / EMU_PER_POINT,
            xfrm.off_y / EMU_PER_POINT,
            xfrm.ext_cx / EMU_PER_POINT,
            xfrm.ext_cy / EMU_PER_POINT,
            xfrm.rot,
            xfrm.flip_h,
            xfrm.flip_v,
        )

    def matrix(self) -> _Affine:
        flip = _scaling(-1.0 if self.flip_h else 1.0, -1.0 if self.flip_v else 1.0)
        return (
            _translation(self.off_x + self.width / 2, self.off_y + self.height / 2)
            @ _rotation(self.rot)
            @ flip
            @ _translation(-self.width / 2, -self.height / 2)
        )


@dataclass(frozen=True)
class _Scope:
    matrix: _Affine = _Affine()
    scale_x: float = 1.0
    scale_y: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0


def _swaps_axes(rot: int) -> bool:
    degrees = rot // _UNITS_PER_DEGREE
    return 45 <= degrees < 135 or 225 <= degrees < 315


def _frame_in(scope: _Scope, xfrm: Xfrm) -> _Frame:
    frame = _Frame.of(xfrm)
    swapped = _swaps_axes(xfrm.rot)
    width = frame.width * (scope.scale_y if swapped else scope.scale_x)
    height = frame.height * (scope.scale_x if swapped else scope.scale_y)
    center_x = (frame.off_x + frame.width / 2 - scope.origin_x) * scope.scale_x
    center_y = (frame.off_y + frame.height / 2 - scope.origin_y) * scope.scale_y
    return replace(
        frame,
        off_x=center_x - width / 2,
        off_y=center_y - height / 2,
        width=width,
        height=height,
    )


def _place(scope: _Scope, xfrm: Xfrm) -> tuple[_Affine, _Frame]:
    frame = _frame_in(scope, xfrm)
    return scope.matrix @ frame.matrix(), frame


def _scope_of_group(group: Xfrm, matrix: _Affine, frame: _Frame) -> _Scope:
    return _Scope(
        matrix,
        frame.width / (group.ch_ext_cx / EMU_PER_POINT) if group.ch_ext_cx else 1.0,
        frame.height / (group.ch_ext_cy / EMU_PER_POINT) if group.ch_ext_cy else 1.0,
        group.ch_off_x / EMU_PER_POINT,
        group.ch_off_y / EMU_PER_POINT,
    )


def _scope_of_ancestors(ancestors: Sequence[Xfrm]) -> _Scope:
    scope = _Scope()
    for group in ancestors:
        matrix, frame = _place(scope, group)
        scope = _scope_of_group(group, matrix, frame)
    return scope


@dataclass(frozen=True)
class ResolvedShape:
    corners: tuple[Point, ...]
    bbox: tuple[float, float, float, float]
    center: Point
    width: float
    height: float
    rotation_deg: float
    mirrored: bool


@dataclass(frozen=True)
class ResolvedConnector:
    start: Point
    end: Point
    box: ResolvedShape


def _as_connector(box: ResolvedShape) -> ResolvedConnector:
    return ResolvedConnector(start=box.corners[0], end=box.corners[2], box=box)


def _describe(matrix: _Affine, frame: _Frame) -> ResolvedShape:
    width, height = frame.width, frame.height
    corners = tuple(
        _round_point(matrix.apply(x, y))
        for x, y in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
    )
    xs, ys = zip(*corners)
    mirrored = matrix.is_mirrored
    angle = math.atan2(-matrix.c, matrix.d) if mirrored else math.atan2(matrix.b, matrix.a)
    return ResolvedShape(
        corners=corners,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        center=_round_point(matrix.apply(width / 2, height / 2)),
        width=_round(width),
        height=_round(height),
        rotation_deg=_round_angle(math.degrees(angle)),
        mirrored=mirrored,
    )


def resolve_shape(xfrm: Xfrm, ancestors: Sequence[Xfrm] = ()) -> ResolvedShape:
    matrix, frame = _place(_scope_of_ancestors(ancestors), xfrm)
    return _describe(matrix, frame)


def resolve_connector(xfrm: Xfrm, ancestors: Sequence[Xfrm] = ()) -> ResolvedConnector:
    return _as_connector(resolve_shape(xfrm, ancestors))


@dataclass(frozen=True)
class ShapeNode:
    xfrm: Xfrm
    name: str = ""
    is_connector: bool = False


@dataclass(frozen=True)
class GroupNode:
    xfrm: Xfrm
    children: tuple[Node, ...] = ()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))


Node = ShapeNode | GroupNode
Resolved = ResolvedShape | ResolvedConnector


def resolve_tree(
    nodes: Sequence[Node],
    ancestors: Sequence[Xfrm] = (),
) -> list[tuple[Node, Resolved]]:
    resolved: list[tuple[Node, Resolved]] = []
    _walk(nodes, _scope_of_ancestors(ancestors), resolved)
    return resolved


def _walk(nodes: Sequence[Node], scope: _Scope, resolved: list[tuple[Node, Resolved]]) -> None:
    for node in nodes:
        matrix, frame = _place(scope, node.xfrm)
        shape = _describe(matrix, frame)
        if isinstance(node, GroupNode):
            resolved.append((node, shape))
            _walk(node.children, _scope_of_group(node.xfrm, matrix, frame), resolved)
        elif node.is_connector:
            resolved.append((node, _as_connector(shape)))
        else:
            resolved.append((node, shape))
