"""
Tests for geometry/transform.py (OOXML EMU -> point normalization and group transforms).

Run:  .venv\\Scripts\\python geometry\\test_transform.py
(plain asserts, no pytest needed; functions are also pytest-compatible)

All expected values are hand-computed in points (y down). They prove internal
consistency with the documented conventions, NOT agreement with real PowerPoint output.
"""

import dataclasses
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry.transform import (  # noqa: E402
    FULL_TURN_UNITS,
    GroupNode,
    ResolvedConnector,
    ResolvedShape,
    ShapeNode,
    Xfrm,
    compose_groups,
    emu_to_pt,
    frame_matrix,
    pt_to_emu,
    resolve_connector,
    resolve_shape,
    resolve_tree,
)

E = lambda pt: int(round(pt * 12700))  # noqa: E731  (points -> EMU fixture helper)

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


# ----------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------
def test_emu_point_conversion():
    assert emu_to_pt(12700) == 1.0
    assert emu_to_pt(914400) == 72.0
    assert emu_to_pt(9525) == 0.75
    assert emu_to_pt("12700") == 1.0
    assert math.copysign(1.0, emu_to_pt(-0.0)) > 0  # no -0.0 output
    assert pt_to_emu(72) == 914400
    assert pt_to_emu(0.75) == 9525


# ----------------------------------------------------------------------
# Single shape and rotation
# ----------------------------------------------------------------------
def test_plain_shape_bbox():
    r = resolve_shape(Xfrm(off_x=E(10), off_y=E(20), ext_cx=E(100), ext_cy=E(50)))
    assert r.bbox == (10.0, 20.0, 110.0, 70.0), r
    assert r.rotation_deg == 0.0
    assert r.flip_h is False and r.flip_v is False
    assert r.width == 100.0 and r.height == 50.0
    assert r.center == (60.0, 45.0)
    assert r.corners == ((10.0, 20.0), (110.0, 20.0), (110.0, 70.0), (10.0, 70.0))


def test_rotation_90_180_270():
    # off(100,100) ext(100,50): centre (150,125), half-size (50,25)
    r90 = resolve_shape(Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(100), ext_cy=E(50), rot=5400000))
    assert r90.bbox == (125.0, 75.0, 175.0, 175.0), r90
    assert r90.corners == ((175.0, 75.0), (175.0, 175.0), (125.0, 175.0), (125.0, 75.0)), r90
    assert r90.center == (150.0, 125.0)
    assert r90.rotation_deg == 90.0
    assert r90.width == 100.0 and r90.height == 50.0
    assert r90.flip_h is False

    r180 = resolve_shape(Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(100), ext_cy=E(50), rot=10800000))
    assert r180.bbox == (100.0, 100.0, 200.0, 150.0), r180  # same box as unrotated
    assert r180.corners == ((200.0, 150.0), (100.0, 150.0), (100.0, 100.0), (200.0, 100.0)), r180
    assert r180.rotation_deg == 180.0
    assert r180.width == 100.0 and r180.height == 50.0

    r270 = resolve_shape(Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(100), ext_cy=E(50), rot=16200000))
    assert r270.bbox == (125.0, 75.0, 175.0, 175.0), r270
    assert r270.corners == ((125.0, 175.0), (125.0, 75.0), (175.0, 75.0), (175.0, 175.0)), r270
    assert r270.rotation_deg == 270.0


# ----------------------------------------------------------------------
# Groups: offset, scale, nesting, rotation, flips
# ----------------------------------------------------------------------
def test_group_offset_and_scale():
    group = Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(200), ext_cy=E(100),
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(100))
    child = Xfrm(off_x=E(50), off_y=E(50), ext_cx=E(50), ext_cy=E(50))
    assert resolve_shape(group).bbox == (100.0, 100.0, 300.0, 200.0)
    assert resolve_shape(child, [group]).bbox == (200.0, 150.0, 300.0, 200.0)


def test_nested_groups_and_resolve_tree():
    outer = Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(200), ext_cy=E(200),
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(100))
    inner = Xfrm(off_x=E(10), off_y=E(10), ext_cx=E(40), ext_cy=E(40),
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(20), ch_ext_cy=E(20))
    shape = Xfrm(off_x=E(5), off_y=E(5), ext_cx=E(10), ext_cy=E(10))

    assert resolve_shape(shape, [outer, inner]).bbox == (140.0, 140.0, 180.0, 180.0)

    tree = [GroupNode(outer, name="outer", children=(
        GroupNode(inner, name="inner", children=(ShapeNode(shape, name="s"),)),
    ))]
    out = resolve_tree(tree)
    assert [type(node).__name__ for node, _ in out] == ["GroupNode", "GroupNode", "ShapeNode"]
    assert [node.name for node, _ in out[:2]] == ["outer", "inner"]
    # group boxes are emitted (pre-order) in the parent space of each group
    assert [res.bbox for _, res in out] == [
        (100.0, 100.0, 300.0, 300.0),
        (120.0, 120.0, 200.0, 200.0),
        (140.0, 140.0, 180.0, 180.0),
    ]
    assert out[2][1] == resolve_shape(shape, [outer, inner])


def test_group_rotation():
    group = Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(100), ext_cy=E(100), rot=5400000,
                 ch_ext_cx=E(100), ch_ext_cy=E(100))
    child = Xfrm(off_x=0, off_y=0, ext_cx=E(50), ext_cy=E(50))
    r = resolve_shape(child, [group])
    assert r.bbox == (150.0, 100.0, 200.0, 150.0), r
    assert r.rotation_deg == 90.0


def test_group_flip_h():
    group = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(100), flip_h=True,
                 ch_ext_cx=E(100), ch_ext_cy=E(100))
    child = Xfrm(off_x=0, off_y=0, ext_cx=E(30), ext_cy=E(30))
    r = resolve_shape(child, [group])
    assert r.bbox == (70.0, 0.0, 100.0, 30.0), r
    assert r.flip_h is True
    assert r.rotation_deg == 0.0


def test_flip_then_rotate_order():
    # Flip inside the box first, then rotate about the centre (ASSUMPTION, see geometry/README.md).
    # The opposite order (rotate, then flip) would give bbox (0,0,10,20) instead.
    group = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(100), rot=5400000, flip_h=True,
                 ch_ext_cx=E(100), ch_ext_cy=E(100))
    child = Xfrm(off_x=0, off_y=0, ext_cx=E(20), ext_cy=E(10))
    r = resolve_shape(child, [group])
    assert r.bbox == (90.0, 80.0, 100.0, 100.0), r
    assert r.flip_h is True
    assert r.rotation_deg == 90.0


# ----------------------------------------------------------------------
# Connectors
# ----------------------------------------------------------------------
def test_connector_endpoints():
    def conn(off_x=0, off_y=0, **kw):
        return resolve_connector(Xfrm(off_x=off_x, off_y=off_y, **kw))

    plain = conn(ext_cx=E(100), ext_cy=E(50))
    assert (plain.start, plain.end) == ((0.0, 0.0), (100.0, 50.0))
    assert isinstance(plain, ResolvedConnector)
    assert isinstance(plain.box, ResolvedShape)

    flip_h = conn(ext_cx=E(100), ext_cy=E(50), flip_h=True)
    assert (flip_h.start, flip_h.end) == ((100.0, 0.0), (0.0, 50.0))

    flip_v = conn(ext_cx=E(100), ext_cy=E(50), flip_v=True)
    assert (flip_v.start, flip_v.end) == ((0.0, 50.0), (100.0, 0.0))

    both = conn(ext_cx=E(100), ext_cy=E(50), flip_h=True, flip_v=True)
    assert (both.start, both.end) == ((100.0, 50.0), (0.0, 0.0))

    horizontal = conn(off_x=E(10), off_y=E(20), ext_cx=E(100), ext_cy=0)
    assert (horizontal.start, horizontal.end) == ((10.0, 20.0), (110.0, 20.0))

    vertical = conn(off_x=E(10), off_y=E(20), ext_cx=0, ext_cy=E(80))
    assert (vertical.start, vertical.end) == ((10.0, 20.0), (10.0, 100.0))


def test_connector_in_rotated_group():
    group = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(100), rot=5400000,
                 ch_ext_cx=E(100), ch_ext_cy=E(100))
    conn_xfrm = Xfrm(off_x=0, off_y=0, ext_cx=E(20), ext_cy=E(10))
    c = resolve_connector(conn_xfrm, [group])
    # Hand-computed: group rot 90 about (50,50) maps (0,0)->(100,0) and (20,10)->(90,20).
    assert c.start == (100.0, 0.0), c
    assert c.end == (90.0, 20.0), c
    # Consistent with the box corners: start = TL, end = BR.
    assert c.start == c.box.corners[0]
    assert c.end == c.box.corners[2]
    # independent path: the box must equal resolve_shape computed separately
    assert c.box == resolve_shape(conn_xfrm, [group])
    assert all(math.isfinite(v) for v in (*c.start, *c.end))


# ----------------------------------------------------------------------
# Equivalences: equal inputs must give equal outputs
# ----------------------------------------------------------------------
def test_equivalences():
    # rot is taken mod 21600000
    assert Xfrm(rot=0) == Xfrm(rot=21600000) == Xfrm(rot=-21600000)
    assert Xfrm(rot=-5400000) == Xfrm(rot=16200000) == Xfrm(rot=37800000)
    # Ticket said 27000000 == 16200000, but 27000000 % 21600000 = 5400000 (90 deg), so pin that.
    assert Xfrm(rot=27000000).rot == 5400000
    box = dict(off_x=E(100), off_y=E(50), ext_cx=E(80), ext_cy=E(40))
    assert resolve_shape(Xfrm(rot=0, **box)) == resolve_shape(Xfrm(rot=21600000, **box))
    assert resolve_shape(Xfrm(rot=-5400000, **box)) == resolve_shape(Xfrm(rot=16200000, **box))

    # flipH+flipV at rot 0 is the same as rot 180 with no flips
    both = resolve_shape(Xfrm(flip_h=True, flip_v=True, **box))
    half_turn = resolve_shape(Xfrm(rot=10800000, **box))
    assert both == half_turn
    assert both.flip_h is False and both.rotation_deg == 180.0

    # flipV alone is flip_h + rotation 180 in canonical form
    fv = resolve_shape(Xfrm(flip_v=True, **box))
    assert fv.flip_h is True and fv.flip_v is False and fv.rotation_deg == 180.0

    # EMU given as numeric strings equals EMU given as ints
    as_str = Xfrm(off_x=str(E(100)), off_y=str(E(50)), ext_cx="1016000", ext_cy="508000")
    as_int = Xfrm(off_x=E(100), off_y=E(50), ext_cx=E(80), ext_cy=E(40))
    assert as_str == as_int
    assert resolve_shape(as_str) == resolve_shape(as_int)

    # flip flags: bool, OOXML text, None
    assert Xfrm(flip_h="1") == Xfrm(flip_h=True) == Xfrm(flip_h="true") == Xfrm(flip_h=True)
    assert Xfrm(flip_v="0") == Xfrm(flip_v=None) == Xfrm(flip_v="") == Xfrm(flip_v=False)

    # identity group (chOff = off, chExt = ext, no rot, no flips) equals no group
    shape = Xfrm(off_x=E(110), off_y=E(110), ext_cx=E(20), ext_cy=E(20))
    identity_group = Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(50), ext_cy=E(50),
                          ch_off_x=E(100), ch_off_y=E(100), ch_ext_cx=E(50), ch_ext_cy=E(50))
    assert resolve_shape(shape, [identity_group]) == resolve_shape(shape)


def test_input_normalization_rules():
    try:
        Xfrm(off_x=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("non-integral float EMU must raise ValueError")
    try:
        Xfrm(flip_h="yes")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown flip text must raise ValueError")
    assert Xfrm(off_x=12700.0).off_x == 12700


# ----------------------------------------------------------------------
# Degenerate cases
# ----------------------------------------------------------------------
def test_degenerate_groups():
    child = Xfrm(off_x=E(10), off_y=E(10), ext_cx=E(20), ext_cy=E(20))

    # ext 0 and chExt 0 on the group: chExt 0 means scale 1, so this is a pure translation
    zero_group = Xfrm(off_x=E(100), off_y=E(100))
    assert resolve_shape(child, [zero_group]).bbox == (110.0, 110.0, 130.0, 130.0)

    # slide-root grpSpPr (all zeros) is the identity
    root = Xfrm()
    assert resolve_shape(child, [root]) == resolve_shape(child)
    assert resolve_shape(child, [root]).bbox == (10.0, 10.0, 30.0, 30.0)

    # ext 0 with chExt != 0: scale 0, children collapse to a point; no exception
    collapsed = Xfrm(off_x=E(100), off_y=E(100), ext_cx=0, ext_cy=0,
                     ch_ext_cx=E(100), ch_ext_cy=E(100))
    c = resolve_shape(child, [collapsed])
    assert c.bbox == (100.0, 100.0, 100.0, 100.0), c
    assert c.width == 0.0 and c.height == 0.0
    assert len(resolve_tree([GroupNode(collapsed, children=(ShapeNode(child),))])) == 2


# ----------------------------------------------------------------------
# XML parsing
# ----------------------------------------------------------------------
def test_from_element_xml():
    el = ET.fromstring(
        f'<a:xfrm xmlns:a="{A_NS}" rot="5400000" flipH="1">'
        '<a:off x="1270000" y="2540000"/><a:ext cx="12700000" cy="6350000"/></a:xfrm>'
    )
    assert Xfrm.from_element(el) == Xfrm(off_x=1270000, off_y=2540000, ext_cx=12700000,
                                         ext_cy=6350000, rot=5400000, flip_h=True)

    # same element under a different namespace prefix
    el_x = ET.fromstring(
        f'<x:xfrm xmlns:x="{A_NS}" rot="5400000" flipH="1">'
        '<x:off x="1270000" y="2540000"/><x:ext cx="12700000" cy="6350000"/></x:xfrm>'
    )
    assert Xfrm.from_element(el_x) == Xfrm.from_element(el)

    # p:xfrm with no children and no attributes gives the zero default
    assert Xfrm.from_element(ET.fromstring(f'<p:xfrm xmlns:p="{P_NS}"/>')) == Xfrm()

    # grpSpPr-style group: xfrm with chOff/chExt inside, plus a sibling fill element to skip
    grp = ET.fromstring(
        f'<p:grpSpPr xmlns:p="{P_NS}" xmlns:a="{A_NS}"><a:xfrm>'
        '<a:off x="1270000" y="1270000"/><a:ext cx="2540000" cy="1270000"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="1270000" cy="1270000"/></a:xfrm>'
        '<a:noFill/></p:grpSpPr>'
    )
    assert Xfrm.from_element(grp) == Xfrm(off_x=1270000, off_y=1270000, ext_cx=2540000,
                                          ext_cy=1270000, ch_off_x=0, ch_off_y=0,
                                          ch_ext_cx=1270000, ch_ext_cy=1270000)


# ----------------------------------------------------------------------
# Convention pinning (NOT ground truth)
# ----------------------------------------------------------------------
def test_convention_group_rot_and_child_rot():
    """ASSUMPTION-PINNING test, not ground truth: pins this module's composition convention.

    A group rotated 60 degrees (uniform scale 1) with a child rotated 30 degrees gives a
    child rotation of 90 degrees. This checks that rotations compose additively here, which
    is the stated convention and not a PowerPoint observation.
    """
    group = Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(100), ext_cy=E(100), rot=3600000,
                 ch_ext_cx=E(100), ch_ext_cy=E(100))
    child = Xfrm(off_x=0, off_y=0, ext_cx=E(50), ext_cy=E(50), rot=1800000)
    assert abs(resolve_shape(child, [group]).rotation_deg - 90.0) < 1e-6


# ----------------------------------------------------------------------
# Non-uniform group scale: box-by-box rule (geometry/README.md, section 4.3).
# Hand-computed fixtures. CONVENTION pinning, NOT verified against PowerPoint.
# ----------------------------------------------------------------------
def _close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) < tol


def _points_close(pts_a, pts_b, tol: float = 1e-6) -> bool:
    return all(_close(p[0], q[0], tol) and _close(p[1], q[1], tol) for p, q in zip(pts_a, pts_b))


def _pure_corners(xfrm, ancestors=()):
    """Pure-matrix reference from the public API: compose_groups(ancestors) @ frame_matrix(xfrm).

    Applied to the raw-ext local corners (0,0), (w,0), (w,h), (0,h).
    """
    m = compose_groups(ancestors) @ frame_matrix(xfrm)
    w, h = xfrm.ext_cx / 12700, xfrm.ext_cy / 12700
    return [m.apply(x, y) for x, y in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))]


def _group_2x1():
    """Group with sx = 2, sy = 1 (ext 200x100 over chExt 100x100), no offset, no rotation."""
    return Xfrm(off_x=0, off_y=0, ext_cx=E(200), ext_cy=E(100),
                ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(100))


def test_nonuniform_group_rotated_child_not_sheared():
    """T1 (convention, hand-computed). Group sx = 2, sy = 1; child ext(100,50), rot 30 deg.

    Child centre (50,25) -> box-local (100,25). Effective size (200,50), no band swap.
    Half-extents: 200/2*cos30 + 50/2*sin30 = 99.102540 and 200/2*sin30 + 50/2*cos30 = 71.650635.
    The pure-matrix product shears the child, so its corners must differ.
    """
    group = _group_2x1()
    child = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=1800000)
    r = resolve_shape(child, [group])
    assert r.center == (100.0, 25.0), r
    assert r.width == 200.0 and r.height == 50.0, r
    assert r.rotation_deg == 30.0 and r.flip_h is False, r
    expected_bbox = (0.89746, -46.650635, 199.10254, 96.650635)
    assert all(_close(a, b) for a, b in zip(r.bbox, expected_bbox)), r.bbox
    assert not _points_close(r.corners, _pure_corners(child, [group]), tol=1e-3), r.corners


def test_nonuniform_group_right_angle_child_band_swap():
    """T2 (convention, hand-computed). Same group; child ext(100,50), rot 90 deg (band).

    Centre (50,25) -> (100,25). Band swap: w' = 100 * sy = 100, h' = 50 * sx = 100.
    Property (c): equals the pure-matrix result for right-angle children.
    """
    group = _group_2x1()
    child = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=5400000)
    r = resolve_shape(child, [group])
    assert r.center == (100.0, 25.0), r
    assert r.width == 100.0 and r.height == 100.0, r
    assert r.rotation_deg == 90.0, r
    assert r.bbox == (50.0, -25.0, 150.0, 75.0), r
    assert _points_close(r.corners, _pure_corners(child, [group])), r.corners


def test_band_boundaries_nonuniform_group():
    """T3 (convention). Same group, child ext(100,50). The band is decided by the whole-degree
    value of the normalized rot: [45,135) and [225,315) swap width and height. Checked via the
    effective local size (width, height)."""
    group = _group_2x1()
    cases = [
        (2_699_999, (200.0, 50.0)),    # 44 deg + 59999 units: no swap
        (2_700_000, (100.0, 100.0)),   # 45 deg: swap
        (8_099_999, (100.0, 100.0)),   # 134 deg + 59999 units: swap
        (8_100_000, (200.0, 50.0)),    # 135 deg: no swap
        (13_499_999, (200.0, 50.0)),   # 224 deg + 59999 units: no swap
        (13_500_000, (100.0, 100.0)),  # 225 deg: swap
        (18_899_999, (100.0, 100.0)),  # 314 deg + 59999 units: swap
        (18_900_000, (200.0, 50.0)),   # 315 deg: no swap
    ]
    for rot, (exp_w, exp_h) in cases:
        child = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=rot)
        r = resolve_shape(child, [group])
        assert (r.width, r.height) == (exp_w, exp_h), (rot, r)


def test_nested_nonuniform_groups_hand_computed():
    """T4 (convention, hand-computed). Outer: sx = 2, sy = 1. Inner (child of outer): ext(100,50),
    rot 30, chExt(100,50) -> effective centre (100,25), size (200,50), rot 30. Inner child scale
    is (2, 1). Shape ext(50,25) at off 0: centre (25,12.5) -> inner box-local (50,12.5), size
    (100,25). Slide centre = (100,25) + R30 * (-50,-12.5) = (62.948730, -10.825318)."""
    outer = _group_2x1()
    inner = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=1800000,
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(50))
    shape = Xfrm(off_x=0, off_y=0, ext_cx=E(50), ext_cy=E(25))
    r = resolve_shape(shape, [outer, inner])
    assert _close(r.center[0], 62.948730) and _close(r.center[1], -10.825318), r
    assert r.width == 100.0 and r.height == 25.0, r
    assert r.rotation_deg == 30.0, r


def test_uniform_scale_rotated_child_equals_pure_matrix():
    """T5 (property a). Uniform scale must equal the phase-1 pure-matrix product, including a
    rotated middle group, a child with a non-zero chOff, and a flipped, rotated leaf."""
    group = Xfrm(off_x=0, off_y=0, ext_cx=E(200), ext_cy=E(200),
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(100))
    child = Xfrm(off_x=E(10), off_y=E(10), ext_cx=E(40), ext_cy=E(20), rot=1800000)
    r = resolve_shape(child, [group])
    assert _points_close(r.corners, _pure_corners(child, [group]), tol=1e-6), r.corners

    outer = Xfrm(off_x=E(20), off_y=E(30), ext_cx=E(200), ext_cy=E(200),
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(100))
    middle = Xfrm(off_x=E(5), off_y=E(8), ext_cx=E(40), ext_cy=E(40), rot=1800000,
                  ch_off_x=E(5), ch_off_y=E(5), ch_ext_cx=E(20), ch_ext_cy=E(20))
    leaf = Xfrm(off_x=E(3), off_y=E(4), ext_cx=E(10), ext_cy=E(6), rot=1200000, flip_v=True)
    deep = resolve_shape(leaf, [outer, middle])
    assert _points_close(deep.corners, _pure_corners(leaf, [outer, middle]), tol=1e-6), deep.corners


def test_rot_0_and_180_children_equal_pure_matrix():
    """T6 (property b). Non-uniform group (sx = 2, sy = 1) with a non-zero chOff and a rotated
    group. Children with rot 0 or 180 and any flip combination equal the pure-matrix result."""
    group = Xfrm(off_x=E(5), off_y=E(7), ext_cx=E(200), ext_cy=E(100), rot=1800000,
                 ch_off_x=E(30), ch_off_y=E(10), ch_ext_cx=E(100), ch_ext_cy=E(100))
    for rot in (0, 10_800_000):
        for flip_h in (False, True):
            for flip_v in (False, True):
                child = Xfrm(off_x=E(40), off_y=E(20), ext_cx=E(30), ext_cy=E(12),
                             rot=rot, flip_h=flip_h, flip_v=flip_v)
                r = resolve_shape(child, [group])
                assert _points_close(r.corners, _pure_corners(child, [group]), tol=1e-6), \
                    (rot, flip_h, flip_v, r.corners)


def test_connector_nonuniform_group_endpoints():
    """T7. Connector ext(100,50), rot 30, in the non-uniform group, all four flip combinations.

    start == corners[0] and end == corners[2] always. Hand-computed without flips:
    start = (100,25) + R30 * (-100,-25) = (25.897460, -46.650635).
    """
    group = _group_2x1()
    for flip_h in (False, True):
        for flip_v in (False, True):
            conn = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=1800000,
                        flip_h=flip_h, flip_v=flip_v)
            c = resolve_connector(conn, [group])
            assert c.start == c.box.corners[0], c
            assert c.end == c.box.corners[2], c
            assert c.box == resolve_shape(conn, [group]), c
            if not flip_h and not flip_v:
                assert _close(c.start[0], 25.897460) and _close(c.start[1], -46.650635), c


def test_tree_matches_chain_nonuniform():
    """T8. resolve_tree over the T4 tree, plus a connector at depth 2, equals resolve_shape /
    resolve_connector with the explicit ancestor chain. Groups are emitted pre-order, and the
    inner group's own box is the effective frame from T4."""
    outer = _group_2x1()
    inner = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=1800000,
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(50))
    shape = Xfrm(off_x=0, off_y=0, ext_cx=E(50), ext_cy=E(25))
    conn = Xfrm(off_x=E(4), off_y=E(2), ext_cx=E(20), ext_cy=E(10), rot=1800000)
    tree = [GroupNode(outer, name="outer", children=(
        GroupNode(inner, name="inner", children=(
            ShapeNode(shape, name="s"),
            ShapeNode(conn, name="c", is_connector=True),
        )),
    ))]
    out = resolve_tree(tree)
    assert [node.name for node, _ in out] == ["outer", "inner", "s", "c"]
    assert out[0][1] == resolve_shape(outer), out[0][1]
    assert out[1][1] == resolve_shape(inner, [outer]), out[1][1]
    inner_box = out[1][1]
    assert (inner_box.center, inner_box.width, inner_box.height, inner_box.rotation_deg) == \
        ((100.0, 25.0), 200.0, 50.0, 30.0), inner_box
    assert out[2][1] == resolve_shape(shape, [outer, inner]), out[2][1]
    assert out[3][1] == resolve_connector(conn, [outer, inner]), out[3][1]
    for _, res in out:
        for v in _floats(res):
            assert not (v == 0.0 and math.copysign(1.0, v) < 0), res
    assert repr(resolve_tree(tree)) == repr(out)


def test_band_group_inside_nonuniform_group():
    """A GROUP in a band angle (rot 90) nested in a non-uniform group (sx = 2, sy = 1).

    Inner effective size: w' = 100 * sy = 100, h' = 50 * sx = 100, so its box is
    (50,-25,150,75) at rot 90. Its child scale is (100/100, 100/50) = (1, 2).
    Shape ext(50,25) at off 0: centre (25,12.5) -> (25,25), effective size (50,50).
    Slide centre = (100,25) + R90 * (25-50, 25-50) = (125, 0).
    """
    outer = _group_2x1()
    inner = Xfrm(off_x=0, off_y=0, ext_cx=E(100), ext_cy=E(50), rot=5400000,
                 ch_off_x=0, ch_off_y=0, ch_ext_cx=E(100), ch_ext_cy=E(50))
    shape = Xfrm(off_x=0, off_y=0, ext_cx=E(50), ext_cy=E(25))
    out = resolve_tree([GroupNode(outer, children=(
        GroupNode(inner, children=(ShapeNode(shape),)),
    ))])
    inner_box = out[1][1]
    assert inner_box.bbox == (50.0, -25.0, 150.0, 75.0), inner_box
    assert inner_box.rotation_deg == 90.0 and inner_box.width == 100.0 and inner_box.height == 100.0
    s = out[2][1]
    assert s.center == (125.0, 0.0), s
    assert s.width == 50.0 and s.height == 50.0 and s.rotation_deg == 90.0, s
    assert s == resolve_shape(shape, [outer, inner])


# ----------------------------------------------------------------------
# Determinism and signed zero
# ----------------------------------------------------------------------
def _floats(obj):
    if isinstance(obj, float):
        yield obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _floats(getattr(obj, f.name))
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _floats(item)


def test_deterministic_and_no_negative_zero():
    outer = Xfrm(off_x=E(100), off_y=E(100), ext_cx=E(100), ext_cy=E(60), rot=5400000,
                 flip_h=True, ch_ext_cx=E(50), ch_ext_cy=E(30))
    tree = [GroupNode(outer, children=(
        ShapeNode(Xfrm(off_x=E(5), off_y=E(5), ext_cx=E(10), ext_cy=E(4), rot=1800000, flip_v=True)),
        ShapeNode(Xfrm(off_x=E(1), off_y=E(2), ext_cx=E(30), ext_cy=0), is_connector=True),
    ))]
    first = resolve_tree(tree)
    second = resolve_tree(tree)
    assert first == second
    assert repr(first) == repr(second)
    for _, res in first:
        for v in _floats(res):
            assert not (v == 0.0 and math.copysign(1.0, v) < 0), res

    # sweep rotations and flip combinations: no -0.0 anywhere, rotation in [0, 360)
    for rot in range(0, FULL_TURN_UNITS, 900_000):
        for flip_h in (False, True):
            for flip_v in (False, True):
                xf = Xfrm(off_x=E(10), off_y=E(10), ext_cx=E(30), ext_cy=E(20),
                          rot=rot, flip_h=flip_h, flip_v=flip_v)
                for res in (resolve_shape(xf), resolve_connector(xf)):
                    for v in _floats(res):
                        assert not (v == 0.0 and math.copysign(1.0, v) < 0), (rot, res)
                    if isinstance(res, ResolvedShape):
                        assert 0.0 <= res.rotation_deg < 360.0, res


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
