import dataclasses
import math
import unittest
import xml.etree.ElementTree as ET

from geometry import (
    EMU_PER_POINT,
    GroupNode,
    ResolvedConnector,
    ResolvedShape,
    ShapeNode,
    Xfrm,
    emu_to_pt,
    resolve_connector,
    resolve_shape,
    resolve_tree,
)

DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
UNITS_PER_DEGREE = 60_000
FULL_TURN_UNITS = 360 * UNITS_PER_DEGREE


def emu(points):
    return round(points * EMU_PER_POINT)


def xfrm(off=(0, 0), ext=(0, 0), rot=0, flip_h=False, flip_v=False, child_off=(0, 0), child_ext=(0, 0)):
    return Xfrm(
        off_x=emu(off[0]),
        off_y=emu(off[1]),
        ext_cx=emu(ext[0]),
        ext_cy=emu(ext[1]),
        rot=round(rot * UNITS_PER_DEGREE),
        flip_h=flip_h,
        flip_v=flip_v,
        ch_off_x=emu(child_off[0]),
        ch_off_y=emu(child_off[1]),
        ch_ext_cx=emu(child_ext[0]),
        ch_ext_cy=emu(child_ext[1]),
    )


def rotated(x, y, degrees):
    angle = math.radians(degrees)
    return x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)


def through_frame(box, x, y):
    width, height = box.ext_cx / EMU_PER_POINT, box.ext_cy / EMU_PER_POINT
    dx = (x - width / 2) * (-1 if box.flip_h else 1)
    dy = (y - height / 2) * (-1 if box.flip_v else 1)
    rx, ry = rotated(dx, dy, box.rot / UNITS_PER_DEGREE)
    return box.off_x / EMU_PER_POINT + width / 2 + rx, box.off_y / EMU_PER_POINT + height / 2 + ry


def through_group(group, x, y):
    scale_x = group.ext_cx / group.ch_ext_cx if group.ch_ext_cx else 1.0
    scale_y = group.ext_cy / group.ch_ext_cy if group.ch_ext_cy else 1.0
    return through_frame(
        group,
        (x - group.ch_off_x / EMU_PER_POINT) * scale_x,
        (y - group.ch_off_y / EMU_PER_POINT) * scale_y,
    )


def matrix_corners(shape, ancestors=()):
    width, height = shape.ext_cx / EMU_PER_POINT, shape.ext_cy / EMU_PER_POINT
    corners = []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        point = through_frame(shape, x, y)
        for group in reversed(ancestors):
            point = through_group(group, *point)
        corners.append(point)
    return corners


def float_values(value):
    if isinstance(value, float):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from float_values(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from float_values(item)


class GeometryTestCase(unittest.TestCase):
    def assertPointsAlmostEqual(self, actual, expected, places=5):
        self.assertEqual(len(actual), len(expected))
        for (actual_x, actual_y), (expected_x, expected_y) in zip(actual, expected):
            self.assertAlmostEqual(actual_x, expected_x, places=places)
            self.assertAlmostEqual(actual_y, expected_y, places=places)

    def assertNoNegativeZero(self, value):
        for number in float_values(value):
            self.assertFalse(number == 0.0 and math.copysign(1.0, number) < 0, value)


class UnitsTest(unittest.TestCase):
    def test_emu_to_pt_uses_12700_emu_per_point(self):
        self.assertEqual(EMU_PER_POINT, 12_700)
        self.assertEqual(emu_to_pt(12_700), 1.0)
        self.assertEqual(emu_to_pt(914_400), 72.0)
        self.assertEqual(emu_to_pt(9_525), 0.75)


class XfrmTest(unittest.TestCase):
    def test_rotation_is_taken_modulo_a_full_turn(self):
        self.assertEqual(Xfrm(rot=0), Xfrm(rot=FULL_TURN_UNITS))
        self.assertEqual(Xfrm(rot=0), Xfrm(rot=-FULL_TURN_UNITS))
        self.assertEqual(Xfrm(rot=-5_400_000), Xfrm(rot=16_200_000))
        self.assertEqual(Xfrm(rot=16_200_000), Xfrm(rot=37_800_000))
        self.assertEqual(Xfrm(rot=27_000_000).rot, 5_400_000)

    def test_numeric_strings_equal_integers(self):
        from_text = Xfrm(off_x="1270000", off_y="635000", ext_cx="1016000", ext_cy="508000")
        from_numbers = Xfrm(off_x=1_270_000, off_y=635_000, ext_cx=1_016_000, ext_cy=508_000)
        self.assertEqual(from_text, from_numbers)
        self.assertEqual(resolve_shape(from_text), resolve_shape(from_numbers))
        self.assertEqual(Xfrm(off_x=12_700.0).off_x, 12_700)

    def test_flip_flags_accept_bool_text_and_none(self):
        for flag in (True, "1", "true"):
            self.assertIs(Xfrm(flip_h=flag).flip_h, True)
        for flag in (False, "0", "false", "", None):
            self.assertIs(Xfrm(flip_v=flag).flip_v, False)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            Xfrm(off_x=1.5)
        with self.assertRaises(ValueError):
            Xfrm(off_x="abc")
        with self.assertRaises(ValueError):
            Xfrm(flip_h="yes")
        with self.assertRaises(TypeError):
            Xfrm(off_x=True)

    def test_from_element_reads_an_xfrm_element(self):
        element = ET.fromstring(
            f'<a:xfrm xmlns:a="{DRAWING_NS}" rot="5400000" flipH="1">'
            '<a:off x="1270000" y="2540000"/><a:ext cx="12700000" cy="6350000"/></a:xfrm>'
        )
        expected = Xfrm(
            off_x=1_270_000, off_y=2_540_000, ext_cx=12_700_000, ext_cy=6_350_000,
            rot=5_400_000, flip_h=True,
        )
        self.assertEqual(Xfrm.from_element(element), expected)

    def test_from_element_ignores_the_namespace_prefix(self):
        element = ET.fromstring(
            f'<x:xfrm xmlns:x="{DRAWING_NS}" rot="5400000" flipH="1">'
            '<x:off x="1270000" y="2540000"/><x:ext cx="12700000" cy="6350000"/></x:xfrm>'
        )
        self.assertEqual(Xfrm.from_element(element).off_y, 2_540_000)
        self.assertTrue(Xfrm.from_element(element).flip_h)

    def test_from_element_reads_a_group_property_element(self):
        element = ET.fromstring(
            f'<p:grpSpPr xmlns:p="{PRESENTATION_NS}" xmlns:a="{DRAWING_NS}"><a:xfrm>'
            '<a:off x="1270000" y="1270000"/><a:ext cx="2540000" cy="1270000"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="1270000" cy="1270000"/></a:xfrm>'
            '<a:noFill/></p:grpSpPr>'
        )
        expected = Xfrm(
            off_x=1_270_000, off_y=1_270_000, ext_cx=2_540_000, ext_cy=1_270_000,
            ch_ext_cx=1_270_000, ch_ext_cy=1_270_000,
        )
        self.assertEqual(Xfrm.from_element(element), expected)

    def test_from_element_defaults_when_nothing_is_set(self):
        self.assertEqual(Xfrm.from_element(ET.fromstring(f'<p:xfrm xmlns:p="{PRESENTATION_NS}"/>')), Xfrm())


class SingleShapeTest(GeometryTestCase):
    def test_unrotated_shape(self):
        shape = resolve_shape(xfrm(off=(10, 20), ext=(100, 50)))
        self.assertEqual(shape.bbox, (10.0, 20.0, 110.0, 70.0))
        self.assertEqual(shape.corners, ((10.0, 20.0), (110.0, 20.0), (110.0, 70.0), (10.0, 70.0)))
        self.assertEqual(shape.center, (60.0, 45.0))
        self.assertEqual((shape.width, shape.height), (100.0, 50.0))
        self.assertEqual(shape.rotation_deg, 0.0)
        self.assertFalse(shape.mirrored)

    def test_right_angle_rotations(self):
        cases = {
            90: ((125.0, 75.0, 175.0, 175.0), ((175.0, 75.0), (175.0, 175.0), (125.0, 175.0), (125.0, 75.0))),
            180: ((100.0, 100.0, 200.0, 150.0), ((200.0, 150.0), (100.0, 150.0), (100.0, 100.0), (200.0, 100.0))),
            270: ((125.0, 75.0, 175.0, 175.0), ((125.0, 175.0), (125.0, 75.0), (175.0, 75.0), (175.0, 175.0))),
        }
        for degrees, (bbox, corners) in cases.items():
            with self.subTest(degrees=degrees):
                shape = resolve_shape(xfrm(off=(100, 100), ext=(100, 50), rot=degrees))
                self.assertEqual(shape.bbox, bbox)
                self.assertEqual(shape.corners, corners)
                self.assertEqual(shape.center, (150.0, 125.0))
                self.assertEqual((shape.width, shape.height), (100.0, 50.0))
                self.assertEqual(shape.rotation_deg, float(degrees))

    def test_equivalent_rotations_resolve_identically(self):
        box = {"off": (100, 50), "ext": (80, 40)}
        self.assertEqual(resolve_shape(xfrm(rot=0, **box)), resolve_shape(xfrm(rot=360, **box)))
        self.assertEqual(resolve_shape(xfrm(rot=-90, **box)), resolve_shape(xfrm(rot=270, **box)))

    def test_both_flips_equal_a_half_turn(self):
        box = {"off": (100, 50), "ext": (80, 40)}
        flipped = resolve_shape(xfrm(flip_h=True, flip_v=True, **box))
        self.assertEqual(flipped, resolve_shape(xfrm(rot=180, **box)))
        self.assertFalse(flipped.mirrored)
        self.assertEqual(flipped.rotation_deg, 180.0)

    def test_vertical_flip_is_a_mirror_with_a_half_turn(self):
        shape = resolve_shape(xfrm(off=(100, 50), ext=(80, 40), flip_v=True))
        self.assertTrue(shape.mirrored)
        self.assertEqual(shape.rotation_deg, 180.0)


class GroupTest(GeometryTestCase):
    def test_offset_and_child_scale(self):
        group = xfrm(off=(100, 100), ext=(200, 100), child_ext=(100, 100))
        child = xfrm(off=(50, 50), ext=(50, 50))
        self.assertEqual(resolve_shape(group).bbox, (100.0, 100.0, 300.0, 200.0))
        self.assertEqual(resolve_shape(child, [group]).bbox, (200.0, 150.0, 300.0, 200.0))

    def test_group_matching_its_child_space_changes_nothing(self):
        shape = xfrm(off=(110, 110), ext=(20, 20))
        identity = xfrm(off=(100, 100), ext=(50, 50), child_off=(100, 100), child_ext=(50, 50))
        self.assertEqual(resolve_shape(shape, [identity]), resolve_shape(shape))

    def test_group_without_child_extent_only_translates(self):
        group = xfrm(off=(100, 100))
        child = xfrm(off=(10, 10), ext=(20, 20))
        self.assertEqual(resolve_shape(child, [group]).bbox, (110.0, 110.0, 130.0, 130.0))

    def test_all_zero_root_group_is_the_identity(self):
        child = xfrm(off=(10, 10), ext=(20, 20))
        self.assertEqual(resolve_shape(child, [Xfrm()]), resolve_shape(child))
        self.assertEqual(resolve_shape(child, [Xfrm()]).bbox, (10.0, 10.0, 30.0, 30.0))

    def test_group_with_zero_extent_collapses_children_to_a_point(self):
        group = xfrm(off=(100, 100), child_ext=(100, 100))
        child = xfrm(off=(10, 10), ext=(20, 20))
        collapsed = resolve_shape(child, [group])
        self.assertEqual(collapsed.bbox, (100.0, 100.0, 100.0, 100.0))
        self.assertEqual((collapsed.width, collapsed.height), (0.0, 0.0))
        self.assertEqual(len(resolve_tree([GroupNode(group, children=(ShapeNode(child),))])), 2)

    def test_nested_groups(self):
        outer = xfrm(off=(100, 100), ext=(200, 200), child_ext=(100, 100))
        inner = xfrm(off=(10, 10), ext=(40, 40), child_ext=(20, 20))
        shape = xfrm(off=(5, 5), ext=(10, 10))
        self.assertEqual(resolve_shape(shape, [outer, inner]).bbox, (140.0, 140.0, 180.0, 180.0))

    def test_tree_emits_each_group_before_its_children(self):
        outer = xfrm(off=(100, 100), ext=(200, 200), child_ext=(100, 100))
        inner = xfrm(off=(10, 10), ext=(40, 40), child_ext=(20, 20))
        shape = xfrm(off=(5, 5), ext=(10, 10))
        tree = [GroupNode(outer, name="outer", children=(
            GroupNode(inner, name="inner", children=(ShapeNode(shape, name="shape"),)),
        ))]
        resolved = resolve_tree(tree)
        self.assertEqual([node.name for node, _ in resolved], ["outer", "inner", "shape"])
        self.assertEqual(
            [result.bbox for _, result in resolved],
            [(100.0, 100.0, 300.0, 300.0), (120.0, 120.0, 200.0, 200.0), (140.0, 140.0, 180.0, 180.0)],
        )
        self.assertEqual(resolved[2][1], resolve_shape(shape, [outer, inner]))

    def test_group_rotation_turns_its_children(self):
        group = xfrm(off=(100, 100), ext=(100, 100), rot=90, child_ext=(100, 100))
        child = resolve_shape(xfrm(ext=(50, 50)), [group])
        self.assertEqual(child.bbox, (150.0, 100.0, 200.0, 150.0))
        self.assertEqual(child.rotation_deg, 90.0)

    def test_group_mirror_flips_its_children(self):
        group = xfrm(ext=(100, 100), flip_h=True, child_ext=(100, 100))
        child = resolve_shape(xfrm(ext=(30, 30)), [group])
        self.assertEqual(child.bbox, (70.0, 0.0, 100.0, 30.0))
        self.assertTrue(child.mirrored)
        self.assertEqual(child.rotation_deg, 0.0)

    def test_flip_is_applied_before_rotation(self):
        group = xfrm(ext=(100, 100), rot=90, flip_h=True, child_ext=(100, 100))
        child = resolve_shape(xfrm(ext=(20, 10)), [group])
        self.assertEqual(child.bbox, (90.0, 80.0, 100.0, 100.0))
        self.assertTrue(child.mirrored)
        self.assertEqual(child.rotation_deg, 90.0)

    def test_rotations_add_across_a_uniform_group(self):
        group = xfrm(off=(100, 100), ext=(100, 100), rot=60, child_ext=(100, 100))
        child = resolve_shape(xfrm(ext=(50, 50), rot=30), [group])
        self.assertAlmostEqual(child.rotation_deg, 90.0, places=6)


class ConnectorTest(GeometryTestCase):
    def test_endpoints_follow_the_flips(self):
        cases = [
            (False, False, (0.0, 0.0), (100.0, 50.0)),
            (True, False, (100.0, 0.0), (0.0, 50.0)),
            (False, True, (0.0, 50.0), (100.0, 0.0)),
            (True, True, (100.0, 50.0), (0.0, 0.0)),
        ]
        for flip_h, flip_v, start, end in cases:
            with self.subTest(flip_h=flip_h, flip_v=flip_v):
                connector = resolve_connector(xfrm(ext=(100, 50), flip_h=flip_h, flip_v=flip_v))
                self.assertIsInstance(connector, ResolvedConnector)
                self.assertIsInstance(connector.box, ResolvedShape)
                self.assertEqual((connector.start, connector.end), (start, end))

    def test_horizontal_and_vertical_connectors(self):
        horizontal = resolve_connector(xfrm(off=(10, 20), ext=(100, 0)))
        vertical = resolve_connector(xfrm(off=(10, 20), ext=(0, 80)))
        self.assertEqual((horizontal.start, horizontal.end), ((10.0, 20.0), (110.0, 20.0)))
        self.assertEqual((vertical.start, vertical.end), ((10.0, 20.0), (10.0, 100.0)))

    def test_connector_in_a_rotated_group(self):
        group = xfrm(ext=(100, 100), rot=90, child_ext=(100, 100))
        shape = xfrm(ext=(20, 10))
        connector = resolve_connector(shape, [group])
        self.assertEqual(connector.start, (100.0, 0.0))
        self.assertEqual(connector.end, (90.0, 20.0))
        self.assertEqual(connector.box, resolve_shape(shape, [group]))

    def test_connector_in_a_non_uniform_group(self):
        group = xfrm(ext=(200, 100), child_ext=(100, 100))
        for flip_h in (False, True):
            for flip_v in (False, True):
                with self.subTest(flip_h=flip_h, flip_v=flip_v):
                    shape = xfrm(ext=(100, 50), rot=30, flip_h=flip_h, flip_v=flip_v)
                    connector = resolve_connector(shape, [group])
                    self.assertEqual(connector.start, connector.box.corners[0])
                    self.assertEqual(connector.end, connector.box.corners[2])
                    self.assertEqual(connector.box, resolve_shape(shape, [group]))
        unflipped = resolve_connector(xfrm(ext=(100, 50), rot=30), [group])
        dx, dy = rotated(-100, -25, 30)
        self.assertPointsAlmostEqual([unflipped.start], [(100 + dx, 25 + dy)])


class NonUniformGroupTest(GeometryTestCase):
    def setUp(self):
        self.outer = xfrm(ext=(200, 100), child_ext=(100, 100))

    def test_rotated_child_is_scaled_before_rotating(self):
        child = xfrm(ext=(100, 50), rot=30)
        shape = resolve_shape(child, [self.outer])
        self.assertEqual(shape.center, (100.0, 25.0))
        self.assertEqual((shape.width, shape.height), (200.0, 50.0))
        self.assertEqual(shape.rotation_deg, 30.0)
        self.assertFalse(shape.mirrored)
        half_width = 100 * math.cos(math.radians(30)) + 25 * math.sin(math.radians(30))
        half_height = 100 * math.sin(math.radians(30)) + 25 * math.cos(math.radians(30))
        expected = (100 - half_width, 25 - half_height, 100 + half_width, 25 + half_height)
        for actual, wanted in zip(shape.bbox, expected):
            self.assertAlmostEqual(actual, wanted, places=5)

    def test_rotated_child_differs_from_plain_matrix_composition(self):
        child = xfrm(ext=(100, 50), rot=30)
        shape = resolve_shape(child, [self.outer])
        gaps = [
            max(abs(a - b) for a, b in zip(actual, wanted))
            for actual, wanted in zip(shape.corners, matrix_corners(child, [self.outer]))
        ]
        self.assertGreater(max(gaps), 1.0)

    def test_right_angle_child_swaps_width_and_height(self):
        child = xfrm(ext=(100, 50), rot=90)
        shape = resolve_shape(child, [self.outer])
        self.assertEqual(shape.center, (100.0, 25.0))
        self.assertEqual((shape.width, shape.height), (100.0, 100.0))
        self.assertEqual(shape.rotation_deg, 90.0)
        self.assertEqual(shape.bbox, (50.0, -25.0, 150.0, 75.0))
        self.assertPointsAlmostEqual(shape.corners, matrix_corners(child, [self.outer]))

    def test_axis_swap_boundaries(self):
        cases = [
            (2_699_999, (200.0, 50.0)),
            (2_700_000, (100.0, 100.0)),
            (8_099_999, (100.0, 100.0)),
            (8_100_000, (200.0, 50.0)),
            (13_499_999, (200.0, 50.0)),
            (13_500_000, (100.0, 100.0)),
            (18_899_999, (100.0, 100.0)),
            (18_900_000, (200.0, 50.0)),
        ]
        for rot, size in cases:
            with self.subTest(rot=rot):
                child = Xfrm(ext_cx=emu(100), ext_cy=emu(50), rot=rot)
                shape = resolve_shape(child, [self.outer])
                self.assertEqual((shape.width, shape.height), size)

    def test_nested_groups_with_a_rotated_inner_group(self):
        inner = xfrm(ext=(100, 50), rot=30, child_ext=(100, 50))
        shape = xfrm(ext=(50, 25))
        resolved = resolve_shape(shape, [self.outer, inner])
        dx, dy = rotated(50 - 100, 12.5 - 25, 30)
        self.assertPointsAlmostEqual([resolved.center], [(100 + dx, 25 + dy)])
        self.assertEqual((resolved.width, resolved.height), (100.0, 25.0))
        self.assertEqual(resolved.rotation_deg, 30.0)

    def test_rotated_group_inside_the_swap_band(self):
        inner = xfrm(ext=(100, 50), rot=90, child_ext=(100, 50))
        shape = xfrm(ext=(50, 25))
        resolved = resolve_tree([GroupNode(self.outer, children=(GroupNode(inner, children=(ShapeNode(shape),)),))])
        inner_box, shape_box = resolved[1][1], resolved[2][1]
        self.assertEqual(inner_box.bbox, (50.0, -25.0, 150.0, 75.0))
        self.assertEqual((inner_box.width, inner_box.height, inner_box.rotation_deg), (100.0, 100.0, 90.0))
        self.assertEqual(shape_box.center, (125.0, 0.0))
        self.assertEqual((shape_box.width, shape_box.height, shape_box.rotation_deg), (50.0, 50.0, 90.0))
        self.assertEqual(shape_box, resolve_shape(shape, [self.outer, inner]))

    def test_unrotated_and_half_turn_children_match_plain_matrix_composition(self):
        group = xfrm(off=(5, 7), ext=(200, 100), rot=30, child_off=(30, 10), child_ext=(100, 100))
        for rot in (0, 180):
            for flip_h in (False, True):
                for flip_v in (False, True):
                    with self.subTest(rot=rot, flip_h=flip_h, flip_v=flip_v):
                        child = xfrm(off=(40, 20), ext=(30, 12), rot=rot, flip_h=flip_h, flip_v=flip_v)
                        shape = resolve_shape(child, [group])
                        self.assertPointsAlmostEqual(shape.corners, matrix_corners(child, [group]))

    def test_uniform_scale_matches_plain_matrix_composition(self):
        group = xfrm(ext=(200, 200), child_ext=(100, 100))
        child = xfrm(off=(10, 10), ext=(40, 20), rot=30)
        self.assertPointsAlmostEqual(
            resolve_shape(child, [group]).corners, matrix_corners(child, [group])
        )
        outer = xfrm(off=(20, 30), ext=(200, 200), child_ext=(100, 100))
        middle = xfrm(off=(5, 8), ext=(40, 40), rot=30, child_off=(5, 5), child_ext=(20, 20))
        leaf = xfrm(off=(3, 4), ext=(10, 6), rot=20, flip_v=True)
        self.assertPointsAlmostEqual(
            resolve_shape(leaf, [outer, middle]).corners, matrix_corners(leaf, [outer, middle])
        )

    def test_tree_matches_the_explicit_ancestor_chain(self):
        inner = xfrm(ext=(100, 50), rot=30, child_ext=(100, 50))
        shape = xfrm(ext=(50, 25))
        connector = xfrm(off=(4, 2), ext=(20, 10), rot=30)
        tree = [GroupNode(self.outer, name="outer", children=(
            GroupNode(inner, name="inner", children=(
                ShapeNode(shape, name="shape"),
                ShapeNode(connector, name="connector", is_connector=True),
            )),
        ))]
        resolved = resolve_tree(tree)
        self.assertEqual([node.name for node, _ in resolved], ["outer", "inner", "shape", "connector"])
        self.assertEqual(resolved[0][1], resolve_shape(self.outer))
        self.assertEqual(resolved[1][1], resolve_shape(inner, [self.outer]))
        self.assertEqual(resolved[2][1], resolve_shape(shape, [self.outer, inner]))
        self.assertEqual(resolved[3][1], resolve_connector(connector, [self.outer, inner]))
        inner_box = resolved[1][1]
        self.assertEqual(
            (inner_box.center, inner_box.width, inner_box.height, inner_box.rotation_deg),
            ((100.0, 25.0), 200.0, 50.0, 30.0),
        )

    def test_tree_starting_inside_ancestors_matches_the_explicit_chain(self):
        inner = xfrm(ext=(100, 50), rot=30, child_ext=(100, 50))
        shape = xfrm(ext=(50, 25))
        resolved = resolve_tree([GroupNode(inner, children=(ShapeNode(shape),))], ancestors=[self.outer])
        self.assertEqual(resolved[0][1], resolve_shape(inner, [self.outer]))
        self.assertEqual(resolved[1][1], resolve_shape(shape, [self.outer, inner]))


class DeterminismTest(GeometryTestCase):
    def test_repeated_resolution_is_identical(self):
        group = xfrm(off=(100, 100), ext=(100, 60), rot=90, flip_h=True, child_ext=(50, 30))
        tree = [GroupNode(group, children=(
            ShapeNode(xfrm(off=(5, 5), ext=(10, 4), rot=30, flip_v=True)),
            ShapeNode(xfrm(off=(1, 2), ext=(30, 0)), is_connector=True),
        ))]
        first, second = resolve_tree(tree), resolve_tree(tree)
        self.assertEqual(first, second)
        self.assertEqual(repr(first), repr(second))
        self.assertNoNegativeZero(first)

    def test_no_negative_zero_and_rotation_stays_in_range(self):
        for rot in range(0, FULL_TURN_UNITS, 900_000):
            for flip_h in (False, True):
                for flip_v in (False, True):
                    with self.subTest(rot=rot, flip_h=flip_h, flip_v=flip_v):
                        box = Xfrm(
                            off_x=emu(10), off_y=emu(10), ext_cx=emu(30), ext_cy=emu(20),
                            rot=rot, flip_h=flip_h, flip_v=flip_v,
                        )
                        shape, connector = resolve_shape(box), resolve_connector(box)
                        self.assertNoNegativeZero(shape)
                        self.assertNoNegativeZero(connector)
                        self.assertTrue(0.0 <= shape.rotation_deg < 360.0)
