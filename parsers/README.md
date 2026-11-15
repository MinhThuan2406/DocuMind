# parsers

Native parsers for office formats. The current package covers PPTX ingestion for GitHub ticket #8, "Build Native PPTX Slide and Shape Tree Ingestion Engine".

## 1. Purpose, scope and layout

The PPTX parser reads a `.pptx` file from its XML parts and produces:

- a typed model of the presentation, its slides and shapes (`pptx_extractor.py`), keeping raw identifiers and transform attributes next to resolved geometry;
- a Unified JSON IR dictionary (`pptx_ir.py`) built from that model.

Covered: slide order, the shape tree of each slide (groups, shapes, connectors, pictures and graphic frames, including tables), resolved geometry through the `geometry` package, connector start and end references, table cells, and position inheritance for placeholders. Not covered: layouts, masters and notes content, node-edge graphs and diagram grouping, which belong to other tickets (section 7).

```
parsers/
  __init__.py              package docstring
  pptx_extractor.py        extract_presentation(path) and the model dataclasses
  pptx_ir.py               extract_ir(path), returns the Unified JSON IR dict
  test_pptx_extractor.py   extractor tests
  test_pptx_ir.py          IR tests
  fixtures/
    build_fixtures.py      writes fixtures/pptx/*.pptx and fixtures/manifest.json
    manifest.json          expected values for every fixture
    test_fixtures.py       checks the fixture files against the manifest
    pptx/                  nine synthetic decks (section 6)
```

## 2. Usage and tests

Run everything from the repository root. The commands work in PowerShell and in cmd.

```python
from parsers.pptx_extractor import extract_presentation
from parsers.pptx_ir import extract_ir

deck = extract_presentation("parsers/fixtures/pptx/slide_order.pptx")
print([slide.number for slide in deck.slides])

ir = extract_ir("parsers/fixtures/pptx/slide_order.pptx")
print(ir["doc_type"], len(ir["elements"]))
```

`extract_presentation(path)` returns a `Presentation`. `extract_ir(path)` returns a plain dict that can be passed to `json.dumps`. Both accept a `str` or a `pathlib.Path`; neither writes files or logs. `pptx_ir` uses a relative import, so import it as `parsers.pptx_ir` from the repository root.

```
.\.venv\Scripts\python -B parsers\test_pptx_extractor.py
.\.venv\Scripts\python -B parsers\test_pptx_ir.py
.\.venv\Scripts\python -B parsers\fixtures\test_fixtures.py
.\.venv\Scripts\python -B -m unittest geometry.test_transform
```

The three scripts print `PASS` or `FAIL` for each test, an `N/N passed` summary, and exit with status 1 on any failure. The unittest run ends with `OK`.

## 3. Extractor model

`extract_presentation` returns frozen dataclasses:

- `Presentation(width_pt, height_pt, slides)`: page size in points from `p:sldSz`; None when that element is absent.
- `Slide(number, slide_id, hidden, shapes)`: `number` is 1-based in `p:sldIdLst` order; `slide_id` is that list entry's `id`; `hidden` is True when the slide's `show` attribute is "0" or "false". Hidden slides are included.
- `Shape(id, name, tag, parent_id, parent_z_order, z_order, text, raw_xfrm, xfrm_source, alternate_content, geometry, start_connection, end_connection, table)`.
- `Connection(target_id, site_index, target_z_order, status)`.
- `Table(num_rows, num_cols, first_row_header, cells)` and `TableCell(row, col, rowspan, colspan, text)`.

Rules:

- Slide order is the order of `p:sldIdLst`, not the order of part names. python-pptx renames slide parts when it loads a file, so part names are not exposed.
- `id` and `name` are the raw `p:cNvPr` values and are never renumbered. Duplicate ids are kept, and each shape keeps its own `z_order`. `name` is "" when absent.
- `z_order` is the 0-based document-order index of the shape within its slide. It counts groups, their children, and shapes in the selected `mc:AlternateContent` branch. `parent_z_order` is the `z_order` of the enclosing `p:grpSp`, or None at top level.
- Shapes are the `p:sp`, `p:cxnSp`, `p:grpSp`, `p:pic` and `p:graphicFrame` children of `p:spTree` and of `p:grpSp`. Other children are skipped.
- `raw_xfrm` holds the transform as written. `off`, `ext`, `chOff` and `chExt` are `{x, y}` or `{cx, cy}` maps of strings; `rot`, `flipH` and `flipV` are strings. Each value is null when absent. The whole field is None when the element has no transform element (`a:xfrm`, or `p:xfrm` on a graphic frame).
- `geometry` comes from the `geometry` package, using the transform and the enclosing group chain. It is None only when `xfrm_source` is "missing". Shapes get `ResolvedShape` (corners, bbox, center, width, height, rotation_deg, mirrored); connectors get `ResolvedConnector` (start, end, box). A vertical flip is expressed as a horizontal flip plus a half turn (geometry/README.md, section 4).
- `xfrm_source` is "explicit" when the element has its own transform; "inherited" when it is a placeholder without one and python-pptx finds a slide layout placeholder with the same `idx`; "missing" otherwise. Inherited frames give position and size only. Rotation and flips are not inherited, and a placeholder whose frame exists only on the master ends up "missing".
- `p:graphicFrame` ignores `rot`, `flipH` and `flipV` for geometry (MS-OI29500). `raw_xfrm` still keeps the values as written.
- `mc:AlternateContent` yields exactly one branch: the first `mc:Choice`, otherwise the `mc:Fallback`, that directly contains a recognised shape. If neither does, nothing is emitted. Shapes read from that branch carry `alternate_content` = `{"branch": "Choice" or "Fallback", "requires": ...}`, where `requires` is the Choice's `Requires` attribute (None for Fallback). `Requires` is recorded and never evaluated.
- Connector references come from `a:stCxn` and `a:endCxn` in `p:cNvCxnSpPr`. Status is "resolved" when exactly one shape on the same slide has that id (`target_z_order` is set), "dangling" when none does, "ambiguous" when more than one does, and "invalid" when `id` or `idx` is missing or not an integer. `site_index` is the raw index and is never converted to coordinates. `id` and `idx` are parsed with Python `int()`, so surrounding spaces and a sign are accepted. A `p:cNvPr` `id` must be an integer; extraction raises if it is not.
- `text` is None without a `p:txBody`. Otherwise paragraphs are joined with a newline. An `a:br` inside a paragraph becomes a vertical tab, as in python-pptx's `paragraph.text`. Text inside `a:fld` is included.
- `table` is set only on a `p:graphicFrame` that holds an `a:tbl`. `num_rows` counts `a:tr`, `num_cols` counts `a:gridCol`, and `first_row_header` is True when `a:tblPr` has `firstRow` "1" or "true".
- Table cells: only origin cells are emitted. A cell whose `hMerge` or `vMerge` is "1" or "true" is a continuation and is skipped, even if it carries spans; the merge flags decide. `rowspan` comes from `rowSpan` and `colspan` from `gridSpan`, both defaulting to 1. A row without `a:tc` yields no cells. Cells have no bounding boxes.

## 4. IR mapping

`extract_ir(path)` returns these top-level keys and no others: `doc_id`, `source_file`, `doc_type`, `metadata`, `elements`, `tables`, `diagrams`.

| Key | Value |
|---|---|
| `doc_id` | file stem |
| `source_file` | file name |
| `doc_type` | "pptx" |
| `diagrams` | always an empty list |

Metadata keys (`metadata`): `page_width`, `page_height` (points; null without `p:sldSz`), `coord_unit` ("pt"), `parser` ("pptx_extractor (python-pptx 1.0.2)", with the installed version), `reading_order_status` ("z_order"), and `slides` (one `{number, slide_id, hidden}` per slide).

Ids are numbered from 1 across the document, in slide order and then z-order: elements as `elem_NNNN` and tables as `tbl_NNNN`, each with a four-digit minimum.

Element keys (`elements[]`):

| Key | Value |
|---|---|
| `id` | `elem_NNNN` |
| `type` | "shape" (`p:sp`), "connector" (`p:cxnSp`), "group" (`p:grpSp`), "picture" (`p:pic`), "table" (graphic frame with a table), "graphic_frame" (other graphic frame) |
| `page` | slide number |
| `bbox` | [x0, y0, x1, y1] in points from the resolved geometry (the box for a connector); null without geometry |
| `reading_order` | z_order within the slide |
| `level` | 1 at top level; parent level + 1 inside a group |
| `parent_id` | element id of the enclosing group, or null |
| `text` | shape text, "" when none; for a table element, "[Table tbl_NNNN: RxC]" |
| `source_ref` | `{file, page, bbox, shape_id}` |
| `metadata` | element metadata, below |
| `table_id` | table elements only |

Element metadata keys (`elements[].metadata`): `tag`, `shape_id`, `shape_name`, `z_order`, `slide_id`, `raw_xfrm`, `xfrm_source`, `rotation_deg`, `mirrored`, `alternate_content`. Table elements also carry `table_id`. Connector elements (`cxnSp`) also carry `start` and `end` (points), `start_connection` and `end_connection`.

Connection keys (`start_connection`, `end_connection`): `target_id`, `site_index`, `status`, and `target_element_id` (the element id of the target when status is "resolved", otherwise null).

Table keys (`tables[]`): `id`, `page`, `bbox` (same as the owning element), `num_rows`, `num_cols`, `cells`, `source_ref`, and `metadata` (`{element_id}` only).

Cell keys (`tables[].cells[]`): `row`, `col`, `rowspan`, `colspan`, `text`, `is_header` (first row, when `firstRow` is set), `is_merged` (rowspan or colspan above 1), and `bbox` (always null).

Differences from the IR specification in `docs/01_architecture_and_ir_schema.md` (branch `feat/structure-and-diagram-recovery`, last changed in commit a3970e5; the file is written in Vietnamese):

- `type` uses the native tag names above, not layout labels such as `section_header`, `text` or `caption`.
- `bbox` is in points, not page pixels; `metadata.coord_unit` says so.
- `reading_order` is the z-order within each slide and restarts at 0 on every slide. `metadata.reading_order_status` is "z_order" to mark this.
- `level` and `parent_id` reflect group nesting, not the semantic hierarchy (title, section).
- Elements have no `confidence`. Tables have no `html`, and table `metadata` has no `caption_id`, `caption_text` or `link_warning`.
- Cell `bbox` is always null.
- `diagrams` is always empty; there is no diagram grouping and no node-edge data.
- `metadata.raw_order` is not emitted; `z_order` plays its role.
- The specification's processing fields (`category`, `description`, `processing_time_sec`, `custom_tags`, `extra`) are not emitted.
- Fields not in the specification: `source_ref.shape_id`, the element metadata keys above, connector `start`, `end`, `start_connection` and `end_connection`, and `metadata.slides`.

## 5. Ticket #8 traceability

Every test named below exists under `parsers/` and passed in the verification run.

| Ticket item | Code | Tests |
|---|---|---|
| DoD: slide and shape ingestion | `pptx_extractor._extract_slide`, `_walk`, `_describe` | `test_shapes_match_manifest_for_every_fixture`, `test_slide_order_follows_presentation_list` |
| DoD: raw shape and connector identifiers preserved | `Shape.id`, `Connection.target_id`, IR `shape_id`, `source_ref.shape_id` | `test_shapes_match_manifest_for_every_fixture`, `test_connection_references_match_manifest_for_every_fixture`, `test_connector_endpoints_and_target_element_ids_for_every_fixture` |
| DoD: geometry metadata normalised | `Shape.geometry` from `geometry.resolve_shape` and `resolve_connector` | `test_geometry_matches_manifest_chain_for_every_fixture`, `test_bbox_and_connector_endpoints_are_hand_computed` |
| DoD: elements mapped into Unified IR | `pptx_ir.extract_ir` | `test_top_level_keys_source_file_and_json_round_trip`, `test_element_ids_pages_reading_order_parents_and_levels_for_every_fixture` |
| Acceptance: slide ordering preserved | `Slide.number`, `Slide.slide_id` from `p:sldIdLst` | `test_slide_order_follows_presentation_list`, `test_slide_order_and_hidden_flags_follow_presentation_order` |
| Acceptance: p:sp and p:cxnSp distinguished | `Shape.tag`, IR `type` | `test_sp_and_cxnsp_stay_distinct`, `test_type_mapping_and_bbox_follow_native_tags` |
| Acceptance: shape coordinates and extents extracted | `Shape.raw_xfrm`, `Shape.geometry`, IR `bbox` | `test_bbox_and_connector_endpoints_are_hand_computed`, `test_geometry_matches_manifest_chain_for_every_fixture` |
| Acceptance: rotation and flip metadata preserved | `raw_xfrm` (`rot`, `flipH`, `flipV`), `rotation_deg`, `mirrored` | `test_flip_and_rotation_keep_raw_values_and_normalise_geometry`, `test_raw_flip_flags_survive_next_to_mirrored_geometry` |
| Acceptance: connector start and end extracted where explicit | `Connection`, IR `start_connection`, `end_connection` | `test_shapes_and_connectors_reference_statuses`, `test_connection_references_match_manifest_for_every_fixture` |
| Acceptance: ungrouped shapes and loose connectors identifiable | `z_order`, IR `elem_NNNN` | `test_duplicate_ids_keep_every_shape_with_distinct_z_order`, `test_element_ids_pages_reading_order_parents_and_levels_for_every_fixture` |
| Deliverable: PPTX fixtures | `parsers/fixtures/` | `test_fixture_files_exist_and_stay_small`, `test_manifest_covers_every_pptx_on_disk` |
| Edge-case regressions | `extract_presentation`, `_paragraph_text`, `_inherited_frames` | `test_missing_slide_size_gives_null_page_size`, `test_line_breaks_become_vertical_tabs_in_text_and_cells`, `test_top_level_content_part_does_not_stop_extraction`, `test_placeholder_inside_alternate_content_inherits_layout_geometry` |

## 6. Test data

Nine fixtures are in `parsers/fixtures/pptx/`, all written by `build_fixtures.py`:

| File | Content |
|---|---|
| `slide_order.pptx` | three slides whose part names differ from presentation order (C, A, B) |
| `hidden_slide.pptx` | two slides; the second has `show="0"` |
| `shapes_and_connectors.pptx` | four shapes and seven connectors: both ends, start only, loose, dangling, flipV, flipH, rot 90 |
| `nested_groups.pptx` | two nested groups with child scaling and a rotated inner group; connectors inside and across groups |
| `flip_rot.pptx` | six shapes with flips and rotations, including rot 390 with flipV |
| `tables_merged.pptx` | 4x4 table with gridSpan, rowSpan, hMerge and vMerge; first row header |
| `placeholders_inherit.pptx` | title placeholder without xfrm (inherited); body placeholder with explicit xfrm |
| `alternate_content.pptx` | one AlternateContent with a Choice (`Requires` "a14") and a Fallback, plus one ordinary shape |
| `duplicate_ids.pptx` | two shapes with id 5, a connector referencing id 5 (ambiguous), and an anchor shape |

`manifest.json` is written by `build_fixtures.py` from the values the builder passes to the XML. It is not parsed back from the saved files. The decks are made with python-pptx plus lxml patches. They were checked against Microsoft PowerPoint 16.0 as described in section 8; they have not been opened in LibreOffice.

Optional local check: the Apache POI slideshow test data at https://github.com/apache/poi/tree/trunk/test-data/slideshow can be run through the extractor on a local copy. Those files are not stored in this repository. License: Apache POI project, license not re-verified here.

## 7. Known limits and assumptions

- The flip-then-rotate order and the connector start and end convention come from the `geometry` package (geometry/README.md, section 12). Section 8 lists what PowerPoint 16.0 confirmed; no other PowerPoint version or product was consulted.
- The AlternateContent branch rule is covered only by synthetic tests. The Apache POI files with AlternateContent keep it outside the shape tree, so they do not exercise the rule. PowerPoint 16.0 opened the synthetic fixture and kept the Choice branch, and a deck saved by PowerPoint held no AlternateContent in its shape tree.
- `Requires` is not evaluated. The first Choice that holds a recognised shape is taken, whatever its `Requires` value.
- A Choice that holds only a nested AlternateContent is not descended into; the Fallback is used instead.
- A `p:contentPart` in the shape tree is skipped; it is not an element.
- Only slide shape trees are read. Layouts, masters and notes are not extracted; a layout is consulted only for placeholder position and size.
- Partial transforms are not completed. An absent `a:off` or `a:ext` is read as 0, `xfrm_source` stays "explicit", and the matching `raw_xfrm` field is null.
- A connector reference is resolved only against shapes on its own slide; a reference to another slide's shape is "dangling".
- `id` and `idx` in connector references are parsed leniently (section 3).
- Element `bbox` is the extent of the resolved corners. For a rotated shape it equals the frame at 0 and 180 degrees, has width and height swapped at 90 and 270 degrees, and is larger at other angles.
- A presentation without `p:sldSz` has null page size; its shapes are still read.
- Hidden slides are included. Their elements are emitted with `page` set, and `metadata.slides` flags them.
- Reading order is the z-order within each slide. There is no node-edge graph and no diagram grouping; those belong to other tickets.
- Rows without cells keep their count in `num_rows` and produce no cell entries.
- A shape without `p:txBody` has `text` None in the model and "" in the IR.
- Verified with python-pptx 1.0.2 and lxml 6.1.3 only.

## 8. Verification against PowerPoint

A read-only check was run on 2026-11-13 with Microsoft PowerPoint 16.0 through COM, on copies of the fixtures and on one deck authored in PowerPoint. No repository file was modified.

Confirmed:

- Flip-then-rotate order: with right-triangle shapes rendered by PowerPoint, the extractor's corners reproduce the rendered shape (IoU 0.99) for `Rot 45 Flip H` and `Rot 390 Flip V`; the opposite order gives 0.14 and 0.21. The other four shapes of `flip_rot.pptx` match at IoU 0.98 to 0.99 but do not tell the two orders apart.
- Connector end: the end arrowhead drawn by PowerPoint lies at the extractor's `end` point for four connectors (flipV, flipH, rot 90, and one connected at both ends). The arrowhead centroid sits about 7 pt back from the tip, so this is a distance check, not an exact position.
- Connections: targets and sites agree for every connected side in the fixtures, including dangling and ambiguous references, and in the PowerPoint-authored deck (9 sides). The COM site number equals `site_index` plus 1.
- Geometry: centre, width, height and rotation agree to 0.0001 pt for the shapes of `nested_groups.pptx` and `shapes_and_connectors.pptx` and for the PowerPoint-authored deck (nested groups, a non-uniform group scale, rotated and flipped shapes), apart from the notation difference below.
- Slide order, hidden flag, table size and origin-cell widths, title placeholder inheritance and the AlternateContent branch choice agree.

Differences and caveats:

- `rotation_deg` differs from the PowerPoint rotation by 180 degrees for vertically flipped shapes, because a vertical flip is expressed as `mirrored` plus a half turn (section 3). Positions are identical.
- PowerPoint renumbers a duplicate shape id in memory (id 5 became 3). The extractor keeps the raw id from the XML. Whether PowerPoint writes the new id back was not tested.
- COM does not expose connector begin and end points, so those positions were derived from the shape frame and flips.
- The geometry of a group that contains other groups cannot be read separately through COM; its children were compared instead.
- Silent repair on open cannot be excluded through COM; no dialog was shown.
- Only PowerPoint 16.0 was used. Table cell positions and PowerPoint-saved AlternateContent were not compared.
