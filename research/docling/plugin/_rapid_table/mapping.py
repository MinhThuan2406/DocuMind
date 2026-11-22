from collections.abc import Sequence

from docling_core.types.doc import BoundingBox, CoordOrigin, TableCell


def _box_bounds(box: Sequence[float]) -> tuple[float, float, float, float]:
    values = [float(v) for v in box]
    if len(values) == 4:
        return values[0], values[1], values[2], values[3]
    if len(values) == 8:
        return min(values[0::2]), min(values[1::2]), max(values[0::2]), max(values[1::2])
    raise ValueError("RapidTable cell bbox must contain 4 or 8 values")


def rapid_cells_to_docling(
    *,
    cell_bboxes: Sequence[Sequence[float]],
    logic_points: Sequence[Sequence[int]],
    crop: BoundingBox,
    scale: float,
    texts: Sequence[str] | None = None,
) -> tuple[list[TableCell], int, int]:
    texts = texts or []
    cells: list[TableCell] = []
    num_rows = num_cols = 0
    for index, (box, logic) in enumerate(zip(cell_bboxes, logic_points)):
        row_start, row_end, col_start, col_end = [int(v) for v in logic]
        l, t, r, b = _box_bounds(box)
        bbox = BoundingBox(
            l=crop.l + l / scale,
            t=crop.t + t / scale,
            r=crop.l + r / scale,
            b=crop.t + b / scale,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        num_rows = max(num_rows, row_end + 1)
        num_cols = max(num_cols, col_end + 1)
        cells.append(
            TableCell(
                bbox=bbox,
                row_span=row_end - row_start + 1,
                col_span=col_end - col_start + 1,
                start_row_offset_idx=row_start,
                end_row_offset_idx=row_end + 1,
                start_col_offset_idx=col_start,
                end_col_offset_idx=col_end + 1,
                text=str(texts[index]) if index < len(texts) else "",
                column_header=row_start == 0,
                row_header=False,
                row_section=False,
            )
        )
    return cells, num_rows, num_cols
