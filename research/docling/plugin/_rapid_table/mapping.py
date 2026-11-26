from collections.abc import Sequence
from html.parser import HTMLParser

from docling_core.types.doc import BoundingBox, CoordOrigin, TableCell
from docling_core.types.doc.page import TextCell


class _TableCellTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._parts: list[str] = []
        self.cells: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.lower() in {"td", "th"}:
            if self._depth == 0:
                self._parts = []
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() not in {"td", "th"} or self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0:
            self.cells.append(" ".join("".join(self._parts).split()))
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._depth > 0:
            self._parts.append(data)


def extract_html_cell_texts(html: str | None) -> list[str]:
    if not html:
        return []
    parser = _TableCellTextParser()
    parser.feed(html)
    parser.close()
    return parser.cells


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
    if scale <= 0:
        raise ValueError("RapidTable scale must be greater than zero")
    if len(cell_bboxes) != len(logic_points):
        raise ValueError(
            "RapidTable cell_bboxes and logic_points must have the same number "
            "of entries"
        )
    texts = texts or []
    cells: list[TableCell] = []
    num_rows = num_cols = 0
    for index, (box, logic) in enumerate(zip(cell_bboxes, logic_points)):
        if len(logic) != 4:
            raise ValueError("RapidTable logic point must contain four indices")
        row_start, row_end, col_start, col_end = [int(v) for v in logic]
        if (
            row_start < 0
            or col_start < 0
            or row_end < row_start
            or col_end < col_start
        ):
            raise ValueError(f"Invalid RapidTable logic point: {list(logic)}")
        l, t, r, b = _box_bounds(box)
        if r <= l or b <= t:
            bbox = None
        else:
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
                # SLANet predicts geometry, not reliable header semantics.
                column_header=False,
                row_header=False,
                row_section=False,
            )
        )
    return cells, num_rows, num_cols


def populate_table_cell_texts(
    table_cells: Sequence[TableCell],
    page_cells: Sequence[TextCell],
    fallback_texts: Sequence[str] | None = None,
) -> None:
    """Fill table text from Docling's merged native/OCR cells, then fallback OCR."""
    fallback_texts = fallback_texts or []
    positioned_page_cells = [
        (cell, cell.rect.to_bounding_box())
        for cell in page_cells
        if cell.text.strip()
    ]

    for index, table_cell in enumerate(table_cells):
        matches: list[tuple[float, float, str]] = []
        if table_cell.bbox is not None:
            target = table_cell.bbox
            for page_cell, source in positioned_page_cells:
                center_x = (source.l + source.r) / 2
                center_y = (source.t + source.b) / 2
                if target.l <= center_x <= target.r and target.t <= center_y <= target.b:
                    matches.append((source.t, source.l, page_cell.text.strip()))

        if matches:
            matches.sort(key=lambda item: (item[0], item[1]))
            table_cell.text = " ".join(text for _, _, text in matches)
        elif index < len(fallback_texts):
            table_cell.text = str(fallback_texts[index]).strip()
