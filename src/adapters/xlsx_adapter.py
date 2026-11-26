from __future__ import annotations

from pathlib import Path

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from src.adapters.base import BaseDocumentAdapter
from src.models.ir import (
    DocumentMetadata,
    Element,
    SourceRef,
    Table,
    TableCell,
    UnifiedDocumentIR,
)


class XlsxAdapter(BaseDocumentAdapter):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".xlsx", ".xlsm", ".xltx", ".xltm")

    def parse(self, file_path: str | Path) -> UnifiedDocumentIR:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        wb = openpyxl.load_workbook(str(path), data_only=False)
        doc_id = path.stem
        metadata = DocumentMetadata(
            title=doc_id,
            pages_or_slides=len(wb.sheetnames),
            parser="openpyxl",
        )

        elements: list[Element] = []
        tables: list[Table] = []
        reading_order = 0

        for sheet_idx, sheet_name in enumerate(wb.sheetnames):
            ws = wb[sheet_name]
            if ws.max_row == 0 or ws.max_column == 0:
                continue

            table, table_elem = self._parse_worksheet(
                ws=ws,
                sheet_name=sheet_name,
                sheet_idx=sheet_idx,
                file_name=path.name,
                start_reading_order=reading_order,
            )

            if table.cells:
                tables.append(table)
                elements.append(table_elem)
                reading_order += 1

        wb.close()

        return UnifiedDocumentIR(
            doc_id=doc_id,
            doc_type="xlsx",
            source_file=path.name,
            metadata=metadata,
            elements=elements,
            tables=tables,
        )

    def _parse_worksheet(
        self,
        ws: Worksheet,
        sheet_name: str,
        sheet_idx: int,
        file_name: str,
        start_reading_order: int,
    ) -> tuple[Table, Element]:
        table_id = f"tbl_{sheet_idx + 1:03d}_{sheet_name.lower().replace(' ', '_')}"
        elem_id = f"elem_{start_reading_order + 1:04d}"

        merged_map: dict[tuple[int, int], tuple[int, int, bool]] = {}
        for rng in ws.merged_cells.ranges:
            min_r, min_c = rng.min_row, rng.min_col
            max_r, max_c = rng.max_row, rng.max_col
            rowspan = max_r - min_r + 1
            colspan = max_c - min_c + 1

            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    is_origin = r == min_r and c == min_c
                    merged_map[(r, c)] = (rowspan, colspan, is_origin)

        cells: list[TableCell] = []
        header_row_vals: list[str] = []

        max_row = ws.max_row
        max_col = ws.max_column

        for c_idx in range(1, max_col + 1):
            val = ws.cell(1, c_idx).value
            header_row_vals.append(
                str(val).strip() if val is not None else f"Col_{c_idx}"
            )

        for r_idx in range(1, max_row + 1):
            for c_idx in range(1, max_col + 1):
                cell_obj = ws.cell(r_idx, c_idx)
                raw_val = cell_obj.value
                text = str(raw_val).strip() if raw_val is not None else ""

                rowspan = 1
                colspan = 1
                is_merged = False

                coord = (r_idx, c_idx)
                if coord in merged_map:
                    is_merged = True
                    span_r, span_c, is_origin = merged_map[coord]
                    if is_origin:
                        rowspan = span_r
                        colspan = span_c
                    else:
                        rowspan = 0
                        colspan = 0

                is_header = r_idx == 1
                col_name = (
                    header_row_vals[c_idx - 1]
                    if c_idx - 1 < len(header_row_vals)
                    else ""
                )
                header_path = [col_name] if col_name else []

                cells.append(
                    TableCell(
                        row=r_idx - 1,
                        col=c_idx - 1,
                        rowspan=rowspan,
                        colspan=colspan,
                        text=text,
                        is_header=is_header,
                        is_merged=is_merged,
                        header_path=header_path,
                    )
                )

        table = Table(
            id=table_id,
            num_rows=max_row,
            num_cols=max_col,
            cells=cells,
            source_ref=SourceRef(
                file=file_name,
                sheet=sheet_name,
                cell=f"A1:{openpyxl.utils.get_column_letter(max_col)}{max_row}",
            ),
            metadata={"sheet_name": sheet_name, "sheet_index": sheet_idx},
        )

        element = Element(
            id=elem_id,
            type="table",
            reading_order=start_reading_order,
            level=1,
            text=f"[Table {table_id}: Sheet '{sheet_name}']",
            table_id=table_id,
            source_ref=SourceRef(
                file=file_name,
                sheet=sheet_name,
            ),
            confidence=1.0,
            metadata={"sheet_name": sheet_name},
        )

        return table, element
