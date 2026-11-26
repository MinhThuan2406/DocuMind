from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import docx
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from src.adapters.base import BaseDocumentAdapter
from src.models.ir import (
    DocumentMetadata,
    Element,
    SourceRef,
    Table,
    TableCell,
    UnifiedDocumentIR,
)


class DocxAdapter(BaseDocumentAdapter):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".docx", ".docm")

    def parse(self, file_path: str | Path) -> UnifiedDocumentIR:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        doc = docx.Document(str(path))
        doc_id = path.stem

        title = doc_id
        if doc.core_properties and doc.core_properties.title:
            prop_title = doc.core_properties.title.strip()
            if prop_title:
                title = prop_title

        elements: list[Element] = []
        tables: list[Table] = []

        reading_order = 0
        paragraph_idx = 0
        table_idx = 0

        for child in doc.element.body:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                p = DocxParagraph(child, doc)
                paragraph_idx += 1
                text = p.text.strip()
                if not text:
                    continue

                reading_order += 1
                elem_type, level = self._classify_paragraph(p)
                elem_id = f"elem_{reading_order:04d}"

                elements.append(
                    Element(
                        id=elem_id,
                        type=elem_type,
                        reading_order=reading_order,
                        level=level,
                        text=p.text,
                        source_ref=SourceRef(
                            file=path.name,
                            paragraph_idx=paragraph_idx,
                        ),
                        metadata={"style": p.style.name if p.style else None},
                    )
                )

            elif tag == "tbl":
                docx_tbl = DocxTable(child, doc)
                table_idx += 1
                reading_order += 1

                ir_table, table_elem = self._parse_table(
                    docx_tbl=docx_tbl,
                    table_idx=table_idx,
                    reading_order=reading_order,
                    file_name=path.name,
                )

                tables.append(ir_table)
                elements.append(table_elem)

        metadata = DocumentMetadata(
            title=title,
            pages_or_slides=None,
            parser="python-docx",
        )

        return UnifiedDocumentIR(
            doc_id=doc_id,
            doc_type="docx",
            source_file=path.name,
            metadata=metadata,
            elements=elements,
            tables=tables,
        )

    def _classify_paragraph(self, p: DocxParagraph) -> tuple[str, int]:
        style_name = p.style.name.lower() if p.style and p.style.name else ""

        if "title" in style_name:
            return "title", 1

        heading_match = re.search(r"heading\s*(\d+)", style_name)
        if heading_match:
            return "heading", int(heading_match.group(1))

        num_pr = p._element.xpath(".//w:numPr")
        if num_pr:
            ilvl_nodes = p._element.xpath(".//w:ilvl")
            if ilvl_nodes:
                val = ilvl_nodes[0].attrib.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "0"
                )
                try:
                    return "list_item", int(val) + 1
                except ValueError:
                    return "list_item", 1
            return "list_item", 1

        return "paragraph", 1

    def _parse_table(
        self,
        docx_tbl: DocxTable,
        table_idx: int,
        reading_order: int,
        file_name: str,
    ) -> tuple[Table, Element]:
        table_id = f"tbl_{table_idx:03d}"
        elem_id = f"elem_{reading_order:04d}"

        rows = docx_tbl.rows
        num_rows = len(rows)
        num_cols = len(docx_tbl.columns) if rows else 0

        cells: list[TableCell] = []
        tc_origin_map: dict[Any, tuple[int, int]] = {}
        tc_spans: dict[tuple[int, int], tuple[int, int]] = {}

        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row.cells):
                tc = cell._tc
                if tc not in tc_origin_map:
                    tc_origin_map[tc] = (r_idx + 1, c_idx + 1)

        for tc, (orig_r, orig_c) in tc_origin_map.items():
            colspan = 0
            for c in range(orig_c - 1, num_cols):
                if rows[orig_r - 1].cells[c]._tc == tc:
                    colspan += 1
                else:
                    break

            rowspan = 0
            for r in range(orig_r - 1, num_rows):
                if rows[r].cells[orig_c - 1]._tc == tc:
                    rowspan += 1
                else:
                    break

            tc_spans[(orig_r, orig_c)] = (max(rowspan, 1), max(colspan, 1))

        header_row_vals: list[str] = []
        md_rows: list[list[str]] = []

        for r_idx, row in enumerate(rows):
            is_tbl_header = bool(row._tr.xpath(".//w:tblHeader")) or r_idx == 0
            row_num = r_idx + 1
            md_row: list[str] = []

            for c_idx, cell in enumerate(row.cells):
                col_num = c_idx + 1
                tc = cell._tc
                orig_r, orig_c = tc_origin_map[tc]
                is_origin = (orig_r == row_num) and (orig_c == col_num)
                raw_text = cell.text.strip()

                if is_origin:
                    rowspan, colspan = tc_spans.get((row_num, col_num), (1, 1))
                    is_merged = rowspan > 1 or colspan > 1
                    cell_text = raw_text
                else:
                    rowspan, colspan = 0, 0
                    is_merged = True
                    cell_text = ""

                header_path: list[str] = []
                if r_idx == 0:
                    header_row_vals.append(raw_text)
                elif c_idx < len(header_row_vals) and header_row_vals[c_idx]:
                    header_path = [header_row_vals[c_idx]]

                cells.append(
                    TableCell(
                        row=row_num,
                        col=col_num,
                        rowspan=rowspan,
                        colspan=colspan,
                        text=cell_text,
                        is_header=is_tbl_header,
                        is_merged=is_merged,
                        header_path=header_path,
                    )
                )
                md_row.append(raw_text.replace("\n", " ") if is_origin else "")

            md_rows.append(md_row)

        markdown_lines: list[str] = []
        if md_rows:
            markdown_lines.append("| " + " | ".join(md_rows[0]) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(md_rows[0])) + " |")
            for row_vals in md_rows[1:]:
                markdown_lines.append("| " + " | ".join(row_vals) + " |")
        markdown_str = "\n".join(markdown_lines)

        source_ref = SourceRef(
            file=file_name,
            cell=f"R1C1:R{num_rows}C{num_cols}",
        )

        ir_table = Table(
            id=table_id,
            num_rows=num_rows,
            num_cols=num_cols,
            cells=cells,
            markdown=markdown_str,
            source_ref=source_ref,
        )

        table_elem = Element(
            id=elem_id,
            type="table",
            reading_order=reading_order,
            table_id=table_id,
            text=markdown_str,
            source_ref=source_ref,
        )

        return ir_table, table_elem
