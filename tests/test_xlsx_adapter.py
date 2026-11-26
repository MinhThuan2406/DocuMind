from pathlib import Path

import pytest

from src import XlsxAdapter


def test_xlsx_adapter_can_handle():
    adapter = XlsxAdapter()
    assert adapter.can_handle("data.xlsx") is True
    assert adapter.can_handle("data.XLSX") is True
    assert adapter.can_handle("data.xlsm") is True
    assert adapter.can_handle("doc.docx") is False
    assert adapter.can_handle("doc.pdf") is False


def test_xlsx_adapter_parse_real_fixture():
    fixture_path = Path("research/dataset_info.xlsx")
    assert fixture_path.is_file(), "research/dataset_info.xlsx must exist as fixture"

    adapter = XlsxAdapter()
    ir = adapter.parse(fixture_path)

    assert ir.doc_id == "dataset_info"
    assert ir.doc_type == "xlsx"
    assert ir.source_file == "dataset_info.xlsx"
    assert ir.metadata.pages_or_slides == 2

    assert len(ir.tables) == 2
    assert len(ir.elements) == 2

    t1 = ir.tables[0]
    assert "document_parsing" in t1.id
    assert t1.num_rows == 16
    assert t1.num_cols == 9
    assert t1.source_ref.sheet == "Document Parsing"
    assert t1.source_ref.cell == "A1:I16"

    headers = [c.text for c in t1.cells if c.is_header]
    assert "Dataset" in headers
    assert "Evaluation" in headers
    assert "License" in headers

    t2 = ir.tables[1]
    assert "relevance" in t2.id
    assert t2.num_rows == 16
    assert t2.num_cols == 8


def test_xlsx_adapter_merged_cells(tmp_path: Path):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MergedSheet"
    ws["A1"] = "Header"
    ws["B1"] = "Value"
    ws["A2"] = "Group"
    ws["B2"] = "Row1"
    ws["B3"] = "Row2"
    ws.merge_cells("A2:A3")

    xlsx_path = tmp_path / "merged.xlsx"
    wb.save(xlsx_path)

    adapter = XlsxAdapter()
    ir = adapter.parse(xlsx_path)
    assert len(ir.tables) == 1
    t = ir.tables[0]
    assert t.num_rows == 3
    assert t.num_cols == 2

    a2 = next(c for c in t.cells if c.row == 1 and c.col == 0)
    assert a2.is_merged is True
    assert a2.rowspan == 2
    assert a2.colspan == 1

    a3 = next(c for c in t.cells if c.row == 2 and c.col == 0)
    assert a3.is_merged is True
    assert a3.rowspan == 0
    assert a3.colspan == 0


def test_xlsx_adapter_file_not_found():
    adapter = XlsxAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.parse("non_existent_file.xlsx")

