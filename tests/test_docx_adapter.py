from pathlib import Path

import pytest

from src import DocxAdapter


def test_docx_adapter_can_handle():
    adapter = DocxAdapter()
    assert adapter.can_handle("doc.docx") is True
    assert adapter.can_handle("doc.DOCX") is True
    assert adapter.can_handle("doc.docm") is True
    assert adapter.can_handle("sheet.xlsx") is False
    assert adapter.can_handle("deck.pptx") is False


def test_docx_adapter_parse_dense_agreement():
    fixture_path = Path("datasets/samples/sample_dense_agreement.docx")
    assert fixture_path.is_file(), "sample_dense_agreement.docx must exist"

    adapter = DocxAdapter()
    ir = adapter.parse(fixture_path)

    assert ir.doc_id == "sample_dense_agreement"
    assert ir.doc_type == "docx"
    assert ir.source_file == "sample_dense_agreement.docx"
    assert ir.metadata.parser == "python-docx"

    assert len(ir.elements) > 200
    assert len(ir.tables) == 2

    t1 = ir.tables[0]
    assert t1.num_rows == 1
    assert t1.num_cols == 1
    assert "Downey Law Group" in t1.cells[0].text

    t2 = ir.tables[1]
    assert t2.num_rows == 6
    assert t2.num_cols == 5
    assert len(t2.cells) == 30
    assert any("Organization" in c.text for c in t2.cells if c.is_header)

    assert ir.elements[0].reading_order == 1
    assert ir.elements[0].type == "table"


def test_docx_adapter_parse_nested_policy():
    fixture_path = Path("datasets/samples/sample_nested_policy.docx")
    assert fixture_path.is_file(), "sample_nested_policy.docx must exist"

    adapter = DocxAdapter()
    ir = adapter.parse(fixture_path)

    assert ir.doc_type == "docx"
    assert len(ir.tables) == 1
    headings = [e for e in ir.elements if e.type == "heading"]
    assert len(headings) >= 20
    assert any(h.level == 1 for h in headings)


def test_docx_adapter_file_not_found():
    adapter = DocxAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.parse("non_existent_file.docx")
