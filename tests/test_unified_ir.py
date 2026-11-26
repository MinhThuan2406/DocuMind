from pathlib import Path

import pytest
from pydantic import ValidationError

from src import (
    BaseDocumentAdapter,
    Diagram,
    DiagramEdge,
    DiagramNode,
    DocumentMetadata,
    Element,
    Finding,
    FindingEvidence,
    NormalizedValue,
    SourceRef,
    Table,
    TableCell,
    UnifiedDocumentIR,
)


def test_source_ref_serialization():
    ref = SourceRef(
        file="spec.docx",
        page=2,
        paragraph_idx=14,
        bbox=[10.0, 20.0, 100.0, 200.0],
    )
    assert ref.file == "spec.docx"
    assert ref.page == 2
    assert ref.paragraph_idx == 14
    assert ref.bbox == [10.0, 20.0, 100.0, 200.0]

    with pytest.raises(ValidationError):
        SourceRef.model_validate({"file": "test.pdf", "unknown_field": 123})


def test_element_and_table_structure():
    cell = TableCell(
        row=0,
        col=0,
        rowspan=2,
        colspan=1,
        text="Component Code",
        is_header=True,
        is_merged=True,
        header_path=["Specification", "General"],
    )
    table = Table(
        id="tbl_001",
        num_rows=4,
        num_cols=3,
        cells=[cell],
        html="<table>...</table>",
    )
    elem = Element(
        id="elem_0001",
        type="table",
        reading_order=0,
        level=2,
        text="[Table tbl_001]",
        table_id="tbl_001",
        source_ref=SourceRef(file="spec.docx", paragraph_idx=5),
    )
    assert table.cells[0].is_merged is True
    assert elem.table_id == "tbl_001"
    assert elem.confidence == 1.0


def test_diagram_graph_models():
    node1 = DiagramNode(id="node_01", text="Gateway ECU", shape_id=4)
    node2 = DiagramNode(id="node_02", text="BCM", shape_id=9)
    edge = DiagramEdge(
        src="node_01",
        dst="node_02",
        direction="directed",
        label="CAN FD",
        connector_id="conn_14",
    )
    diag = Diagram(id="diag_01", nodes=[node1, node2], edges=[edge])
    assert len(diag.nodes) == 2
    assert diag.edges[0].src == "node_01"
    assert diag.edges[0].dst == "node_02"


def test_finding_and_evidence():
    evidence_a = FindingEvidence(
        file="spec.docx",
        locator="table[3].r5c4",
        raw="500 ms",
        normalized=NormalizedValue(value=500, unit="ms", qualifier="Normal @ 25 C"),
    )
    evidence_b = FindingEvidence(
        file="test.xlsx",
        locator="Sheet2!D12",
        raw="0.8 s",
        normalized=NormalizedValue(value=800, unit="ms", qualifier="Normal @ 25 C"),
    )
    finding = Finding(
        finding_id="F-014",
        type="numerical_conflict",
        description="Timeout discrepancy",
        status="confirmed",
        evidence=[evidence_a, evidence_b],
    )
    assert finding.finding_id == "F-014"
    assert len(finding.evidence) == 2
    assert finding.evidence[0].normalized.value == 500
    assert finding.evidence[1].normalized.value == 800


def test_unified_document_ir_roundtrip():
    ir = UnifiedDocumentIR(
        doc_id="EV_Braking_System_Specs",
        doc_type="pptx",
        source_file="EV_Braking_System_Specs.pptx",
        metadata=DocumentMetadata(
            title="EV Braking Control Architecture",
            pages_or_slides=5,
            coordinate_unit="pt",
        ),
        elements=[
            Element(
                id="elem_0001",
                type="shape",
                reading_order=0,
                level=1,
                text="Brake Control Unit",
                bbox=[120.5, 80.0, 320.0, 160.0],
                source_ref=SourceRef(file="EV_Braking_System_Specs.pptx", slide=2),
            )
        ],
    )
    json_str = ir.model_dump_json()
    reconstructed = UnifiedDocumentIR.model_validate_json(json_str)
    assert reconstructed.doc_id == ir.doc_id
    assert reconstructed.elements[0].text == "Brake Control Unit"
    assert reconstructed.elements[0].source_ref.slide == 2


def test_base_document_adapter_interface():
    class DummyDocxAdapter(BaseDocumentAdapter):
        @property
        def supported_extensions(self) -> tuple[str, ...]:
            return (".docx",)

        def parse(self, file_path):
            return UnifiedDocumentIR(
                doc_id=Path(file_path).stem,
                doc_type="docx",
                source_file=str(file_path),
            )

    adapter = DummyDocxAdapter()
    assert adapter.can_handle("document.docx") is True
    assert adapter.can_handle("document.DOCX") is True
    assert adapter.can_handle("document.xlsx") is False
    assert adapter.can_handle("document.pdf") is False

    res = adapter.parse("test_sample.docx")
    assert res.doc_id == "test_sample"
    assert res.doc_type == "docx"
