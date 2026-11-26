from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell: str | None = None
    paragraph_idx: int | None = None
    shape_name: str | None = None
    connector_id: str | None = None
    bbox: list[float] | None = None


class NormalizedValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    unit: str | None = None
    qualifier: str | None = None


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str | None = None
    pages_or_slides: int | None = None
    coordinate_unit: str = "pt"
    parser: str | None = None
    category: str | None = None
    description: str | None = None
    page_width: float | None = None
    page_height: float | None = None


class Element(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    reading_order: int
    level: int = 1
    parent_id: str | None = None
    text: str = ""
    bbox: list[float] | None = None
    page: int | None = None
    source_ref: SourceRef | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    table_id: str | None = None
    normalized: NormalizedValue | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableCell(BaseModel):
    model_config = ConfigDict(extra="allow")

    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    is_header: bool = False
    is_merged: bool = False
    bbox: list[float] | None = None
    header_path: list[str] = Field(default_factory=list)


class Table(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    num_rows: int
    num_cols: int
    cells: list[TableCell] = Field(default_factory=list)
    page: int | None = None
    bbox: list[float] | None = None
    html: str | None = None
    markdown: str | None = None
    source_ref: SourceRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagramNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    text: str = ""
    shape_id: Any | None = None
    bbox: list[float] | None = None
    connection_points: dict[str, Any] = Field(default_factory=dict)
    source_ref: SourceRef | None = None


class DiagramEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    src: str
    dst: str
    direction: str = "directed"
    label: str | None = None
    connector_id: Any | None = None
    source_ref: SourceRef | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Diagram(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    page: int | None = None
    bbox: list[float] | None = None
    cropped_image: str | None = None
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    merged_element_ids: list[str] = Field(default_factory=list)
    text_summary: str | None = None
    source_ref: SourceRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FindingEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    file: str
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell: str | None = None
    locator: str | None = None
    text: str | None = None
    raw: str | None = None
    normalized: NormalizedValue | None = None
    bbox: list[float] | None = None


class Finding(BaseModel):
    model_config = ConfigDict(extra="allow")

    finding_id: str
    type: str
    description: str
    severity: str = "medium"
    status: str = "confirmed"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entity: str | None = None
    evidence: list[FindingEvidence] = Field(default_factory=list)
    limitations: str | None = None


class UnifiedDocumentIR(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_id: str
    doc_type: str
    source_file: str | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    elements: list[Element] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    diagrams: list[Diagram] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
