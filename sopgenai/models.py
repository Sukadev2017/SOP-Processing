from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class TextSpan(BaseModel):
    text: str

    bbox: Optional[BoundingBox] = None

    font: Optional[str] = None
    size: Optional[float] = None

    color: Optional[str] = None

    bold: bool = False
    italic: bool = False


class DocumentElement(BaseModel):
    element_id: str

    type: str

    page_number: Optional[int] = None

    order: int

    text: Optional[str] = None

    bbox: Optional[BoundingBox] = None

    spans: List[TextSpan] = Field(
        default_factory=list
    )

    style_ref: Optional[str] = None

    semantic_type: Optional[str] = None

    extraction_method: Optional[str] = None

    confidence: Optional[float] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


class TableCell(BaseModel):
    row: int
    column: int

    text: str

    bbox: Optional[BoundingBox] = None

    row_span: int = 1
    column_span: int = 1


class TableElement(BaseModel):
    table_id: str

    page_number: int

    order: int

    rows: int
    columns: int

    cells: List[TableCell] = Field(
        default_factory=list
    )

    bbox: Optional[BoundingBox] = None

    extraction_method: str = "pdfplumber"


class ImageElement(BaseModel):
    image_id: str

    page_number: int

    order: int

    asset_path: Optional[str] = None

    bbox: Optional[BoundingBox] = None

    width: Optional[int] = None
    height: Optional[int] = None

    ocr_text: Optional[str] = None

    ai_description: Optional[str] = None


class DocumentSection(BaseModel):
    section_id: str

    parent_section_id: Optional[str] = None

    section_number: Optional[str] = None

    heading: str

    heading_level: int = 1

    order: int

    elements: List[DocumentElement] = Field(
        default_factory=list
    )

    tables: List[TableElement] = Field(
        default_factory=list
    )

    images: List[ImageElement] = Field(
        default_factory=list
    )

    children: List["DocumentSection"] = Field(
        default_factory=list
    )


class DocumentMetadata(BaseModel):
    document_id: str

    file_name: str
    file_type: str

    title: Optional[str] = None

    document_type: Optional[str] = None

    document_number: Optional[str] = None

    version: Optional[str] = None

    status: Optional[str] = None

    scope: Optional[str] = None

    business_domain: Optional[str] = None

    process: Optional[str] = None

    application: Optional[str] = None

    language: str = "en"

    file_hash: Optional[str] = None


class Role(BaseModel):
    role_id: str

    name: str

    responsibilities: List[str] = Field(
        default_factory=list
    )

    competence: Optional[str] = None

    source_element_ids: List[str] = Field(
        default_factory=list
    )


class ProcessStep(BaseModel):
    step_id: str

    sequence: int

    activity: str

    role: Optional[str] = None

    input: Optional[str] = None
    output: Optional[str] = None

    system: Optional[str] = None

    decision: Optional[str] = None

    source_element_ids: List[str] = Field(
        default_factory=list
    )


class Requirement(BaseModel):
    requirement_id: str

    text: str

    requirement_type: Optional[str] = None

    mandatory: bool = False

    source_element_ids: List[str] = Field(
        default_factory=list
    )


class Definition(BaseModel):
    term: str

    full_form: Optional[str] = None

    definition: str

    source_element_ids: List[str] = Field(
        default_factory=list
    )


class SemanticContent(BaseModel):

    purpose: List[str] = Field(
        default_factory=list
    )

    applicability: List[str] = Field(
        default_factory=list
    )

    roles: List[Role] = Field(
        default_factory=list
    )

    process_steps: List[ProcessStep] = Field(
        default_factory=list
    )

    requirements: List[Requirement] = Field(
        default_factory=list
    )

    definitions: List[Definition] = Field(
        default_factory=list
    )

    references: List[str] = Field(
        default_factory=list
    )

    prerequisites: List[str] = Field(
        default_factory=list
    )


class KnowledgeUnit(BaseModel):

    knowledge_unit_id: str

    document_id: str

    section_id: Optional[str] = None

    element_ids: List[str] = Field(
        default_factory=list
    )

    knowledge_type: str

    title: str

    text: str

    document_type: Optional[str] = None

    authority_level: int = 0

    is_current: bool = True

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


class CanonicalDocument(BaseModel):

    schema_version: str = "2.0"

    metadata: DocumentMetadata

    sections: List[DocumentSection] = Field(
        default_factory=list
    )

    elements: List[DocumentElement] = Field(
        default_factory=list
    )

    tables: List[TableElement] = Field(
        default_factory=list
    )

    images: List[ImageElement] = Field(
        default_factory=list
    )

    styles: Dict[str, Any] = Field(
        default_factory=dict
    )

    semantic_content: SemanticContent = Field(
        default_factory=SemanticContent
    )

    knowledge_units: List[KnowledgeUnit] = Field(
        default_factory=list
    )

    relationships: List[Dict[str, Any]] = Field(
        default_factory=list
    )

    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict
    )
