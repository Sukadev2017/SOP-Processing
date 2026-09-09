from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
)


class BBox(BaseModel):

    x0: float
    y0: float
    x1: float
    y1: float


class TextRun(BaseModel):

    text: str

    font: Optional[str] = None

    size: Optional[float] = None

    color: Optional[str] = None

    bold: bool = False

    italic: bool = False

    bbox: Optional[BBox] = None


class Element(BaseModel):

    element_id: str

    type: str

    page: int

    order: int

    text: str = ""

    bbox: Optional[BBox] = None

    runs: List[TextRun] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    provenance: Dict[str, Any] = Field(
        default_factory=dict
    )


class Section(BaseModel):

    section_id: str

    heading: str

    number: Optional[str] = None

    level: int = 1

    elements: List[str] = Field(
        default_factory=list
    )


class CanonicalDocument(BaseModel):

    schema_version: str = "3.0"

    document_id: str

    source_file: str

    source_sha256: str

    document_type: Optional[str] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    elements: List[Element] = Field(
        default_factory=list
    )

    sections: List[Section] = Field(
        default_factory=list
    )

    assets: List[Dict[str, Any]] = Field(
        default_factory=list
    )

    semantic: Dict[str, Any] = Field(
        default_factory=dict
    )


class KnowledgeUnit(BaseModel):

    id: str

    document_id: str

    title: str

    text: str

    source_element_ids: List[str] = Field(
        default_factory=list
    )

    document_type: str = "UNKNOWN"

    authority: int = 0

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


class RetrievedEvidence(BaseModel):

    score: float

    knowledge_unit_id: str

    document_id: str

    document_type: str

    authority: int

    text: str

    source_element_ids: List[str] = Field(
        default_factory=list
    )


class Mapping(BaseModel):

    target_id: str

    target_heading: str

    source_element_ids: List[str] = Field(
        default_factory=list
    )

    transformed_content: str = ""

    confidence: float = 0.0

    state: str = "SME_REQUIRED"

    evidence: List[
        Dict[str, Any]
    ] = Field(
        default_factory=list
    )
