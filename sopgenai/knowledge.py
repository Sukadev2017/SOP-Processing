from __future__ import annotations

import uuid

from .models import (
    CanonicalDocument,
    KnowledgeUnit,
)


AUTHORITY = {
    "POLICY": 100,
    "ORGANIZATIONAL_STANDARD": 90,
    "ARCHITECTURE_STANDARD": 80,
    "APPROVED_SOP": 70,
    "PROCESS_GUIDELINE": 60,
    "SOP": 50,
    "HISTORICAL_SOP": 30,
}


class KnowledgeBuilder:

    def build(
        self,
        document: CanonicalDocument,
    ):

        units = []

        document_type = (
            document.metadata.document_type
            or "UNKNOWN"
        ).upper()

        authority = AUTHORITY.get(
            document_type,
            0,
        )

        for element in document.elements:

            if not element.text:
                continue

            text = element.text.strip()

            if len(text) < 20:
                continue

            unit = KnowledgeUnit(
                knowledge_unit_id=(
                    "KU-"
                    + uuid.uuid4()
                    .hex[:12]
                    .upper()
                ),

                document_id=(
                    document
                    .metadata
                    .document_id
                ),

                element_ids=[
                    element.element_id
                ],

                knowledge_type=(
                    element.semantic_type
                    or "DOCUMENT_CONTENT"
                ),

                title=(
                    text[:100]
                ),

                text=text,

                document_type=(
                    document_type
                ),

                authority_level=(
                    authority
                ),

                metadata={
                    "page":
                        element.page_number,

                    "process":
                        document
                        .metadata
                        .process,

                    "application":
                        document
                        .metadata
                        .application,
                },
            )

            units.append(unit)

        document.knowledge_units = units

        return units
