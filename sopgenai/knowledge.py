from __future__ import annotations

import uuid

from .models import (
    CanonicalDocument,
    KnowledgeUnit,
)


class KnowledgeBuilder:

    def __init__(
        self,
        authority_config: dict,
    ):

        self.authority = (
            authority_config
        )

    def build(
        self,
        document: CanonicalDocument,
    ):

        units = []

        document_type = (
            document.document_type
            or "UNKNOWN"
        ).upper()

        authority = (
            self.authority.get(
                document_type,
                0,
            )
        )

        # --------------------------------
        # Prefer semantic knowledge units
        # --------------------------------

        for semantic_type, values in (
            document.semantic.items()
        ):

            if not isinstance(
                values,
                list,
            ):
                continue

            for item in values:

                if isinstance(
                    item,
                    dict,
                ):

                    text = (
                        item.get(
                            "text"
                        )
                        or item.get(
                            "statement"
                        )
                        or item.get(
                            "definition"
                        )
                        or item.get(
                            "activity"
                        )
                        or str(item)
                    )

                    evidence = (
                        item.get(
                            "evidence_element_ids",
                            [],
                        )
                    )

                else:

                    text = str(
                        item
                    )

                    evidence = []

                if not text.strip():
                    continue

                units.append(
                    KnowledgeUnit(
                        id=(
                            "KU-"
                            + uuid.uuid4()
                            .hex[:12]
                            .upper()
                        ),

                        document_id=(
                            document
                            .document_id
                        ),

                        title=(
                            semantic_type
                        ),

                        text=text,

                        source_element_ids=(
                            evidence
                        ),

                        document_type=(
                            document_type
                        ),

                        authority=(
                            authority
                        ),

                        metadata={
                            "semantic_type":
                                semantic_type
                        },
                    )
                )

        # --------------------------------
        # Raw fallback
        # --------------------------------

        if not units:

            for element in (
                document.elements
            ):

                if not (
                    element.text.strip()
                ):
                    continue

                units.append(
                    KnowledgeUnit(
                        id=(
                            "KU-"
                            + uuid.uuid4()
                            .hex[:12]
                            .upper()
                        ),

                        document_id=(
                            document
                            .document_id
                        ),

                        title=(
                            element.type
                        ),

                        text=(
                            element.text
                        ),

                        source_element_ids=[
                            element.element_id
                        ],

                        document_type=(
                            document_type
                        ),

                        authority=(
                            authority
                        ),

                        metadata={
                            "page":
                                element.page
                        },
                    )
                )

        document.metadata[
            "knowledge_unit_count"
        ] = len(
            units
        )

        return units
