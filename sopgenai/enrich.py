from __future__ import annotations

from .llm import (
    LLMProvider,
)

from .models import (
    CanonicalDocument,
)


SYSTEM_PROMPT = """
You are an enterprise controlled-document analysis engine.

Analyze only the supplied source evidence.

Do not invent missing information.

Identify:

- purpose
- applicability
- definitions
- abbreviations
- prerequisites
- roles
- responsibilities
- process steps
- mandatory requirements
- recommendations
- references
- associated documents
- systems/applications
- warnings
- attention items
- conflicts
- ambiguities

Every semantic item must include evidence_element_ids.

If evidence is insufficient, do not infer the answer.

Return JSON with these top-level arrays:

purpose
applicability
definitions
abbreviations
prerequisites
roles
responsibilities
process_steps
requirements
recommendations
references
associated_documents
systems
warnings
conflicts
"""


class SemanticEnricher:

    def __init__(
        self,
        llm: LLMProvider,
    ):

        self.llm = llm

    def enrich(
        self,
        document: CanonicalDocument,
    ):

        elements = [
            {
                "element_id":
                    element.element_id,

                "page":
                    element.page,

                "type":
                    element.type,

                "text":
                    element.text,
            }
            for element
            in document.elements
            if element.text.strip()
        ]

        payload = {
            "document": {
                "document_id":
                    document.document_id,

                "document_type":
                    document.document_type,

                "source_file":
                    document.source_file,
            },

            "elements":
                elements,
        }

        result = (
            self.llm
            .json_completion(
                SYSTEM_PROMPT,
                payload,
            )
        )

        document.semantic = (
            result
        )

        return result
