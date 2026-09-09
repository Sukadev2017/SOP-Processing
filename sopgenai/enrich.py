from __future__ import annotations

from .llm import LLMProvider
from .models import CanonicalDocument


SYSTEM_PROMPT = """
You are an enterprise SOP document-analysis engine.

Extract semantic information only from the supplied
document evidence.

Do not invent missing information.

Identify:

1. purpose
2. applicability
3. roles and responsibilities
4. process steps
5. prerequisites
6. mandatory requirements
7. definitions and abbreviations
8. references
9. warnings or attention items
10. conflicts or ambiguous statements

Every extracted item must contain source element IDs.

Return JSON only.
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

        evidence = []

        for element in document.elements:

            evidence.append(
                {
                    "element_id":
                        element.element_id,

                    "page":
                        element.page_number,

                    "text":
                        element.text,
                }
            )

        payload = {
            "document": {
                "document_id":
                    document.metadata.document_id,

                "document_type":
                    document.metadata.document_type,
            },

            "elements": evidence,
        }

        result = (
            self.llm
            .structured_completion(
                SYSTEM_PROMPT,
                payload,
            )
        )

        document.extraction_metadata[
            "semantic_enrichment"
        ] = result

        return result
