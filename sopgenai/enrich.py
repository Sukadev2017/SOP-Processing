from __future__ import annotations

from .llm import LLMProvider

from .models import (
    CanonicalDocument,
)


SYSTEM_PROMPT = """
You are analyzing a portion of an enterprise controlled
document.

Analyze ONLY the supplied elements.

Do not invent missing information.

Extract semantic information into these categories:

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

Every extracted item must contain:

{
  "text": "...",
  "evidence_element_ids": ["..."]
}

Return valid JSON only.
"""


SEMANTIC_KEYS = [

    "purpose",

    "applicability",

    "definitions",

    "abbreviations",

    "prerequisites",

    "roles",

    "responsibilities",

    "process_steps",

    "requirements",

    "recommendations",

    "references",

    "associated_documents",

    "systems",

    "warnings",

    "conflicts",
]


class SemanticEnricher:

    def __init__(
        self,
        llm: LLMProvider,
        config=None,
    ):

        self.llm = llm

        config = (
            config
            or {}
        )

        self.batch_size = (
            config.get(
                "batch_size",
                30,
            )
        )

        self.max_batch_chars = (
            config.get(
                "max_batch_chars",
                18000,
            )
        )

    def _create_batches(
        self,
        elements,
    ):

        batch = []

        characters = 0

        for element in elements:

            item = {

                "element_id":
                    element.element_id,

                "page":
                    element.page,

                "type":
                    element.type,

                "text":
                    element.text,
            }

            item_size = len(
                element.text
            )

            should_flush = (

                batch

                and (

                    len(batch)
                    >= self.batch_size

                    or

                    (
                        characters
                        + item_size
                    )
                    > self.max_batch_chars
                )
            )

            if should_flush:

                yield batch

                batch = []

                characters = 0

            batch.append(
                item
            )

            characters += (
                item_size
            )

        if batch:

            yield batch

    def enrich(
        self,
        document:
            CanonicalDocument,
    ):

        elements = [

            element

            for element
            in document.elements

            if (
                element.text
                .strip()
            )
        ]

        batches = list(
            self._create_batches(
                elements
            )
        )

        merged = {

            key: []

            for key
            in SEMANTIC_KEYS
        }

        total = len(
            batches
        )

        for (
            index,
            batch,
        ) in enumerate(
            batches,
            1,
        ):

            print(
                f"[Semantic] "
                f"Batch {index}/{total} "
                f"({len(batch)} elements)"
            )

            payload = {

                "document": {

                    "document_id":
                        document.document_id,

                    "source_file":
                        document.source_file,

                    "document_type":
                        document.document_type,
                },

                "elements":
                    batch,
            }

            response = (
                self.llm
                .json_completion(
                    SYSTEM_PROMPT,
                    payload,
                )
            )

            for key in (
                SEMANTIC_KEYS
            ):

                values = (
                    response.get(
                        key,
                        [],
                    )
                )

                if isinstance(
                    values,
                    list,
                ):

                    merged[
                        key
                    ].extend(
                        values
                    )

        document.semantic = (
            merged
        )

        return merged
