from __future__ import annotations

from .llm import (
    LLMProvider,
)

from .models import (
    CanonicalDocument,
    KnowledgeUnit,
    Mapping,
)


TARGET_SECTIONS = [

    ("1", "PURPOSE"),

    ("2", "APPLICABILITY"),

    (
        "3",
        "DEFINITIONS & ABBREVIATIONS",
    ),

    (
        "4",
        "IMPLEMENTATION AND/OR PRE-REQUISITES",
    ),

    (
        "5",
        "ROLES & RESPONSIBILITIES",
    ),

    (
        "6",
        "PROCESS",
    ),

    (
        "7",
        "ASSOCIATED DOCUMENTS",
    ),

    (
        "8",
        "REFERENCES",
    ),

    (
        "9",
        "DISTRIBUTION OF CONTROLLED PRINTS/COPIES (OPTIONAL)",
    ),

    (
        "10",
        "DOCUMENT HISTORY",
    ),
]


SYSTEM_PROMPT = """
You are migrating an existing SOP into a controlled GP
document template.

SOURCE RULES

1. The source SOP is the primary source for existing
   operational content.

2. Retrieved enterprise policies, standards, architecture
   standards, guidelines, glossaries and approved SOPs may
   provide additional evidence.

3. Do not silently overwrite source content when retrieved
   evidence conflicts with it.

4. Identify conflicts explicitly.

5. Do not invent missing operational information.

6. Every generated section must retain evidence.

7. Use one of these states:

   SOURCE_SUPPORTED
   REFERENCE_DERIVED
   CONFLICT
   SME_REQUIRED

8. Use active, concise language.

9. Use "must" for mandatory requirements.

10. Use "should" for recommendations.

11. Section 6 PROCESS should be how-style operational
    content.

Return JSON:

{
  "mappings": [
    {
      "target_id": "...",
      "target_heading": "...",
      "source_element_ids": [],
      "transformed_content": "...",
      "confidence": 0.0,
      "state": "...",
      "evidence": []
    }
  ]
}
"""


class TemplateMapper:

    def __init__(
        self,
        llm: LLMProvider,
    ):

        self.llm = llm

    def map(
        self,
        source: CanonicalDocument,
        retrieved: dict[
            str,
            list[
                tuple[
                    float,
                    KnowledgeUnit,
                ]
            ],
        ],
    ):

        source_elements = [
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
            in source.elements
            if element.text.strip()
        ]

        retrieval_context = {}

        for (
            section_id,
            results,
        ) in retrieved.items():

            retrieval_context[
                section_id
            ] = [
                {
                    "score":
                        score,

                    "knowledge_unit_id":
                        unit.id,

                    "document_id":
                        unit.document_id,

                    "document_type":
                        unit.document_type,

                    "authority":
                        unit.authority,

                    "text":
                        unit.text,

                    "source_element_ids":
                        unit
                        .source_element_ids,
                }
                for score, unit
                in results
            ]

        payload = {

            "target_sections": [
                {
                    "target_id":
                        section_id,

                    "target_heading":
                        heading,
                }
                for section_id, heading
                in TARGET_SECTIONS
            ],

            "source_elements":
                source_elements,

            "retrieved_enterprise_evidence":
                retrieval_context,
        }

        response = (
            self.llm
            .json_completion(
                SYSTEM_PROMPT,
                payload,
            )
        )

        mappings = []

        for item in (
            response.get(
                "mappings",
                []
            )
        ):

            mappings.append(
                Mapping(
                    **item
                )
            )

        return mappings
