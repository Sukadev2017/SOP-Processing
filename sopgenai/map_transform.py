from __future__ import annotations

from .llm import LLMProvider
from .models import CanonicalDocument


TARGET_SECTIONS = [
    {
        "id": "1",
        "title": "PURPOSE"
    },
    {
        "id": "2",
        "title": "APPLICABILITY"
    },
    {
        "id": "3",
        "title":
            "DEFINITIONS & ABBREVIATIONS"
    },
    {
        "id": "4",
        "title":
            "IMPLEMENTATION AND/OR "
            "PRE-REQUISITES"
    },
    {
        "id": "5",
        "title":
            "ROLES & RESPONSIBILITIES"
    },
    {
        "id": "6",
        "title": "PROCESS"
    },
    {
        "id": "7",
        "title":
            "ASSOCIATED DOCUMENTS"
    },
    {
        "id": "8",
        "title": "REFERENCES"
    },
    {
        "id": "9",
        "title":
            "DISTRIBUTION OF CONTROLLED "
            "PRINTS/COPIES"
    },
    {
        "id": "10",
        "title": "DOCUMENT HISTORY"
    },
]


MAPPING_PROMPT = """
Map the source SOP evidence into the supplied target
SOP template sections.

Rules:

- Preserve source meaning.
- Do not invent information.
- Multiple source elements may map to one target.
- One source element may contribute to multiple targets.
- Section 6 PROCESS should contain how-style operational
  information.
- Missing information must be marked SME_REQUIRED.
- Return source element IDs for every mapping.
- Assign confidence from 0 to 1.
- Return JSON only.
"""


class GenAITemplateMapper:

    def __init__(
        self,
        llm: LLMProvider,
    ):

        self.llm = llm

    def map(
        self,
        source: CanonicalDocument,
    ):

        elements = [
            {
                "element_id":
                    element.element_id,

                "page":
                    element.page_number,

                "text":
                    element.text,
            }
            for element
            in source.elements
            if element.text
        ]

        payload = {
            "target_sections":
                TARGET_SECTIONS,

            "source_elements":
                elements,
        }

        return (
            self.llm
            .structured_completion(
                MAPPING_PROMPT,
                payload,
            )
        )
