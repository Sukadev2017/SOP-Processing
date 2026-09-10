from __future__ import annotations

from .models import (
    CanonicalDocument,
    Mapping,
    Section,
)


SYSTEM_PROMPT = """
You are constructing ONE section of a controlled SOP.

You will receive:

1. One target section extracted from the actual approved
   Word template.

2. The template's instructions/content associated with
   that section.

3. A small set of evidence retrieved from the source SOP
   and enterprise knowledge.

Use ONLY supplied evidence.

Do not invent missing information.

Determine whether the evidence supports content for this
specific target section.

Preserve source_element_ids.

If the evidence is insufficient:
state = SME_REQUIRED.

If evidence conflicts:
state = CONFLICT.

Allowed states:

SOURCE_SUPPORTED
REFERENCE_DERIVED
CONFLICT
SME_REQUIRED

Use active and concise language.

Use "must" for mandatory requirements.

Use "should" for recommendations.

Return exactly:

{
  "mapping": {
    "target_id": "...",
    "target_heading": "...",
    "target_level": 1,
    "source_element_ids": [],
    "transformed_content": "...",
    "confidence": 0.0,
    "state": "SOURCE_SUPPORTED",
    "evidence": []
  }
}
"""


class TemplateMapper:

    def __init__(
        self,
        llm,
        generation_config=None,
    ):

        self.llm = llm

        self.config = (
            generation_config
            or {}
        )

        self.evidence_per_section = (
            self.config.get(
                "evidence_per_section",
                6,
            )
        )

    @staticmethod
    def target_sections(
        template_document:
            CanonicalDocument,
    ):

        sections = sorted(

            template_document.sections,

            key=lambda section:
                section.order,
        )

        if not sections:

            raise ValueError(
                "No sections were detected "
                "in the target template. "
                "Check Word heading styles "
                "or heading detection configuration."
            )

        return sections

    def map_section(
        self,
        target_section:
            Section,
        retrieval_results,
    ):

        evidence = []

        for (
            score,
            unit,
        ) in retrieval_results[
            :self.evidence_per_section
        ]:

            evidence.append(
                {
                    "retrieval_score":
                        float(score),

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
            )

        target_id = (
            target_section.number
            or target_section.section_id
        )

        payload = {

            "target_section": {

                "target_id":
                    target_id,

                "target_heading":
                    target_section.heading,

                "target_level":
                    target_section.level,

                "template_instructions":
                    target_section.text[
                        :3000
                    ],
            },

            "evidence":
                evidence,
        }

        print(
            "[Mapping] "
            f"{target_id} "
            f"{target_section.heading} "
            f"with {len(evidence)} "
            "evidence units"
        )

        response = (
            self.llm
            .json_completion(
                SYSTEM_PROMPT,
                payload,
            )
        )

        item = (
            response.get(
                "mapping",
                {},
            )
        )

        # -------------------------------------------------
        # Target identity comes from the actual template,
        # not from the LLM.
        # -------------------------------------------------

        item[
            "target_id"
        ] = target_id

        item[
            "target_heading"
        ] = (
            target_section.heading
        )

        item[
            "target_level"
        ] = (
            target_section.level
        )

        item.setdefault(
            "source_element_ids",
            [],
        )

        item.setdefault(
            "transformed_content",
            "",
        )

        item.setdefault(
            "confidence",
            0.0,
        )

        item.setdefault(
            "state",
            "SME_REQUIRED",
        )

        item.setdefault(
            "evidence",
            evidence,
        )

        return Mapping(
            **item
        )

    def map(
        self,
        template_document,
        retrieved,
    ):

        mappings = []

        target_sections = (
            self.target_sections(
                template_document
            )
        )

        total = len(
            target_sections
        )

        for (
            index,
            section,
        ) in enumerate(
            target_sections,
            1,
        ):

            target_id = (
                section.number
                or section.section_id
            )

            print(
                f"[Mapping] "
                f"Section {index}/{total}: "
                f"{target_id} "
                f"{section.heading}"
            )

            results = (
                retrieved.get(
                    target_id,
                    [],
                )
            )

            mapping = (
                self.map_section(
                    section,
                    results,
                )
            )

            mappings.append(
                mapping
            )

        return mappings
