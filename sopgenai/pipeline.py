from __future__ import annotations

import json
from pathlib import Path

from .extract_pdf import PDFExtractor
from .extract_docx import DOCXExtractor

from .enrich import SemanticEnricher

from .knowledge import KnowledgeBuilder

from .map_transform import (
    GenAITemplateMapper,
)

from .validate import (
    DeterministicValidator,
)

from .render import TemplateRenderer


def save_json(
    path,
    payload,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(
        payload,
        "model_dump",
    ):
        payload = (
            payload.model_dump()
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


class SOPPipeline:

    def __init__(
        self,
        llm,
        output_dir: Path,
    ):

        self.llm = llm

        self.output_dir = (
            output_dir
        )

    def run(
        self,
        source_pdf: Path,
        template_docx: Path,
    ):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------
        # PDF extraction
        # ---------------------

        pdf_extractor = (
            PDFExtractor()
        )

        source = (
            pdf_extractor.extract(
                source_pdf,
                document_type="SOP",
            )
        )

        # ---------------------
        # Template extraction
        # ---------------------

        template_extractor = (
            DOCXExtractor()
        )

        template = (
            template_extractor.extract(
                template_docx,
                document_type="TEMPLATE",
            )
        )

        # ---------------------
        # GenAI enrichment
        # ---------------------

        enricher = (
            SemanticEnricher(
                self.llm
            )
        )

        semantic = (
            enricher.enrich(
                source
            )
        )

        # ---------------------
        # Knowledge units
        # ---------------------

        knowledge_builder = (
            KnowledgeBuilder()
        )

        knowledge_units = (
            knowledge_builder.build(
                source
            )
        )

        # ---------------------
        # GenAI template mapping
        # ---------------------

        mapper = (
            GenAITemplateMapper(
                self.llm
            )
        )

        mapping = mapper.map(
            source
        )

        # ---------------------
        # Validation
        # ---------------------

        validator = (
            DeterministicValidator()
        )

        validation = (
            validator.validate(
                mapping
            )
        )

        # ---------------------
        # Persistence
        # ---------------------

        save_json(
            self.output_dir
            / "source_structure.json",
            source,
        )

        save_json(
            self.output_dir
            / "template_structure.json",
            template,
        )

        save_json(
            self.output_dir
            / "semantic_content.json",
            semantic,
        )

        save_json(
            self.output_dir
            / "knowledge_units.json",
            [
                unit.model_dump()
                for unit
                in knowledge_units
            ],
        )

        save_json(
            self.output_dir
            / "mapping.json",
            mapping,
        )

        save_json(
            self.output_dir
            / "validation.json",
            validation,
        )

        # ---------------------
        # Render
        # ---------------------

        renderer = (
            TemplateRenderer()
        )

        generated = (
            renderer.render(
                template_docx,
                mapping,
                self.output_dir
                / "Draft_SOP.docx",
            )
        )

        return {
            "source":
                source,

            "template":
                template,

            "semantic":
                semantic,

            "mapping":
                mapping,

            "validation":
                validation,

            "generated":
                generated,
        }
