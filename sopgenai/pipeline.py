from __future__ import annotations

import json

from pathlib import Path

from .config import (
    Configuration,
)

from .extract_pdf import (
    PDFExtractor,
)

from .extract_docx import (
    DOCXExtractor,
)

from .llm import (
    LLMFactory,
)

from .enrich import (
    SemanticEnricher,
)

from .knowledge import (
    KnowledgeBuilder,
)

from .retrieval import (
    HybridRetriever,
)

from .map_transform import (
    TARGET_SECTIONS,
    TemplateMapper,
)

from .validate import (
    Validator,
)

from .render import (
    ControlledDOCXRenderer,
)


def save_json(
    path: Path,
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

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


class SOPPipeline:

    def __init__(
        self,
        config: Configuration,
    ):

        self.config = config

        # ----------------------------
        # LLM
        # ----------------------------

        self.llm = (
            LLMFactory.create(
                config.llm
            )
        )

        # ----------------------------
        # Semantic enrichment
        # ----------------------------

        self.enricher = (
            SemanticEnricher(
                self.llm
            )
        )

        # ----------------------------
        # Knowledge builder
        # ----------------------------

        self.knowledge_builder = (
            KnowledgeBuilder(
                config.authority
            )
        )

    def extract(
        self,
        path: Path,
        document_type: str | None = None,
    ):

        suffix = (
            path.suffix
            .lower()
        )

        if suffix == ".pdf":

            return (
                PDFExtractor(
                    self.config
                    .extraction
                )
                .extract(
                    path,
                    document_type,
                )
            )

        if suffix == ".docx":

            return (
                DOCXExtractor()
                .extract(
                    path,
                    document_type,
                )
            )

        raise ValueError(
            "Unsupported document "
            f"type: {suffix}"
        )

    def run(
        self,
        source_path: Path,
        template_path: Path,
        knowledge_files: list[
            Path
        ],
        output_path: Path,
    ):

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==================================================
        # 1. SOURCE SOP
        # ==================================================

        source_document = (
            self.extract(
                source_path,
                "SOP",
            )
        )

        self.enricher.enrich(
            source_document
        )

        # ==================================================
        # 2. TEMPLATE
        # ==================================================

        template_document = (
            self.extract(
                template_path,
                "TEMPLATE",
            )
        )

        # ==================================================
        # 3. ENTERPRISE KNOWLEDGE
        # ==================================================

        knowledge_units = []

        for knowledge_file in (
            knowledge_files
        ):

            knowledge_document = (
                self.extract(
                    knowledge_file
                )
            )

            self.enricher.enrich(
                knowledge_document
            )

            units = (
                self
                .knowledge_builder
                .build(
                    knowledge_document
                )
            )

            knowledge_units.extend(
                units
            )

        # Source SOP also becomes searchable evidence.

        source_units = (
            self
            .knowledge_builder
            .build(
                source_document
            )
        )

        knowledge_units.extend(
            source_units
        )

        # ==================================================
        # 4. HYBRID RETRIEVAL
        # ==================================================

        retriever = (
            HybridRetriever(
                embedding_config=(
                    self.config
                    .embedding
                ),

                retrieval_config=(
                    self.config
                    .retrieval
                ),
            )
        )

        retriever.index(
            knowledge_units
        )

        retrieved = {}

        for (
            section_id,
            heading,
        ) in TARGET_SECTIONS:

            query = (
                f"{heading}. "
                "Find relevant SOP content, "
                "policies, standards, roles, "
                "requirements, terminology, "
                "process information and "
                "approved reference material."
            )

            retrieved[
                section_id
            ] = (
                retriever.search(
                    query
                )
            )

        # ==================================================
        # 5. TEMPLATE MAPPING / GENERATION
        # ==================================================

        mapper = (
            TemplateMapper(
                self.llm
            )
        )

        mappings = (
            mapper.map(
                source_document,
                retrieved,
            )
        )

        # ==================================================
        # 6. VALIDATION
        # ==================================================

        validation_report = (
            Validator()
            .validate(
                mappings
            )
        )

        # ==================================================
        # 7. PERSIST ARTIFACTS
        # ==================================================

        save_json(
            output_path
            / "source_structure.json",

            source_document,
        )

        save_json(
            output_path
            / "template_structure.json",

            template_document,
        )

        save_json(
            output_path
            / "semantic_content.json",

            source_document.semantic,
        )

        save_json(
            output_path
            / "knowledge_units.json",

            [
                unit.model_dump()
                for unit
                in knowledge_units
            ],
        )

        retrieval_json = {}

        for (
            section_id,
            results,
        ) in retrieved.items():

            retrieval_json[
                section_id
            ] = [
                {
                    "score":
                        score,

                    "knowledge_unit":
                        unit.model_dump(),
                }
                for score, unit
                in results
            ]

        save_json(
            output_path
            / "retrieval_results.json",

            retrieval_json,
        )

        save_json(
            output_path
            / "mapping.json",

            [
                mapping.model_dump()
                for mapping
                in mappings
            ],
        )

        save_json(
            output_path
            / "validation_report.json",

            validation_report,
        )

        # ==================================================
        # 8. DOCX
        # ==================================================

        generated_document = (
            ControlledDOCXRenderer()
            .render(
                template_path=(
                    template_path
                ),

                mappings=(
                    mappings
                ),

                output_path=(
                    output_path
                    / "Draft_Migrated_SOP.docx"
                ),
            )
        )

        return {
            "generated_document":
                generated_document,

            "validation":
                validation_report,

            "mappings":
                mappings,
        }
