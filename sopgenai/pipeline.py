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
    TemplateMapper,
)

from .validate import (
    Validator,
)

from .render import (
    ControlledDOCXRenderer,
)


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
        config:
            Configuration,
    ):

        self.config = config

        # ---------------------------------------------
        # Lazy LLM initialization.
        #
        # --extract-only never initializes the LLM.
        # ---------------------------------------------

        self.llm = None

        self.enricher = None

        self.knowledge_builder = (
            KnowledgeBuilder(
                config.authority
            )
        )

    def initialize_llm(
        self,
    ):

        if self.llm is not None:

            return

        provider = (
            self.config.llm[
                "provider"
            ]
        )

        print(
            "[LLM] "
            f"Initializing {provider}"
        )

        self.llm = (
            LLMFactory.create(
                self.config.llm
            )
        )

        self.enricher = (
            SemanticEnricher(

                self.llm,

                self.config
                .app
                .get(
                    "semantic_enrichment",
                    {},
                ),
            )
        )

    # =====================================================
    # EXTRACTION
    # =====================================================

    def extract(
        self,
        path,
        document_type,
    ):

        if not path.exists():

            raise FileNotFoundError(
                path
            )

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
                DOCXExtractor(
                    self.config
                    .app
                    .get(
                        "template",
                        {},
                    )
                )
                .extract(
                    path,
                    document_type,
                )
            )

        raise ValueError(
            f"Unsupported file: {path}"
        )

    # =====================================================
    # EXTRACTION ONLY
    # =====================================================

    def run_extraction_only(
        self,
        source_path,
        template_path,
        output_path,
    ):

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "[1/2] Extracting source..."
        )

        source = self.extract(
            source_path,
            "SOP",
        )

        print(
            "[2/2] Extracting template..."
        )

        template = self.extract(
            template_path,
            "TEMPLATE",
        )

        save_json(
            output_path
            / "source_structure.json",
            source,
        )

        save_json(
            output_path
            / "template_structure.json",
            template,
        )

        print(
            "[DONE] Extraction complete."
        )

        print(
            "[Template] "
            f"Detected {len(template.sections)} "
            "sections."
        )

        for section in (
            template.sections
        ):

            print(
                "  ",
                section.number
                or section.section_id,
                section.heading,
            )

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def run(
        self,
        source_path,
        template_path,
        knowledge_files,
        output_path,
    ):

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =============================================
        # 1. EXTRACTION
        # =============================================

        print(
            "\n[1/8] "
            "Extracting source SOP..."
        )

        source = self.extract(
            source_path,
            "SOP",
        )

        print(
            "\n[2/8] "
            "Extracting target template..."
        )

        template = self.extract(
            template_path,
            "TEMPLATE",
        )

        save_json(
            output_path
            / "source_structure.json",
            source,
        )

        save_json(
            output_path
            / "template_structure.json",
            template,
        )

        # ---------------------------------------------
        # Dynamic template sections
        # ---------------------------------------------

        target_sections = (
            TemplateMapper
            .target_sections(
                template
            )
        )

        print(
            "\n[Template] "
            f"Detected "
            f"{len(target_sections)} "
            "target sections:"
        )

        for section in (
            target_sections
        ):

            print(
                "   ",
                section.number
                or section.section_id,
                section.heading,
            )

        # =============================================
        # 2. LLM
        # =============================================

        self.initialize_llm()

        # =============================================
        # 3. BATCHED SOURCE ENRICHMENT
        # =============================================

        print(
            "\n[3/8] "
            "Semantic enrichment..."
        )

        self.enricher.enrich(
            source
        )

        save_json(
            output_path
            / "semantic_content.json",
            source.semantic,
        )

        # =============================================
        # 4. KNOWLEDGE UNITS
        # =============================================

        print(
            "\n[4/8] "
            "Building knowledge units..."
        )

        units = (
            self.knowledge_builder
            .build(
                source
            )
        )

        for (
            index,
            knowledge_file,
        ) in enumerate(
            knowledge_files,
            1,
        ):

            print(
                "[Knowledge] "
                f"{index}/"
                f"{len(knowledge_files)} "
                f"{knowledge_file}"
            )

            knowledge_document = (
                self.extract(
                    knowledge_file,
                    "UNKNOWN",
                )
            )

            self.enricher.enrich(
                knowledge_document
            )

            units.extend(
                self
                .knowledge_builder
                .build(
                    knowledge_document
                )
            )

        save_json(
            output_path
            / "knowledge_units.json",

            [
                unit.model_dump()

                for unit
                in units
            ],
        )

        # =============================================
        # 5. RETRIEVAL
        # =============================================

        print(
            "\n[5/8] "
            "Building retrieval index..."
        )

        retriever = (
            HybridRetriever(

                self.config
                .embedding,

                self.config
                .app[
                    "retrieval"
                ],
            )
        )

        retriever.index(
            units
        )

        retrieved = {}

        total_sections = len(
            target_sections
        )

        # ---------------------------------------------
        # One retrieval query for each ACTUAL
        # template section.
        # ---------------------------------------------

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
                "[Retrieval] "
                f"{index}/"
                f"{total_sections}: "
                f"{target_id} "
                f"{section.heading}"
            )

            query = (
                f"Target SOP section: "
                f"{section.heading}. "
                f"Template instructions: "
                f"{section.text[:1000]}. "
                "Find the most relevant "
                "source SOP and enterprise "
                "evidence for this section."
            )

            retrieved[
                target_id
            ] = (
                retriever.search(
                    query
                )
            )

        save_json(
            output_path
            / "retrieval_results.json",

            {

                target_id: [

                    {
                        "score":
                            score,

                        "knowledge_unit":
                            unit.model_dump(),
                    }

                    for (
                        score,
                        unit,
                    )
                    in results
                ]

                for (
                    target_id,
                    results,
                )
                in retrieved.items()
            },
        )

        # =============================================
        # 6. ONE LLM CALL PER TARGET SECTION
        # =============================================

        print(
            "\n[6/8] "
            "Mapping target sections..."
        )

        mapper = (
            TemplateMapper(

                self.llm,

                self.config
                .generation,
            )
        )

        mappings = (
            mapper.map(
                template,
                retrieved,
            )
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

        # =============================================
        # 7. VALIDATION
        # =============================================

        print(
            "\n[7/8] "
            "Validating mappings..."
        )

        validation = (
            Validator()
            .validate(
                mappings,
                target_sections,
            )
        )

        save_json(
            output_path
            / "validation_report.json",
            validation,
        )

        # =============================================
        # 8. RENDERING
        # =============================================

        print(
            "\n[8/8] "
            "Rendering Word document..."
        )

        generated_document = (
            ControlledDOCXRenderer()
            .render(

                template_path,

                template,

                mappings,

                output_path
                / "Draft_Migrated_SOP.docx",
            )
        )

        print(
            "\n[DONE] "
            "SOP processing completed."
        )

        return {

            "generated_document":
                generated_document,

            "validation":
                validation,

            "mappings":
                mappings,
        }
