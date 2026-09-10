from __future__ import annotations

import json

from pathlib import Path

from .config import Configuration

from .extract_pdf import PDFExtractor

from .extract_docx import DOCXExtractor

from .llm import LLMFactory

from .enrich import SemanticEnricher

from .knowledge import KnowledgeBuilder

from .retrieval import HybridRetriever

from .map_transform import (
    TARGET_SECTIONS,
    TemplateMapper,
)

from .validate import Validator

from .render import ControlledDOCXRenderer


# ==========================================================
# JSON UTILITY
# ==========================================================


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


# ==========================================================
# SOP PIPELINE
# ==========================================================


class SOPPipeline:

    def __init__(
        self,
        config: Configuration,
    ):

        self.config = config

        # --------------------------------------------------
        # IMPORTANT
        #
        # Do not initialize an LLM here.
        #
        # This allows extraction-only mode to work without:
        #
        # Ollama
        # Hugging Face
        # OpenAI
        # Anthropic
        # API keys
        #
        # --------------------------------------------------

        self.llm = None

        self.enricher = None

        # Knowledge builder does not require LLM.

        self.knowledge_builder = (
            KnowledgeBuilder(
                config.authority
            )
        )

    # ======================================================
    # LLM INITIALIZATION
    # ======================================================

    def initialize_llm(
        self,
    ):

        if self.llm is not None:
            return

        print(
            "Initializing LLM provider..."
        )

        self.llm = (
            LLMFactory.create(
                self.config.llm
            )
        )

        self.enricher = (
            SemanticEnricher(
                self.llm
            )
        )

    # ======================================================
    # DOCUMENT EXTRACTION
    # ======================================================

    def extract(
        self,
        path: Path,
        document_type=None,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        suffix = (
            path.suffix
            .lower()
        )

        # --------------------------------------------------
        # PDF
        # --------------------------------------------------

        if suffix == ".pdf":

            extractor = (
                PDFExtractor(
                    self.config
                    .extraction
                )
            )

            return extractor.extract(
                path,
                document_type,
            )

        # --------------------------------------------------
        # DOCX
        # --------------------------------------------------

        if suffix == ".docx":

            extractor = (
                DOCXExtractor()
            )

            return extractor.extract(
                path,
                document_type,
            )

        raise ValueError(
            "Unsupported document type: "
            f"{suffix}. "
            "Supported types are PDF and DOCX."
        )

    # ======================================================
    # EXTRACTION-ONLY PIPELINE
    # ======================================================

    def run_extraction_only(
        self,
        source_path: Path,
        template_path: Path,
        output_path: Path,
    ):

        """
        Extract only the source SOP and target template.

        Generated files:

            source_structure.json
            template_structure.json

        This method DOES NOT initialize or call:

            Ollama
            Hugging Face
            OpenAI
            Anthropic
            Sentence Transformers
            FAISS
            BM25
            Template Mapping
            GenAI Generation
            Validation
            DOCX Renderer
        """

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # SOURCE
        # --------------------------------------------------

        print(
            "Extracting source document:"
        )

        print(
            source_path
        )

        source_document = (
            self.extract(
                source_path,
                "SOP",
            )
        )

        # --------------------------------------------------
        # TEMPLATE
        # --------------------------------------------------

        print(
            "Extracting template document:"
        )

        print(
            template_path
        )

        template_document = (
            self.extract(
                template_path,
                "TEMPLATE",
            )
        )

        # --------------------------------------------------
        # OUTPUT PATHS
        # --------------------------------------------------

        source_output = (
            output_path
            / "source_structure.json"
        )

        template_output = (
            output_path
            / "template_structure.json"
        )

        # --------------------------------------------------
        # SAVE SOURCE
        # --------------------------------------------------

        save_json(
            source_output,
            source_document,
        )

        # --------------------------------------------------
        # SAVE TEMPLATE
        # --------------------------------------------------

        save_json(
            template_output,
            template_document,
        )

        return {

            "source_structure":
                source_output,

            "template_structure":
                template_output,
        }

    # ======================================================
    # FULL GENAI PIPELINE
    # ======================================================

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

        # --------------------------------------------------
        # Initialize LLM only for full pipeline
        # --------------------------------------------------

        self.initialize_llm()

        # ==================================================
        # 1. SOURCE SOP
        # ==================================================

        print(
            "Processing source SOP..."
        )

        source = self.extract(
            source_path,
            "SOP",
        )

        self.enricher.enrich(
            source
        )

        # ==================================================
        # 2. TEMPLATE
        # ==================================================

        print(
            "Processing template..."
        )

        template = self.extract(
            template_path,
            "TEMPLATE",
        )

        # ==================================================
        # 3. ENTERPRISE KNOWLEDGE
        # ==================================================

        units = []

        for file in (
            knowledge_files
        ):

            print(
                "Processing knowledge document:",
                file,
            )

            document = (
                self.extract(
                    file
                )
            )

            self.enricher.enrich(
                document
            )

            document_units = (
                self
                .knowledge_builder
                .build(
                    document
                )
            )

            units.extend(
                document_units
            )

        # ==================================================
        # 4. SOURCE SOP KNOWLEDGE
        # ==================================================

        source_units = (
            self
            .knowledge_builder
            .build(
                source
            )
        )

        units.extend(
            source_units
        )

        # ==================================================
        # 5. HYBRID RETRIEVAL
        # ==================================================

        print(
            "Building hybrid retrieval index..."
        )

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
            units
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
        # 6. TEMPLATE MAPPING / GENERATION
        # ==================================================

        print(
            "Mapping source content "
            "to target template..."
        )

        mapper = (
            TemplateMapper(
                self.llm
            )
        )

        mappings = (
            mapper.map(
                source,
                retrieved,
            )
        )

        # ==================================================
        # 7. VALIDATION
        # ==================================================

        print(
            "Validating generated content..."
        )

        report = (
            Validator()
            .validate(
                mappings
            )
        )

        # ==================================================
        # 8. JSON OUTPUTS
        # ==================================================

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

        save_json(
            output_path
            / "semantic_content.json",
            source.semantic,
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

        save_json(
            output_path
            / "retrieval_results.json",

            {

                section_id: [

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
                    section_id,
                    results,
                )
                in retrieved.items()
            },
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
            report,
        )

        # ==================================================
        # 9. DOCX RENDERING
        # ==================================================

        print(
            "Generating DOCX..."
        )

        generated = (
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
                generated,

            "validation":
                report,

            "mappings":
                mappings,
        }
