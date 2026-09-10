from __future__ import annotations

import argparse
from pathlib import Path

from sopgenai.config import Configuration
from sopgenai.pipeline import SOPPipeline


def arguments():

    parser = argparse.ArgumentParser(
        description="Enterprise GenAI SOP Migration Platform"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source SOP PDF or DOCX",
    )

    parser.add_argument(
        "--template",
        required=True,
        help="Target GP DOCX template",
    )

    parser.add_argument(
        "--knowledge",
        nargs="*",
        default=[],
        help="Enterprise knowledge files",
    )

    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory",
    )

    parser.add_argument(
        "--config",
        default="./config/app_config.yaml",
        help="Application configuration file",
    )

    parser.add_argument(
        "--rules",
        default="./config/bi_gp_rules.yaml",
        help="GP rules configuration file",
    )

    # =====================================================
    # EXTRACTION-ONLY OPTION
    # =====================================================

    parser.add_argument(
        "--extract-only",
        action="store_true",
        help=(
            "Extract only source_structure.json and "
            "template_structure.json. "
            "No LLM, RAG, mapping, validation or generation."
        ),
    )

    return parser.parse_args()


def main():

    args = arguments()

    # =====================================================
    # LOAD CONFIGURATION
    # =====================================================

    configuration = Configuration(
        app_config_path=Path(
            args.config
        ),
        rules_config_path=Path(
            args.rules
        ),
    )

    # =====================================================
    # CREATE PIPELINE
    # =====================================================

    pipeline = SOPPipeline(
        configuration
    )

    # =====================================================
    # EXTRACTION-ONLY MODE
    # =====================================================

    if args.extract_only:

        result = pipeline.run_extraction_only(

            source_path=Path(
                args.source
            ),

            template_path=Path(
                args.template
            ),

            output_path=Path(
                args.output
            ),
        )

        print()
        print(
            "Extraction completed successfully."
        )

        print(
            "Source structure:",
            result[
                "source_structure"
            ],
        )

        print(
            "Template structure:",
            result[
                "template_structure"
            ],
        )

        return

    # =====================================================
    # FULL SOP PIPELINE
    # =====================================================

    result = pipeline.run(

        source_path=Path(
            args.source
        ),

        template_path=Path(
            args.template
        ),

        knowledge_files=[
            Path(file)
            for file
            in args.knowledge
        ],

        output_path=Path(
            args.output
        ),
    )

    print()
    print(
        "SOP processing completed."
    )

    print(
        "Generated document:",
        result[
            "generated_document"
        ],
    )

    print(
        "Validation passed:",
        result[
            "validation"
        ][
            "passed"
        ],
    )


if __name__ == "__main__":

    main()
