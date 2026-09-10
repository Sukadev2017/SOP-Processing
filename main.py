from __future__ import annotations

import argparse

from pathlib import Path

from sopgenai.config import (
    Configuration,
)

from sopgenai.pipeline import (
    SOPPipeline,
)


def arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Enterprise GenAI "
            "SOP Migration Platform"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--knowledge",
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--output",
        default="./output",
    )

    parser.add_argument(
        "--config",
        default=(
            "./config/"
            "app_config.yaml"
        ),
    )

    parser.add_argument(
        "--rules",
        default=(
            "./config/"
            "bi_gp_rules.yaml"
        ),
    )

    parser.add_argument(
        "--extract-only",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = arguments()

    config = Configuration(

        Path(
            args.config
        ),

        Path(
            args.rules
        ),
    )

    pipeline = (
        SOPPipeline(
            config
        )
    )

    if args.extract_only:

        pipeline.run_extraction_only(

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

        return

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
