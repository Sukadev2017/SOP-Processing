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
            "Enterprise GenAI SOP "
            "Migration Platform"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Source SOP PDF or DOCX"
        ),
    )

    parser.add_argument(
        "--template",
        required=True,
        help=(
            "Target GP DOCX template"
        ),
    )

    parser.add_argument(
        "--knowledge",
        nargs="*",
        default=[],
        help=(
            "Policies, standards, "
            "guidelines, glossaries, "
            "architecture documents and "
            "approved SOPs"
        ),
    )

    parser.add_argument(
        "--output",
        default="./output",
    )

    parser.add_argument(
        "--config",
        default=(
            "./config/app_config.yaml"
        ),
    )

    parser.add_argument(
        "--rules",
        default=(
            "./config/bi_gp_rules.yaml"
        ),
    )

    return parser.parse_args()


def main():

    args = arguments()

    configuration = (
        Configuration(
            app_config_path=Path(
                args.config
            ),

            rules_config_path=Path(
                args.rules
            ),
        )
    )

    pipeline = (
        SOPPipeline(
            configuration
        )
    )

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

    print(
        "\nSOP processing completed."
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





# import argparse
# import os

# from pathlib import Path

# from sopgenai.llm import (
#     OpenAICompatibleLLM,
# )

# from sopgenai.pipeline import (
#     SOPPipeline,
# )


# def arguments():

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--source",
#         required=True,
#     )

#     parser.add_argument(
#         "--template",
#         required=True,
#     )

#     parser.add_argument(
#         "--output",
#         default="./output",
#     )

#     parser.add_argument(
#         "--model",
#         required=True,
#     )

#     parser.add_argument(
#         "--base-url",
#         default=None,
#     )

#     return parser.parse_args()


# def main():

#     args = arguments()

#     api_key = os.environ.get(
#         "LLM_API_KEY"
#     )

#     if not api_key:

#         raise RuntimeError(
#             "Set LLM_API_KEY "
#             "environment variable."
#         )

#     llm = OpenAICompatibleLLM(
#         model=args.model,
#         api_key=api_key,
#         base_url=args.base_url,
#     )

#     pipeline = SOPPipeline(
#         llm=llm,
#         output_dir=Path(
#             args.output
#         ),
#     )

#     result = pipeline.run(
#         source_pdf=Path(
#             args.source
#         ),
#         template_docx=Path(
#             args.template
#         ),
#     )

#     print(
#         "Generated:",
#         result["generated"],
#     )


# if __name__ == "__main__":
#     main()
