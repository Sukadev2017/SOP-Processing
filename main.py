import argparse
import os

from pathlib import Path

from sopgenai.llm import (
    OpenAICompatibleLLM,
)

from sopgenai.pipeline import (
    SOPPipeline,
)


def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="./output",
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--base-url",
        default=None,
    )

    return parser.parse_args()


def main():

    args = arguments()

    api_key = os.environ.get(
        "LLM_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "Set LLM_API_KEY "
            "environment variable."
        )

    llm = OpenAICompatibleLLM(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
    )

    pipeline = SOPPipeline(
        llm=llm,
        output_dir=Path(
            args.output
        ),
    )

    result = pipeline.run(
        source_pdf=Path(
            args.source
        ),
        template_docx=Path(
            args.template
        ),
    )

    print(
        "Generated:",
        result["generated"],
    )


if __name__ == "__main__":
    main()
