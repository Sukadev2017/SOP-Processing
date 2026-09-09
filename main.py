import argparse
import os

from pathlib import Path

from sopgenai.llm import HuggingFaceLLM
from sopgenai.pipeline import SOPPipeline


def arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Source SOP PDF",
    )

    parser.add_argument(
        "--template",
        required=True,
        help="Target SOP DOCX template",
    )

    parser.add_argument(
        "--output",
        default="./output",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model ID",
    )

    return parser.parse_args()


def main():

    args = arguments()

    hf_token = os.environ.get(
        "HF_TOKEN"
    )

    if not hf_token:
        raise RuntimeError(
            "Set the HF_TOKEN environment variable."
        )

    llm = HuggingFaceLLM(
        model=args.model,
        api_token=hf_token,
        max_tokens=4096,
        temperature=0.0,
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
