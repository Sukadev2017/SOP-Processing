from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document


class TemplateRenderer:

    def render(
        self,
        template_path: Path,
        mapping: dict,
        output_path: Path,
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            template_path,
            output_path,
        )

        document = Document(
            str(output_path)
        )

        generated = {}

        for item in mapping.get(
            "mappings",
            []
        ):

            generated[
                str(
                    item.get(
                        "target_section_id"
                    )
                )
            ] = item.get(
                "generated_content",
                "",
            )

        for paragraph in document.paragraphs:

            text = (
                paragraph.text
                .strip()
                .upper()
            )

            for section_id, content in (
                generated.items()
            ):

                prefix = (
                    section_id + " "
                )

                if text.startswith(
                    prefix
                ):

                    if not content:
                        continue

                    new_para = (
                        paragraph
                        .insert_paragraph_before(
                            ""
                        )
                    )

                    new_para.add_run(
                        content
                    )

        document.save(
            str(output_path)
        )

        return output_path
