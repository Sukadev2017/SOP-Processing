from __future__ import annotations

import re
import shutil

from pathlib import Path

from docx import Document

from docx.oxml import (
    OxmlElement,
)

from docx.text.paragraph import (
    Paragraph,
)

from .models import (
    Mapping,
)


class ControlledDOCXRenderer:

    def render(
        self,
        template_path: Path,
        mappings: list[
            Mapping
        ],
        output_path: Path,
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Never modify original template.

        shutil.copy2(
            template_path,
            output_path,
        )

        document = Document(
            str(output_path)
        )

        mapping_lookup = {
            mapping.target_id:
                mapping
            for mapping
            in mappings
        }

        for paragraph in list(
            document.paragraphs
        ):

            heading_text = (
                paragraph.text
                .strip()
            )

            match = re.match(
                r"^(\d+)\s+(.+)$",
                heading_text,
            )

            if not match:
                continue

            section_id = (
                match.group(1)
            )

            if (
                section_id
                not in mapping_lookup
            ):
                continue

            mapping = (
                mapping_lookup[
                    section_id
                ]
            )

            content = (
                mapping
                .transformed_content
                .strip()
            )

            if not content:
                continue

            new_xml_paragraph = (
                OxmlElement(
                    "w:p"
                )
            )

            paragraph._p.addnext(
                new_xml_paragraph
            )

            new_paragraph = (
                Paragraph(
                    new_xml_paragraph,
                    paragraph._parent,
                )
            )

            try:

                new_paragraph.style = (
                    document.styles[
                        "Normal"
                    ]
                )

            except KeyError:
                pass

            new_paragraph.add_run(
                content
            )

        document.save(
            str(output_path)
        )

        return output_path
