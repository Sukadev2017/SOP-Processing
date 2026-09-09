from __future__ import annotations

import hashlib
import uuid

from pathlib import Path

from docx import Document

from .models import (
    CanonicalDocument,
    Element,
    TextRun,
)


def sha256(
    path: Path,
):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


class DOCXExtractor:

    def extract(
        self,
        path: Path,
        document_type: str | None = None,
    ):

        document = Document(
            str(path)
        )

        elements = []

        order = 0

        # ----------------------------
        # Paragraphs
        # ----------------------------

        for paragraph in (
            document.paragraphs
        ):

            text = (
                paragraph.text.strip()
            )

            if not text:
                continue

            order += 1

            runs = []

            for run in (
                paragraph.runs
            ):

                color = None

                if (
                    run.font.color
                    and run.font.color.rgb
                ):
                    color = (
                        f"#{run.font.color.rgb}"
                    )

                runs.append(
                    TextRun(
                        text=run.text,

                        font=(
                            run.font.name
                        ),

                        size=(
                            run.font.size.pt
                            if run.font.size
                            else None
                        ),

                        color=color,

                        bold=bool(
                            run.bold
                        ),

                        italic=bool(
                            run.italic
                        ),
                    )
                )

            elements.append(
                Element(
                    element_id=(
                        f"DOCX-TXT-"
                        f"{order:06d}"
                    ),

                    type="text",

                    page=0,

                    order=order,

                    text=text,

                    runs=runs,

                    metadata={
                        "style": (
                            paragraph
                            .style
                            .name
                            if paragraph.style
                            else None
                        )
                    },

                    provenance={
                        "extractor":
                            "python-docx"
                    },
                )
            )

        # ----------------------------
        # Tables
        # ----------------------------

        for table_index, table in enumerate(
            document.tables,
            start=1,
        ):

            order += 1

            rows = [
                [
                    cell.text.strip()
                    for cell
                    in row.cells
                ]
                for row
                in table.rows
            ]

            elements.append(
                Element(
                    element_id=(
                        f"DOCX-TABLE-"
                        f"{table_index:03d}"
                    ),

                    type="table",

                    page=0,

                    order=order,

                    text="\n".join(
                        " | ".join(
                            row
                        )
                        for row
                        in rows
                    ),

                    metadata={
                        "rows":
                            rows
                    },

                    provenance={
                        "extractor":
                            "python-docx"
                    },
                )
            )

        return CanonicalDocument(
            document_id=(
                "DOC-"
                + uuid.uuid4()
                .hex[:12]
                .upper()
            ),

            source_file=(
                path.name
            ),

            source_sha256=(
                sha256(path)
            ),

            document_type=(
                document_type
            ),

            elements=elements,
        )
