from __future__ import annotations

import hashlib
import io
import uuid

from pathlib import Path
from typing import List

import fitz
import pdfplumber
import pytesseract

from PIL import Image
from pypdf import PdfReader

from .models import (
    BBox,
    CanonicalDocument,
    Element,
    TextRun,
)


def sha256(
    path: Path,
) -> str:

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


def color_to_hex(
    value: int,
) -> str:

    return f"#{value:06X}"


class PDFExtractor:

    def __init__(
        self,
        extraction_config: dict,
    ):

        pdf_config = (
            extraction_config[
                "pdf"
            ]
        )

        ocr = pdf_config[
            "ocr"
        ]

        self.ocr_enabled = (
            ocr.get(
                "enabled",
                True,
            )
        )

        self.ocr_dpi = ocr.get(
            "dpi",
            300,
        )

        self.min_native_chars = (
            ocr.get(
                "minimum_native_characters",
                40,
            )
        )

        self.min_ocr_confidence = (
            ocr.get(
                "minimum_confidence",
                40,
            )
        )

    def extract(
        self,
        path: Path,
        document_type: str | None = None,
    ) -> CanonicalDocument:

        pymupdf_document = (
            fitz.open(
                str(path)
            )
        )

        pypdf_reader = (
            PdfReader(
                str(path)
            )
        )

        elements: List[
            Element
        ] = []

        assets = []

        order = 0

        with pdfplumber.open(
            str(path)
        ) as plumber:

            for page_number, page in enumerate(
                pymupdf_document,
                start=1,
            ):

                native_text = []

                raw = page.get_text(
                    "dict"
                )

                for block in raw.get(
                    "blocks",
                    [],
                ):

                    if (
                        block.get(
                            "type"
                        )
                        != 0
                    ):
                        continue

                    for line in block.get(
                        "lines",
                        [],
                    ):

                        line_runs = []

                        line_text = []

                        for span in line.get(
                            "spans",
                            [],
                        ):

                            text = span.get(
                                "text",
                                "",
                            )

                            if not text:
                                continue

                            line_text.append(
                                text
                            )

                            bbox = span.get(
                                "bbox"
                            )

                            font = span.get(
                                "font",
                                "",
                            )

                            line_runs.append(
                                TextRun(
                                    text=text,

                                    font=font,

                                    size=span.get(
                                        "size"
                                    ),

                                    color=(
                                        color_to_hex(
                                            span.get(
                                                "color",
                                                0,
                                            )
                                        )
                                    ),

                                    bold=(
                                        "bold"
                                        in font.lower()
                                    ),

                                    italic=(
                                        "italic"
                                        in font.lower()
                                    ),

                                    bbox=(
                                        BBox(
                                            x0=bbox[0],
                                            y0=bbox[1],
                                            x1=bbox[2],
                                            y1=bbox[3],
                                        )
                                        if bbox
                                        else None
                                    ),
                                )
                            )

                        combined = (
                            "".join(
                                line_text
                            ).strip()
                        )

                        if not combined:
                            continue

                        native_text.append(
                            combined
                        )

                        order += 1

                        bbox = line.get(
                            "bbox"
                        )

                        elements.append(
                            Element(
                                element_id=(
                                    f"P{page_number}"
                                    f"-TXT-"
                                    f"{order:06d}"
                                ),

                                type="text",

                                page=(
                                    page_number
                                ),

                                order=order,

                                text=combined,

                                bbox=(
                                    BBox(
                                        x0=bbox[0],
                                        y0=bbox[1],
                                        x1=bbox[2],
                                        y1=bbox[3],
                                    )
                                    if bbox
                                    else None
                                ),

                                runs=(
                                    line_runs
                                ),

                                provenance={
                                    "extractor":
                                        "pymupdf"
                                },
                            )
                        )

                # ------------------------
                # TABLE EXTRACTION
                # ------------------------

                tables = (
                    plumber
                    .pages[
                        page_number - 1
                    ]
                    .extract_tables()
                    or []
                )

                for table_index, table in enumerate(
                    tables,
                    start=1,
                ):

                    rows = [
                        [
                            ""
                            if cell is None
                            else str(cell)
                            for cell
                            in row
                        ]
                        for row
                        in table
                    ]

                    order += 1

                    elements.append(
                        Element(
                            element_id=(
                                f"P{page_number}"
                                f"-TABLE-"
                                f"{table_index:03d}"
                            ),

                            type="table",

                            page=(
                                page_number
                            ),

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
                                    "pdfplumber"
                            },
                        )
                    )

                # ------------------------
                # OCR FALLBACK
                # ------------------------

                page_native_text = (
                    " ".join(
                        native_text
                    ).strip()
                )

                if (
                    self.ocr_enabled
                    and len(
                        page_native_text
                    )
                    < self.min_native_chars
                ):

                    pixmap = (
                        page.get_pixmap(
                            dpi=(
                                self.ocr_dpi
                            ),
                            alpha=False,
                        )
                    )

                    image = Image.open(
                        io.BytesIO(
                            pixmap.tobytes(
                                "png"
                            )
                        )
                    )

                    ocr_result = (
                        pytesseract
                        .image_to_data(
                            image,
                            output_type=(
                                pytesseract
                                .Output
                                .DICT
                            ),
                        )
                    )

                    words = []

                    for index, text in enumerate(
                        ocr_result[
                            "text"
                        ]
                    ):

                        text = (
                            text.strip()
                        )

                        if not text:
                            continue

                        try:
                            confidence = float(
                                ocr_result[
                                    "conf"
                                ][index]
                            )
                        except (
                            TypeError,
                            ValueError,
                        ):
                            confidence = -1

                        if (
                            confidence
                            >= self.min_ocr_confidence
                        ):
                            words.append(
                                text
                            )

                    if words:

                        order += 1

                        elements.append(
                            Element(
                                element_id=(
                                    f"P{page_number}"
                                    "-OCR"
                                ),

                                type=(
                                    "ocr_text"
                                ),

                                page=(
                                    page_number
                                ),

                                order=order,

                                text=" ".join(
                                    words
                                ),

                                provenance={
                                    "extractor":
                                        "tesseract",

                                    "dpi":
                                        self.ocr_dpi,
                                },
                            )
                        )

                # ------------------------
                # IMAGE REFERENCES
                # ------------------------

                for image_index, image in enumerate(
                    page.get_images(
                        full=True
                    ),
                    start=1,
                ):

                    assets.append(
                        {
                            "asset_id": (
                                f"P{page_number}"
                                f"-IMG-"
                                f"{image_index}"
                            ),

                            "page":
                                page_number,

                            "xref":
                                image[0],

                            "type":
                                "embedded_image",
                        }
                    )

        metadata = {
            str(key).strip("/"):
                value
            for key, value
            in (
                pypdf_reader.metadata
                or {}
            ).items()
        }

        # pypdf independent extraction
        # is retained as fallback/audit information.

        metadata[
            "pypdf_text_pages"
        ] = [
            (
                page.extract_text()
                or ""
            )[:1000]
            for page
            in pypdf_reader.pages
        ]

        pymupdf_document.close()

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

            metadata=metadata,

            elements=elements,

            assets=assets,
        )
