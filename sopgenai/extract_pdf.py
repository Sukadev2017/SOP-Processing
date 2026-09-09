from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

import fitz
import pdfplumber
import pytesseract

from PIL import Image
from pypdf import PdfReader

from .models import (
    BoundingBox,
    CanonicalDocument,
    DocumentElement,
    DocumentMetadata,
    ImageElement,
    TableCell,
    TableElement,
    TextSpan,
)


def file_hash(path: Path) -> str:

    sha = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha.update(chunk)

    return sha.hexdigest()


def color_to_hex(color: int) -> str:

    return f"#{color:06X}"


class PDFExtractor:

    def __init__(
        self,
        ocr_enabled=True,
        ocr_min_chars=50,
        render_dpi=250,
    ):

        self.ocr_enabled = ocr_enabled

        self.ocr_min_chars = ocr_min_chars

        self.render_dpi = render_dpi

    def extract(
        self,
        pdf_path: Path,
        document_type: str = "SOP",
    ) -> CanonicalDocument:

        document_id = (
            f"DOC-{uuid.uuid4().hex[:12].upper()}"
        )

        metadata = self._metadata(
            pdf_path,
            document_id,
            document_type,
        )

        elements = self._extract_pymupdf(
            pdf_path
        )

        tables = self._extract_tables(
            pdf_path
        )

        images = self._extract_images(
            pdf_path
        )

        return CanonicalDocument(
            metadata=metadata,
            elements=elements,
            tables=tables,
            images=images,
            extraction_metadata={
                "primary_parser": "PyMuPDF",
                "metadata_parser": "pypdf",
                "table_parser": "pdfplumber",
                "ocr": "Tesseract",
            },
        )

    def _metadata(
        self,
        path,
        document_id,
        document_type,
    ):

        reader = PdfReader(str(path))

        info = reader.metadata or {}

        return DocumentMetadata(
            document_id=document_id,
            file_name=path.name,
            file_type="pdf",
            title=info.get("/Title"),
            document_type=document_type,
            file_hash=file_hash(path),
        )

    def _extract_pymupdf(
        self,
        path: Path,
    ):

        doc = fitz.open(str(path))

        elements = []

        order = 0

        for page_index, page in enumerate(doc):

            page_number = page_index + 1

            page_dict = page.get_text(
                "dict"
            )

            page_text = []

            for block in page_dict["blocks"]:

                if block.get("type") != 0:
                    continue

                for line in block.get(
                    "lines", []
                ):

                    spans = []

                    text_parts = []

                    for span in line.get(
                        "spans", []
                    ):

                        text = span.get(
                            "text", ""
                        )

                        if not text.strip():
                            continue

                        text_parts.append(text)

                        bbox = span.get(
                            "bbox"
                        )

                        spans.append(
                            TextSpan(
                                text=text,
                                bbox=BoundingBox(
                                    x0=bbox[0],
                                    y0=bbox[1],
                                    x1=bbox[2],
                                    y1=bbox[3],
                                ),
                                font=span.get(
                                    "font"
                                ),
                                size=span.get(
                                    "size"
                                ),
                                color=color_to_hex(
                                    span.get(
                                        "color",
                                        0,
                                    )
                                ),
                                bold=(
                                    "bold"
                                    in span.get(
                                        "font",
                                        "",
                                    ).lower()
                                ),
                                italic=(
                                    "italic"
                                    in span.get(
                                        "font",
                                        "",
                                    ).lower()
                                ),
                            )
                        )

                    line_text = "".join(
                        text_parts
                    ).strip()

                    if not line_text:
                        continue

                    order += 1

                    page_text.append(
                        line_text
                    )

                    bbox = line.get(
                        "bbox"
                    )

                    elements.append(
                        DocumentElement(
                            element_id=(
                                f"EL-{order:07d}"
                            ),
                            type="text",
                            page_number=page_number,
                            order=order,
                            text=line_text,
                            bbox=BoundingBox(
                                x0=bbox[0],
                                y0=bbox[1],
                                x1=bbox[2],
                                y1=bbox[3],
                            ),
                            spans=spans,
                            extraction_method=(
                                "PyMuPDF"
                            ),
                        )
                    )

            combined = "\n".join(
                page_text
            )

            if (
                self.ocr_enabled
                and len(combined.strip())
                < self.ocr_min_chars
            ):

                ocr_text = self._ocr_page(
                    page
                )

                if ocr_text.strip():

                    order += 1

                    elements.append(
                        DocumentElement(
                            element_id=(
                                f"EL-{order:07d}"
                            ),
                            type="ocr_text",
                            page_number=page_number,
                            order=order,
                            text=ocr_text,
                            extraction_method=(
                                "Tesseract"
                            ),
                            confidence=None,
                        )
                    )

        doc.close()

        return elements

    def _ocr_page(
        self,
        page,
    ):

        zoom = self.render_dpi / 72

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = Image.open(
            io.BytesIO(
                pix.tobytes("png")
            )
        )

        return pytesseract.image_to_string(
            image
        )

    def _extract_tables(
        self,
        path: Path,
    ):

        tables = []

        table_counter = 0

        with pdfplumber.open(
            str(path)
        ) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):

                extracted = (
                    page.extract_tables()
                    or []
                )

                for table in extracted:

                    table_counter += 1

                    cells = []

                    row_count = len(table)

                    col_count = max(
                        (
                            len(row)
                            for row in table
                            if row
                        ),
                        default=0,
                    )

                    for row_index, row in enumerate(
                        table
                    ):

                        if not row:
                            continue

                        for col_index, value in enumerate(
                            row
                        ):

                            cells.append(
                                TableCell(
                                    row=row_index,
                                    column=col_index,
                                    text=(
                                        value or ""
                                    ).strip(),
                                )
                            )

                    tables.append(
                        TableElement(
                            table_id=(
                                f"TABLE-"
                                f"{table_counter:05d}"
                            ),
                            page_number=(
                                page_number
                            ),
                            order=(
                                table_counter
                            ),
                            rows=row_count,
                            columns=col_count,
                            cells=cells,
                            extraction_method=(
                                "pdfplumber"
                            ),
                        )
                    )

        return tables

    def _extract_images(
        self,
        path: Path,
    ):

        doc = fitz.open(str(path))

        images = []

        counter = 0

        for page_index, page in enumerate(doc):

            for image_info in (
                page.get_images(
                    full=True
                )
            ):

                counter += 1

                xref = image_info[0]

                image_data = (
                    doc.extract_image(
                        xref
                    )
                )

                image_bytes = (
                    image_data["image"]
                )

                try:

                    image = Image.open(
                        io.BytesIO(
                            image_bytes
                        )
                    )

                    ocr_text = (
                        pytesseract
                        .image_to_string(
                            image
                        )
                    )

                    width, height = (
                        image.size
                    )

                except Exception:

                    ocr_text = ""

                    width = None
                    height = None

                images.append(
                    ImageElement(
                        image_id=(
                            f"IMG-{counter:05d}"
                        ),
                        page_number=(
                            page_index + 1
                        ),
                        order=counter,
                        width=width,
                        height=height,
                        ocr_text=ocr_text,
                    )
                )

        doc.close()

        return images
