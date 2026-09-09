from pathlib import Path
import hashlib
import uuid

from docx import Document

from .models import (
    CanonicalDocument,
    DocumentElement,
    DocumentMetadata,
    TextSpan,
)


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


class DOCXExtractor:

    def extract(
        self,
        path: Path,
        document_type="TEMPLATE",
    ):

        doc = Document(str(path))

        document_id = (
            f"DOC-{uuid.uuid4().hex[:12].upper()}"
        )

        elements = []

        counter = 0

        for paragraph in doc.paragraphs:

            text = paragraph.text.strip()

            if not text:
                continue

            counter += 1

            spans = []

            for run in paragraph.runs:

                spans.append(
                    TextSpan(
                        text=run.text,
                        font=run.font.name,
                        size=(
                            run.font.size.pt
                            if run.font.size
                            else None
                        ),
                        bold=bool(run.bold),
                        italic=bool(
                            run.italic
                        ),
                    )
                )

            elements.append(
                DocumentElement(
                    element_id=(
                        f"EL-{counter:06d}"
                    ),
                    type="paragraph",
                    order=counter,
                    text=text,
                    spans=spans,
                    style_ref=(
                        paragraph.style.name
                        if paragraph.style
                        else None
                    ),
                    extraction_method=(
                        "python-docx"
                    ),
                )
            )

        return CanonicalDocument(
            metadata=DocumentMetadata(
                document_id=document_id,
                file_name=path.name,
                file_type="docx",
                document_type=(
                    document_type
                ),
                file_hash=sha256(path),
            ),
            elements=elements,
        )
