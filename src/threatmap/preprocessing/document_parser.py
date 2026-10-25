"""
Document parser for plain text, uploaded TXT files, and PDF threat reports.
"""

import hashlib
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.document_parser")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB safety limit


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf if available."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except ImportError:
        logger.warning("pypdf is not installed. PDF parsing unavailable.")
        raise ImportError(
            "PDF extraction requires 'pypdf'. Install it with: pip install pypdf"
        )
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise ValueError(f"Could not parse PDF content: {e}")


def parse_document(
    content: Union[str, bytes, Path],
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parses document input into (document_id, raw_text).
    Supports raw text strings, bytes, and file paths.
    """
    if isinstance(content, Path):
        filename = filename or content.name
        if not content.exists():
            raise FileNotFoundError(f"Document file not found: {content}")
        file_bytes = content.read_bytes()
    elif isinstance(content, bytes):
        file_bytes = content
    elif isinstance(content, str):
        # Treat as plain text
        raw_text = content.strip()
        if not raw_text:
            raise ValueError("Input document text cannot be empty.")
        doc_id = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]
        return doc_id, raw_text
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size exceeds maximum allowed threshold of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # Determine file type
    is_pdf = filename and filename.lower().endswith(".pdf")

    if is_pdf:
        raw_text = extract_text_from_pdf(file_bytes).strip()
    else:
        try:
            raw_text = file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            raw_text = file_bytes.decode("latin-1", errors="replace").strip()

    if not raw_text:
        raise ValueError("Document appears to be empty after extraction.")

    doc_id = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]
    return doc_id, raw_text
