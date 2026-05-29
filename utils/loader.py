"""
utils/loader.py
Handles text extraction from PDF, DOCX, and TXT files.
"""

import PyPDF2
import docx
import io


def extract_text_from_pdf(file) -> str:
    """Extract all text from a PDF file object."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_docx(file) -> str:
    """Extract all paragraph text from a DOCX file object."""
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    return text


def extract_text_from_txt(file) -> str:
    """Decode and return plain text from a TXT file object."""
    return file.read().decode("utf-8")


def extract_text(file) -> str | None:
    """
    Auto-detect file type by extension and extract text.

    Args:
        file: A file-like object with a `.name` attribute (e.g. Streamlit UploadedFile).

    Returns:
        Extracted text as a string, or None if the file type is unsupported.
    """
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        return None


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    Args:
        text:       The full document text.
        chunk_size: Number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks
                    to preserve context at boundaries.

    Returns:
        A list of text chunk strings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
