"""Step 1 of indexing: turn files (MD, TXT, DOCX, PDF) into plain text.

Retrieval can only be as good as the extracted text, so each format gets a dedicated loader:
- DOCX: walk paragraphs AND tables in document order, so a table stays next to the paragraph
  that introduces it (python-docx's `doc.paragraphs` alone would silently drop tables).
- PDF: pypdf's "layout" mode keeps table columns aligned instead of mashing cells together.

All text is then normalized. That normalized string is what we store and what chunk offsets
and evaluation evidence spans point into, so normalization must be deterministic.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

SUPPORTED = {".md", ".txt", ".docx", ".pdf"}


# @dataclass generates __init__, __eq__ and __repr__ from the annotated fields; frozen=True
# makes it immutable. .NET analogy: `public record LoadedDoc(Path Path, string DocType, ...)`.
@dataclass(frozen=True)
class LoadedDoc:
    path: Path
    doc_type: str
    text: str  # normalized text; chunk offsets and evidence spans refer to this string


def _normalize(text: str) -> str:
    """Unify line endings, strip trailing spaces, collapse 3+ newlines into one blank line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _load_docx(path: Path) -> str:
    parts: list[str] = []
    # iter_inner_content() yields Paragraph and Table objects in the order they appear.
    for block in docx.Document(str(path)).iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                parts.append(block.text)
        elif isinstance(block, Table):
            # One line per row, cells joined with " | " (a light Markdown-like table format
            # that both embedding models and LLMs read well).
            rows = [" | ".join(c.text.strip() for c in row.cells) for row in block.rows]
            parts.append("\n".join(rows))
    return "\n\n".join(parts)


def _load_pdf(path: Path) -> str:
    # A generator expression (like a lazy LINQ Select) over pages.
    pages = (
        page.extract_text(extraction_mode="layout") or "" for page in PdfReader(str(path)).pages
    )
    return "\n\n".join(p.strip() for p in pages if p.strip())


def load_document(path: Path) -> LoadedDoc:
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        text = path.read_text(encoding="utf-8")
    elif ext == ".docx":
        text = _load_docx(path)
    elif ext == ".pdf":
        text = _load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    return LoadedDoc(path=path, doc_type=ext.lstrip("."), text=_normalize(text))


def load_corpus(root: Path) -> list[LoadedDoc]:
    """Load a single file or every supported file under a directory (sorted, stable)."""
    if root.is_file():
        return [load_document(root)]
    # rglob("*") = recursive search, like Directory.EnumerateFiles(root, "*", AllDirectories).
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    return [load_document(p) for p in files]
