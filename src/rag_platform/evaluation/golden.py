"""Golden set: the fixed list of test questions every pipeline version is measured against.

File format (data/eval/golden.jsonl, one JSON object per line - "JSON Lines"):

  {"id": "q01", "question": "...", "answerable": true, "reference_answer": "...",
   "evidence": [{"doc": "data/sample_corpus/x.pdf", "quote": "exact text that answers it"}]}

Why evidence spans and not chunk ids? Chunk ids change every time chunking changes, which would
make runs with different chunk sizes incomparable. Instead each question points at the TEXT
that answers it; at load time the quote is located in the document's normalized text and turned
into a Span (doc + char offsets). A retrieved chunk is "relevant" if its span overlaps one.

Quotes are matched whitespace-insensitively, because PDF extraction wraps lines and pads table
columns with spaces. A quote that is missing or appears twice is an error: the label would be
wrong or ambiguous, so we fail loudly instead of silently scoring garbage.

Unanswerable questions (answerable=false, no evidence) measure whether the system correctly
says "I don't know" instead of inventing an answer.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Span:
    doc: str  # document path as stored in SQLite (e.g. "data/sample_corpus/x.pdf")
    start: int
    end: int


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    answerable: bool
    reference_answer: str
    evidence: list[Span]


def find_span(doc: str, text: str, quote: str) -> Span:
    """Locate `quote` in `text`, treating any run of whitespace as equivalent."""
    # Build a regex from the quote's words: re.escape makes characters like "(" or "." literal,
    # and r"\s+" between words matches spaces, newlines or padding of any length.
    # e.g. "M = 32" -> r"M\s+=\s+32", which also matches "M =\n 32" from a wrapped PDF line.
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        problem = "not found" if not matches else f"found {len(matches)} times"
        raise ValueError(f"evidence quote {problem} in {doc}: {quote[:60]!r}")
    return Span(doc, matches[0].start(), matches[0].end())


def load_golden(path: Path, doc_texts: dict[str, str]) -> list[GoldenItem]:
    """Parse the golden set and resolve every evidence quote against the indexed documents."""
    items: list[GoldenItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        spans = []
        for ev in raw.get("evidence", []):
            if ev["doc"] not in doc_texts:
                raise ValueError(
                    f"{path}:{line_no} ({raw['id']}): document not indexed: {ev['doc']}"
                )
            try:
                spans.append(find_span(ev["doc"], doc_texts[ev["doc"]], ev["quote"]))
            except ValueError as e:
                # Re-raise with file/line/question id added. `from e` chains the original
                # error, like `throw new X("...", innerException)` in C#.
                raise ValueError(f"{path}:{line_no} ({raw['id']}): {e}") from e
        if raw["answerable"] and not spans:
            raise ValueError(f"{path}:{line_no} ({raw['id']}): answerable question needs evidence")
        items.append(
            GoldenItem(
                id=raw["id"],
                question=raw["question"],
                answerable=raw["answerable"],
                reference_answer=raw.get("reference_answer", ""),
                evidence=spans,
            )
        )
    return items
