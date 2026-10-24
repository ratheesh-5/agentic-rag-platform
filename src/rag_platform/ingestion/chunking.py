"""Step 2 of indexing: split each document's text into chunks.

Why chunk at all? An embedding is ONE vector per text. A whole document squeezed into one
vector blurs many topics together, so search becomes imprecise; small chunks each get a sharp
vector, and only the relevant chunks are sent to the LLM (cheaper, less noise).

We use LangChain's RecursiveCharacterTextSplitter: it tries to cut at paragraph breaks first,
then line breaks, then sentence ends, then spaces, and only cuts mid-word as a last resort.
That keeps paragraphs and table rows intact whenever they fit in `chunk_size`.

Every chunk keeps its [start, end) character offsets into the document text. Offsets let
citations point to an exact span, and let evaluation check "did we retrieve the evidence?"
independently of how the text was chunked.
"""

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_platform.config import RagConfig


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int  # char offsets into the document's normalized text
    end: int
    text_hash: str  # sha256 of chunk text; key for the embedding cache


def text_hash(text: str) -> str:
    """SHA-256 of the text. Same text -> same hash, so we can detect changes and reuse
    embeddings without comparing whole strings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, cfg: RagConfig) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # tried in this order, coarsest first
        add_start_index=True,  # record where each chunk starts in the original text
    )
    chunks = []
    # enumerate() yields (index, item) pairs, like LINQ .Select((item, i) => ...).
    for i, d in enumerate(splitter.create_documents([text])):
        start = d.metadata["start_index"]
        chunks.append(
            Chunk(
                index=i,
                text=d.page_content,
                start=start,
                end=start + len(d.page_content),
                text_hash=text_hash(d.page_content),
            )
        )
    return chunks
