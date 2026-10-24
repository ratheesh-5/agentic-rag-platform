"""Analyse mode (plan C14, phase 3): questions across ALL documents of a collection.

Normal answering takes the top 5 passages - fine for "what is X?", but a question like "compare
the 2025 and 2026 remote-work policies" or "what is the overall picture of this case?" needs every
document, and 5 passages often come from only one or two of them.

Map-reduce, the standard pattern for multi-document questions (LangChain's map-reduce chain,
LlamaIndex's tree summarize):
1. select (code, no AI): score every passage of the collection against the question, keep each
   document's best PASSAGES_PER_DOCUMENT passages, and the MAX_DOCUMENTS best documents;
2. map (1 AI call per document): "what does THIS document say about the question?" -> a short
   note citing its passages, or "nothing relevant";
3. reduce (1 AI call): combine the notes into one cited answer (comparisons, differences, the
   overall picture).
The code decides which documents and passages are read; the model only reads and writes - the
project's rule "deterministic work in plain Python, the LLM only where it adds value".

Nothing is dropped silently: the answer reports how many documents were read, how many had
nothing relevant, how many weren't read (over the limit) and how many failed (a provider error
on one document doesn't lose the others). Same safety as fast mode: below the collection's
similarity threshold -> "I couldn't find this" without any AI call; citations are checked in code.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from rag_platform.generation.answer import (
    BELOW_THRESHOLD,
    CITATION,
    UNTRUSTED_PASSAGES,
    Answer,
    LlmAnswer,
    valid_labels,
)
from rag_platform.retrieval.retriever import RetrievedChunk, embed_query, hydrate
from rag_platform.store.db import Store
from rag_platform.store.vector_index import VectorIndex

MAX_DOCUMENTS = 10  # cost: one AI call per document (+1); free tier ~14 calls per minute
PASSAGES_PER_DOCUMENT = 4  # a document's best passages - its whole text would cost too much
NOTHING_RELEVANT = "no document says anything about this"

MAP_PROMPT = (
    """You read passages from ONE document and note what they say about the question.
- Write a short note (1-4 sentences) with only the facts these passages state about the
  question, citing each fact with the passage label, e.g. [P2].
- If the passages say nothing about the question, set relevant to false and leave note empty.
- Never use outside knowledge or guess."""
    + UNTRUSTED_PASSAGES
)

REDUCE_PROMPT = (
    """You combine notes taken from several documents into ONE answer to the question.
- Use only facts from the notes and keep their passage labels as citations, e.g. [P3].
- Compare, contrast or summarise across documents when the question asks for it; name the
  document a fact comes from when that helps (e.g. "the 2026 policy ...").
- If the notes only partly answer the question, answer that part and say what is missing.
- Never use outside knowledge or guess."""
    + UNTRUSTED_PASSAGES
)


class DocumentNote(BaseModel):
    """Structured output of the map step (one document)."""

    relevant: bool = Field(description="True if these passages say something about the question")
    note: str = Field(description="The relevant facts with [Pn] citations; empty if not relevant")


@dataclass
class Reading:
    """What analyse mode read, for the answer's "N documents read" line."""

    read: int = 0  # documents whose passages were given to the model
    relevant: int = 0  # ... and that had something on the question
    skipped: int = 0  # over MAX_DOCUMENTS: not read
    failed: int = 0  # the model call failed for that document


def select_passages(
    store: Store, index: VectorIndex, embedder: Embeddings, question: str
) -> tuple[list[list[RetrievedChunk]], int]:
    """Each document's best passages, best documents first; plus how many documents didn't
    make the cut. No AI call: one question embedding and a vector search over the collection."""
    hits = index.search(embed_query(embedder, question), max(len(index.ids), 1))
    by_doc: dict[str, list[RetrievedChunk]] = {}
    for chunk in hydrate(store, hits):  # hits are best-first, so each doc keeps its best ones
        passages = by_doc.setdefault(chunk.doc_path, [])
        if len(passages) < PASSAGES_PER_DOCUMENT:
            passages.append(chunk)
    documents = sorted(by_doc.values(), key=lambda ps: ps[0].score, reverse=True)
    # Within a document, keep the passages in reading order: easier for the model to follow.
    chosen = [sorted(ps, key=lambda c: c.start) for ps in documents[:MAX_DOCUMENTS]]
    return chosen, max(len(documents) - MAX_DOCUMENTS, 0)


def analyse_question(
    question: str,
    store: Store,
    index: VectorIndex,
    embedder: Embeddings,
    llm: BaseChatModel,
    similarity_threshold: float,
) -> Answer:
    """Answer a question from every document of the collection (map-reduce, see above)."""
    documents, skipped = select_passages(store, index, embedder, question)
    best = max((p.score for ps in documents for p in ps), default=0.0)
    if best < similarity_threshold:  # nothing in the collection is about this: no AI call
        return Answer(question, False, "", abstain_reason=BELOW_THRESHOLD, route="analyse")

    context: list[RetrievedChunk] = []  # every passage shown to the model; global [P] labels
    notes: list[str] = []
    reading = Reading(skipped=skipped)
    for passages in documents:
        first = len(context)  # this document's passages become [P first+1] ... globally
        context.extend(passages)
        name = Path(passages[0].doc_path).name
        labels = [f"P{first + i}" for i in range(1, len(passages) + 1)]
        shown = "\n\n".join(
            f"[{label}]\n{p.text}" for label, p in zip(labels, passages, strict=True)
        )
        try:
            out: DocumentNote = llm.with_structured_output(DocumentNote).invoke(
                [
                    ("system", MAP_PROMPT),
                    (
                        "human",
                        f"Document: {name}\n\nPassages:\n\n{shown}\n\nQuestion: {question}\n\n"
                        f"Cite only these labels: {', '.join(f'[{x}]' for x in labels)}.",
                    ),
                ],
                config={"run_name": "analyse_document", "metadata": {"purpose": "analyse"}},
            )
        except Exception:  # noqa: BLE001 - one document's failure must not lose the others
            reading.failed += 1
            continue
        reading.read += 1
        if out.relevant and out.note.strip():
            reading.relevant += 1
            notes.append(f"Document: {name}\nNote: {out.note.strip()}")
    if reading.failed and not reading.read:
        raise RuntimeError("analyse: every document call failed")  # -> 503 like any answer

    route = _route(reading)
    if not notes:
        return Answer(
            question, False, "", context=context, abstain_reason=NOTHING_RELEVANT, route=route
        )
    combined: LlmAnswer = llm.with_structured_output(LlmAnswer).invoke(
        [
            ("system", REDUCE_PROMPT),
            (
                "human",
                "Notes:\n\n" + "\n\n".join(notes) + f"\n\nQuestion: {question}\n\n"
                f"Cite only these passage labels: {valid_labels(len(context))}.",
            ),
        ],
        config={"run_name": "analyse_combine", "metadata": {"purpose": "analyse"}},
    )
    valid = range(1, len(context) + 1)  # citations checked in code, as in fast mode
    text = CITATION.sub(
        lambda m: f"[{m.group(1)}]" if int(m.group(1)) in valid else "", combined.answer
    )
    citations = sorted({int(n) for n in CITATION.findall(text)})
    if not combined.answerable or not citations:
        return Answer(
            question, False, "", context=context, abstain_reason=NOTHING_RELEVANT, route=route
        )
    return Answer(
        question, True, text.strip(), citations, context, complete=combined.complete, route=route
    )


def _route(r: Reading) -> str:
    """E.g. "analyse: 5 documents read, 2 without anything on this" (shown with the answer)."""
    parts = [f"{r.read} document{'s' if r.read != 1 else ''} read"]
    if r.read - r.relevant:
        parts.append(f"{r.read - r.relevant} without anything on this")
    if r.skipped:
        parts.append(f"{r.skipped} more not read (limit {MAX_DOCUMENTS})")
    if r.failed:
        parts.append(f"{r.failed} could not be read")
    return "analyse: " + ", ".join(parts)
