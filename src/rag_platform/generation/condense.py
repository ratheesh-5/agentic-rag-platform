"""Conversation memory: turn a follow-up question into a standalone question.

Retrieval searches with the question's words. A follow-up such as "and what about its overlap?"
contains none of the words that identify the topic ("chunk", "default"), so searching with it
finds nothing useful. Before retrieval, an LLM rewrites the follow-up using the recent turns:

    user:      What is the team's default chunk size?
    assistant: 800 characters [1].
    user:      and its overlap?         ->  "What is the team's default chunk overlap?"

This is "question condensing", the standard way to give a RAG chatbot memory. The rest of the
pipeline (retrieve -> generate -> validate citations) stays unchanged and answers the standalone
question, so memory works for both fast and agentic mode.

Cost: one extra LLM call, and only for follow-ups (the first question of a conversation is used
as-is). Only the last few turns are sent, which keeps the prompt small and stops old topics
leaking into new questions.
"""

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

CONDENSE_PROMPT = """You rewrite the user's latest message into a standalone question for a
document search engine, using the conversation only to resolve references (it, that, the same,
what about ..., and ...). Keep names, codes and numbers exactly. If the latest message is already
a complete question on its own, return it unchanged. Do not answer it."""

MEMORY_TURNS = 3  # user/assistant pairs of history given to the rewriter


class StandaloneQuestion(BaseModel):
    question: str = Field(description="The standalone question")


def condense_question(history: list[dict], question: str, llm: BaseChatModel) -> str:
    """`history` = previous messages, oldest first, as {"role": ..., "content": ...}."""
    if not history:
        return question  # first message: nothing to resolve, no LLM call
    turns = history[-2 * MEMORY_TURNS :]
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in turns)
    out: StandaloneQuestion = llm.with_structured_output(StandaloneQuestion).invoke(
        [
            ("system", CONDENSE_PROMPT),
            ("human", f"Conversation:\n{transcript}\n\nLatest message: {question}"),
        ],
        config={"run_name": "condense_question", "metadata": {"purpose": "rewrite"}},
    )
    rewritten = out.question.strip()
    return rewritten or question  # never lose the user's question to an empty rewrite
