"""Answer-quality evaluation with RAGAS (an open-source RAG evaluation library).

Retrieval metrics in metrics.py tell us whether the right TEXT was found (deterministic, using
the golden evidence spans). RAGAS uses an LLM as judge to grade the ANSWER and the CONTEXT:

| Metric             | Question it answers                                    | Needs reference |
|--------------------|--------------------------------------------------------|-----------------|
| faithfulness       | Is every claim in the answer supported by the context? | no              |
|                    | (low = hallucination)                                  |                 |
| answer_relevancy   | Does the answer address the question asked?            | no              |
| context_precision  | Are the useful chunks ranked above the useless ones?   | yes             |
| context_recall     | Does the context contain everything the reference says? | yes            |
| answer_correctness | Does the answer agree with the reference answer?       | yes             |

All scores are 0..1. How they work, briefly: faithfulness splits the answer into atomic claims
and asks the LLM to verify each against the context; answer_relevancy generates questions the
answer would answer and compares them with the real question by embedding similarity;
answer_correctness combines claim-level overlap with the reference and embedding similarity.
This means several LLM calls per question - RAGAS is thorough but not cheap.

Adapters: RAGAS metrics expect its own LLM/embedding interfaces. Instead of letting RAGAS
create a second Gemini client (which would bypass our rate limiter), two small adapters wrap
the LangChain models from llm.py. .NET analogy: the Adapter pattern - implement RAGAS's
interface by delegating to an object we already have.
"""

import math

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag_platform.evaluation.metrics import RAGAS_METRICS


class LangChainRagasLLM(InstructorBaseRagasLLM):
    """RAGAS LLM interface -> LangChain chat model. RAGAS passes a prompt and the Pydantic
    class it wants back; LangChain's with_structured_output does exactly that.

    `class X(Base)` = inheritance; RAGAS declares generate/agenerate as abstract methods, so we
    must implement both (like implementing an abstract class in C#). `async def` + `await` work
    like C# async/await: RAGAS runs its metrics asynchronously and calls agenerate().
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.calls = 0  # LLM calls made, to show evaluation cost

    def generate(self, prompt, response_model):
        self.calls += 1
        return self.llm.with_structured_output(response_model).invoke(prompt)

    async def agenerate(self, prompt, response_model):
        self.calls += 1
        return await self.llm.with_structured_output(response_model).ainvoke(prompt)


class LangChainRagasEmbeddings(BaseRagasEmbedding):
    """RAGAS embedding interface -> LangChain embeddings. RAGAS compares short questions and
    answers here, so the query-side embedding is the right one."""

    def __init__(self, embeddings: Embeddings):
        super().__init__()  # run the base class constructor (like `: base()` in C#)
        self.embeddings = embeddings

    def embed_text(self, text, **kwargs):
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text, **kwargs):
        return await self.embeddings.aembed_query(text)


class RagasEvaluator:
    """Scores one answered question with the five RAGAS metrics."""

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        self.llm = LangChainRagasLLM(llm)
        emb = LangChainRagasEmbeddings(embeddings)
        self.faithfulness = Faithfulness(llm=self.llm)
        # strictness=1: generate one reverse question instead of three (3x fewer LLM calls).
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=emb, strictness=1)
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)
        self.answer_correctness = AnswerCorrectness(llm=self.llm, embeddings=emb)

    def score(self, question: str, answer: str, contexts: list[str], reference: str) -> dict:
        """Return {metric: 0..1 or None}. Reference-based metrics are skipped (None) when the
        question has no reference answer (e.g. the system answered an unanswerable question)."""
        # metric.score(...) is RAGAS's synchronous wrapper around its async ascore(); it blocks
        # until the metric's LLM calls finish (each call waits for the shared rate limiter).
        scores = {
            "faithfulness": self.faithfulness.score(
                user_input=question, response=answer, retrieved_contexts=contexts
            ),
            "answer_relevancy": self.answer_relevancy.score(user_input=question, response=answer),
        }
        if reference:
            scores["context_precision"] = self.context_precision.score(
                user_input=question, reference=reference, retrieved_contexts=contexts
            )
            scores["context_recall"] = self.context_recall.score(
                user_input=question, retrieved_contexts=contexts, reference=reference
            )
            scores["answer_correctness"] = self.answer_correctness.score(
                user_input=question, response=answer, reference=reference
            )
        return {name: _value(scores.get(name)) for name in RAGAS_METRICS}


def _value(result) -> float | None:
    """MetricResult -> float; RAGAS returns NaN when it cannot score (e.g. no claims found)."""
    if result is None or result.value is None or math.isnan(result.value):
        return None
    return round(float(result.value), 4)
