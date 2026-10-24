"""Shared test helpers. pytest loads conftest.py automatically for every test in this folder.

A `@pytest.fixture` function provides a value to any test that names it as a parameter
(`def test_x(fake_embedder): ...`) - dependency injection for tests, similar to xUnit
fixtures / constructor injection. Each test gets a fresh instance.

Unit tests never call real AI APIs (slow, costly, non-deterministic). Instead they use fakes
that implement the same interfaces as the real models (see llm.py).
"""

import hashlib
import re

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbedder(Embeddings):
    """Deterministic bag-of-words hashing embedder: shared words -> higher cosine. No API calls."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls: list[str] = []  # every text sent to embed_documents

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLlm:
    """Stands in for a chat model: with_structured_output(schema).invoke() returns a scripted reply.

    `reply` can be:
    - one object                      -> returned for every call;
    - {SchemaClass: reply}            -> reply chosen by the requested output schema
                                         (the agentic graph asks for several schemas);
    - a list (alone or as dict value) -> successive calls pop the next reply, so a test can
                                         script e.g. "regenerate" first, then "accept";
    - an exception (anywhere above)   -> raised, like a provider error.
    Every call's messages and schema are recorded in `prompts` / `schemas`.
    """

    def __init__(self, reply=None, tool_replies=None):
        self.reply = reply
        self.tool_replies = list(tool_replies or [])  # AIMessages returned by bind_tools().invoke
        self.prompts: list = []
        self.schemas: list = []

    def with_structured_output(self, schema):
        return _BoundFakeLlm(self, schema)

    def bind_tools(self, tools):
        """Tool calling: each invoke() returns the next scripted AIMessage (with tool_calls),
        then a plain "done" message once the script is used up."""
        return _ToolFakeLlm(self)


class _ToolFakeLlm:
    def __init__(self, llm: FakeLlm):
        self.llm = llm

    def invoke(self, messages, config=None):  # config: run metadata (ignored)
        from langchain_core.messages import AIMessage

        self.llm.prompts.append(messages)
        return self.llm.tool_replies.pop(0) if self.llm.tool_replies else AIMessage("done")


class _BoundFakeLlm:
    def __init__(self, llm: FakeLlm, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, messages, config=None):  # config: run metadata (ignored)
        self.llm.prompts.append(messages)
        self.llm.schemas.append(self.schema)
        reply = self.llm.reply
        if isinstance(reply, dict):
            reply = reply[self.schema]
        reply = reply.pop(0) if isinstance(reply, list) else reply
        if isinstance(reply, Exception):  # scripted provider failure
            raise reply
        return reply

    async def ainvoke(self, messages):  # async variant (RAGAS calls models asynchronously)
        return self.invoke(messages)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
