"""Request and response models of the REST API (pydantic).

FastAPI validates every request body against these classes (wrong type, missing field, value out
of range -> automatic 422 with details) and serializes responses through them, so the OpenAPI
documentation at /docs is generated from the same definitions. .NET analogy: DTOs with data
annotations, validated by model binding and documented by Swashbuckle.
"""

from typing import Literal

from pydantic import BaseModel, Field

from rag_platform.config import DEFAULT_COLLECTION_ID


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # Optional overrides of rag_config.json for this request; None = use the active config.
    mode: Literal["fast", "agentic", "auto"] | None = None
    retrieval: Literal["dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    collection_id: int = DEFAULT_COLLECTION_ID  # which collection to search


class Source(BaseModel):
    number: int  # the [n] used in the answer text
    document: str
    start: int  # character range of the cited passage in the document's text
    end: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answerable: bool
    answer: str
    sources: list[Source]
    abstain_reason: str
    mode: str
    steps: list[str]  # agentic mode: what the workflow did


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: int | None = None  # None = start a new conversation
    # The collection a NEW conversation searches; a conversation keeps its collection, so this
    # is ignored for follow-ups.
    collection_id: int = DEFAULT_COLLECTION_ID
    # Developer overrides of rag_config.json (the UI shows them only in development builds).
    mode: Literal["fast", "agentic", "auto"] | None = None
    retrieval: Literal["dense", "hybrid"] | None = None
    # Analyse mode (generation/analyse.py): read every document of the collection instead of the
    # top passages - for comparisons and overviews. Signed-in users and admins only.
    analyse: bool = False


class CollectionRef(BaseModel):
    id: int
    name: str


class ChatMessage(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    # Assistant messages only:
    answerable: bool | None = None
    sources: list[Source] = []
    standalone_question: str | None = None  # how the follow-up was understood (memory)
    abstain_reason: str = ""
    steps: list[str] = []
    # auto mode: "fast" or "auto → agentic (reason)"; analyse mode: "analyse: N documents
    # read, ..." (shown under the answer); empty in fast/agentic mode
    route: str = ""
    # "I couldn't find this" here, but these other collections have matching passages.
    found_in: list[CollectionRef] = []
    feedback: int | None = None  # 1 helpful, -1 not helpful
    seconds: float | None = None  # assistant: how long the answer took (older messages: none)
    # Online-eval scores (only for sampled answers; filled in after the response is sent).
    faithfulness: float | None = None
    answer_relevancy: float | None = None


class ModeChange(BaseModel):
    """Admin › Settings: switch the answer mode (fast / agentic / auto)."""

    mode: Literal["fast", "agentic", "auto"]


class ChatProvider(BaseModel):
    """Admin › Settings "Answer model": a chat provider and whether it is set up (key in .env).
    The key itself is never sent."""

    name: Literal["gemini", "groq"]
    model: str
    configured: bool


class ModelChange(BaseModel):
    """Admin › Settings: which provider writes answers first; fallback to the other one."""

    chat_provider: Literal["gemini", "groq"]
    chat_fallback: bool


ModelKind = Literal["gemini", "groq", "judge"]


class ModelCheckItem(BaseModel):
    """One question of a model quality check."""

    id: str  # golden-set id, e.g. "q06"
    question: str
    ok: bool
    note: str  # "cited the right document", "refused correctly", ...


class ModelCheck(BaseModel):
    """Quality check of one answer model (plan G7): a few demo questions answered with exactly
    that model; it must pass before the model can be used for answers."""

    kind: Literal["gemini", "groq"]
    model: str
    state: Literal["running", "done", "failed"]
    passed: int = 0
    total: int = 0
    ok: bool = False  # passed >= the minimum
    error: str | None = None
    started_at: str
    finished_at: str | None = None
    items: list[ModelCheckItem] = []


class ModelSlot(BaseModel):
    """One model choice in Admin › Settings › Models."""

    kind: ModelKind
    configured: bool  # the provider has its API key in .env
    current: str  # the model in use
    default: str  # the .env model (used when nothing is chosen)
    backup: str = ""  # answer models: same-provider backup ("" = none; never for the judge)
    options: list[str]  # models the provider offers today ([] = list not available)


class ModelsInfo(BaseModel):
    slots: list[ModelSlot]  # gemini, groq, judge
    embedding: str  # shown, never changed here (needs re-indexing every document)
    checks: list[ModelCheck]  # the latest check per model
    check_minimum: int  # passes needed out of the check questions


class ModelsChange(BaseModel):
    """Chosen models; "" = the .env default."""

    gemini_model: str = ""
    groq_model: str = ""
    judge_model: str = ""
    gemini_backup_model: str = ""  # "" = no backup
    groq_backup_model: str = ""


class ModelRequest(BaseModel):
    kind: ModelKind
    model: str = Field(min_length=1, max_length=100)


class ModelTestResult(BaseModel):
    ok: bool
    message: str  # the model's reply, or why it failed
    seconds: float


class ChatResponse(BaseModel):
    conversation_id: int
    collection_id: int  # the collection this conversation searches
    title: str
    user_message: ChatMessage
    assistant_message: ChatMessage


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    messages: int
    collection_id: int
    collection: str  # the collection's name


class Conversation(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    collection_id: int
    collection: str
    messages: list[ChatMessage]


class Feedback(BaseModel):
    rating: Literal[1, -1] | None  # None clears the feedback


class Topic(BaseModel):
    title: str
    description: str = ""
    examples: list[str] = []  # questions the documents can answer (shown as clickable examples)
    # File names the examples are answered from. GET /assistant shows the topic only while at
    # least one of them is indexed (a deleted document's examples would be unanswerable).
    # Empty = always shown.
    documents: list[str] = []
    # True for topics GET /assistant adds for uploaded documents no hand-written topic covers:
    # the welcome screen lists those compactly (100 uploads must not mean 100 cards).
    generated: bool = False


class AssistantProfile(BaseModel):
    """What the chatbot covers, from assistant.json: shown to new users on the welcome screen and
    when a question is outside the documents. Keeps the platform domain-independent: another
    knowledge base only needs another assistant.json, no UI change."""

    name: str = "Knowledge Assistant"
    description: str = "I answer questions from the indexed documents and cite my sources."
    topics: list[Topic] = []
    # From the server settings (not assistant.json): chats without activity this long are deleted
    # (plan B10); 0 = kept. The chat shows it as a privacy note.
    chat_retention_days: int = 0


class UserInfo(BaseModel):
    """A user signed in with Google (plan B1), as shown in the UI's account menu."""

    email: str
    name: str
    picture: str  # Google profile photo URL ("" if none)


class Me(BaseModel):
    role: Literal["admin", "user", "guest"]  # user = signed in with Google, not an admin
    user: UserInfo | None = None


class AuthConfig(BaseModel):
    """Public sign-in settings for the UI. An empty client id = Google sign-in is off."""

    google_client_id: str


class GoogleSignIn(BaseModel):
    # The ID token (a signed JWT) Google's sign-in button gave the browser.
    credential: str = Field(min_length=20, max_length=8000)


class SignInResponse(BaseModel):
    token: str  # our session token: send as "Authorization: Bearer <token>"
    expires_at: str
    role: Literal["admin", "user"]
    user: UserInfo


class Collection(BaseModel):
    """A named group of documents; a chat searches exactly one collection (plan section I)."""

    id: int
    name: str
    description: str
    documents: int
    passages: int
    is_default: bool  # the demo collection: evaluation works on it; it can't be deleted
    is_private: bool  # only its owner sees it (phase 2)
    can_manage: bool  # the caller may upload into / rename / delete it
    # Similarity threshold used for its questions and where it comes from: "measured" (demo
    # collection, rag_config.json), "manual" (admin override), "automatic" (probe questions,
    # retrieval/calibration.py) or "default" (global value, nothing measured yet).
    threshold: float
    threshold_source: Literal["measured", "manual", "automatic", "default"]
    threshold_override: float | None = None  # the admin's manual value, if any


class CollectionIn(BaseModel):
    """Create or rename a collection (admin)."""

    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=300)
    # Manual similarity threshold (admins, rarely needed); None = automatic. Not for the demo
    # collection, whose value changes only through measured optimizer recommendations.
    threshold: float | None = Field(default=None, ge=0.3, le=0.95)
    # Admins: create a private collection of their own instead of a shared one. Other signed-in
    # users always create private collections.
    private: bool = False


class PrivateSummary(BaseModel):
    """All an admin may know about users' private collections: totals, never content."""

    collections: int
    users: int
    documents: int
    passages: int


class DocumentInfo(BaseModel):
    id: int
    path: str
    doc_type: str
    collection_id: int
    chunks: int
    chunker: str
    ingested_at: str
    # Lines that look like instructions to an AI (prompt-injection scan, plan B6); [] = clean.
    # A warning only: the document is indexed, the admin checks it.
    warnings: list[str] = []


class UploadResponse(BaseModel):
    document: DocumentInfo
    status: Literal["added", "updated", "unchanged"]
    embeddings_new: int
    embeddings_cached: int


class EvalRunSummary(BaseModel):
    id: int
    created_at: str
    label: str
    metrics: dict
    kind: Literal["full", "check", "experiment"] = "full"  # full eval / quality check / optimizer
    hidden: bool = False  # left out of the admin list by default (old test runs)
    current: bool = False  # measured with the active rag_config.json
    gates: Literal["pass", "fail", "warn", "n/a"] = "n/a"  # release gates, overall
    test_set: str = ""  # file name of the golden set used


class HiddenFlag(BaseModel):
    hidden: bool


class Recommendation(BaseModel):
    """The optimizer's latest recommendation and the human decision on it."""

    id: int
    created_at: str
    status: Literal["pending", "applied", "rejected", "superseded"]
    decided_at: str | None
    report: dict  # changes, recommended_config, baseline/winner metrics, summary, rationale, risks
    stale: bool = False  # the active config changed since: its numbers no longer apply


class OptimizerRun(BaseModel):
    """The optimizer started from the UI: it runs in the background, the page polls this."""

    state: Literal["idle", "running", "done", "failed"] = "idle"
    test_set: str = ""
    steps: list[str] = []  # progress lines, as the CLI prints them
    # done: "recommendation" (waits in Settings) | "no-change" | "nothing-to-improve"
    outcome: str = ""
    recommendation_id: int | None = None
    error: str = ""
    started_at: str | None = None
    finished_at: str | None = None


class QualityTest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    # The collection to search (like a new chat); the demo collection if not given.
    collection_id: int = DEFAULT_COLLECTION_ID


class TestPassage(BaseModel):
    document: str
    score: float  # cosine similarity to the question
    cited: bool  # used as a citation in the answer
    text: str


class QualityTestResult(BaseModel):
    """One admin test question: the answer AND its quality, scored on the spot."""

    question: str
    answerable: bool
    answer: str
    abstain_reason: str
    sources: list[Source]  # cited passages, numbered as in the answer
    passages: list[TestPassage]  # everything the generator read, with search scores
    faithfulness: float | None
    answer_relevancy: float | None
    judge_calls: int
    judge_model: str = ""  # who scored it (Admin › Settings › Models)
    score_error: str | None
    seconds: float
