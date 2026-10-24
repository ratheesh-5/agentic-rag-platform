"""Command-line interface: `uv run rag <command>`.

Typer builds the CLI from plain functions: each `@app.command()` function becomes a subcommand,
its parameters become arguments/options (`top_k` -> `--top-k`), and the docstring becomes the
help text. .NET analogy: System.CommandLine, but driven by type hints.

`rag = "rag_platform.cli:app"` in pyproject.toml is what makes the `rag` command exist.

Heavy imports (LangChain, FAISS, Google SDK) are done inside each command so that `rag --help`
and `rag inspect` start quickly. This file only wires things together; the logic lives in
ingestion/, store/, retrieval/ and generation/ so it can be unit-tested without the CLI.
"""

from pathlib import Path

import typer

from rag_platform.config import DEFAULT_COLLECTION_ID, RagConfig, Settings

app = typer.Typer(help="Agentic RAG platform", no_args_is_help=True)


# Set by the global --trace-langsmith option (read by _settings()).
_trace_langsmith = False


@app.callback()
def main(
    trace_langsmith: bool = typer.Option(
        False,
        "--trace-langsmith",
        help="Send LangSmith traces for this command (needs LANGSMITH_* in .env). Off by "
        "default for the CLI so batch runs don't use up the free monthly trace allowance.",
    ),
) -> None:
    """Agentic RAG platform."""
    global _trace_langsmith  # module-level flag, like a static field set at startup
    _trace_langsmith = trace_langsmith


def _config(**overrides) -> RagConfig:
    """Active config (rag_config.json) + command-line overrides; options left at None keep the
    file's value. Shows a one-line error instead of a pydantic traceback for a bad value."""
    from pydantic import ValidationError

    from rag_platform.config import load_rag_config

    try:
        return load_rag_config(**overrides)
    except ValidationError as e:
        err = e.errors()[0]
        where = err["loc"][0] if err["loc"] else "config"  # model-level errors have no field
        typer.echo(f"Invalid {where}: {err['msg']}", err=True)
        raise typer.Exit(1) from None


def _settings() -> Settings:
    s = Settings()
    if not s.gemini_api_key:
        typer.echo("GEMINI_API_KEY is not set (copy .env.example to .env).", err=True)
        raise typer.Exit(1)
    if _trace_langsmith:
        from rag_platform.observability import configure_tracing

        if configure_tracing(s):  # before any model is created (langsmith caches its env)
            typer.echo(f"LangSmith tracing on (project {s.langsmith_project}).")
        else:
            typer.echo("--trace-langsmith ignored: LANGSMITH_TRACING/API_KEY not set.", err=True)
    return s


def _collection_id(store, name: str, create: bool = False) -> int:
    """--collection NAME -> its id ("" = the demo collection); `create` makes a missing one."""
    if not name:
        return DEFAULT_COLLECTION_ID
    found = store.collection_by_name(name)
    if found:
        return found["id"]
    if create:
        typer.echo(f'New collection "{name}".')
        return store.create_collection(name)
    names = ", ".join(c["name"] for c in store.list_collections() if not c["owner"])
    typer.echo(f'No collection named "{name}" (existing: {names}).', err=True)
    raise typer.Exit(1)


@app.command()
def check() -> None:
    """Verify the API keys: one chat call per chat provider (Gemini, Groq if set) and one
    embedding call."""
    from rag_platform.llm import get_chat_model, get_embedder

    s = _settings()
    try:
        for name, model in get_chat_model(s).models.items():  # each provider directly
            reply = model.invoke("Reply with the single word OK.").text
            typer.echo(f"chat model ({name}): {model_name(model)} -> {reply.strip()!r}")
        vec = get_embedder(s).embed_query("hello")
    except Exception as e:  # surface any API/auth error as a clean failure
        typer.echo(f"Check failed: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"embedding model: {s.embedding_model} -> {len(vec)} dims")


def model_name(model) -> str:
    """The model id of a LangChain chat model (Gemini and Groq name the field differently)."""
    return getattr(model, "model_name", None) or getattr(model, "model", "")
    typer.echo("OK")


@app.command()
def inspect(
    path: Path = typer.Argument(Path("data/sample_corpus"), help="File or directory"),
    chunk_size: int | None = typer.Option(None, help="Max chars per chunk [config]"),
    overlap: int | None = typer.Option(None, help="Chunk overlap in chars [config]"),
    show: str = typer.Option("", help="Only show chunks of the file with this name"),
) -> None:
    """Load documents and show how they are chunked (no API calls)."""
    from rag_platform.ingestion.chunking import chunk_text
    from rag_platform.ingestion.loaders import load_corpus

    cfg = _config(chunk_size=chunk_size, chunk_overlap=overlap)
    for doc in load_corpus(path):
        chunks = chunk_text(doc.text, cfg)
        typer.echo(
            f"{doc.path.name}  [{doc.doc_type}]  {len(doc.text)} chars  {len(chunks)} chunks"
        )
        if show and doc.path.name == show:
            for c in chunks:
                typer.echo(f"--- chunk {c.index} [{c.start}:{c.end}] {c.text_hash[:8]}")
                typer.echo(c.text)


@app.command()
def ingest(
    path: Path = typer.Argument(Path("data/sample_corpus"), help="File or directory"),
    collection: str = typer.Option(
        "", help="Collection name (created if new); default: the demo collection"
    ),
) -> None:
    """Incrementally index documents into SQLite + FAISS (embeds only new/changed chunks)."""
    from rag_platform.ingestion.pipeline import ingest as run_ingest
    from rag_platform.llm import get_embedder
    from rag_platform.store.db import Store

    s = _settings()
    store = Store(s.db_path)
    try:
        cid = _collection_id(store, collection, create=True)
        index_path = s.collection_index_path(cid)
        st = run_ingest(path, store, index_path, get_embedder(s), s.embedding_model, _config(), cid)
        docs, chunks = store.counts(cid)
        store.set_auto_threshold(cid, None)  # documents changed: the API measures it again
    finally:
        store.close()
    typer.echo(
        f"docs:   {st.added} added, {st.updated} updated, {st.unchanged} unchanged,"
        f" {st.deleted} deleted"
    )
    typer.echo(f"chunks: {st.chunks_added} added, {st.chunks_removed} removed")
    typer.echo(f"embeddings: {st.embedded} new, {st.cache_hits} from cache")
    typer.echo(f"store: {docs} docs, {chunks} chunks  ({s.index_dir})")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the indexed documents"),
    top_k: int | None = typer.Option(None, help="Chunks to retrieve [config]"),
    mode: str | None = typer.Option(
        None, help="fast, agentic (LangGraph) or auto (fast, agentic when needed) [config]"
    ),
    retrieval: str | None = typer.Option(None, help="dense or hybrid (vector + BM25) [config]"),
    trace: bool = typer.Option(False, help="Print the agentic workflow steps"),
    show_context: bool = typer.Option(False, help="Print the passages the answer was built from"),
    collection: str = typer.Option("", help="Collection to search; default: the demo collection"),
) -> None:
    """Answer a question with numbered citations."""
    from rag_platform.answering import build_answerer
    from rag_platform.ingestion.pipeline import open_index
    from rag_platform.llm import get_chat_model, get_embedder
    from rag_platform.retrieval.bm25 import BM25Index
    from rag_platform.store.db import Store

    s = _settings()
    cfg = _config(top_k=top_k, mode=mode, retrieval=retrieval)
    store = Store(s.db_path)
    try:
        cid = _collection_id(store, collection)
        index = open_index(store, s.collection_index_path(cid), s.embedding_model, cid)
        bm25 = BM25Index.from_store(store, cid) if cfg.retrieval == "hybrid" else None
        answerer = build_answerer(store, index, get_embedder(s), get_chat_model(s), cfg, bm25)
        ans = answerer(question)
    finally:
        store.close()

    if trace and ans.steps:
        typer.echo("Steps:")
        for step in ans.steps:
            typer.echo(f"  - {step}")
        typer.echo("")
    if ans.answerable:
        typer.echo(ans.text)
        typer.echo("\nSources:")
        for n, c in zip(ans.citations, ans.cited_chunks, strict=True):
            typer.echo(f"  [{n}] {c.doc_path} [{c.start}:{c.end}]  score {c.score:.3f}")
    else:
        typer.echo(f"I don't know - the documents don't answer this ({ans.abstain_reason}).")
    if show_context:
        typer.echo("\nRetrieved:")
        for i, c in enumerate(ans.context, start=1):
            typer.echo(f"--- [{i}] {c.doc_path} [{c.start}:{c.end}] score {c.score:.3f}")
            typer.echo(c.text)


@app.command()
def delete(path: Path = typer.Argument(..., help="Indexed document to remove")) -> None:
    """Remove a document from the index (no API calls; the source file is not touched)."""
    from rag_platform.ingestion.pipeline import delete as run_delete
    from rag_platform.ingestion.pipeline import doc_key
    from rag_platform.store.db import Store

    s = Settings()
    store = Store(s.db_path)
    try:
        row = store.get_document(doc_key(path))
        cid = row["collection_id"] if row else DEFAULT_COLLECTION_ID
        removed = run_delete(path, store, s.collection_index_path(cid), s.embedding_model, cid)
        docs, chunks = store.counts()
    finally:
        store.close()
    if removed is None:
        typer.echo(f"Not indexed: {path}", err=True)
        raise typer.Exit(1)
    typer.echo(f"deleted {path} ({removed} chunks)")
    typer.echo(f"store: {docs} docs, {chunks} chunks  ({s.index_dir})")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to listen on (0.0.0.0 = all)"),
    port: int = typer.Option(8000, help="TCP port"),
    reload: bool = typer.Option(False, help="Restart on code changes (development)"),
) -> None:
    """Start the REST API with Uvicorn (docs at http://HOST:PORT/docs)."""
    import uvicorn  # the ASGI web server that hosts the FastAPI app (like Kestrel in .NET)

    if not Settings().api_key_list:
        typer.echo("Warning: API_KEYS is empty - every protected endpoint will return 503.")
    from rag_platform.observability import LOG_CONFIG

    # JSON log lines on stdout (plan D1); the request middleware replaces uvicorn's access log.
    uvicorn.run(
        "rag_platform.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_config=LOG_CONFIG,
        access_log=False,
    )


# --- rag optimize -------------------------------------------------------------------------------


def _stream(graph, graph_input, thread: dict) -> dict | None:
    """Run the graph, printing each step as it happens. Returns the interrupt payload if the
    graph paused for human approval, else None (finished)."""
    for update in graph.stream(graph_input, thread, stream_mode="updates"):
        for node, change in update.items():
            if node == "__interrupt__":
                return change[0].value
            for step in (change or {}).get("steps", []):
                typer.echo(f"  - {step}")
    return None


def _print_report(report: dict) -> None:
    typer.echo("\n=== Optimization report ===")
    typer.echo(f"{'experiment':<16} {'recall':>7} {'mrr':>7} {'safe_thr':>9}  changes")
    for e in report["experiments"]:
        m = e["metrics"]
        typer.echo(
            f"{e['label']:<16} {_fmt(m['recall_at_k']):>7} {_fmt(m['mrr']):>7}"
            f" {_fmt(m['safe_threshold']):>9}  {e['changes'] or '-'}"
        )
    typer.echo(f"\nWinner: {report['winner']}")
    typer.echo(f"Recommended changes: {report['changes'] or 'none'}")
    typer.echo(f"\nSummary:   {report['summary']}")
    typer.echo(f"Rationale: {report['rationale']}")
    typer.echo(f"Risks:     {report['risks']}\n")


@app.command()
def optimize(
    golden: Path = typer.Option(Path("data/eval/golden.jsonl"), help="Golden set (JSONL)"),
    experiments_per_round: int = typer.Option(3, help="Experiment budget per optimizer round"),
    rounds: int = typer.Option(2, help="Maximum optimizer rounds"),
    limit: int = typer.Option(0, help="Only the first N golden questions (0 = all)"),
    decision: str = typer.Option(
        "ask", help="ask | approve | reject | later (decide in the admin UI: Settings page)"
    ),
) -> None:
    """Multi-agent optimizer: evaluate, diagnose, run retrieval experiments, recommend a config,
    and write it to rag_config.json only after your approval. Experiments use no chat calls."""
    from langgraph.types import Command

    from rag_platform.agents.optimizer_graph.experiments import ExperimentRunner
    from rag_platform.agents.optimizer_graph.graph import build_optimizer_graph
    from rag_platform.config import RAG_CONFIG_FILE
    from rag_platform.llm import get_chat_model, get_embedder

    if decision not in ("ask", "approve", "reject", "later"):
        typer.echo("--decision must be ask, approve, reject or later.", err=True)
        raise typer.Exit(1)
    s = _settings()
    runner = ExperimentRunner(s, get_embedder(s), golden, _config(), limit, log=typer.echo)
    try:
        graph = build_optimizer_graph(
            runner, get_chat_model(s), RAG_CONFIG_FILE, experiments_per_round, rounds
        )
        thread = {"configurable": {"thread_id": "optimize"}}  # identifies this run's saved state
        typer.echo("Optimizing ...")
        paused = _stream(graph, {"steps": []}, thread)
        if paused is not None:  # waiting for human approval
            _print_report(paused["report"])
            # Saved so the decision can also be made (or reviewed) in the admin UI.
            rec_id = runner.main.save_recommendation(paused["report"])
            if decision == "later":
                typer.echo(f"\nSaved as recommendation {rec_id}: approve or reject it in the UI.")
                return
            approved = decision == "approve" or (
                decision == "ask" and typer.confirm("Apply this configuration?", default=False)
            )
            _stream(graph, Command(resume=approved), thread)  # continue from the interrupt
            runner.main.decide_recommendation(rec_id, "applied" if approved else "rejected")
        state = graph.get_state(thread).values
    finally:
        runner.close()

    outcome = state.get("decision")
    if outcome is None:
        typer.echo("\nSupervisor finished: retrieval already optimal, nothing to change.")
    elif outcome == "no-change":
        _print_report(state["report"])
        typer.echo("The current configuration is already the best: nothing to change.")
    elif outcome == "approved":
        extra = " Main index re-chunked (cached vectors reused)." if state.get("reindexed") else ""
        typer.echo(f"\nApplied: {RAG_CONFIG_FILE} updated.{extra}")
    else:
        typer.echo("\nRejected: configuration unchanged.")


# --- rag eval <run|list|compare> --------------------------------------------------------------
# A Typer sub-app gives nested commands, like `dotnet ef migrations add`.
eval_app = typer.Typer(help="Evaluate the pipeline on the golden set", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

# Metrics shown by `eval run` / `eval compare`, in display order.
REPORT_METRICS = [
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "mrr",
    "abstention_accuracy",
    "false_abstention_rate",
    "unanswerable_answered_rate",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "latency_avg_s",
    "latency_p95_s",
    "top_score_answerable",
    "top_score_unanswerable",
    "calibrated_threshold",
    "calibrated_abstention_accuracy",
    "safe_threshold",
    "unanswerable_below_safe",
]


# Short labels for the per-question progress line.
SHORT = {
    "faithfulness": "faith",
    "answer_relevancy": "relev",
    "context_precision": "c_prec",
    "context_recall": "c_rec",
    "answer_correctness": "correct",
}


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


@eval_app.command("run")
def eval_run(
    golden: Path = typer.Option(Path("data/eval/golden.jsonl"), help="Golden set (JSONL)"),
    top_k: int | None = typer.Option(None, help="Chunks to retrieve [config]"),
    threshold: float | None = typer.Option(None, help="Abstention threshold [config]"),
    mode: str | None = typer.Option(
        None, help="Pipeline to evaluate: fast, agentic or auto [config]"
    ),
    retrieval: str | None = typer.Option(None, help="dense or hybrid [config]"),
    retrieval_only: bool = typer.Option(
        False, help="No LLM at all: retrieval metrics + similarity-gate abstention (cheap)"
    ),
    label: str = typer.Option("", help="Free-text name for this run"),
    ragas: bool = typer.Option(True, help="Score answers with RAGAS (--no-ragas to skip)"),
    limit: int = typer.Option(0, help="Only the first N questions (0 = all)"),
    resume: int = typer.Option(0, help="Continue an incomplete run (same settings, missing Qs)"),
) -> None:
    """Run the golden set through the pipeline (-> RAGAS) and store the run in SQLite.

    A run that stops early (e.g. API quota exhausted) is still saved, marked incomplete;
    `--resume <run id>` evaluates only its missing questions and saves the combined run.
    """
    from dataclasses import asdict

    from rag_platform.answering import build_answerer
    from rag_platform.evaluation.golden import load_golden
    from rag_platform.evaluation.metrics import QuestionResult
    from rag_platform.evaluation.ragas_eval import RagasEvaluator
    from rag_platform.evaluation.runner import retrieval_only_answerer, run_eval, run_metrics
    from rag_platform.ingestion.chunking import text_hash
    from rag_platform.ingestion.pipeline import open_index
    from rag_platform.llm import get_chat_model, get_embedder, get_judge_model
    from rag_platform.retrieval.query_cache import CachedQueryEmbedder
    from rag_platform.retrieval.retriever import build_retriever
    from rag_platform.store.db import Store

    s = _settings()
    cfg = _config(top_k=top_k, similarity_threshold=threshold, mode=mode, retrieval=retrieval)
    store = Store(s.db_path)
    try:
        previous: list[QuestionResult] = []
        prior_ragas_calls = 0
        if resume:
            # Take every setting from the run being resumed, so the combined run is consistent.
            prior = store.get_eval_run(resume)
            if prior is None or prior["metrics"].get("complete", True):
                typer.echo(f"Run {resume} not found or already complete.", err=True)
                raise typer.Exit(1)
            cfg = RagConfig(**prior["config"]["rag"])
            golden, limit = Path(prior["config"]["golden"]), prior["config"]["n_questions"]
            ragas, label = prior["config"]["judge_model"] is not None, prior["label"]
            retrieval_only = prior["config"].get("retrieval_only", False)
            previous = [QuestionResult.from_dict(r) for r in prior["results"]]
            # Models come from the current environment: refuse to mix two models in one run.
            now = {
                "chat_model": None if retrieval_only else s.chat_model,
                "embedding_model": s.embedding_model,
                "judge_model": (s.judge_model or s.chat_model) if ragas else None,
            }
            mismatch = {k: v for k, v in now.items() if prior["config"].get(k) != v}
            if previous and mismatch:
                for k, v in mismatch.items():
                    typer.echo(
                        f"{k}: run {resume} used {prior['config'].get(k)}, now {v}", err=True
                    )
                typer.echo("Set the same model(s) (CHAT_MODEL / JUDGE_MODEL) to resume.", err=True)
                raise typer.Exit(1)
            prior_ragas_calls = prior["metrics"].get("ragas_llm_calls", 0)
        # The config recorded with the run must describe the index actually searched.
        expected = f"{cfg.chunk_size}/{cfg.chunk_overlap}"
        chunkers = store.chunkers(DEFAULT_COLLECTION_ID)  # the golden set's collection
        if chunkers != {expected}:
            typer.echo(f"Index chunking {chunkers} != config {expected}; run `rag ingest`.")
            raise typer.Exit(1)
        items = load_golden(golden, store.document_texts())
        if limit:
            items = items[:limit]
        n_total = len(items)
        done = {r.id for r in previous}
        items = [i for i in items if i.id not in done]  # resume: only the missing questions
        index = open_index(store, s.faiss_path, s.embedding_model)

        # A nested function (closure) used as the progress callback: prints one line per
        # question. "!!" marks a wrong answer/abstain decision.
        def show(r) -> None:
            ok = r.predicted_answerable == r.answerable
            did = "answered" if r.predicted_answerable else "abstained"
            line = f"{r.id:<5} {'ok' if ok else '!!'} {did:<9}"
            if r.retrieval:
                line += f"  hit={r.retrieval['hit']:.0f} rr={r.retrieval['rr']:.2f}"
            if r.ragas:
                line += "  " + " ".join(f"{SHORT[k]}={_fmt(v)}" for k, v in r.ragas.items())
            line += f"  {r.latency_s:.1f}s"
            typer.echo(line)

        embedder = get_embedder(s)
        # Question embeddings are cached in SQLite: re-running the golden set costs no calls.
        # (Only for retrieval - RAGAS calls the embedder from other threads, so it gets the
        # plain one; a SQLite connection may only be used by the thread that created it.)
        cached = CachedQueryEmbedder(embedder, store, s.embedding_model)
        if retrieval_only:
            ragas = False
            retriever = build_retriever(store, index, cached, cfg)
            answerer = retrieval_only_answerer(retriever, cfg.similarity_threshold)
        else:
            answerer = build_answerer(store, index, cached, get_chat_model(s), cfg)
        evaluator = RagasEvaluator(get_judge_model(s), embedder) if ragas else None
        what = "retrieval-only" if retrieval_only else f"mode={cfg.mode}"
        typer.echo(
            f"Evaluating {len(items)} questions ({what}, {cfg.retrieval}, top_k={cfg.top_k})"
            + (f", resuming run {resume} ({len(previous)} already done)" if resume else "")
            + " ..."
        )
        new, error = run_eval(items, answerer, cfg.top_k, ragas=evaluator, on_result=show)
        results = previous + new
        metrics = run_metrics(results, cfg.similarity_threshold, n_total, error)
        metrics["ragas_llm_calls"] = prior_ragas_calls + (evaluator.llm.calls if evaluator else 0)
        if resume:
            metrics["resumed_from"] = resume
        config = {
            "rag": cfg.model_dump(),
            "retrieval_only": retrieval_only,
            "chat_model": None if retrieval_only else s.chat_model,
            "embedding_model": s.embedding_model,
            "judge_model": (s.judge_model or s.chat_model) if ragas else None,
            "golden": golden.as_posix(),
            "golden_hash": text_hash(golden.read_text(encoding="utf-8")),
            "n_questions": n_total,
        }
        run_id = store.save_eval_run(label, config, metrics, [asdict(r) for r in results])
    finally:
        store.close()

    if error:
        typer.echo(f"\nStopped early: {error}")
        typer.echo(
            f"Run {run_id} saved INCOMPLETE ({len(results)}/{n_total} questions). "
            f"Continue later with: rag eval run --resume {run_id}"
        )
    else:
        typer.echo(f"\nRun {run_id} saved ({len(results)}/{n_total} questions). Metrics:")
    for name in REPORT_METRICS:
        typer.echo(f"  {name:<32} {_fmt(metrics.get(name))}")
    typer.echo(f"  {'ragas_llm_calls':<32} {metrics['ragas_llm_calls']}")
    if error:
        raise typer.Exit(1)


@eval_app.command("list")
def eval_list() -> None:
    """List stored eval runs with their headline metrics."""
    from rag_platform.store.db import Store

    store = Store(Settings().db_path)
    try:
        runs = store.list_eval_runs()
    finally:
        store.close()
    typer.echo(
        f"{'id':>3}  {'created':<20} {'Qs':>6} {'recall':>7} {'mrr':>7} {'abst':>7}"
        f" {'correct':>8}  label"
    )
    for r in runs:
        m = r["metrics"]
        # Qs = questions evaluated; "28*" = incomplete run (resume it with --resume <id>)
        qs = f"{m.get('n_questions')}" + ("*" if m.get("complete") is False else "")
        typer.echo(
            f"{r['id']:>3}  {r['created_at']:<20} {qs:>6} {_fmt(m.get('recall_at_k')):>7}"
            f" {_fmt(m.get('mrr')):>7} {_fmt(m.get('abstention_accuracy')):>7}"
            f" {_fmt(m.get('answer_correctness')):>8}  {r['label']}"
        )


@eval_app.command("compare")
def eval_compare(run_a: int, run_b: int) -> None:
    """Compare two eval runs: config differences, metric deltas, questions whose outcome changed.

    If the runs cover different questions (e.g. one used --limit), both are re-summarized on the
    questions they share, so the comparison stays apples-to-apples.
    """
    from rag_platform.evaluation.metrics import QuestionResult, summarize
    from rag_platform.store.db import Store

    store = Store(Settings().db_path)
    try:
        a, b = store.get_eval_run(run_a), store.get_eval_run(run_b)
    finally:
        store.close()
    if a is None or b is None:
        typer.echo("Run not found.", err=True)
        raise typer.Exit(1)

    # Merge the nested "rag" settings into the top level so both runs' configs can be
    # compared key by key. {**a, **b} builds a new dict from two dicts.
    def flat(cfg: dict) -> dict:
        return {**{k: v for k, v in cfg.items() if k != "rag"}, **cfg["rag"]}

    fa, fb = flat(a["config"]), flat(b["config"])
    typer.echo("Config differences:")
    diffs = [k for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)]
    for k in diffs:
        typer.echo(f"  {k:<24} {fa.get(k)!s:<24} -> {fb.get(k)}")
    if not diffs:
        typer.echo("  (none)")

    ma, mb = a["metrics"], b["metrics"]
    ids_a, ids_b = {r["id"] for r in a["results"]}, {r["id"] for r in b["results"]}
    if ids_a != ids_b:
        common = ids_a & ids_b  # set intersection
        typer.echo(f"\nRuns cover different questions: comparing on the {len(common)} they share.")
        ma = summarize([QuestionResult.from_dict(r) for r in a["results"] if r["id"] in common])
        mb = summarize([QuestionResult.from_dict(r) for r in b["results"] if r["id"] in common])

    typer.echo(f"\n{'metric':<32} {'run ' + str(run_a):>9} {'run ' + str(run_b):>9} {'delta':>8}")
    for name in REPORT_METRICS:
        va, vb = ma.get(name), mb.get(name)
        delta = f"{vb - va:+.3f}" if va is not None and vb is not None else "-"
        typer.echo(f"{name:<32} {_fmt(va):>9} {_fmt(vb):>9} {delta:>8}")

    # A question's "outcome" = right/wrong answer-vs-abstain decision + whether retrieval hit.
    # Listing the questions whose outcome changed shows exactly what a config change fixed/broke.
    def outcome(r: dict) -> str:
        ok = "ok" if r["predicted_answerable"] == r["answerable"] else "wrong-abstain"
        hit = r["retrieval"]["hit"] if r.get("retrieval") else None
        return ok + ("" if hit is None else f"/hit={hit:.0f}")

    by_id = {r["id"]: r for r in a["results"]}
    changed = [
        (r["id"], outcome(by_id[r["id"]]), outcome(r))
        for r in b["results"]
        if r["id"] in by_id and outcome(by_id[r["id"]]) != outcome(r)
    ]
    typer.echo(f"\nQuestions whose outcome changed: {len(changed)}")
    for qid, oa, ob in changed:
        typer.echo(f"  {qid:<5} {oa:<20} -> {ob}")


# --- rag gate (CI evaluation gate) ------------------------------------------------------------


@app.command()
def gate(
    export: bool = typer.Option(
        False, "--export", help="Refresh the embedding snapshot from the local index"
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Record the current metrics as the new baseline"
    ),
) -> None:
    """CI quality gate: retrieval-only evaluation of the sample corpus with the active config.
    Exit code 1 if a release gate is missed or a metric dropped more than 0.02 below the
    baseline. Uses data/eval/embedding_snapshot.db, so normally no API key is needed."""
    import json
    import os

    from rag_platform.evaluation.gate import (
        BASELINE,
        GATED,
        SNAPSHOT,
        MissingVectors,
        OfflineEmbeddings,
        export_snapshot,
        run_gate,
    )

    s = Settings()
    if export:
        n = export_snapshot(s.db_path, SNAPSHOT, s.embedding_model)
        size = SNAPSHOT.stat().st_size / 1024
        typer.echo(f"Snapshot written: {SNAPSHOT} ({n} vectors, {size:.0f} KB).")
        return
    # Real embeddings only if a key exists (new texts); otherwise the snapshot must suffice.
    if s.gemini_api_key:
        from rag_platform.llm import get_embedder

        embedder = get_embedder(s)
    else:
        embedder = OfflineEmbeddings()
    try:
        result = run_gate(_config(), embedder, s.embedding_model)
    except MissingVectors as e:
        # GitHub Actions shows "::warning::" lines as annotations on the run.
        prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "WARNING: "
        typer.echo(f"{prefix}Quality gate skipped: {e}. Run `rag gate --export` locally "
                   "(or set GEMINI_API_KEY) after changing the chunking.")  # fmt: skip
        return
    for c in result.checks:
        if c.status != "n/a":
            op = ">=" if c.op == ">=" else "<="
            typer.echo(f"  {c.status.upper():5} {c.label:20} {c.value:.3f} {op} {c.threshold}")
    for d in result.drops:
        typer.echo(f"  FAIL  regression: {d}")
    if update_baseline:
        BASELINE.write_text(
            json.dumps({k: round(result.metrics[k], 4) for k in GATED}, indent=2) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"Baseline updated: {BASELINE}")
    if not result.passed:
        typer.echo("Quality gate FAILED.", err=True)
        raise typer.Exit(1)
    typer.echo("Quality gate passed.")
