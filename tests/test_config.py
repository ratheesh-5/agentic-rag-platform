from rag_platform.config import RagConfig, Settings


def test_rag_config_defaults():
    cfg = RagConfig()
    assert cfg.chunk_overlap < cfg.chunk_size
    assert cfg.top_k > 0


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    assert Settings().gemini_api_key == "abc"


def test_chat_models_get_a_timeout_and_few_retries():
    # A slow provider must not hang a question for minutes (library default: 6 retries, no
    # timeout). Building the client makes no network call.
    from rag_platform.llm import get_chat_model, get_judge_model

    s = Settings(
        _env_file=None,
        gemini_api_key="x",
        groq_api_key="y",
        chat_timeout_seconds=15,
        chat_max_retries=1,
    )
    chat = get_chat_model(s)
    assert list(chat.models) == ["gemini", "groq"]
    for model in (*chat.models.values(), get_judge_model(s)):
        timeout = getattr(model, "timeout", None) or getattr(model, "request_timeout", None)
        assert timeout == 15 and model.max_retries == 1
    defaults = Settings(_env_file=None, gemini_api_key="x")
    assert (defaults.chat_timeout_seconds, defaults.chat_max_retries) == (20, 2)


def test_chat_limiter_allows_a_question_s_two_calls_without_waiting():
    # A chat question makes 2 calls (rewrite + answer). With a burst of 2 saved up while idle,
    # both go out at once; the 3rd waits. Fast numbers so the test takes ~0.5 s, no real model.
    import time

    from rag_platform.llm import _rate_limiter

    limiter = _rate_limiter("test-model", requests_per_minute=602, burst=2)  # refill 10/s
    time.sleep(0.5)  # idle: the bucket fills up to the burst size
    assert limiter.acquire(blocking=False) and limiter.acquire(blocking=False)
    assert not limiter.acquire(blocking=False)  # bucket empty: the next call would wait

    # Defaults: burst 2 and a steady 12/min, so one minute never exceeds the 14/min cap.
    s = Settings(_env_file=None, gemini_api_key="x")
    default = _rate_limiter("other-model", s.chat_requests_per_minute, s.chat_burst)
    assert default.max_bucket_size == 2
    assert abs(default.requests_per_second - 12 / 60) < 1e-9
