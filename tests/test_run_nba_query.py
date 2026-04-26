"""Tests for scripts/run_nba_query.py."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from scripts.run_nba_query import (
    DEFAULT_END_TIME,
    OPENAI_MODEL_OPTIONS,
    DEFAULT_QUESTION,
    DEFAULT_START_TIME,
    DEFAULT_YOUTUBE_URL,
    _prompt_model_choice,
    build_mock_vlm_result,
    build_parser,
    ensure_non_demo_env_configured,
    run_nba_query,
    save_frame_grid,
    validate_question,
)


def _make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        youtube_url=DEFAULT_YOUTUBE_URL,
        start_time=DEFAULT_START_TIME,
        end_time=DEFAULT_END_TIME,
        question=DEFAULT_QUESTION,
        n_frames=8,
        output_dir=str(tmp_path),
        no_show=True,
        demo=False,
        rulebook_pdf_path="docs/2023-24-NBA-Season-Official-Playing-Rules.pdf",
        vector_store_dir=str(tmp_path / "vector_store"),
    )


def test_parser_has_expected_defaults():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.youtube_url == DEFAULT_YOUTUBE_URL
    assert args.start_time == DEFAULT_START_TIME
    assert args.end_time == DEFAULT_END_TIME
    assert args.question == DEFAULT_QUESTION
    assert args.demo is False


def test_build_mock_vlm_result_uses_expected_shape():
    result = build_mock_vlm_result(DEFAULT_QUESTION, 7)

    assert result["question"] == DEFAULT_QUESTION
    assert "play_narration" in result
    assert "frame_observations" in result
    assert len(result["frame_observations"]) == 7


def test_build_parser_has_embedding_defaults():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.rulebook_pdf_path is not None
    assert args.vector_store_dir is not None


def test_validate_question_accepts_non_empty():
    assert validate_question("  Was this a travel?  ") == "Was this a travel?"


def test_validate_question_rejects_empty():
    with pytest.raises(ValueError, match="Question/query must be a non-empty string"):
        validate_question("   ")


def test_save_frame_grid_creates_file(tmp_path: Path):
    frames = [Image.new("RGB", (100, 60), color=(i * 20, 0, 0)) for i in range(6)]
    output_path = tmp_path / "grid.jpg"

    saved = save_frame_grid(frames, output_path)

    assert saved == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_run_nba_query_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    args = _make_args(tmp_path)

    fake_frames = [Image.new("RGB", (120, 80), color=(0, i * 20, 0)) for i in range(8)]

    def _fake_extract(video_id, start_sec, end_sec, n_frames, output_dir):
        assert video_id == "9fFWawcJXUw"
        assert start_sec == 21.0
        assert end_sec == 29.0
        assert n_frames == 8
        extract_dir = Path(output_dir)
        assert extract_dir.parent == tmp_path
        assert extract_dir.name
        return fake_frames

    monkeypatch.setattr(
        "scripts.run_nba_query.extract_frames_from_youtube",
        _fake_extract,
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.ensure_non_demo_env_configured",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.describe_frames_with_vlm",
        lambda frames, question, frame_timestamps_sec=None: {
            "n_frames": len(frames),
            "frame_timestamps_sec": frame_timestamps_sec,
            "question": question,
            "play_narration": "Mock sequence summary.",
            "frame_observations": [],
            "query_relevant_signals": [],
            "uncertainties": [],
        },
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.ensure_rulebook_embeddings",
        lambda rulebook_pdf_path, vector_store_dir, chunk_size_words=180, overlap_words=40: {
            "vector_store_dir": vector_store_dir,
            "built": False,
            "index_path": str(tmp_path / "vector_store" / "faiss.index"),
            "chunks_path": str(tmp_path / "vector_store" / "chunks.jsonl"),
            "metadata_path": str(tmp_path / "vector_store" / "metadata.json"),
        },
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.build_vlm_output_embedding",
        lambda vlm_result, output_dir: {
            "vlm_embedding_path": str(tmp_path / "vlm_output_embedding.npy"),
            "vlm_embedding_text_path": str(tmp_path / "vlm_output_text.json"),
            "vlm_embedding_dim": 384,
        },
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.retrieve_rulebook_chunks",
        lambda query, top_k, vector_store_dir: [
            {"text": "Rule 10 Section I: Travelling.", "section_title": "Rule 10", "score": 0.91},
        ],
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.request_reasoning_ruling",
        lambda question, vlm_result, rag_chunks, timeout_sec=60: {
            "question": question,
            "applicable_rules": [
                {"rule": "Rule 4, Section III(b)", "application": "Gather appears at F03."},
            ],
            "likely_gather_frame": "F03",
            "gather_confidence": "medium",
            "post_gather_step_count_visible": "unclear",
            "pivot_foot": "unclear",
            "travel_ruling": "inconclusive",
            "ruling_confidence": "medium",
            "reasoning": "Insufficient visible post-gather foot contacts.",
            "facts_required_for_travel_call": ["Exact post-gather first and second floor contacts."],
            "limitations": ["Sampled frames omit intermediate motion."],
        },
    )

    result = run_nba_query(args)

    assert result["video_id"] == "9fFWawcJXUw"
    assert result["start_sec"] == 21.0
    assert result["end_sec"] == 29.0
    assert result["question"] == DEFAULT_QUESTION
    assert result["n_frames"] == 8
    assert len(result["frame_timestamps_sec"]) == 8
    assert "Frame index" in result["vlm_prompt"]
    assert result["vlm_error"] is None
    assert result["rulebook_embedding_store"]["built"] is False
    assert result["vlm_result"] is not None
    assert result["vlm_result"]["question"] == DEFAULT_QUESTION
    assert result["vlm_embedding"]["vlm_embedding_dim"] == 384
    assert result["vlm_embedding_error"] is None
    assert result["rag_query"] != ""
    assert result["rag_chunks"] is not None
    assert len(result["rag_chunks"]) == 1
    assert result["rag_error"] is None
    assert result["ruling"] is not None
    assert result["ruling"]["travel_ruling"] == "inconclusive"
    assert result["ruling_error"] is None
    run_output_dir = Path(result["output_dir"])
    assert run_output_dir.parent == tmp_path
    assert run_output_dir.exists()

    for i in range(1, 9):
        assert (run_output_dir / f"frame_{i:02d}.jpg").exists()
    assert (run_output_dir / "frame_grid.jpg").exists()


def test_run_nba_query_vlm_failure_continues_to_reasoning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When VLM fails, the script should continue and still attempt automated reasoning."""
    args = _make_args(tmp_path)

    fake_frames = [Image.new("RGB", (120, 80), color=(i * 30, 0, 0)) for i in range(8)]

    monkeypatch.setattr(
        "scripts.run_nba_query.extract_frames_from_youtube",
        lambda video_id, start_sec, end_sec, n_frames, output_dir: fake_frames,
    )
    monkeypatch.setattr("scripts.run_nba_query.ensure_non_demo_env_configured", lambda: None)
    monkeypatch.setattr(
        "scripts.run_nba_query.describe_frames_with_vlm",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("VLM HTTP 401: invalid API key")),
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.ensure_rulebook_embeddings",
        lambda rulebook_pdf_path, vector_store_dir: {
            "vector_store_dir": vector_store_dir,
            "built": False,
            "index_path": str(tmp_path / "faiss.index"),
            "chunks_path": str(tmp_path / "chunks.jsonl"),
            "metadata_path": str(tmp_path / "metadata.json"),
        },
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.retrieve_rulebook_chunks",
        lambda query, top_k, vector_store_dir: [
            {"text": "Rule 10.", "section_title": "Rule 10", "score": 0.80},
        ],
    )

    reasoning_called_with_none_vlm = {}

    def _fake_reasoning(question, vlm_result, rag_chunks, timeout_sec=60):
        reasoning_called_with_none_vlm["vlm_result"] = vlm_result
        raise RuntimeError("Reasoning HTTP 401: invalid API key")

    monkeypatch.setattr("scripts.run_nba_query.request_reasoning_ruling", _fake_reasoning)

    result = run_nba_query(args)

    # VLM failed — error captured, result is None
    assert result["vlm_result"] is None
    assert "401" in result["vlm_error"]

    # Reasoning was still attempted (with vlm_result=None) and its error is captured
    assert reasoning_called_with_none_vlm["vlm_result"] is None
    assert result["ruling"] is None
    assert "401" in result["ruling_error"]

    # Script did not raise — it continued gracefully


def test_run_nba_query_demo_uses_mock_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    args = _make_args(tmp_path)
    args.demo = True
    args.youtube_url = "https://www.youtube.com/watch?v=dummy_should_be_ignored"
    args.start_time = "1:00"
    args.end_time = "1:05"
    args.question = "This should be replaced by demo defaults"

    fake_frames = [Image.new("RGB", (120, 80), color=(0, i * 20, 0)) for i in range(6)]

    monkeypatch.setattr(
        "scripts.run_nba_query.load_demo_frames",
        lambda output_dir: fake_frames,
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.describe_frames_with_vlm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live VLM should not be called")),
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.ensure_rulebook_embeddings",
        lambda rulebook_pdf_path, vector_store_dir, chunk_size_words=180, overlap_words=40: {
            "vector_store_dir": vector_store_dir,
            "built": True,
            "index_path": str(tmp_path / "vector_store" / "faiss.index"),
            "chunks_path": str(tmp_path / "vector_store" / "chunks.jsonl"),
            "metadata_path": str(tmp_path / "vector_store" / "metadata.json"),
        },
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.build_vlm_output_embedding",
        lambda vlm_result, output_dir: {
            "vlm_embedding_path": str(tmp_path / "vlm_output_embedding.npy"),
            "vlm_embedding_text_path": str(tmp_path / "vlm_output_text.json"),
            "vlm_embedding_dim": 384,
        },
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.retrieve_rulebook_chunks",
        lambda query, top_k, vector_store_dir: [
            {"text": "Rule 10 Section I: Travelling.", "section_title": "Rule 10", "score": 0.88},
        ],
    )
    monkeypatch.setattr(
        "scripts.run_nba_query.request_reasoning_ruling",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("automated reasoner should not be called in demo mode")),
    )

    result = run_nba_query(args)

    assert result["vlm_error"] is None
    assert result["youtube_url"] == DEFAULT_YOUTUBE_URL
    assert result["start_time"] == DEFAULT_START_TIME
    assert result["end_time"] == DEFAULT_END_TIME
    assert result["question"] == DEFAULT_QUESTION
    assert len(result["vlm_result"]["frame_observations"]) == 7
    assert result["vlm_result"]["question"] is not None
    assert result["vlm_embedding"]["vlm_embedding_dim"] == 384
    assert result["vlm_embedding_error"] is None
    assert result["rag_query"] != ""
    assert result["rag_chunks"] is not None
    assert result["rag_error"] is None
    assert result["ruling"] is not None
    assert result["ruling_error"] is None


def test_run_nba_query_invalid_url_raises(tmp_path: Path):
    args = _make_args(tmp_path)
    args.youtube_url = "https://example.com/not-youtube"

    with pytest.raises(ValueError, match="Not a recognised YouTube URL"):
        run_nba_query(args)


def test_run_nba_query_invalid_interval_raises(tmp_path: Path):
    args = _make_args(tmp_path)
    args.start_time = "0:29"
    args.end_time = "0:21"

    with pytest.raises(ValueError, match="must be strictly before"):
        run_nba_query(args)


def test_run_nba_query_empty_question_raises(tmp_path: Path):
    args = _make_args(tmp_path)
    args.question = "   "

    with pytest.raises(ValueError, match="Question/query must be a non-empty string"):
        run_nba_query(args)


def test_ensure_non_demo_env_configured_raises_when_non_interactive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)
    monkeypatch.delenv("REASONER_MODEL", raising=False)
    monkeypatch.setattr("scripts.run_nba_query._load_env_file", lambda env_path=None: None)
    monkeypatch.setattr("scripts.run_nba_query.sys.stdin.isatty", lambda: False)

    with pytest.raises(RuntimeError, match="Missing required environment variables for live mode"):
        ensure_non_demo_env_configured()


def test_prompt_model_choice_returns_predefined_option(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    selected = _prompt_model_choice("VLM_MODEL")
    assert selected == OPENAI_MODEL_OPTIONS[0]


# ---------------------------------------------------------------------------
# Real API call test — skipped by default.
# Run with:  pytest --run-api-tests tests/test_run_nba_query.py::test_api_calls_work
# WARNING: this makes real HTTP requests to the OpenAI API and will charge your account.
# ---------------------------------------------------------------------------

def _prompt_api_key() -> str:
    """Return OPENAI_API_KEY from env, or prompt the user securely."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    print("\nOPENAI_API_KEY not set in environment.", file=sys.stderr)
    key = getpass.getpass("Enter your OpenAI API key (input hidden): ").strip()
    if not key:
        pytest.skip("No API key provided — skipping API test.")
    return key


def _load_repo_env_vars() -> None:
    """Load KEY=VALUE pairs from repo .env if present and not already set."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def _confirm_charges(model: str) -> None:
    """Warn the user and let them opt out before any API call is made."""
    print(
        f"\n⚠  This test will make TWO real API calls to OpenAI using model '{model}'.\n"
        "   These calls are intentionally minimal (~50 tokens each) but WILL charge your account.\n",
        file=sys.stderr,
    )
    answer = input("Proceed? [y/N]: ").strip().lower()
    if answer != "y":
        pytest.skip("User opted out of API charges — skipping API test.")


def _post_minimal(api_key: str, model: str, messages: list[dict], label: str) -> dict:
    """POST a minimal chat completion, retrying with max_completion_tokens if needed."""
    def _attempt(payload_dict: dict) -> dict:
        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload_dict).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        from urllib.error import HTTPError as _HTTPError
        try:
            with urlopen(req, timeout=30) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
        except _HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {details}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pytest.fail(f"{label} response was not valid JSON:\n{body}")

    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "none",
        "max_completion_tokens": 512,
    }
    try:
        return _attempt(payload)
    except RuntimeError as exc:
        err = str(exc).lower()
        if "reasoning_effort" in err and "unsupported" in err:
            payload_low_effort = {**payload, "reasoning_effort": "low"}
            try:
                return _attempt(payload_low_effort)
            except RuntimeError as exc2:
                pytest.fail(f"{label} API call failed: {exc2}")

        if "response_format" in err and "unsupported_parameter" in err:
            payload_no_response_format = {k: v for k, v in payload.items() if k != "response_format"}
            try:
                return _attempt(payload_no_response_format)
            except RuntimeError as exc2:
                pytest.fail(f"{label} API call failed: {exc2}")

        if "max_completion_tokens" in err and "unsupported_parameter" in err:
            payload_no_max_completion_tokens = {k: v for k, v in payload.items() if k != "max_completion_tokens"}
            try:
                return _attempt(payload_no_max_completion_tokens)
            except RuntimeError as exc2:
                pytest.fail(f"{label} API call failed: {exc2}")

        pytest.fail(f"{label} API call failed: {exc}")


@pytest.mark.api
def test_api_calls_work() -> None:
    """Send minimal real requests to the OpenAI API for both VLM and reasoning.

    Skipped automatically when running the full test suite.
    Run manually with:
        pytest --run-api-tests tests/test_run_nba_query.py::test_api_calls_work
    """
    _load_repo_env_vars()

    required_model_env = ("VLM_MODEL", "REASONER_MODEL")
    missing_model_env = [
        name for name in required_model_env if not os.environ.get(name, "").strip()
    ]
    if missing_model_env:
        pytest.skip(
            "Missing required environment variables for API test: "
            + ", ".join(missing_model_env)
            + ". Set them and rerun."
        )

    api_key = _prompt_api_key()

    model = os.environ["VLM_MODEL"].strip()

    _confirm_charges(model)

    # ── VLM call: use a real demo frame image from the repository ───────
    repo_root = Path(__file__).resolve().parents[1]
    demo_frame_path = repo_root / "data" / "processed" / "demo_frames" / "frame_01.jpg"
    if not demo_frame_path.exists():
        pytest.skip(f"Demo image not found: {demo_frame_path}")

    b64 = base64.b64encode(demo_frame_path.read_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    vlm_messages = [
        {
            "role": "system",
            "content": (
                "Return JSON only with exactly this shape: "
                '{"player_count": <integer>}.'
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Count visible basketball players in this frame and return only JSON: "
                        '{"player_count": <integer>}'
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
            ],
        },
    ]
    vlm_response = _post_minimal(api_key, model, vlm_messages, "VLM")

    assert "choices" in vlm_response, f"VLM response missing 'choices': {vlm_response}"
    vlm_text = vlm_response["choices"][0]["message"]["content"]
    finish_reason = vlm_response["choices"][0].get("finish_reason")
    usage = vlm_response.get("usage", {})
    assert isinstance(vlm_text, str) and vlm_text.strip(), (
        "VLM returned empty content. "
        f"finish_reason={finish_reason!r}, usage={usage!r}, full_response={vlm_response!r}"
    )

    # Parse strict JSON or first JSON object embedded in surrounding text.
    vlm_text_stripped = vlm_text.strip()
    try:
        vlm_json = json.loads(vlm_text_stripped)
    except json.JSONDecodeError:
        start = vlm_text_stripped.find("{")
        end = vlm_text_stripped.rfind("}")
        assert start != -1 and end != -1 and end > start, (
            f"VLM did not return JSON content: {vlm_text_stripped!r}"
        )
        vlm_json = json.loads(vlm_text_stripped[start : end + 1])

    assert isinstance(vlm_json, dict), f"VLM JSON is not an object: {vlm_json}"
    assert "player_count" in vlm_json, f"VLM JSON missing player_count: {vlm_json}"
    assert isinstance(vlm_json["player_count"], int), (
        f"player_count must be int, got {type(vlm_json['player_count']).__name__}: {vlm_json}"
    )
    assert vlm_json["player_count"] >= 0, f"player_count must be non-negative: {vlm_json}"
    print(f"\n  VLM response  : {vlm_json!r}", file=sys.stderr)

    # ── Reasoning call: text-only, minimal prompt ────────────────────────
    reasoner_model = os.environ["REASONER_MODEL"].strip()

    reasoning_messages = [
        {
            "role": "system",
            "content": "Return JSON only with exactly this shape: {\"ack\": \"OK\"}.",
        },
        {"role": "user", "content": "Acknowledge and return json."},
    ]
    reasoning_response = _post_minimal(api_key, reasoner_model, reasoning_messages, "Reasoning")

    assert "choices" in reasoning_response, (
        f"Reasoning response missing 'choices': {reasoning_response}"
    )
    reasoning_text = reasoning_response["choices"][0]["message"]["content"]
    assert isinstance(reasoning_text, str) and reasoning_text.strip(), (
        "Reasoning returned empty content"
    )
    try:
        reasoning_json = json.loads(reasoning_text)
    except json.JSONDecodeError:
        start = reasoning_text.find("{")
        end = reasoning_text.rfind("}")
        assert start != -1 and end != -1 and end > start, (
            f"Reasoning did not return JSON content: {reasoning_text!r}"
        )
        reasoning_json = json.loads(reasoning_text[start : end + 1])

    assert isinstance(reasoning_json, dict), f"Reasoning JSON is not an object: {reasoning_json}"
    assert reasoning_json.get("ack") == "OK", f"Reasoning JSON unexpected: {reasoning_json}"
    print(f"  Reasoning response: {reasoning_json!r}", file=sys.stderr)
    print("  Both API calls succeeded.", file=sys.stderr)
