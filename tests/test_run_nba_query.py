"""Tests for scripts/run_nba_query.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from PIL import Image

from scripts.run_nba_query import (
    DEFAULT_END_TIME,
    DEFAULT_QUESTION,
    DEFAULT_START_TIME,
    DEFAULT_YOUTUBE_URL,
    build_mock_vlm_result,
    build_parser,
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
        assert output_dir == str(tmp_path)
        return fake_frames

    monkeypatch.setattr(
        "scripts.run_nba_query.extract_frames_from_youtube",
        _fake_extract,
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

    for i in range(1, 9):
        assert (tmp_path / f"frame_{i:02d}.jpg").exists()
    assert (tmp_path / "frame_grid.jpg").exists()


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
