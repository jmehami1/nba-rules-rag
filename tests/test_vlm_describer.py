"""Tests for VLM frame describer."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from PIL import Image

from nba_rules_rag.vlm_describer import (
    build_vlm_user_prompt,
    describe_frame_with_vlm,
    describe_frames_with_vlm,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_describe_frames_with_vlm_happy_path(monkeypatch: pytest.MonkeyPatch):
    frames = [
        Image.new("RGB", (120, 80), color=(10, 20, 30)),
        Image.new("RGB", (120, 80), color=(20, 30, 40)),
    ]

    fake_payload = {
        "choices": [
            {
                "message": {
                    "content": """{
  \"question\": \"Was this a travel?\",
  \"play_narration\": \"Player in blue receives the ball, gathers control, and takes two steps toward the basket.\",
  \"frame_observations\": [
    {\"frame_id\": \"F01\", \"timestamp_sec\": 21.0, \"description\": \"Player dribbling, ball at floor level.\", \"change_from_previous\": \"N/A — first frame\"},
    {\"frame_id\": \"F02\", \"timestamp_sec\": 23.0, \"description\": \"Ball raised to hip; right foot planted.\", \"change_from_previous\": \"Ball lifted off floor, gather initiated.\"}
  ],
  \"query_relevant_signals\": [\"Two advancing steps visible after control\"],
  \"uncertainties\": [\"Gather instant is partially occluded\"]
}"""
                }
            }
        ]
    }

    def _fake_urlopen(_request, timeout=60):
        assert timeout == 60
        return _FakeHTTPResponse(fake_payload)

    monkeypatch.setenv("HF_TOKEN", "fake-token")
    with patch("nba_rules_rag.vlm_describer.urlopen", side_effect=_fake_urlopen):
        result = describe_frames_with_vlm(
            frames=frames,
            question="Was this a travel?",
            frame_timestamps_sec=[21.0, 23.0],
            model="test-model",
        )

        assert result["n_frames"] == 2
        assert result["frame_timestamps_sec"] == [21.0, 23.0]
        assert result["play_narration"] is not None
        assert result["frame_observations"][0]["change_from_previous"] == "N/A — first frame"


def test_describe_frame_with_vlm_requires_token():
    frame = Image.new("RGB", (120, 80), color=(10, 20, 30))

    with pytest.raises(RuntimeError, match="Missing API token"):
        describe_frame_with_vlm(frame=frame, question="Was this a travel?", token=None)


def test_describe_frame_with_vlm_requires_question(monkeypatch: pytest.MonkeyPatch):
    frame = Image.new("RGB", (120, 80), color=(10, 20, 30))
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("VLM_MODEL", "test-model")

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        describe_frame_with_vlm(frame=frame, question="   ")


def test_describe_frames_with_vlm_requires_frames(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setenv("VLM_MODEL", "test-model")

    with pytest.raises(ValueError, match="frames must contain at least one image"):
        describe_frames_with_vlm(frames=[], question="Was this a travel?")


def test_build_vlm_user_prompt_includes_timestamps():
    prompt = build_vlm_user_prompt("Was this a travel?", [1.0, 2.5, 4.0])

    assert "Question to answer: Was this a travel?" in prompt
    assert "1.00s" in prompt
    assert "4.00s" in prompt
    assert "change_from_previous" in prompt
    assert "play_narration" in prompt
    assert "query_relevant_signals" in prompt
