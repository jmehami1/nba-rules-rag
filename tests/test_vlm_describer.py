"""Tests for VLM frame describer."""

from __future__ import annotations

import json
from urllib.error import HTTPError
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


class _FakeHTTPErrorResponse:
    def __init__(self, payload: dict, code: int = 400):
        self.payload = payload
        self.code = code

    def __call__(self, req, timeout=60):
        raise HTTPError(
            url=getattr(req, "full_url", "https://api.openai.com/v1/chat/completions"),
            code=self.code,
            msg="Bad Request",
            hdrs=None,
            fp=_BytesPayload(self.payload),
        )


class _BytesPayload:
    def __init__(self, payload: dict):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def close(self) -> None:
        return None


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

    monkeypatch.setenv("OPENAI_API_KEY", "fake-token")
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
    monkeypatch.setenv("OPENAI_API_KEY", "fake-token")
    monkeypatch.setenv("VLM_MODEL", "test-model")

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        describe_frame_with_vlm(frame=frame, question="   ")


def test_describe_frames_with_vlm_requires_frames(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-token")
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


def test_describe_frames_with_vlm_plain_text_fallback(monkeypatch: pytest.MonkeyPatch):
    frames = [Image.new("RGB", (120, 80), color=(10, 20, 30))]

    fake_payload = {
        "choices": [
            {
                "message": {
                    "content": "Player gathers at the wing, takes two controlled steps, then stops."
                }
            }
        ]
    }

    def _fake_urlopen(_request, timeout=60):
        assert timeout == 60
        return _FakeHTTPResponse(fake_payload)

    monkeypatch.setenv("OPENAI_API_KEY", "fake-token")
    with patch("nba_rules_rag.vlm_describer.urlopen", side_effect=_fake_urlopen):
        result = describe_frames_with_vlm(
            frames=frames,
            question="Was this a travel?",
            frame_timestamps_sec=[21.0],
            model="test-model",
        )

        assert result["question"] == "Was this a travel?"
        assert "Player gathers" in result["play_narration"]
        assert result["frame_observations"] == []
        assert "parse_warning" in result


def test_describe_frames_with_vlm_output_text_parts(monkeypatch: pytest.MonkeyPatch):
    frames = [Image.new("RGB", (120, 80), color=(10, 20, 30))]

    fake_payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"question":"Was this a travel?","play_narration":"Valid JSON via output_text","frame_observations":[],"query_relevant_signals":[],"uncertainties":[]}',
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setenv("OPENAI_API_KEY", "fake-token")
    with patch("nba_rules_rag.vlm_describer.urlopen", return_value=_FakeHTTPResponse(fake_payload)):
        result = describe_frames_with_vlm(
            frames=frames,
            question="Was this a travel?",
            frame_timestamps_sec=[21.0],
            model="test-model",
        )

    assert result["play_narration"] == "Valid JSON via output_text"


def test_describe_frames_with_vlm_retries_without_response_format(monkeypatch: pytest.MonkeyPatch):
    frames = [Image.new("RGB", (120, 80), color=(10, 20, 30))]
    call_count = {"n": 0}

    unsupported_response_format_error = {
        "error": {
            "message": "Unsupported parameter: 'response_format' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": "unsupported_parameter",
        }
    }

    success_payload = {
        "choices": [
            {
                "message": {
                    "content": '{"question":"Was this a travel?","play_narration":"Retry succeeded","frame_observations":[],"query_relevant_signals":[],"uncertainties":[]}'
                }
            }
        ]
    }

    def _fake_urlopen(req, timeout=60):
        call_count["n"] += 1
        request_body = json.loads(req.data.decode("utf-8"))
        if call_count["n"] == 1:
            assert "response_format" in request_body
            return _FakeHTTPErrorResponse(unsupported_response_format_error)(req, timeout)
        assert "response_format" not in request_body
        return _FakeHTTPResponse(success_payload)

    monkeypatch.setenv("OPENAI_API_KEY", "fake-token")
    with patch("nba_rules_rag.vlm_describer.urlopen", side_effect=_fake_urlopen):
        result = describe_frames_with_vlm(
            frames=frames,
            question="Was this a travel?",
            frame_timestamps_sec=[21.0],
            model="test-model",
        )

    assert call_count["n"] == 2
    assert result["play_narration"] == "Retry succeeded"
