"""VLM inference helpers for extracting structured facts from frames."""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

DEFAULT_VLM_MODEL: str | None = None  # set via VLM_MODEL env var or pass explicitly
DEFAULT_VLM_API_BASE = "https://api.openai.com/v1/chat/completions"


_VLM_SYSTEM_PROMPT = (
	"You are an expert NBA basketball analyst reviewing a sequence of video frames "
	"to produce a factual, evidence-based narration that will be used to answer a "
	"specific officiating question. Your narration will feed directly into a rules "
	"reasoning engine, so accuracy and specificity matter more than fluency.\n\n"
	"Guidelines:\n"
	"- Describe only what is visually observable. Do not speculate beyond what the frames show.\n"
	"- Use precise NBA terminology (gather, pivot foot, live dribble, euro step, etc.).\n"
	"- For every player action note: body position, foot placement, ball position, "
	"  and contact with other players.\n"
	"- Each frame observation must describe what changed or progressed since the previous frame.\n"
	"- The play_narration must be a single coherent chronological account written as if "
	"  dictating to an officiating review panel — factual, present-tense, sequential.\n"
	"- Explicitly flag anything that is occluded, ambiguous, or outside the frame.\n"
	"- Do not estimate steps unless a foot visibly contacts the floor after the gather. "
	"  If a foot is already on the floor in the gather frame, describe it but do not count it "
	"  as a step unless the rule-based stage can determine it is the first post-gather floor contact.\n"
	"- Return strict JSON only. No markdown fences, no extra prose."
)


def build_vlm_user_prompt(question: str, frame_timestamps_sec: list[float] | None = None) -> str:
	"""Build the full user prompt used for VLM extraction."""
	clean_question = (question or "").strip()

	timestamps_block = ""
	if frame_timestamps_sec:
		lines = [
			f"  F{i:02d}: {ts:.2f}s"
			for i, ts in enumerate(frame_timestamps_sec, start=1)
		]
		timestamps_block = (
			"Frame index → clip timestamp mapping:\n"
			+ "\n".join(lines)
			+ "\n\n"
		)

	schema = (
		"{\n"
		'  "question": "Restate the input question verbatim.",\n'
		'  "play_narration": '
		'"Chronological present-tense narration of the full play sequence as seen across all frames. '
		'Written for an NBA officiating review panel. Must directly address the question.",\n'
		'  "frame_observations": [\n'
		'    {\n'
		'      "frame_id": "F01",\n'
		'      "timestamp_sec": 0.0,\n'
		'      "description": "What is happening in this frame — player positions, ball location, foot placement.",\n'
		'      "change_from_previous": "What changed since the previous frame (write \\"N/A — first frame\\" for F01)."\n'
		'    }\n'
		"  ],\n"
		'  "query_relevant_signals": [\n'
		'    "Concise bullet of any observable fact directly relevant to answering the question."\n'
		"  ],\n"
		'  "uncertainties": [\n'
		'    "Anything occluded, ambiguous, or not clearly visible that would affect the answer."\n'
		"  ]\n"
		"}"
	)

	return (
		f"Question to answer: {clean_question}\n\n"
		f"{timestamps_block}"
		f"You are given {len(frame_timestamps_sec) if frame_timestamps_sec else 'N'} "
		f"ordered frames (F01 is earliest, last frame is most recent) from the same basketball clip.\n\n"
		"Analyze the temporal progression across all frames as a continuous sequence — "
		"not as isolated snapshots.\n\n"
		f"Return JSON only matching this schema:\n{schema}"
	)


def _image_to_data_url(frame: Image.Image) -> str:
	"""Convert a PIL image to a base64 JPEG data URL."""
	buffer = BytesIO()
	frame.save(buffer, format="JPEG", quality=92)
	raw = buffer.getvalue()
	b64 = base64.b64encode(raw).decode("ascii")
	return f"data:image/jpeg;base64,{b64}"


def _extract_text_from_response(payload: dict) -> str:
	"""Extract text from OpenAI-compatible chat completion JSON."""
	choices = payload.get("choices")
	if not choices:
		raise RuntimeError("VLM response missing 'choices'.")

	message = choices[0].get("message", {})
	content = message.get("content")

	if isinstance(content, str):
		return content.strip()

	if isinstance(content, list):
		text_parts: list[str] = []
		for part in content:
			if isinstance(part, dict) and part.get("type") == "text":
				text = part.get("text", "")
				if text:
					text_parts.append(str(text).strip())
		joined = "\n".join([p for p in text_parts if p])
		if joined:
			return joined

	raise RuntimeError("VLM response did not contain readable text content.")


def _extract_json_block(text: str) -> dict:
	"""Extract the first JSON object from model text output."""
	stripped = text.strip()
	try:
		parsed = json.loads(stripped)
		if isinstance(parsed, dict):
			return parsed
	except json.JSONDecodeError:
		pass

	start = stripped.find("{")
	end = stripped.rfind("}")
	if start != -1 and end != -1 and end > start:
		candidate = stripped[start : end + 1]
		try:
			parsed = json.loads(candidate)
		except json.JSONDecodeError as exc:
			raise RuntimeError("VLM response did not include valid JSON.") from exc
		if isinstance(parsed, dict):
			return parsed

	raise RuntimeError("VLM response did not include a JSON object.")


def _post_chat_completion(payload: dict, token: str, timeout_sec: int) -> dict:
	"""Post payload to OpenAI-compatible chat completion endpoint."""
	api_base = os.getenv("VLM_API_BASE", DEFAULT_VLM_API_BASE)
	body = json.dumps(payload).encode("utf-8")

	request = Request(
		api_base,
		data=body,
		headers={
			"Authorization": f"Bearer {token}",
			"Content-Type": "application/json",
		},
		method="POST",
	)

	try:
		with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
			response_body = response.read().decode("utf-8")
	except HTTPError as exc:
		details = exc.read().decode("utf-8", errors="replace")
		raise RuntimeError(
			f"VLM request failed with HTTP {exc.code}: {details}"
		) from exc
	except URLError as exc:
		raise RuntimeError(f"VLM request failed: {exc.reason}") from exc

	try:
		return json.loads(response_body)
	except json.JSONDecodeError as exc:
		raise RuntimeError("VLM response was not valid JSON.") from exc


def describe_frames_with_vlm(
	frames: list[Image.Image],
	question: str,
	frame_timestamps_sec: list[float] | None = None,
	model: str | None = None,
	token: str | None = None,
	timeout_sec: int = 60,
) -> dict:
	"""Send ordered frames + question to VLM and return structured extraction.

	The output is designed for downstream reasoning/RAG stages.
	"""
	if not frames:
		raise ValueError("frames must contain at least one image")
	if any(not isinstance(frame, Image.Image) for frame in frames):
		raise ValueError("all frames must be PIL.Image.Image instances")
	if frame_timestamps_sec is not None and len(frame_timestamps_sec) != len(frames):
		raise ValueError("frame_timestamps_sec must match the number of frames")

	clean_question = (question or "").strip()
	if not clean_question:
		raise ValueError("question must be a non-empty string")

	resolved_token = token or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
	if not resolved_token:
		raise RuntimeError(
			"Missing API token. Set OPENAI_API_KEY or HF_TOKEN env var, or pass token explicitly."
		)

	resolved_model = model or os.getenv("VLM_MODEL")
	if not resolved_model:
		raise ValueError(
			"No VLM model specified. Set VLM_MODEL env var or pass model explicitly."
		)

	user_text = build_vlm_user_prompt(clean_question, frame_timestamps_sec)

	user_content: list[dict] = [{"type": "text", "text": user_text}]
	for i, frame in enumerate(frames, start=1):
		frame_id = f"F{i:02d}"
		if frame_timestamps_sec is None:
			frame_label = f"Frame {frame_id}"
		else:
			frame_label = f"Frame {frame_id} at t={frame_timestamps_sec[i - 1]:.2f}s"
		user_content.append({"type": "text", "text": frame_label})
		user_content.append(
			{
				"type": "image_url",
				"image_url": {"url": _image_to_data_url(frame)},
			}
		)

	payload = {
		"model": resolved_model,
		"messages": [
			{
				"role": "system",
				"content": _VLM_SYSTEM_PROMPT,
			},
			{
				"role": "user",
				"content": user_content,
			},
		],
		"temperature": 0.1,
		"max_tokens": 1500,
	}

	response_json = _post_chat_completion(payload, resolved_token, timeout_sec)
	text = _extract_text_from_response(response_json)
	structured = _extract_json_block(text)

	return {
		"n_frames": len(frames),
		"frame_timestamps_sec": frame_timestamps_sec,
		**structured,
	}


def describe_frame_with_vlm(
	frame: Image.Image,
	question: str,
	model: str | None = None,
	token: str | None = None,
	timeout_sec: int = 60,
) -> dict:
	"""Compatibility wrapper for single-frame callers.

	This now calls the multi-frame extractor with a single item list.
	"""
	multi = describe_frames_with_vlm(
		frames=[frame],
		question=question,
		frame_timestamps_sec=[0.0],
		model=model,
		token=token,
		timeout_sec=timeout_sec,
	)
	return multi
