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


def _is_openai_endpoint(api_base: str) -> bool:
	"""Return True if the configured endpoint appears to be OpenAI-hosted."""
	return "api.openai.com" in (api_base or "").lower()


def _resolve_vlm_token(explicit_token: str | None, api_base: str) -> str | None:
	"""Resolve API token based on endpoint.

	For OpenAI-hosted endpoints, only OPENAI_API_KEY is accepted to avoid
	accidentally sending an HF token to OpenAI.
	"""
	if explicit_token:
		return explicit_token

	openai_key = os.getenv("OPENAI_API_KEY")
	if _is_openai_endpoint(api_base):
		return openai_key

	return openai_key or os.getenv("HF_TOKEN")


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

	first_choice = choices[0]
	message = first_choice.get("message", {})
	content = message.get("content")

	# Some OpenAI-compatible providers return plain text at choice level.
	choice_text = first_choice.get("text")
	if isinstance(choice_text, str) and choice_text.strip():
		return choice_text.strip()

	if isinstance(content, str):
		if content.strip():
			return content.strip()
		refusal = message.get("refusal")
		if isinstance(refusal, str) and refusal.strip():
			return refusal.strip()
		return ""

	if isinstance(content, list):
		text_parts: list[str] = []
		for part in content:
			if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
				text = part.get("text", "")
				if text:
					text_parts.append(str(text).strip())
		joined = "\n".join([p for p in text_parts if p])
		if joined:
			return joined
		refusal = message.get("refusal")
		if isinstance(refusal, str) and refusal.strip():
			return refusal.strip()
		return ""

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


def _fallback_structured_response(question: str, text: str) -> dict:
	"""Build a minimal structured response when the model returns plain text.

	This keeps the pipeline running even when a model ignores the JSON-only instruction.
	"""
	clean_text = (text or "").strip()
	if not clean_text:
		clean_text = "VLM returned empty content; no structured JSON could be parsed."

	return {
		"question": question,
		"play_narration": clean_text,
		"frame_observations": [],
		"query_relevant_signals": [],
		"uncertainties": [
			"Model response was plain text instead of JSON; parsed using fallback wrapper.",
		],
		"parse_warning": "VLM response was not valid JSON; fallback wrapper applied.",
	}


def _post_chat_completion(payload: dict, token: str, timeout_sec: int) -> dict:
	"""Post payload to OpenAI-compatible chat completion endpoint."""
	api_base = os.getenv("VLM_API_BASE", DEFAULT_VLM_API_BASE)

	def _do_post(request_payload: dict) -> dict:
		body = json.dumps(request_payload).encode("utf-8")
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

	try:
		return _do_post(payload)
	except RuntimeError as exc:
		err = str(exc).lower()

		# Some models reject reasoning_effort="none"; try "low" first.
		if (
			"reasoning_effort" in err
			and "unsupported" in err
			and payload.get("reasoning_effort") == "none"
		):
			retry_payload = {**payload, "reasoning_effort": "low"}
			return _do_post(retry_payload)

		# If reasoning_effort itself is unsupported, remove it.
		if (
			"reasoning_effort" in err
			and "unsupported" in err
			and "reasoning_effort" in payload
		):
			retry_payload = {**payload}
			retry_payload.pop("reasoning_effort", None)
			return _do_post(retry_payload)

		# Some providers do not support response_format.
		if (
			"unsupported_parameter" in err
			and "response_format" in err
			and "response_format" in payload
		):
			retry_payload = {**payload}
			retry_payload.pop("response_format", None)
			return _do_post(retry_payload)

		# Some models reject max_completion_tokens and require max_tokens.
		if (
			"unsupported_parameter" in err
			and "max_completion_tokens" in err
			and "max_completion_tokens" in payload
		):
			retry_payload = {
				**payload,
				"max_tokens": payload["max_completion_tokens"],
			}
			retry_payload.pop("max_completion_tokens", None)
			return _do_post(retry_payload)

		# If max_completion_tokens is unsupported and max_tokens fallback fails,
		# try once without explicit completion token caps.
		if (
			"unsupported_parameter" in err
			and "max_completion_tokens" in err
			and "max_completion_tokens" in payload
		):
			retry_payload = {**payload}
			retry_payload.pop("max_completion_tokens", None)
			return _do_post(retry_payload)

		# Some models reject max_tokens and require max_completion_tokens.
		if (
			"unsupported_parameter" in err
			and "max_tokens" in err
			and "max_tokens" in payload
		):
			retry_payload = {
				**payload,
				"max_completion_tokens": payload["max_tokens"],
			}
			retry_payload.pop("max_tokens", None)
			return _do_post(retry_payload)
		raise


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

	api_base = os.getenv("VLM_API_BASE", DEFAULT_VLM_API_BASE)
	resolved_token = _resolve_vlm_token(token, api_base)
	if not resolved_token:
		if _is_openai_endpoint(api_base):
			raise RuntimeError("Missing API token. Set OPENAI_API_KEY env var, or pass token explicitly.")
		raise RuntimeError("Missing API token. Set OPENAI_API_KEY or HF_TOKEN env var, or pass token explicitly.")

	resolved_model = model or os.getenv("VLM_MODEL")
	if not resolved_model:
		raise ValueError(
			"No VLM model specified. Set VLM_MODEL env var or pass model explicitly."
		)

	reasoning_effort = (os.getenv("VLM_REASONING_EFFORT", "none") or "none").strip()
	max_completion_tokens_raw = (os.getenv("VLM_MAX_COMPLETION_TOKENS", "2048") or "2048").strip()
	try:
		max_completion_tokens = int(max_completion_tokens_raw)
	except ValueError as exc:
		raise ValueError(
			f"Invalid VLM_MAX_COMPLETION_TOKENS value: {max_completion_tokens_raw!r}. Expected integer."
		) from exc
	if max_completion_tokens <= 0:
		raise ValueError("VLM_MAX_COMPLETION_TOKENS must be a positive integer.")

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
		"response_format": {"type": "json_object"},
		"reasoning_effort": reasoning_effort,
		"max_completion_tokens": max_completion_tokens,
	}

	response_json = _post_chat_completion(payload, resolved_token, timeout_sec)
	text = _extract_text_from_response(response_json)
	try:
		structured = _extract_json_block(text)
	except RuntimeError:
		structured = _fallback_structured_response(clean_question, text)

	return {
		"n_frames": len(frames),
		"frame_timestamps_sec": frame_timestamps_sec,
		"_raw_vlm_response_text": text,
		"_raw_vlm_response_json": response_json,
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
