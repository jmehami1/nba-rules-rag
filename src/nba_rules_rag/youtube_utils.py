"""YouTube URL parsing and timestamp validation utilities."""

import re
from urllib.parse import urlparse, parse_qs


def parse_youtube_url(url: str) -> str:
    """Extract and return the 11-character video ID from a YouTube URL.

    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID
        - http variants and mobile (m.youtube.com)

    Returns:
        The video ID string.

    Raises:
        ValueError: if the URL is not a recognised YouTube URL or has no valid video ID.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower().lstrip("www.")

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "m.youtube.com"):
        if parsed.path.startswith("/shorts/"):
            parts = parsed.path.split("/")
            video_id = parts[2] if len(parts) > 2 else ""
        else:
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [None])[0] or ""
    else:
        raise ValueError(f"Not a recognised YouTube URL: {url!r}")

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Could not extract a valid video ID from URL: {url!r}")

    return video_id


def parse_timestamp(ts: str) -> float:
    """Parse a timestamp string to total seconds.

    Accepts:
        - "SS" or "SS.s"
        - "MM:SS" or "MM:SS.s"
        - "HH:MM:SS" or "HH:MM:SS.s"

    Returns:
        Total seconds as a float.

    Raises:
        ValueError: if the format is not recognised or values are out of range.
    """
    parts = ts.strip().split(":")
    try:
        if len(parts) == 1:
            hours, minutes, seconds = 0, 0, float(parts[0])
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), float(parts[1])
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        else:
            raise ValueError(f"Unrecognised timestamp format: {ts!r}")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unrecognised timestamp format: {ts!r}") from exc

    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError(f"Timestamp values out of range: {ts!r}")

    return hours * 3600 + minutes * 60 + seconds


def validate_interval(start_ts: str, end_ts: str) -> tuple[float, float]:
    """Parse and validate a start/end timestamp pair.

    Returns:
        (start_sec, end_sec) as floats.

    Raises:
        ValueError: if timestamps are invalid or start >= end.
    """
    start_sec = parse_timestamp(start_ts)
    end_sec = parse_timestamp(end_ts)
    if start_sec >= end_sec:
        raise ValueError(
            f"start_time ({start_ts!r}) must be strictly before end_time ({end_ts!r})"
        )
    return start_sec, end_sec
