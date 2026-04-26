"""Frame extraction from a YouTube video segment using yt-dlp and OpenCV."""

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
from PIL import Image


def _resize_to_360p(rgb_frame) -> Image.Image:
    """Resize an RGB numpy frame to max height 360 while preserving aspect ratio."""
    h, w = rgb_frame.shape[:2]
    target_h = 360
    if h <= target_h:
        return Image.fromarray(rgb_frame)

    scale = target_h / float(h)
    target_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def download_video_segment(
    video_id: str,
    start_sec: float,
    end_sec: float,
    output_dir: str | None = None,
) -> str:
    """Download a section of a YouTube video using yt-dlp.

    Args:
        video_id: 11-character YouTube video ID.
        start_sec: Clip start in seconds.
        end_sec: Clip end in seconds.
        output_dir: Directory to save the file. A temp directory is used if None.

    Returns:
        Absolute path to the downloaded .mp4 file.

    Raises:
        RuntimeError: if yt-dlp exits with a non-zero return code.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    output_path = str(Path(output_dir) / f"{video_id}.mp4")
    url = f"https://www.youtube.com/watch?v={video_id}"

    # yt-dlp section syntax: *start_sec-end_sec
    section = f"*{start_sec}-{end_sec}"

    cmd = [
        "yt-dlp",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "-o", output_path,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n{result.stderr}"
        )

    return output_path


def extract_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = 8,
) -> list[Image.Image]:
    """Sample n_frames evenly from a local video file over [start_sec, end_sec].

    Args:
        video_path: Path to the video file.
        start_sec: Start of the sampling interval (seconds from start of file).
        end_sec: End of the sampling interval (seconds from start of file).
        n_frames: Target number of frames; clamped to the range [5, 10].

    Returns:
        List of PIL Images in chronological order.

    Raises:
        FileNotFoundError: if video_path does not exist.
        ValueError: if the video cannot be opened or no frames can be read.
    """
    n_frames = max(5, min(10, n_frames))

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    duration = end_sec - start_sec
    # Evenly spaced sample times within the interval (inclusive of both endpoints)
    if n_frames == 1:
        sample_times = [start_sec]
    else:
        step = duration / (n_frames - 1)
        sample_times = [start_sec + i * step for i in range(n_frames)]

    frames: list[Image.Image] = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(_resize_to_360p(rgb))

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path!r}")

    return frames


def extract_frames_from_youtube(
    video_id: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = 8,
    output_dir: str | None = None,
) -> list[Image.Image]:
    """Download a YouTube video segment and return evenly sampled frames.

    This is the main entry point that combines download and frame extraction.

    Args:
        video_id: 11-character YouTube video ID.
        start_sec: Clip start in seconds.
        end_sec: Clip end in seconds.
        n_frames: Number of frames to sample (clamped to 5-10).
        output_dir: Optional directory for the downloaded video file.

    Returns:
        List of PIL Images.
    """
    video_path = download_video_segment(video_id, start_sec, end_sec, output_dir)
    # yt-dlp --download-sections trims the file to the section, so it starts at t=0
    duration = end_sec - start_sec
    return extract_frames(video_path, 0.0, duration, n_frames)
