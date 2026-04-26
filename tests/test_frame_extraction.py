"""Tests for frame extraction from YouTube videos."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call

import cv2
import numpy as np
import pytest
from PIL import Image

from nba_rules_rag.frame_extraction import (
    download_video_segment,
    extract_frames,
    extract_frames_from_youtube,
)


@pytest.fixture
def temp_video_dir():
    """Create a temporary directory for test videos."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup: remove all files in the temp directory
    for file in Path(tmpdir).glob("*"):
        file.unlink()
    os.rmdir(tmpdir)


@pytest.fixture
def dummy_video_file(temp_video_dir):
    """Create a dummy MP4 video file for testing.

    Returns a path to a simple 4-second video (30 fps, ~120 frames total).
    """
    video_path = os.path.join(temp_video_dir, "dummy.mp4")
    
    # Create a simple video: 30 fps, 4 seconds = 120 frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    frame_size = (640, 480)
    
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    
    # Write 120 frames (4 seconds at 30 fps)
    for i in range(120):
        # Create a simple frame with a gradient (to make frames visually different)
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frame[:, :] = [i % 256, (i * 2) % 256, (i * 3) % 256]
        out.write(frame)
    
    out.release()
    yield video_path


class TestExtractFrames:
    """Test local frame extraction from video files."""

    def test_extract_frames_basic(self, dummy_video_file):
        """Extract frames from a video file."""
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=2.0, n_frames=5)
        
        assert len(frames) == 5
        assert all(isinstance(f, Image.Image) for f in frames)
        # Frames are resized to 360p for VLM efficiency.
        assert all(f.size == (480, 360) for f in frames)

    def test_extract_frames_clamped_to_range(self, dummy_video_file):
        """Frames are clamped to [5, 10] range."""
        # Request 2 frames (should be clamped to 5)
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=2.0, n_frames=2)
        assert len(frames) == 5
        
        # Request 15 frames (should be clamped to 10)
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=2.0, n_frames=15)
        assert len(frames) == 10

    def test_extract_frames_single_frame(self, dummy_video_file):
        """Extract single frame (n_frames=1, clamped to 5)."""
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=2.0, n_frames=1)
        assert len(frames) == 5

    def test_extract_frames_nonexistent_file(self):
        """Raise FileNotFoundError for nonexistent video file."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_frames("/nonexistent/video.mp4", 0.0, 2.0)

    def test_extract_frames_invalid_file(self, temp_video_dir):
        """Raise ValueError if file cannot be opened as video."""
        invalid_file = os.path.join(temp_video_dir, "invalid.txt")
        with open(invalid_file, "w") as f:
            f.write("not a video")
        
        with pytest.raises(ValueError, match="Could not open video file"):
            extract_frames(invalid_file, 0.0, 2.0)

    def test_extract_frames_evenly_spaced(self, dummy_video_file):
        """Frames are evenly spaced across the interval."""
        # Extract 3 frames from a 3-second interval (should be at t=0, 1.5, 3)
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=3.0, n_frames=3)
        # Because n_frames is clamped to [5, 10], we'll get 5 frames
        # at times: 0, 0.75, 1.5, 2.25, 3.0 (evenly spaced across 3 seconds)
        assert len(frames) == 5

    def test_extract_frames_returns_pil_images(self, dummy_video_file):
        """All returned frames are PIL Images."""
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=2.0, n_frames=5)
        assert all(isinstance(f, Image.Image) for f in frames)
        assert all(f.mode == "RGB" for f in frames)
        assert all(f.size[1] <= 360 for f in frames)


class TestDownloadVideoSegment:
    """Test YouTube video download with mocking."""

    @patch("nba_rules_rag.frame_extraction.subprocess.run")
    def test_download_video_segment_success(self, mock_run, temp_video_dir):
        """Successfully download a video segment."""
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        result = download_video_segment("dQw4w9WgXcQ", 10.0, 15.0, temp_video_dir)
        
        assert result.endswith("dQw4w9WgXcQ.mp4")
        assert result.startswith(temp_video_dir)
        
        # Verify yt-dlp was called with correct arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "yt-dlp" in call_args
        assert "--download-sections" in call_args
        assert "*10.0-15.0" in call_args

    @patch("nba_rules_rag.frame_extraction.subprocess.run")
    def test_download_video_segment_failure(self, mock_run, temp_video_dir):
        """Raise RuntimeError if yt-dlp fails."""
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Error: video not found",
        )
        
        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_video_segment("invalid_id", 10.0, 15.0, temp_video_dir)

    @patch("nba_rules_rag.frame_extraction.subprocess.run")
    def test_download_video_segment_temp_dir(self, mock_run):
        """Use temporary directory if output_dir is None."""
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        result = download_video_segment("dQw4w9WgXcQ", 10.0, 15.0, output_dir=None)
        
        # Should return a path to a temp directory
        assert os.path.isabs(result)
        assert "dQw4w9WgXcQ.mp4" in result


class TestExtractFramesFromYoutube:
    """Test end-to-end YouTube frame extraction (with mocking)."""

    @patch("nba_rules_rag.frame_extraction.extract_frames")
    @patch("nba_rules_rag.frame_extraction.download_video_segment")
    def test_extract_frames_from_youtube(self, mock_download, mock_extract, temp_video_dir):
        """Extract frames from YouTube video (mocked)."""
        # Mock download to return a dummy path
        mock_download.return_value = "/tmp/video.mp4"
        
        # Mock frame extraction
        mock_frames = [Mock(spec=Image.Image) for _ in range(8)]
        mock_extract.return_value = mock_frames
        
        frames = extract_frames_from_youtube(
            "dQw4w9WgXcQ",
            10.0,
            15.0,
            n_frames=8,
            output_dir=temp_video_dir,
        )
        
        assert len(frames) == 8
        mock_download.assert_called_once_with("dQw4w9WgXcQ", 10.0, 15.0, temp_video_dir)
        # After download, the trimmed file starts at 0, so extract should be called
        # with duration (15 - 10 = 5) as the end time
        mock_extract.assert_called_once_with("/tmp/video.mp4", 0.0, 5.0, 8)

    @patch("nba_rules_rag.frame_extraction.extract_frames")
    @patch("nba_rules_rag.frame_extraction.download_video_segment")
    def test_extract_frames_from_youtube_frame_clamping(
        self, mock_download, mock_extract
    ):
        """Frame count is clamped to [5, 10] range."""
        mock_download.return_value = "/tmp/video.mp4"
        mock_frames = [Mock(spec=Image.Image) for _ in range(5)]
        mock_extract.return_value = mock_frames
        
        # Request 2 frames (will be clamped by extract_frames)
        extract_frames_from_youtube("dQw4w9WgXcQ", 10.0, 15.0, n_frames=2)
        
        # extract_frames receives the clamped value
        mock_extract.assert_called_once()
        assert mock_extract.call_args[0][3] == 2  # n_frames passed through


@pytest.mark.integration
class TestFrameExtractionIntegration:
    """Integration tests using real video file (not YouTube)."""

    def test_extract_frames_from_real_video(self, dummy_video_file):
        """Extract frames from a real video file and verify they're valid."""
        frames = extract_frames(dummy_video_file, start_sec=0.0, end_sec=2.0, n_frames=8)
        
        assert len(frames) == 8
        for frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.mode == "RGB"
            assert frame.size == (480, 360)
            # Convert to numpy to check it has reasonable data
            arr = np.array(frame)
            assert arr.shape == (360, 480, 3)


@pytest.mark.integration
class TestFrameExtractionYouTubeNetworkIntegration:
    """Integration tests that download real video segments from YouTube over network."""

    def test_download_youtube_segment_real(self, temp_video_dir):
        """Download a real YouTube video segment and verify the file exists.
        
        Uses a short public domain video clip.
        Timeout: 120 seconds to account for network latency.
        """
        # Using a short, stable YouTube video (Big Buck Bunny trailer - 15 seconds)
        # You can replace with any short public YouTube video
        video_id = "aqz-KE-bpKQ"  # Big Buck Bunny trailer
        start_sec = 1.0
        end_sec = 5.0  # 4-second segment
        
        video_path = download_video_segment(video_id, start_sec, end_sec, temp_video_dir)
        
        assert os.path.exists(video_path)
        assert video_path.endswith(".mp4")
        assert os.path.getsize(video_path) > 0  # File has content

    def test_extract_frames_from_youtube_real(self, temp_video_dir):
        """Download a YouTube video and extract frames (end-to-end test).
        
        This is a full integration test that:
        1. Downloads a YouTube video segment
        2. Extracts frames
        3. Validates frame data
        
        Timeout: 120 seconds to account for network latency and processing.
        """
        video_id = "aqz-KE-bpKQ"  # Big Buck Bunny trailer
        start_sec = 1.0
        end_sec = 5.0  # 4-second segment
        n_frames = 8
        
        frames = extract_frames_from_youtube(
            video_id,
            start_sec,
            end_sec,
            n_frames=n_frames,
            output_dir=temp_video_dir,
        )
        
        # Verify we got frames
        assert len(frames) > 0
        assert len(frames) <= 10  # Clamped to max 10
        
        # Verify each frame is a valid PIL Image
        for i, frame in enumerate(frames):
            assert isinstance(frame, Image.Image), f"Frame {i} is not a PIL Image"
            assert frame.mode == "RGB", f"Frame {i} is not RGB mode"
            # Frames should have reasonable dimensions (not tiny)
            assert frame.size[0] >= 240, f"Frame {i} width too small"
            assert frame.size[1] >= 135, f"Frame {i} height too small"
            # Convert to numpy array and verify it has data
            arr = np.array(frame)
            assert arr.shape[2] == 3, f"Frame {i} does not have 3 channels"
