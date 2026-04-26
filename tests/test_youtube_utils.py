"""Tests for YouTube URL parsing and timestamp validation."""

import pytest

from nba_rules_rag.youtube_utils import (
    parse_youtube_url,
    parse_timestamp,
    validate_interval,
)


class TestParseYoutubeUrl:
    """Test YouTube URL parsing."""

    def test_standard_youtube_url(self):
        """Parse standard youtube.com watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert parse_youtube_url(url) == "dQw4w9WgXcQ"

    def test_youtu_be_short_url(self):
        """Parse youtu.be shortened URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert parse_youtube_url(url) == "dQw4w9WgXcQ"

    def test_youtube_shorts_url(self):
        """Parse YouTube Shorts URL."""
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert parse_youtube_url(url) == "dQw4w9WgXcQ"

    def test_mobile_url(self):
        """Parse mobile YouTube URL."""
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert parse_youtube_url(url) == "dQw4w9WgXcQ"

    def test_http_url(self):
        """Parse http (not https) YouTube URL."""
        url = "http://youtube.com/watch?v=dQw4w9WgXcQ"
        assert parse_youtube_url(url) == "dQw4w9WgXcQ"

    def test_invalid_url_domain(self):
        """Raise ValueError for non-YouTube URL."""
        with pytest.raises(ValueError, match="Not a recognised YouTube URL"):
            parse_youtube_url("https://vimeo.com/123456789")

    def test_invalid_url_no_video_id(self):
        """Raise ValueError when no video ID present."""
        with pytest.raises(ValueError, match="Could not extract a valid video ID"):
            parse_youtube_url("https://www.youtube.com/watch?v=")

    def test_invalid_video_id_format(self):
        """Raise ValueError for invalid video ID format (must be 11 chars)."""
        with pytest.raises(ValueError, match="Could not extract a valid video ID"):
            parse_youtube_url("https://www.youtube.com/watch?v=toolong123456789")

    def test_valid_video_id_chars(self):
        """Accept 11-char video IDs with letters, numbers, underscore, dash."""
        # All valid character classes should work
        url = "https://www.youtube.com/watch?v=abcDEF_-123"
        assert parse_youtube_url(url) == "abcDEF_-123"


class TestParseTimestamp:
    """Test timestamp parsing."""

    def test_seconds_only(self):
        """Parse single seconds value."""
        assert parse_timestamp("30") == 30.0

    def test_seconds_with_decimal(self):
        """Parse seconds with decimal point."""
        assert parse_timestamp("30.5") == 30.5

    def test_minutes_seconds(self):
        """Parse MM:SS format."""
        assert parse_timestamp("01:30") == 90.0

    def test_minutes_seconds_with_decimal(self):
        """Parse MM:SS.s format."""
        assert parse_timestamp("01:30.5") == 90.5

    def test_hours_minutes_seconds(self):
        """Parse HH:MM:SS format."""
        assert parse_timestamp("1:05:30") == 3930.0

    def test_hours_minutes_seconds_with_decimal(self):
        """Parse HH:MM:SS.s format."""
        assert parse_timestamp("1:05:30.5") == 3930.5

    def test_zero_timestamp(self):
        """Parse zero timestamp."""
        assert parse_timestamp("00:00") == 0.0

    def test_leading_zeros(self):
        """Handle leading zeros correctly."""
        assert parse_timestamp("00:05") == 5.0

    def test_minutes_out_of_range(self):
        """Raise ValueError for minutes >= 60."""
        with pytest.raises(ValueError, match="Timestamp values out of range"):
            parse_timestamp("01:60")

    def test_seconds_out_of_range(self):
        """Raise ValueError for seconds >= 60."""
        with pytest.raises(ValueError, match="Timestamp values out of range"):
            parse_timestamp("01:30:60")

    def test_invalid_format_too_many_colons(self):
        """Raise ValueError for invalid format with too many colons."""
        with pytest.raises(ValueError, match="Unrecognised timestamp format"):
            parse_timestamp("1:2:3:4")

    def test_invalid_format_non_numeric(self):
        """Raise ValueError for non-numeric components."""
        with pytest.raises(ValueError, match="Unrecognised timestamp format"):
            parse_timestamp("1:2a:30")

    def test_timestamp_with_whitespace(self):
        """Handle leading/trailing whitespace."""
        assert parse_timestamp("  00:41  ") == 41.0


class TestValidateInterval:
    """Test start/end timestamp interval validation."""

    def test_valid_interval(self):
        """Return both timestamps as seconds for valid interval."""
        start, end = validate_interval("00:10", "00:15")
        assert start == 10.0
        assert end == 15.0

    def test_valid_interval_with_decimals(self):
        """Valid interval with decimal seconds."""
        start, end = validate_interval("00:10.5", "00:15.5")
        assert start == 10.5
        assert end == 15.5

    def test_start_equals_end(self):
        """Raise ValueError when start equals end."""
        with pytest.raises(
            ValueError,
            match="start_time.*must be strictly before end_time",
        ):
            validate_interval("00:10", "00:10")

    def test_start_after_end(self):
        """Raise ValueError when start is after end."""
        with pytest.raises(
            ValueError,
            match="start_time.*must be strictly before end_time",
        ):
            validate_interval("00:15", "00:10")

    def test_invalid_start_timestamp(self):
        """Raise ValueError if start timestamp is invalid."""
        with pytest.raises(ValueError):
            validate_interval("invalid", "00:10")

    def test_invalid_end_timestamp(self):
        """Raise ValueError if end timestamp is invalid."""
        with pytest.raises(ValueError):
            validate_interval("00:10", "invalid")
