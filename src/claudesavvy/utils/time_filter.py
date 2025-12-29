"""Utilities for filtering data by time ranges."""

from datetime import datetime, timedelta
from typing import Optional
from dateutil import parser as date_parser


class TimeFilter:
    """Handles time-based filtering for usage data."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """
        Initialize time filter.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
        """
        self.start_time = start_time
        self.end_time = end_time or datetime.now()

    @classmethod
    def from_preset(cls, preset: str) -> "TimeFilter":
        """
        Create a TimeFilter from a preset string.

        Args:
            preset: One of 'today', 'week', 'month', 'quarter', 'year', or 'all'

        Returns:
            TimeFilter instance
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if preset == "today":
            return cls(start_time=today_start, end_time=now)
        elif preset == "week":
            week_start = today_start - timedelta(days=7)
            return cls(start_time=week_start, end_time=now)
        elif preset == "month":
            month_start = today_start - timedelta(days=30)
            return cls(start_time=month_start, end_time=now)
        elif preset == "quarter":
            # Calculate start of current quarter
            # Q1: Jan-Mar (1-3), Q2: Apr-Jun (4-6), Q3: Jul-Sep (7-9), Q4: Oct-Dec (10-12)
            current_month = now.month
            quarter_start_month = ((current_month - 1) // 3) * 3 + 1
            quarter_start = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            return cls(start_time=quarter_start, end_time=now)
        elif preset == "year":
            year_start = today_start - timedelta(days=365)
            return cls(start_time=year_start, end_time=now)
        elif preset == "all":
            return cls(start_time=None, end_time=now)
        else:
            raise ValueError(f"Unknown preset: {preset}")

    @classmethod
    def from_since(cls, since_str: str) -> "TimeFilter":
        """
        Create a TimeFilter from a 'since' date string.

        Args:
            since_str: Date string in various formats (YYYY-MM-DD, etc.)

        Returns:
            TimeFilter instance
        """
        start_time = date_parser.parse(since_str)
        return cls(start_time=start_time, end_time=datetime.now())

    def matches_timestamp_ms(self, timestamp_ms: int) -> bool:
        """
        Check if a millisecond timestamp falls within the filter range.

        Args:
            timestamp_ms: Timestamp in milliseconds since epoch

        Returns:
            True if timestamp is within range
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return self.matches_datetime(dt)

    def matches_datetime(self, dt: datetime) -> bool:
        """
        Check if a datetime falls within the filter range.

        Args:
            dt: Datetime to check

        Returns:
            True if datetime is within range
        """
        # Convert to naive datetime for comparison (remove timezone info)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        if self.start_time and dt < self.start_time:
            return False
        if self.end_time and dt > self.end_time:
            return False
        return True

    def matches_iso_string(self, iso_str: str) -> bool:
        """
        Check if an ISO format timestamp string falls within the filter range.

        Args:
            iso_str: ISO format timestamp string

        Returns:
            True if timestamp is within range
        """
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return self.matches_datetime(dt)

    def get_description(self) -> str:
        """Get a human-readable description of the time range."""
        if self.start_time is None:
            return "All time"

        now = datetime.now()
        delta = now - self.start_time

        # Check if it's current quarter
        current_month = now.month
        quarter_start_month = ((current_month - 1) // 3) * 3 + 1
        quarter_start = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)

        if self.start_time == quarter_start:
            quarter_num = (quarter_start_month - 1) // 3 + 1
            return f"Q{quarter_num} {now.year}"
        elif delta.days == 0:
            return "Today"
        elif delta.days == 7:
            return "Last 7 days"
        elif delta.days == 30:
            return "Last 30 days"
        elif delta.days == 365:
            return "Last year"
        else:
            return f"Since {self.start_time.strftime('%Y-%m-%d')}"
