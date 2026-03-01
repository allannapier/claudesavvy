"""Utilities for filtering data by time ranges."""

from datetime import date, datetime, timedelta, timezone
from typing import Optional
from dateutil import parser as date_parser


def _local_midnight() -> datetime:
    """Return today's local midnight as a UTC-aware datetime."""
    now_local = datetime.now().astimezone()
    midnight_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_local.astimezone(timezone.utc)


def _last_monday_midnight() -> datetime:
    """Return the most recent Monday's local midnight as a UTC-aware datetime."""
    now_local = datetime.now().astimezone()
    days_since_monday = now_local.weekday()  # 0=Mon, 6=Sun
    monday_local = now_local - timedelta(days=days_since_monday)
    monday_midnight = monday_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return monday_midnight.astimezone(timezone.utc)


def _this_month_start() -> datetime:
    """Return the 1st of the current month at local midnight as a UTC-aware datetime."""
    now_local = datetime.now().astimezone()
    first_of_month = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return first_of_month.astimezone(timezone.utc)


class TimeFilter:
    """Handles time-based filtering for usage data."""

    def __init__(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        """
        Initialize time filter.

        Args:
            start_time: Start of time range (inclusive), UTC-aware
            end_time: End of time range (inclusive), UTC-aware
        """
        self.start_time = start_time
        self.end_time = end_time or datetime.now(timezone.utc)
        self._preset: Optional[str] = None  # set by from_preset for accurate descriptions

    @classmethod
    def from_preset(cls, preset: str) -> "TimeFilter":
        """
        Create a TimeFilter from a preset string.

        Args:
            preset: One of '15min', 'today', 'this_week', '7days', 'this_month',
                    '3months', 'quarter', 'year', or 'all'

        Returns:
            TimeFilter instance
        """
        now = datetime.now(timezone.utc)

        instance: "TimeFilter"
        if preset == "15min":
            instance = cls(start_time=now - timedelta(minutes=15), end_time=now)
        elif preset == "today":
            instance = cls(start_time=_local_midnight(), end_time=now)
        elif preset == "this_week":
            instance = cls(start_time=_last_monday_midnight(), end_time=now)
        elif preset == "7days":
            instance = cls(start_time=now - timedelta(days=7), end_time=now)
        elif preset == "this_month":
            instance = cls(start_time=_this_month_start(), end_time=now)
        elif preset == "3months":
            instance = cls(start_time=now - timedelta(days=91), end_time=now)
        elif preset == "quarter":
            now_local = datetime.now().astimezone()
            quarter_start_month = ((now_local.month - 1) // 3) * 3 + 1
            quarter_start_local = now_local.replace(
                month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            instance = cls(start_time=quarter_start_local.astimezone(timezone.utc), end_time=now)
        elif preset == "year":
            instance = cls(start_time=now - timedelta(days=365), end_time=now)
        elif preset == "all":
            instance = cls(start_time=None, end_time=now)
        else:
            raise ValueError(f"Unknown preset: {preset}")
        instance._preset = preset
        return instance

    @classmethod
    def from_since(cls, since_str: str) -> "TimeFilter":
        """
        Create a TimeFilter from a 'since' date string.

        Args:
            since_str: Date string in various formats (YYYY-MM-DD, etc.)

        Returns:
            TimeFilter instance
        """
        now = datetime.now(timezone.utc)
        start_time = date_parser.parse(since_str)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        return cls(start_time=start_time, end_time=now)

    @classmethod
    def from_range(cls, start: date, end: date) -> "TimeFilter":
        """
        Create a TimeFilter from an explicit date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive, extended to end of day)

        Returns:
            TimeFilter instance

        Raises:
            ValueError: If end is before start or range exceeds 2 years
        """
        if end < start:
            raise ValueError("end date must be on or after start date")
        if (end - start).days > 730:
            raise ValueError("date range cannot exceed 2 years")

        # Start of start day in UTC; end of end day in UTC (23:59:59.999999, inclusive)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, 999999, tzinfo=timezone.utc)

        return cls(start_time=start_dt, end_time=end_dt)

    def matches_timestamp_ms(self, timestamp_ms: int) -> bool:
        """
        Check if a millisecond timestamp falls within the filter range.

        Args:
            timestamp_ms: Timestamp in milliseconds since epoch

        Returns:
            True if timestamp is within range
        """
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return self.matches_datetime(dt)

    def matches_datetime(self, dt: datetime) -> bool:
        """
        Check if a datetime falls within the filter range.

        Args:
            dt: Datetime to check (naive datetimes are assumed to be UTC)

        Returns:
            True if datetime is within range
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

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

    def get_previous_period(self) -> "TimeFilter":
        """
        Get a TimeFilter for the equivalent previous period.

        Returns:
            TimeFilter for the previous period
        """
        if self.start_time is None:
            return TimeFilter(start_time=None, end_time=None)

        duration = self.end_time - self.start_time
        prev_end = self.start_time
        prev_start = prev_end - duration

        return TimeFilter(start_time=prev_start, end_time=prev_end)

    def get_description(self) -> str:
        """Get a human-readable description of the time range."""
        if self.start_time is None:
            return "All time"

        _preset_labels = {
            "15min": "Last 15 minutes",
            "today": "Today",
            "this_week": "This week",
            "7days": "Last 7 days",
            "this_month": "This month",
            "3months": "Last 3 months",
            "quarter": "This quarter",
            "year": "Last year",
            "all": "All time",
        }
        if self._preset and self._preset in _preset_labels:
            return _preset_labels[self._preset]

        # Custom date range: use actual date strings
        start_str = self.start_time.strftime('%Y-%m-%d')
        end_str = self.end_time.strftime('%Y-%m-%d')
        if start_str == end_str:
            return f"Since {start_str}"
        return f"{start_str} – {end_str}"
