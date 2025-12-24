"""Parser for Claude Code session files containing token usage data."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from ..utils.time_filter import TimeFilter


@dataclass
class TokenUsage:
    """Represents token usage for a single message."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens
        )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache tokens."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def cache_efficiency_percentage(self) -> float:
        """Calculate cache efficiency as percentage of cache reads."""
        total_cacheable = self.cache_creation_input_tokens + self.cache_read_input_tokens
        if total_cacheable == 0:
            return 0.0
        return (self.cache_read_input_tokens / total_cacheable) * 100


@dataclass
class SessionMessage:
    """Represents a single message in a session."""

    role: str
    timestamp: str
    session_id: str
    cwd: Optional[str] = None
    usage: Optional[TokenUsage] = None
    model: Optional[str] = None

    @property
    def datetime(self) -> Optional[datetime]:
        """Get datetime from ISO timestamp.

        Returns:
            datetime object or None if timestamp is invalid/empty
        """
        if not self.timestamp or not self.timestamp.strip():
            return None
        try:
            return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMessage":
        """Create SessionMessage from parsed JSON dict."""
        # Check for nested message structure (newer format)
        message = data.get("message", {})

        # Try to get usage from nested message first, then fall back to top level
        usage_data = message.get("usage") or data.get("usage")
        usage = None
        if usage_data:
            usage = TokenUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0)
            )

        # Get role from message or top level
        role = message.get("role") or data.get("role", "")

        # Get model from message or top level
        model = message.get("model") or data.get("model")

        return cls(
            role=role,
            timestamp=data.get("timestamp", ""),
            session_id=data.get("sessionId", ""),
            cwd=data.get("cwd"),
            usage=usage,
            model=model
        )


@dataclass
class SessionStats:
    """Aggregated statistics for session(s)."""

    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    message_count: int = 0
    session_ids: set[str] = field(default_factory=set)
    projects: set[str] = field(default_factory=set)
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    model_usage: dict[str, TokenUsage] = field(default_factory=dict)  # Tokens per model

    def add_message(self, message: SessionMessage):
        """Add a message to the statistics."""
        if message.usage:
            self.total_tokens += message.usage

            # Track tokens per model
            if message.model:
                if message.model not in self.model_usage:
                    self.model_usage[message.model] = TokenUsage()
                self.model_usage[message.model] += message.usage

        self.message_count += 1
        self.session_ids.add(message.session_id)

        if message.cwd:
            self.projects.add(message.cwd)

        # Only update timestamps if we have a valid datetime
        msg_dt = message.datetime
        if msg_dt is not None:
            if self.earliest_timestamp is None or msg_dt < self.earliest_timestamp:
                self.earliest_timestamp = msg_dt
            if self.latest_timestamp is None or msg_dt > self.latest_timestamp:
                self.latest_timestamp = msg_dt

    @property
    def session_count(self) -> int:
        """Number of unique sessions."""
        return len(self.session_ids)

    @property
    def project_count(self) -> int:
        """Number of unique projects."""
        return len(self.projects)


class SessionParser:
    """Parser for Claude Code session JSONL files."""

    def __init__(self, session_files: list[Path]):
        """
        Initialize session parser.

        Args:
            session_files: List of session JSONL file paths
        """
        self.session_files = session_files

    def parse_file(
        self,
        session_file: Path,
        time_filter: Optional[TimeFilter] = None
    ) -> Iterator[SessionMessage]:
        """
        Parse a single session file and yield messages.

        Args:
            session_file: Path to session JSONL file
            time_filter: Optional time filter

        Yields:
            SessionMessage instances
        """
        if not session_file.exists():
            return

        with open(session_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    message = SessionMessage.from_dict(data)

                    # Skip messages with invalid timestamps
                    if not message.timestamp or not message.timestamp.strip():
                        continue

                    # Apply time filter
                    if time_filter and not time_filter.matches_iso_string(message.timestamp):
                        continue

                    yield message

                except (json.JSONDecodeError, ValueError):
                    # Skip malformed lines
                    continue

    def parse_all(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None
    ) -> Iterator[SessionMessage]:
        """
        Parse all session files and yield messages.

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Yields:
            SessionMessage instances
        """
        for session_file in self.session_files:
            for message in self.parse_file(session_file, time_filter=time_filter):
                # Apply project filter
                if project_filter and message.cwd != project_filter:
                    continue

                yield message

    def get_stats(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None
    ) -> SessionStats:
        """
        Get aggregated statistics from all sessions.

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Returns:
            SessionStats with aggregated data
        """
        stats = SessionStats()

        for message in self.parse_all(time_filter=time_filter, project_filter=project_filter):
            stats.add_message(message)

        return stats

    def get_project_stats(
        self,
        time_filter: Optional[TimeFilter] = None
    ) -> dict[str, SessionStats]:
        """
        Get per-project statistics.

        Args:
            time_filter: Optional time filter

        Returns:
            Dict mapping project paths to SessionStats
        """
        project_stats: dict[str, SessionStats] = {}

        for message in self.parse_all(time_filter=time_filter):
            if not message.cwd:
                continue

            if message.cwd not in project_stats:
                project_stats[message.cwd] = SessionStats()

            project_stats[message.cwd].add_message(message)

        # Sort by total tokens (descending)
        return dict(
            sorted(
                project_stats.items(),
                key=lambda x: x[1].total_tokens.total_input_tokens + x[1].total_tokens.output_tokens,
                reverse=True
            )
        )

    def get_daily_stats(
        self,
        days: int = 7,
        time_filter: Optional[TimeFilter] = None
    ) -> dict[str, SessionStats]:
        """
        Get token statistics aggregated by day.

        Args:
            days: Number of days to include (default 7)
            time_filter: Optional time filter

        Returns:
            Dict mapping date strings (YYYY-MM-DD) to SessionStats
        """
        from datetime import timedelta

        # Initialize dict for each day
        daily_stats: dict[str, SessionStats] = {}
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for i in range(days, -1, -1):  # Include today
            day = today - timedelta(days=i)
            date_key = day.strftime('%Y-%m-%d')
            daily_stats[date_key] = SessionStats()

        # Parse messages and bucket by day
        for message in self.parse_all(time_filter=time_filter):
            msg_dt = message.datetime
            if msg_dt:
                date_key = msg_dt.strftime('%Y-%m-%d')
                if date_key in daily_stats:
                    daily_stats[date_key].add_message(message)

        return daily_stats
