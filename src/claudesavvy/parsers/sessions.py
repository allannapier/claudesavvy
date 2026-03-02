"""Parser for Claude Code session files containing token usage data."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, List, Set

from ..utils.time_filter import TimeFilter


@dataclass
class TokenUsage:
    """Represents token usage for a single message."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    tool_output_tokens: int = 0  # Tokens from persisted tool outputs (Claude Code 2.1.0+)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            tool_output_tokens=self.tool_output_tokens + other.tool_output_tokens,
        )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache tokens and tool outputs."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.tool_output_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Total all token types."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.tool_output_tokens
        )

    @property
    def cache_efficiency_percentage(self) -> float:
        """Calculate cache efficiency as percentage of cache reads."""
        total_cacheable = (
            self.cache_creation_input_tokens + self.cache_read_input_tokens
        )
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
    team_name: Optional[str] = None

    @property
    def datetime(self) -> Optional[datetime]:
        """Get datetime from ISO timestamp.

        Returns:
            datetime object or None if timestamp is invalid/empty
        """
        if not self.timestamp or not self.timestamp.strip():
            return None
        try:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
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
                cache_creation_input_tokens=usage_data.get(
                    "cache_creation_input_tokens", 0
                ),
                cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
            )

        # Get role from message or top level
        role = message.get("role") or data.get("role", "")

        # Get model from message or top level
        model = message.get("model") or data.get("model")

        # Get team name from top level (present when Teams feature is invoked)
        team_name = data.get("teamName")

        return cls(
            role=role,
            timestamp=data.get("timestamp", ""),
            session_id=data.get("sessionId", ""),
            cwd=data.get("cwd"),
            usage=usage,
            model=model,
            team_name=team_name,
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
    team_usage: dict[str, TokenUsage] = field(default_factory=dict)  # Tokens per team

    def add_message(self, message: SessionMessage):
        """Add a message to the statistics."""
        if message.usage:
            self.total_tokens += message.usage

            # Track tokens per model
            if message.model:
                if message.model not in self.model_usage:
                    self.model_usage[message.model] = TokenUsage()
                self.model_usage[message.model] += message.usage

            # Track tokens per team
            if message.team_name:
                if message.team_name not in self.team_usage:
                    self.team_usage[message.team_name] = TokenUsage()
                self.team_usage[message.team_name] += message.usage

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
        self, session_file: Path, time_filter: Optional[TimeFilter] = None
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

        with open(session_file, "r", encoding="utf-8") as f:
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
                    if time_filter and not time_filter.matches_iso_string(
                        message.timestamp
                    ):
                        continue

                    yield message

                except (json.JSONDecodeError, ValueError):
                    # Skip malformed lines
                    continue

    def parse_all(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
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
        project_filter: Optional[str] = None,
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

        for message in self.parse_all(
            time_filter=time_filter, project_filter=project_filter
        ):
            stats.add_message(message)

        return stats

    def get_project_stats(
        self, time_filter: Optional[TimeFilter] = None
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
                key=lambda x: x[1].total_tokens.total_input_tokens
                + x[1].total_tokens.output_tokens,
                reverse=True,
            )
        )

    def get_team_stats(
        self, time_filter: Optional[TimeFilter] = None
    ) -> dict[str, SessionStats]:
        """
        Get per-team statistics when Teams feature is invoked.

        Args:
            time_filter: Optional time filter

        Returns:
            Dict mapping team names to SessionStats
        """
        team_stats: dict[str, SessionStats] = {}

        for message in self.parse_all(time_filter=time_filter):
            if not message.team_name:
                continue

            if message.team_name not in team_stats:
                team_stats[message.team_name] = SessionStats()

            team_stats[message.team_name].add_message(message)

        # Sort by total tokens (descending)
        return dict(
            sorted(
                team_stats.items(),
                key=lambda x: x[1].total_tokens.total_input_tokens
                + x[1].total_tokens.output_tokens,
                reverse=True,
            )
        )

    def get_daily_stats(
        self, days: int = 7, time_filter: Optional[TimeFilter] = None
    ) -> dict[str, SessionStats]:
        """
        Get token statistics aggregated by day.

        Args:
            days: Number of days to include (default 7), including today
            time_filter: Optional time filter

        Returns:
            Dict mapping date strings (YYYY-MM-DD) to SessionStats
        """
        from datetime import timedelta

        # Initialize dict for each day
        daily_stats: dict[str, SessionStats] = {}
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for i in range(days - 1, -1, -1):  # days entries, including today
            day = today - timedelta(days=i)
            date_key = day.strftime("%Y-%m-%d")
            daily_stats[date_key] = SessionStats()

        # Parse messages and bucket by day
        for message in self.parse_all(time_filter=time_filter):
            msg_dt = message.datetime
            if msg_dt:
                date_key = msg_dt.strftime("%Y-%m-%d")
                if date_key in daily_stats:
                    daily_stats[date_key].add_message(message)

        return daily_stats

    def get_daily_cost_trend(
        self, days: int = 7, time_filter: Optional[TimeFilter] = None
    ) -> dict[str, dict[str, float]]:
        """
        Get daily cost statistics for trend visualization.

        Args:
            days: Number of days to include (default 7), including today
            time_filter: Optional time filter

        Returns:
            Dict mapping date strings (YYYY-MM-DD) to cost breakdowns with keys:
            - input_cost: Cost of input tokens
            - output_cost: Cost of output tokens
            - cache_write_cost: Cost of cache creation
            - cache_read_cost: Cost of cache reads
            - total_cost: Total cost for the day
        """
        # Define pricing constants locally to avoid cyclic import
        DEFAULT_PRICING = {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "cache_write_per_mtok": 3.75,
            "cache_read_per_mtok": 0.30,
        }

        daily_stats = self.get_daily_stats(days=days, time_filter=time_filter)

        daily_costs = {}
        for date_str, stats in daily_stats.items():
            # Calculate costs for each token type using default pricing
            # (This is an approximation since model-specific pricing varies)
            input_cost = (
                stats.total_tokens.input_tokens / 1_000_000
            ) * DEFAULT_PRICING["input_per_mtok"]
            output_cost = (
                stats.total_tokens.output_tokens / 1_000_000
            ) * DEFAULT_PRICING["output_per_mtok"]
            cache_write_cost = (
                stats.total_tokens.cache_creation_input_tokens / 1_000_000
            ) * DEFAULT_PRICING["cache_write_per_mtok"]
            cache_read_cost = (
                stats.total_tokens.cache_read_input_tokens / 1_000_000
            ) * DEFAULT_PRICING["cache_read_per_mtok"]

            daily_costs[date_str] = {
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "cache_write_cost": round(cache_write_cost, 4),
                "cache_read_cost": round(cache_read_cost, 4),
                "total_cost": round(
                    input_cost + output_cost + cache_write_cost + cache_read_cost, 4
                ),
            }

        return daily_costs

    def get_tool_output_files(
        self,
        time_filter: Optional[TimeFilter] = None,
    ) -> dict[str, Set[str]]:
        """
        Extract file paths from tool_result messages for Claude Code 2.1.0+.

        In Claude Code 2.1.0+, large tool outputs are saved to disk and referenced
        by file path in the session JSONL. This method extracts those file paths
        so we can count their tokens separately.

        Args:
            time_filter: Optional time filter

        Returns:
            Dict mapping project paths to set of tool output file paths
        """
        project_files: dict[str, Set[str]] = {}

        for session_file in self.session_files:
            if not session_file.exists():
                continue

            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        timestamp = data.get("timestamp", "")

                        # Apply time filter
                        if time_filter and timestamp:
                            if not time_filter.matches_iso_string(timestamp):
                                continue

                        # Get project path
                        cwd = data.get("cwd")
                        if not cwd:
                            continue

                        # Look for tool_result messages with file references
                        message = data.get("message", {})
                        content = message.get("content", [])

                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "tool_result":
                                    # Get the text content which may contain file paths
                                    result_content = item.get("content", [])
                                    if isinstance(result_content, list):
                                        for sub_item in result_content:
                                            if isinstance(sub_item, dict) and sub_item.get("type") == "text":
                                                text = sub_item.get("text", "")
                                                # Look for file paths in tool-results directories
                                                if "tool-results" in text:
                                                    # Extract file paths using regex
                                                    file_paths = re.findall(r'/[^ \n]+tool-results/[^ \n]+\.txt', text)
                                                    if file_paths:
                                                        if cwd not in project_files:
                                                            project_files[cwd] = set()
                                                        project_files[cwd].update(file_paths)

                    except (json.JSONDecodeError, ValueError):
                        continue

        return project_files

    def scan_tool_output_files(
        self,
        time_filter: Optional[TimeFilter] = None,
        base_dir: Optional[Path] = None,
    ) -> int:
        """
        Scan tool-results directories and count tokens from persisted tool outputs.

        Claude Code 2.1.0+ stores large tool outputs in separate files in the
        tool-results directories. This method scans those directories and counts
        tokens based on file modification dates.

        Args:
            time_filter: Optional time filter (filters by file modification date)
            base_dir: Base directory to scan (defaults to ~/.claude)

        Returns:
            Total token count from tool output files
        """
        base = base_dir or Path.home() / ".claude"
        cutoff_time = 0

        if time_filter and time_filter.start_time:
            cutoff_time = int(time_filter.start_time.timestamp())

        total_tokens = 0
        processed_dirs = set()

        # Find all tool-results directories
        for tool_dir in base.rglob("tool-results"):
            if str(tool_dir) in processed_dirs:
                continue
            processed_dirs.add(str(tool_dir))

            for txt_file in tool_dir.glob("*.txt"):
                try:
                    mtime = int(txt_file.stat().st_mtime)

                    # Apply time filter based on file modification time
                    if cutoff_time and mtime < cutoff_time:
                        continue

                    # Count tokens: ~4 characters per token
                    file_tokens = txt_file.stat().st_size // 4
                    total_tokens += file_tokens

                except (OSError, IOError):
                    continue

        return total_tokens

    def count_tool_output_tokens(
        self,
        time_filter: Optional[TimeFilter] = None,
    ) -> int:
        """
        Count tokens from persisted tool output files (Claude Code 2.1.0+).

        Since Claude Code 2.1.0, large tool outputs are stored in separate files
        rather than in the session JSON. This method counts tokens from those files
        using a rough approximation (4 chars per token).

        Args:
            time_filter: Optional time filter

        Returns:
            Total token count from tool output files
        """
        # First try to extract from session files
        project_files = self.get_tool_output_files(time_filter=time_filter)

        total_chars = 0
        processed_files: Set[str] = set()

        for project_path, files in project_files.items():
            for file_path in files:
                if file_path in processed_files:
                    continue
                processed_files.add(file_path)

                try:
                    file_path_obj = Path(file_path)
                    if file_path_obj.exists():
                        file_size = file_path_obj.stat().st_size
                        # Rough approximation: 4 characters per token
                        total_chars += file_size
                except (OSError, IOError):
                    continue

        # If no file references found in sessions, scan directories directly
        if total_chars == 0:
            return self.scan_tool_output_files(time_filter=time_filter)

        return total_chars // 4

    def get_stats_with_tool_outputs(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
    ) -> SessionStats:
        """
        Get aggregated statistics including persisted tool outputs (Claude Code 2.1.0+).

        This is the main method to use for accurate token counting with Claude Code 2.1.0+.

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Returns:
            SessionStats with tool output tokens included
        """
        stats = self.get_stats(time_filter=time_filter, project_filter=project_filter)

        # Add tool output tokens
        tool_output_tokens = self.count_tool_output_tokens(time_filter=time_filter)
        if tool_output_tokens > 0:
            stats.total_tokens.tool_output_tokens = tool_output_tokens

        return stats

    def get_conversation_stats(
        self,
        time_filter: Optional[TimeFilter] = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Get per-conversation statistics grouped by session_id.

        Args:
            time_filter: Optional time filter
            limit: Maximum number of conversations to return (sorted by total_tokens desc; cost ordering is done in the service layer)

        Returns:
            List of dicts with per-conversation stats, sorted by total_tokens desc
        """
        from pathlib import Path as _Path

        # Map: session_id -> accumulated data
        conv: dict[str, dict] = {}

        for message in self.parse_all(time_filter=time_filter):
            sid = message.session_id
            if not sid:
                continue

            if sid not in conv:
                conv[sid] = {
                    "session_id": sid,
                    "project": message.cwd or "",
                    "start_time": None,
                    "end_time": None,
                    "message_count": 0,
                    "turn_count": 0,
                    "total_tokens": TokenUsage(),
                    "model_usage": {},
                    "peak_context_tokens": 0,
                    "context_per_turn": [],
                }

            c = conv[sid]

            # Update project (use last seen cwd)
            if message.cwd:
                c["project"] = message.cwd

            # Update timestamps
            msg_dt = message.datetime
            if msg_dt is not None:
                if c["start_time"] is None or msg_dt < c["start_time"]:
                    c["start_time"] = msg_dt
                if c["end_time"] is None or msg_dt > c["end_time"]:
                    c["end_time"] = msg_dt

            c["message_count"] += 1

            if message.role == "user":
                c["turn_count"] += 1

            if message.usage:
                c["total_tokens"] += message.usage

                # Per-model tracking
                if message.model:
                    if message.model not in c["model_usage"]:
                        c["model_usage"][message.model] = TokenUsage()
                    c["model_usage"][message.model] += message.usage

                # Track context growth on assistant messages.
                # total_input_tokens = input + cache_creation + cache_read
                # which equals the full context window size sent to the model.
                if message.role == "assistant":
                    ctx_size = message.usage.total_input_tokens
                    if ctx_size > 0:
                        c["context_per_turn"].append(ctx_size)
                        if ctx_size > c["peak_context_tokens"]:
                            c["peak_context_tokens"] = ctx_size

        # Build final list with derived fields
        result = []
        for c in conv.values():
            start = c["start_time"]
            end = c["end_time"]
            if start and end:
                duration_minutes = (end - start).total_seconds() / 60.0
            else:
                duration_minutes = 0.0

            total_tokens: TokenUsage = c["total_tokens"]
            total_cacheable = (
                total_tokens.cache_creation_input_tokens
                + total_tokens.cache_read_input_tokens
            )
            cache_efficiency = (
                (total_tokens.cache_read_input_tokens / total_cacheable * 100)
                if total_cacheable > 0
                else 0.0
            )

            project = c["project"]
            result.append(
                {
                    "session_id": c["session_id"],
                    "project": project,
                    "project_name": _Path(project).name if project else "Unknown",
                    "start_time": start,
                    "end_time": end,
                    "duration_minutes": round(duration_minutes, 1),
                    "message_count": c["message_count"],
                    "turn_count": c["turn_count"],
                    "total_tokens": total_tokens,
                    "model_usage": c["model_usage"],
                    "peak_context_tokens": c["peak_context_tokens"],
                    "context_per_turn": c["context_per_turn"],
                    "cache_efficiency": round(cache_efficiency, 1),
                }
            )

        # Sort by total tokens (descending) — cost ordering done in service layer
        result.sort(
            key=lambda x: x["total_tokens"].total_tokens,
            reverse=True,
        )

        return result[:limit]

    def get_project_daily_stats(
        self,
        days: int = 7,
        time_filter: Optional[TimeFilter] = None,
        max_projects: int = 10,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Get daily cost statistics per project for trend visualization.

        Args:
            days: Number of days to include (default 7), including today
            time_filter: Optional time filter
            max_projects: Maximum number of top projects to include

        Returns:
            Dict mapping project names to date strings to cost data:
            {
                "project_name": {
                    "2024-01-01": {"total_cost": 0.50, ...},
                    ...
                }
            }
        """
        from datetime import timedelta

        # Define pricing constants locally to avoid cyclic import
        DEFAULT_PRICING = {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "cache_write_per_mtok": 3.75,
            "cache_read_per_mtok": 0.30,
        }

        # First, get overall project stats to find top projects by total cost
        all_project_stats = self.get_project_stats(time_filter=time_filter)

        # Calculate total cost for each project
        project_costs = {}
        for project_path, stats in all_project_stats.items():
            input_cost = (
                stats.total_tokens.input_tokens / 1_000_000
            ) * DEFAULT_PRICING["input_per_mtok"]
            output_cost = (
                stats.total_tokens.output_tokens / 1_000_000
            ) * DEFAULT_PRICING["output_per_mtok"]
            cache_write_cost = (
                stats.total_tokens.cache_creation_input_tokens / 1_000_000
            ) * DEFAULT_PRICING["cache_write_per_mtok"]
            cache_read_cost = (
                stats.total_tokens.cache_read_input_tokens / 1_000_000
            ) * DEFAULT_PRICING["cache_read_per_mtok"]
            project_costs[project_path] = (
                input_cost + output_cost + cache_write_cost + cache_read_cost
            )

        # Get top projects
        top_projects = sorted(project_costs.items(), key=lambda x: x[1], reverse=True)[
            :max_projects
        ]

        # Initialize result structure
        result = {}
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Initialize dates
        date_keys = []
        for i in range(days - 1, -1, -1):
            day = today - timedelta(days=i)
            date_keys.append(day.strftime("%Y-%m-%d"))

        # Collect top project paths for efficient lookup (filter out falsy values)
        project_paths = [
            project_path for project_path, _ in top_projects if project_path
        ]
        project_paths_set = set(project_paths)

        # Prepare per-project, per-day stats for all top projects in a single pass
        project_stats_map: dict[str, dict[str, SessionStats]] = {}
        for project_path in project_paths:
            project_stats_map[project_path] = {
                date_key: SessionStats() for date_key in date_keys
            }

        # Single pass over all messages, grouped by project and date (O(n) complexity)
        for message in self.parse_all(time_filter=time_filter):
            project_path = message.cwd
            if not project_path or project_path not in project_paths_set:
                continue

            msg_dt = message.datetime
            if not msg_dt:
                continue

            date_key = msg_dt.strftime("%Y-%m-%d")
            project_stats = project_stats_map.get(project_path)
            if project_stats and date_key in project_stats:
                project_stats[date_key].add_message(message)

        # For each top project, convert collected stats to daily costs
        for project_path, _ in top_projects:
            # Get short project name for display (works for both Unix and Windows paths)
            project_name = Path(project_path).name

            project_stats = project_stats_map.get(project_path, {})

            # Convert to costs
            daily_costs = {}
            for date_key, stats in project_stats.items():
                input_cost = (
                    stats.total_tokens.input_tokens / 1_000_000
                ) * DEFAULT_PRICING["input_per_mtok"]
                output_cost = (
                    stats.total_tokens.output_tokens / 1_000_000
                ) * DEFAULT_PRICING["output_per_mtok"]
                cache_write_cost = (
                    stats.total_tokens.cache_creation_input_tokens / 1_000_000
                ) * DEFAULT_PRICING["cache_write_per_mtok"]
                cache_read_cost = (
                    stats.total_tokens.cache_read_input_tokens / 1_000_000
                ) * DEFAULT_PRICING["cache_read_per_mtok"]

                daily_costs[date_key] = {
                    "input_cost": round(input_cost, 4),
                    "output_cost": round(output_cost, 4),
                    "cache_write_cost": round(cache_write_cost, 4),
                    "cache_read_cost": round(cache_read_cost, 4),
                    "total_cost": round(
                        input_cost + output_cost + cache_write_cost + cache_read_cost, 4
                    ),
                }

            result[project_name] = daily_costs

        return result


# Per-model pricing rates (inline to avoid circular imports with analyzers.tokens)
_SUBAGENT_MODEL_RATES: dict = {
    "claude-opus-4-6": {"input_per_mtok": 5.0, "output_per_mtok": 25.0, "cache_write_per_mtok": 6.25, "cache_read_per_mtok": 0.50},
    "claude-opus-4-5-20251101": {"input_per_mtok": 5.0, "output_per_mtok": 25.0, "cache_write_per_mtok": 6.25, "cache_read_per_mtok": 0.50},
    "claude-opus-4-20250514": {"input_per_mtok": 15.0, "output_per_mtok": 75.0, "cache_write_per_mtok": 18.75, "cache_read_per_mtok": 1.50},
    "claude-sonnet-4-6": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "cache_write_per_mtok": 3.75, "cache_read_per_mtok": 0.30},
    "claude-sonnet-4-5-20250929": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "cache_write_per_mtok": 3.75, "cache_read_per_mtok": 0.30},
    "claude-haiku-4-6": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "cache_write_per_mtok": 1.25, "cache_read_per_mtok": 0.10},
    "claude-haiku-4-5-20251001": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "cache_write_per_mtok": 1.25, "cache_read_per_mtok": 0.10},
}
_SUBAGENT_DEFAULT_RATES: dict = {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "cache_write_per_mtok": 3.75, "cache_read_per_mtok": 0.30}


@dataclass
class SubAgentExchange:
    """Represents a complete sub-agent exchange (Task tool invocation and result)."""

    # Identifiers
    agent_id: str
    session_id: str
    project: Optional[str] = None

    # Timing
    timestamp: str = ""
    duration_ms: Optional[int] = None

    # Sub-agent metadata
    subagent_type: str = ""
    description: str = ""
    prompt: str = ""
    result_text: str = ""

    # Token usage for the sub-agent's work
    subagent_usage: Optional[TokenUsage] = None
    total_tokens: int = 0
    total_tool_use_count: int = 0

    # Token usage for the parent's Task invocation message
    parent_usage: Optional[TokenUsage] = None

    # Model attribution (populated by reading subagent JSONL files)
    model: Optional[str] = None  # Primary model used by this subagent
    model_usage: dict = field(default_factory=dict)  # {model_id: TokenUsage}

    # Status
    status: str = "completed"  # completed, error, etc.

    @property
    def datetime(self) -> Optional[datetime]:
        """Get datetime from ISO timestamp."""
        if not self.timestamp or not self.timestamp.strip():
            return None
        try:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    @property
    def subagent_cost(self) -> float:
        """Calculate estimated cost using model-aware pricing from subagent JSONL data."""
        def _calc(usage: "TokenUsage", rates: dict) -> float:
            return (
                (usage.input_tokens / 1_000_000) * rates["input_per_mtok"]
                + (usage.output_tokens / 1_000_000) * rates["output_per_mtok"]
                + (usage.cache_creation_input_tokens / 1_000_000) * rates["cache_write_per_mtok"]
                + (usage.cache_read_input_tokens / 1_000_000) * rates["cache_read_per_mtok"]
            )

        # Prefer per-model usage breakdown (most accurate)
        if self.model_usage:
            return sum(
                _calc(usage, _SUBAGENT_MODEL_RATES.get(model_id, _SUBAGENT_DEFAULT_RATES))
                for model_id, usage in self.model_usage.items()
            )

        # Fall back to aggregate subagent_usage with model-aware rates
        if not self.subagent_usage:
            return 0.0
        rates = _SUBAGENT_MODEL_RATES.get(self.model or "", _SUBAGENT_DEFAULT_RATES)
        return _calc(self.subagent_usage, rates)

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.duration_ms:
            return self.duration_ms / 1000.0
        return 0.0


class SubAgentParser:
    """Parser for extracting sub-agent exchange data from session files."""

    def __init__(self, session_files: List[Path], subagent_file_map: Optional[dict] = None):
        """
        Initialize sub-agent parser.

        Args:
            session_files: List of session JSONL file paths
            subagent_file_map: Optional dict mapping agent_id -> subagent JSONL Path
        """
        self.session_files = session_files
        self.subagent_file_map: dict[str, Path] = subagent_file_map or {}

    def parse_exchanges(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
    ) -> List[SubAgentExchange]:
        """
        Parse all session files and extract sub-agent exchanges.

        Looks for tool_result messages with toolUseResult.agentId to identify
        completed sub-agent calls. Also directly parses subagent JSONL files for
        agents not captured via the parent session (e.g. Teams teammate agents).

        Args:
            time_filter: Optional time filter
            project_filter: Optional project path to filter by

        Returns:
            List of SubAgentExchange instances
        """
        exchanges = []

        for session_file in self.session_files:
            # Skip agent-*.jsonl files (these are the sub-agent sessions themselves)
            if session_file.name.startswith("agent-"):
                continue

            file_exchanges = self._parse_file(session_file, time_filter, project_filter)
            exchanges.extend(file_exchanges)

        # Also parse subagent files not captured via parent session (e.g. Teams agents)
        found_agent_ids = {ex.agent_id for ex in exchanges}
        for agent_id in self.subagent_file_map:
            if agent_id in found_agent_ids:
                continue
            exchange = self._parse_subagent_file_directly(agent_id, time_filter, project_filter)
            if exchange:
                exchanges.append(exchange)

        # Sort by timestamp (most recent first)
        exchanges.sort(key=lambda x: x.timestamp, reverse=True)
        return exchanges

    def _parse_file(
        self,
        session_file: Path,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
    ) -> List[SubAgentExchange]:
        """Parse a single session file for sub-agent exchanges."""
        exchanges = []

        if not session_file.exists():
            return exchanges

        # First pass: collect Task tool_use messages for prompt/description
        task_invocations: dict[str, dict] = {}  # tool_use_id -> data

        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    message = data.get("message", {})
                    content = message.get("content", [])

                    # Look for Task tool_use messages
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("name") == "Task":
                                tool_id = item.get("id", "")
                                input_data = item.get("input", {})
                                task_invocations[tool_id] = {
                                    "subagent_type": input_data.get(
                                        "subagent_type", ""
                                    ),
                                    "description": input_data.get("description", ""),
                                    "prompt": input_data.get("prompt", ""),
                                    "parent_usage": message.get("usage"),
                                    "timestamp": data.get("timestamp", ""),
                                }

                except (json.JSONDecodeError, ValueError):
                    continue

        # Second pass: find tool_result messages with toolUseResult
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Check for toolUseResult with agentId
                    tool_result = data.get("toolUseResult")
                    if not tool_result or not isinstance(tool_result, dict):
                        continue
                    if not tool_result.get("agentId"):
                        continue

                    timestamp = data.get("timestamp", "")

                    # Apply time filter
                    if time_filter and timestamp:
                        if not time_filter.matches_iso_string(timestamp):
                            continue

                    # Apply project filter
                    project = data.get("cwd")
                    if project_filter and project != project_filter:
                        continue

                    # Get the tool_use_id to match with invocation
                    message = data.get("message", {})
                    content = message.get("content", [])
                    tool_use_id = None
                    if isinstance(content, list):
                        for item in content:
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "tool_result"
                            ):
                                tool_use_id = item.get("tool_use_id")
                                break

                    # Get invocation data
                    invocation = (
                        task_invocations.get(tool_use_id, {}) if tool_use_id else {}
                    )

                    # Extract sub-agent usage
                    usage_data = tool_result.get("usage")
                    subagent_usage = None
                    if usage_data:
                        subagent_usage = TokenUsage(
                            input_tokens=usage_data.get("input_tokens", 0),
                            output_tokens=usage_data.get("output_tokens", 0),
                            cache_creation_input_tokens=usage_data.get(
                                "cache_creation_input_tokens", 0
                            ),
                            cache_read_input_tokens=usage_data.get(
                                "cache_read_input_tokens", 0
                            ),
                        )

                    # Extract parent usage from invocation
                    parent_usage = None
                    parent_usage_data = invocation.get("parent_usage")
                    if parent_usage_data:
                        parent_usage = TokenUsage(
                            input_tokens=parent_usage_data.get("input_tokens", 0),
                            output_tokens=parent_usage_data.get("output_tokens", 0),
                            cache_creation_input_tokens=parent_usage_data.get(
                                "cache_creation_input_tokens", 0
                            ),
                            cache_read_input_tokens=parent_usage_data.get(
                                "cache_read_input_tokens", 0
                            ),
                        )

                    # Extract result text
                    result_text = ""
                    result_content = tool_result.get("content", [])
                    if isinstance(result_content, list):
                        for item in result_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                # Skip the agentId line
                                if not text.startswith("agentId:"):
                                    result_text = text
                                    break

                    agent_id = tool_result.get("agentId", "")

                    # Load model attribution from the subagent's own JSONL file
                    model_usage = self._load_model_usage_from_file(agent_id)
                    primary_model = self._primary_model(model_usage)

                    exchange = SubAgentExchange(
                        agent_id=agent_id,
                        session_id=data.get("sessionId", ""),
                        project=project,
                        timestamp=invocation.get("timestamp", timestamp),
                        duration_ms=tool_result.get("totalDurationMs"),
                        subagent_type=invocation.get("subagent_type", ""),
                        description=invocation.get("description", ""),
                        prompt=invocation.get("prompt", tool_result.get("prompt", "")),
                        result_text=result_text,
                        subagent_usage=subagent_usage,
                        total_tokens=tool_result.get("totalTokens") or 0,
                        total_tool_use_count=tool_result.get("totalToolUseCount") or 0,
                        parent_usage=parent_usage,
                        status=tool_result.get("status", "completed"),
                        model=primary_model,
                        model_usage=model_usage,
                    )

                    exchanges.append(exchange)

                except (json.JSONDecodeError, ValueError):
                    continue

        return exchanges

    def _load_model_usage_from_file(self, agent_id: str) -> dict:
        """
        Read a subagent JSONL file and aggregate token usage per model.

        Args:
            agent_id: The agent ID (hex string matching agent-<id>.jsonl filename)

        Returns:
            Dict mapping model_id -> TokenUsage, or empty dict if file not found
        """
        agent_file = self.subagent_file_map.get(agent_id)
        if not agent_file or not agent_file.exists():
            return {}

        model_usage: dict[str, TokenUsage] = {}

        with open(agent_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    message = data.get("message", {})
                    if message.get("role") != "assistant":
                        continue
                    model_id = message.get("model")
                    usage_data = message.get("usage")
                    if not model_id or not usage_data:
                        continue
                    usage = TokenUsage(
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
                        cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
                    )
                    if model_id not in model_usage:
                        model_usage[model_id] = TokenUsage()
                    model_usage[model_id] = model_usage[model_id] + usage
                except (json.JSONDecodeError, ValueError):
                    continue

        return model_usage

    def _primary_model(self, model_usage: dict) -> Optional[str]:
        """Return the model with the highest output token count, or None."""
        if not model_usage:
            return None
        return max(model_usage, key=lambda m: model_usage[m].output_tokens)

    def _parse_subagent_file_directly(
        self,
        agent_id: str,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
    ) -> Optional["SubAgentExchange"]:
        """
        Parse a subagent JSONL file directly to create a SubAgentExchange.

        Used for agents not captured via the parent session's toolUseResult (e.g.
        Teams teammate agents where toolUseResult.agentId is absent).
        """
        agent_file = self.subagent_file_map.get(agent_id)
        if not agent_file or not agent_file.exists():
            return None

        first_ts: Optional[str] = None
        last_ts: Optional[str] = None
        cwd: Optional[str] = None
        session_id: str = ""
        first_user_content: str = ""
        model_usage: dict[str, TokenUsage] = {}
        total_tool_use_count: int = 0

        with open(agent_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                ts = data.get("timestamp", "")
                if ts:
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts

                if not cwd:
                    cwd = data.get("cwd")
                if not session_id:
                    session_id = data.get("sessionId", "")

                msg = data.get("message", {})
                role = msg.get("role")
                content = msg.get("content", [])

                if role == "user" and not first_user_content:
                    if isinstance(content, str):
                        first_user_content = content[:300]
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                first_user_content = item.get("text", "")[:300]
                                break

                if role == "assistant":
                    model_id = msg.get("model")
                    usage_data = msg.get("usage")
                    if model_id and usage_data:
                        usage = TokenUsage(
                            input_tokens=usage_data.get("input_tokens", 0),
                            output_tokens=usage_data.get("output_tokens", 0),
                            cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
                            cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
                        )
                        if model_id not in model_usage:
                            model_usage[model_id] = TokenUsage()
                        model_usage[model_id] = model_usage[model_id] + usage

                    if isinstance(content, list):
                        total_tool_use_count += sum(
                            1 for item in content
                            if isinstance(item, dict) and item.get("type") == "tool_use"
                        )

        if not first_ts:
            return None

        # Apply time filter
        if time_filter and not time_filter.matches_iso_string(first_ts):
            return None

        # Apply project filter
        if project_filter and cwd != project_filter:
            return None

        # Duration
        duration_ms: Optional[int] = None
        if first_ts and last_ts and first_ts != last_ts:
            try:
                t1 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                duration_ms = int((t2 - t1).total_seconds() * 1000)
            except (ValueError, AttributeError):
                pass

        # Aggregate total usage
        total_usage = TokenUsage()
        for u in model_usage.values():
            total_usage = total_usage + u

        # Detect subagent type and extract readable description from Teams agents
        description = first_user_content[:100]
        if "<teammate-message" in first_user_content:
            # Extract summary="..." and teammate_id="..." from the tag
            summary_match = re.search(r'summary="([^"]*)"', first_user_content)
            id_match = re.search(r'teammate_id="([^"]*)"', first_user_content)
            teammate_id = id_match.group(1) if id_match else "teammate"
            subagent_type = teammate_id
            if summary_match:
                description = summary_match.group(1)
            else:
                description = f"[{teammate_id}] teammate agent"
        else:
            subagent_type = "unknown"

        primary_model = self._primary_model(model_usage)

        return SubAgentExchange(
            agent_id=agent_id,
            session_id=session_id,
            project=cwd,
            timestamp=first_ts,
            duration_ms=duration_ms,
            subagent_type=subagent_type,
            description=description,
            prompt=first_user_content,
            subagent_usage=total_usage,
            total_tokens=total_usage.total_tokens,
            total_tool_use_count=total_tool_use_count,
            model=primary_model,
            model_usage=model_usage,
        )

    def get_exchange_stats(self, time_filter: Optional[TimeFilter] = None) -> dict:
        """
        Get aggregated statistics for sub-agent exchanges.

        Returns:
            Dict with aggregate statistics
        """
        exchanges = self.parse_exchanges(time_filter=time_filter)

        if not exchanges:
            return {
                "total_exchanges": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_tokens_per_exchange": 0,
                "avg_duration_seconds": 0.0,
                "by_type": {},
            }

        total_tokens = sum(e.total_tokens for e in exchanges)
        total_cost = sum(e.subagent_cost for e in exchanges)
        total_duration = sum(e.duration_ms or 0 for e in exchanges)

        # Group by subagent_type
        by_type: dict[str, dict] = {}
        for e in exchanges:
            type_name = e.subagent_type or "unknown"
            if type_name not in by_type:
                by_type[type_name] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                }
            by_type[type_name]["count"] += 1
            by_type[type_name]["total_tokens"] += e.total_tokens
            by_type[type_name]["total_cost"] += e.subagent_cost

        return {
            "total_exchanges": len(exchanges),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 2),
            "avg_tokens_per_exchange": round(total_tokens / len(exchanges))
            if exchanges
            else 0,
            "avg_duration_seconds": round(total_duration / len(exchanges) / 1000, 1)
            if exchanges
            else 0.0,
            "by_type": by_type,
        }
