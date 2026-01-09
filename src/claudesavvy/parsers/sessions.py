"""Parser for Claude Code session files containing token usage data."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, List

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
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
        )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache tokens."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
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

        return cls(
            role=role,
            timestamp=data.get("timestamp", ""),
            session_id=data.get("sessionId", ""),
            cwd=data.get("cwd"),
            usage=usage,
            model=model,
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
        """Calculate estimated cost for sub-agent work using default Sonnet pricing."""
        if not self.subagent_usage:
            return 0.0
        pricing = {
            "input_per_mtok": 3.0,
            "output_per_mtok": 15.0,
            "cache_write_per_mtok": 3.75,
            "cache_read_per_mtok": 0.30,
        }
        input_cost = (self.subagent_usage.input_tokens / 1_000_000) * pricing[
            "input_per_mtok"
        ]
        output_cost = (self.subagent_usage.output_tokens / 1_000_000) * pricing[
            "output_per_mtok"
        ]
        cache_write_cost = (
            self.subagent_usage.cache_creation_input_tokens / 1_000_000
        ) * pricing["cache_write_per_mtok"]
        cache_read_cost = (
            self.subagent_usage.cache_read_input_tokens / 1_000_000
        ) * pricing["cache_read_per_mtok"]
        return input_cost + output_cost + cache_write_cost + cache_read_cost

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.duration_ms:
            return self.duration_ms / 1000.0
        return 0.0


class SubAgentParser:
    """Parser for extracting sub-agent exchange data from session files."""

    def __init__(self, session_files: List[Path]):
        """
        Initialize sub-agent parser.

        Args:
            session_files: List of session JSONL file paths
        """
        self.session_files = session_files

    def parse_exchanges(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
    ) -> List[SubAgentExchange]:
        """
        Parse all session files and extract sub-agent exchanges.

        Looks for tool_result messages with toolUseResult.agentId to identify
        completed sub-agent calls.

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

                    exchange = SubAgentExchange(
                        agent_id=tool_result.get("agentId", ""),
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
                    )

                    exchanges.append(exchange)

                except (json.JSONDecodeError, ValueError):
                    continue

        return exchanges

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
