"""Parser for tool usage including sub-agents from Claude Code sessions."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from ..utils.time_filter import TimeFilter


@dataclass
class ToolInvocation:
    """Represents a single tool invocation."""

    tool_name: str
    timestamp: str
    session_id: str
    project: str
    input_params: dict = field(default_factory=dict)

    # For Task tool (sub-agents)
    subagent_type: Optional[str] = None
    description: Optional[str] = None

    # Token usage for this invocation
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def datetime(self) -> datetime:
        """Get datetime from ISO timestamp."""
        return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))

    @property
    def total_tokens(self) -> int:
        """Total tokens used for this invocation."""
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class ToolStats:
    """Aggregated statistics for a tool."""

    tool_name: str
    invocation_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    sessions: set[str] = field(default_factory=set)
    projects: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all invocations."""
        return (self.total_input_tokens + self.total_output_tokens +
                self.total_cache_read_tokens + self.total_cache_write_tokens)

    @property
    def session_count(self) -> int:
        """Number of unique sessions."""
        return len(self.sessions)


class ToolUsageParser:
    """Parser for extracting tool usage from session files."""

    def __init__(self, session_files: list[Path]):
        """
        Initialize tool usage parser.

        Args:
            session_files: List of session JSONL file paths
        """
        self.session_files = session_files

    def parse_file(
        self,
        session_file: Path,
        time_filter: Optional[TimeFilter] = None
    ) -> Iterator[ToolInvocation]:
        """
        Parse a single session file and yield tool invocations.

        Args:
            session_file: Path to session JSONL file
            time_filter: Optional time filter

        Yields:
            ToolInvocation instances
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

                    # Only process assistant messages with tool use
                    if data.get('type') != 'assistant':
                        continue

                    message = data.get('message', {})
                    content = message.get('content', [])
                    timestamp = data.get('timestamp', '')
                    session_id = data.get('sessionId', '')
                    project = data.get('cwd', '')

                    # Apply time filter
                    if time_filter and timestamp:
                        if not time_filter.matches_iso_string(timestamp):
                            continue

                    # Get token usage for this message
                    usage = message.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    cache_read = usage.get('cache_read_input_tokens', 0)
                    cache_write = usage.get('cache_creation_input_tokens', 0)

                    # Extract tool uses from content
                    for item in content:
                        if item.get('type') == 'tool_use':
                            tool_name = item.get('name', '')
                            tool_input = item.get('input', {})

                            invocation = ToolInvocation(
                                tool_name=tool_name,
                                timestamp=timestamp,
                                session_id=session_id,
                                project=project,
                                input_params=tool_input,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cache_read_tokens=cache_read,
                                cache_write_tokens=cache_write
                            )

                            # Extract sub-agent info if this is a Task tool
                            if tool_name == 'Task':
                                invocation.subagent_type = tool_input.get('subagent_type', '')
                                invocation.description = tool_input.get('description', '')

                            yield invocation

                except (json.JSONDecodeError, ValueError, KeyError):
                    # Skip malformed lines
                    continue

    def parse_all(
        self,
        time_filter: Optional[TimeFilter] = None
    ) -> Iterator[ToolInvocation]:
        """
        Parse all session files and yield tool invocations.

        Args:
            time_filter: Optional time filter

        Yields:
            ToolInvocation instances
        """
        for session_file in self.session_files:
            yield from self.parse_file(session_file, time_filter=time_filter)

    def get_tool_stats(
        self,
        time_filter: Optional[TimeFilter] = None
    ) -> dict[str, ToolStats]:
        """
        Get aggregated statistics for all tools.

        Args:
            time_filter: Optional time filter

        Returns:
            Dict mapping tool names to ToolStats
        """
        stats: dict[str, ToolStats] = defaultdict(lambda: ToolStats(tool_name=""))

        for invocation in self.parse_all(time_filter=time_filter):
            tool_name = invocation.tool_name

            if tool_name not in stats:
                stats[tool_name] = ToolStats(tool_name=tool_name)

            tool_stats = stats[tool_name]
            tool_stats.invocation_count += 1
            tool_stats.total_input_tokens += invocation.input_tokens
            tool_stats.total_output_tokens += invocation.output_tokens
            tool_stats.total_cache_read_tokens += invocation.cache_read_tokens
            tool_stats.total_cache_write_tokens += invocation.cache_write_tokens
            tool_stats.sessions.add(invocation.session_id)
            if invocation.project:
                tool_stats.projects.add(invocation.project)

        # Sort by invocation count (descending)
        return dict(
            sorted(stats.items(), key=lambda x: x[1].invocation_count, reverse=True)
        )

    def get_subagent_stats(
        self,
        time_filter: Optional[TimeFilter] = None
    ) -> dict[str, ToolStats]:
        """
        Get statistics specifically for sub-agents (Task tool with subagent_type).

        Args:
            time_filter: Optional time filter

        Returns:
            Dict mapping subagent types to ToolStats
        """
        stats: dict[str, ToolStats] = defaultdict(lambda: ToolStats(tool_name=""))

        for invocation in self.parse_all(time_filter=time_filter):
            if invocation.tool_name != 'Task' or not invocation.subagent_type:
                continue

            agent_type = invocation.subagent_type

            if agent_type not in stats:
                stats[agent_type] = ToolStats(tool_name=agent_type)

            agent_stats = stats[agent_type]
            agent_stats.invocation_count += 1
            agent_stats.total_input_tokens += invocation.input_tokens
            agent_stats.total_output_tokens += invocation.output_tokens
            agent_stats.total_cache_read_tokens += invocation.cache_read_tokens
            agent_stats.total_cache_write_tokens += invocation.cache_write_tokens
            agent_stats.sessions.add(invocation.session_id)
            if invocation.project:
                agent_stats.projects.add(invocation.project)

        # Sort by invocation count (descending)
        return dict(
            sorted(stats.items(), key=lambda x: x[1].invocation_count, reverse=True)
        )
