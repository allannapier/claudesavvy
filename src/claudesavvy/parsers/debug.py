"""Parser for Claude Code debug logs to extract MCP server usage."""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MCPServerStats:
    """Statistics for MCP server usage."""

    server_name: str
    connection_count: int = 0
    tool_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    projects: set[str] = field(default_factory=set)  # Projects using this MCP

    @property
    def tool_call_count(self) -> int:
        """Total number of tool calls."""
        return len(self.tool_calls)

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)

    @property
    def unique_tools(self) -> set[str]:
        """Set of unique tool names called."""
        return set(self.tool_calls)


class DebugLogParser:
    """Parser for Claude Code debug logs."""

    # Regex patterns for extracting information
    MCP_CONNECTION_PATTERN = re.compile(r'MCP server "([^"]+)":\s*(Initializing|Starting|connected|started)', re.IGNORECASE)
    MCP_TOOL_PATTERN = re.compile(r'mcp__([^_]+)__([^",\s\]]+)')
    ERROR_PATTERN = re.compile(r'(ERROR|Error|error):\s*(.+)')

    def __init__(self, debug_files: list[Path], project_map: Optional[dict[str, str]] = None):
        """
        Initialize debug log parser.

        Args:
            debug_files: List of debug log file paths
            project_map: Optional dict mapping session IDs to project paths
        """
        self.debug_files = debug_files
        self.project_map = project_map or {}

    def parse_file(self, debug_file: Path) -> dict[str, MCPServerStats]:
        """
        Parse a single debug log file.

        Args:
            debug_file: Path to debug log file

        Returns:
            Dict mapping server names to MCPServerStats
        """
        stats: dict[str, MCPServerStats] = defaultdict(
            lambda: MCPServerStats(server_name="")
        )

        if not debug_file.exists():
            return {}

        # Get project path from session ID
        session_id = debug_file.stem  # filename without extension
        project_path = self.project_map.get(session_id, "global")

        try:
            with open(debug_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Find MCP server connections
            for match in self.MCP_CONNECTION_PATTERN.finditer(content):
                server_name = match.group(1)
                if server_name not in stats:
                    stats[server_name] = MCPServerStats(server_name=server_name)
                stats[server_name].connection_count += 1
                stats[server_name].projects.add(project_path)

            # Find MCP tool calls (matches mcp__servername__toolname)
            for match in self.MCP_TOOL_PATTERN.finditer(content):
                server_name = match.group(1)
                tool_name = match.group(2)
                full_tool_name = f"mcp__{server_name}__{tool_name}"

                if server_name not in stats:
                    stats[server_name] = MCPServerStats(server_name=server_name)
                stats[server_name].tool_calls.append(full_tool_name)
                stats[server_name].projects.add(project_path)

            # Find errors (basic error detection)
            for match in self.ERROR_PATTERN.finditer(content):
                error_msg = match.group(2).strip()
                # Try to associate with a server if mentioned
                for server_name in stats.keys():
                    if server_name in error_msg:
                        stats[server_name].errors.append(error_msg)
                        break

        except Exception:
            # Skip files that can't be read
            pass

        return dict(stats)

    def get_all_mcp_stats(self) -> dict[str, MCPServerStats]:
        """
        Get aggregated MCP server statistics from all debug logs.

        Returns:
            Dict mapping server names to aggregated MCPServerStats
        """
        aggregated: dict[str, MCPServerStats] = {}

        for debug_file in self.debug_files:
            file_stats = self.parse_file(debug_file)

            for server_name, stats in file_stats.items():
                if server_name not in aggregated:
                    aggregated[server_name] = MCPServerStats(server_name=server_name)

                aggregated[server_name].connection_count += stats.connection_count
                aggregated[server_name].tool_calls.extend(stats.tool_calls)
                aggregated[server_name].errors.extend(stats.errors)
                aggregated[server_name].projects.update(stats.projects)

        # Sort by tool call count (descending)
        return dict(
            sorted(
                aggregated.items(),
                key=lambda x: x[1].tool_call_count,
                reverse=True
            )
        )

    def get_total_error_count(self) -> int:
        """
        Get total number of errors across all logs.

        Returns:
            Total error count
        """
        stats = self.get_all_mcp_stats()
        return sum(s.error_count for s in stats.values())
