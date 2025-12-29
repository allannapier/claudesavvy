"""Integration analyzer for MCP server usage."""

from dataclasses import dataclass

from ..parsers.debug import DebugLogParser, MCPServerStats


@dataclass
class IntegrationSummary:
    """Summary of MCP integration usage."""

    total_servers: int
    total_tool_calls: int
    total_connections: int
    total_errors: int
    top_servers: list[tuple[str, MCPServerStats]]

    @property
    def has_integrations(self) -> bool:
        """Check if any integrations are being used."""
        return self.total_servers > 0


class IntegrationAnalyzer:
    """Analyzer for MCP server integration usage."""

    def __init__(self, debug_parser: DebugLogParser):
        """
        Initialize integration analyzer.

        Args:
            debug_parser: Parser for debug logs
        """
        self.debug_parser = debug_parser

    def get_summary(self, limit: int = 10) -> IntegrationSummary:
        """
        Get summary of MCP integration usage.

        Args:
            limit: Maximum number of top servers to include

        Returns:
            IntegrationSummary with aggregated data
        """
        all_stats = self.debug_parser.get_all_mcp_stats()

        total_tool_calls = sum(s.tool_call_count for s in all_stats.values())
        total_connections = sum(s.connection_count for s in all_stats.values())
        total_errors = sum(s.error_count for s in all_stats.values())

        # Get top servers by tool call count
        top_servers = list(all_stats.items())[:limit]

        return IntegrationSummary(
            total_servers=len(all_stats),
            total_tool_calls=total_tool_calls,
            total_connections=total_connections,
            total_errors=total_errors,
            top_servers=top_servers
        )

    def get_server_details(self) -> dict[str, MCPServerStats]:
        """
        Get detailed statistics for all MCP servers.

        Returns:
            Dict mapping server names to MCPServerStats
        """
        return self.debug_parser.get_all_mcp_stats()
