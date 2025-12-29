"""Analyzer for Claude Code features (sub-agents, skills, MCPs)."""

from dataclasses import dataclass
from typing import Optional

from ..parsers.tools import ToolUsageParser, ToolStats
from ..parsers.skills import SkillsParser, SkillInfo, ConfigurationParser
from ..parsers.debug import DebugLogParser, MCPServerStats
from ..utils.time_filter import TimeFilter


@dataclass
class FeaturesSummary:
    """Summary of Claude Code features usage."""

    # Sub-agents
    subagent_stats: dict[str, ToolStats]
    total_subagent_calls: int
    unique_subagents_used: int

    # Skills
    installed_skills: list[SkillInfo]
    total_skills: int

    # MCPs
    mcp_stats: dict[str, MCPServerStats]
    enabled_mcps: dict[str, bool]
    total_mcp_servers: int

    # Tools (general)
    tool_stats: dict[str, ToolStats]
    total_tool_calls: int

    # Configuration
    always_thinking_enabled: bool
    enabled_plugins: dict[str, bool]


class FeaturesAnalyzer:
    """Analyzer for Claude Code features."""

    def __init__(
        self,
        tool_parser: ToolUsageParser,
        skills_parser: SkillsParser,
        debug_parser: DebugLogParser,
        config_parser: ConfigurationParser,
        time_filter: Optional[TimeFilter] = None
    ):
        """
        Initialize features analyzer.

        Args:
            tool_parser: Parser for tool usage
            skills_parser: Parser for skills
            debug_parser: Parser for debug logs (MCP data)
            config_parser: Parser for configuration
            time_filter: Optional time filter
        """
        self.tool_parser = tool_parser
        self.skills_parser = skills_parser
        self.debug_parser = debug_parser
        self.config_parser = config_parser
        self.time_filter = time_filter

    def get_summary(self) -> FeaturesSummary:
        """
        Get comprehensive features summary.

        Returns:
            FeaturesSummary with all feature data
        """
        # Get sub-agent stats
        subagent_stats = self.tool_parser.get_subagent_stats(time_filter=self.time_filter)
        total_subagent_calls = sum(s.invocation_count for s in subagent_stats.values())
        unique_subagents = len(subagent_stats)

        # Get tool stats
        tool_stats = self.tool_parser.get_tool_stats(time_filter=self.time_filter)
        total_tool_calls = sum(s.invocation_count for s in tool_stats.values())

        # Get skills
        installed_skills = self.skills_parser.get_installed_skills()
        total_skills = len(installed_skills)

        # Get MCP stats
        mcp_stats = self.debug_parser.get_all_mcp_stats()
        total_mcp_servers = len(mcp_stats)

        # Get configuration
        enabled_plugins = self.config_parser.get_enabled_plugins()
        always_thinking = self.config_parser.is_always_thinking_enabled()

        # Try to correlate enabled plugins with MCP servers
        # Plugins might enable MCP servers
        enabled_mcps = {}
        for plugin_name, enabled in enabled_plugins.items():
            if 'mcp' in plugin_name.lower():
                enabled_mcps[plugin_name] = enabled

        return FeaturesSummary(
            subagent_stats=subagent_stats,
            total_subagent_calls=total_subagent_calls,
            unique_subagents_used=unique_subagents,
            installed_skills=installed_skills,
            total_skills=total_skills,
            mcp_stats=mcp_stats,
            enabled_mcps=enabled_mcps,
            total_mcp_servers=total_mcp_servers,
            tool_stats=tool_stats,
            total_tool_calls=total_tool_calls,
            always_thinking_enabled=always_thinking,
            enabled_plugins=enabled_plugins
        )

    def get_top_tools(self, limit: int = 10) -> list[tuple[str, ToolStats]]:
        """
        Get most frequently used tools.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of (tool_name, ToolStats) tuples
        """
        tool_stats = self.tool_parser.get_tool_stats(time_filter=self.time_filter)
        return list(tool_stats.items())[:limit]

    def get_top_subagents(self, limit: int = 10) -> list[tuple[str, ToolStats]]:
        """
        Get most frequently used sub-agents.

        Args:
            limit: Maximum number of sub-agents to return

        Returns:
            List of (subagent_type, ToolStats) tuples
        """
        subagent_stats = self.tool_parser.get_subagent_stats(time_filter=self.time_filter)
        return list(subagent_stats.items())[:limit]
