"""Dashboard service for Claude Monitor web interface.

This service provides a clean interface for web routes to access dashboard data
by wrapping existing parsers and analyzers. It reuses all existing business logic
while returning data as dicts/dataclasses suitable for web rendering.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from ...utils.paths import get_claude_paths, ClaudeDataPaths
from ...utils.time_filter import TimeFilter
from ...utils.pricing import PricingSettings
from ...parsers.history import HistoryParser
from ...parsers.sessions import SessionParser, SubAgentParser
from ...parsers.debug import DebugLogParser
from ...parsers.files import FileHistoryParser
from ...parsers.tools import ToolUsageParser
from ...parsers.skills import SkillsParser, ConfigurationParser
from ...parsers.configuration_scanner import ConfigurationScanner
from ...analyzers.usage import UsageAnalyzer, UsageSummary
from ...analyzers.tokens import (
    TokenAnalyzer,
    TokenSummary,
    get_model_display_name,
    DEFAULT_PRICING,
)
from ...analyzers.integrations import IntegrationAnalyzer, IntegrationSummary
from ...analyzers.features import FeaturesAnalyzer, FeaturesSummary
from ...analyzers.configuration import ConfigurationAnalyzer
from ...analyzers.project_analyzer import ProjectAnalyzer


class DashboardService:
    """Service for providing dashboard data to web routes.

    Initializes all parsers and analyzers, then provides methods to retrieve
    aggregated data suitable for web rendering.
    """

    def __init__(self, claude_data_paths: Optional[ClaudeDataPaths] = None):
        """
        Initialize dashboard service with parsers and analyzers.

        Args:
            claude_data_paths: Optional ClaudeDataPaths instance. If not provided,
                             uses default Claude data directory.
        """
        # Use provided paths or get default
        if claude_data_paths is None:
            claude_data_paths = get_claude_paths()

        self.paths = claude_data_paths

        # Initialize parsers
        self._history_parser = HistoryParser(self.paths.history_file)
        self._session_parser = SessionParser(self.paths.get_project_session_files())
        self._file_parser = FileHistoryParser(
            self.paths.file_history_dir, self.paths.get_project_session_files()
        )
        self._tool_parser = ToolUsageParser(self.paths.get_project_session_files())
        self._skills_parser = SkillsParser(self.paths.base_dir / "skills")
        self._config_parser = ConfigurationParser(self.paths.base_dir)

        # Initialize sub-agent parser
        self._subagent_parser = SubAgentParser(self.paths.get_project_session_files())

        # Build project map for debug logs
        project_map = self._build_project_map()
        self._debug_parser = DebugLogParser(
            self.paths.get_debug_log_files(), project_map
        )

        # Initialize pricing settings
        self._pricing_settings = PricingSettings(self.paths.base_dir)

        # Initialize analyzers (no time filter by default)
        self._usage_analyzer = UsageAnalyzer(
            self._history_parser, self._session_parser, time_filter=None
        )
        self._token_analyzer = TokenAnalyzer(
            self._session_parser,
            time_filter=None,
            pricing_settings=self._pricing_settings,
        )
        self._integration_analyzer = IntegrationAnalyzer(self._debug_parser)
        self._features_analyzer = FeaturesAnalyzer(
            self._tool_parser,
            self._skills_parser,
            self._debug_parser,
            self._config_parser,
            time_filter=None,
        )

        # Initialize configuration scanner and analyzer
        self._config_scanner = ConfigurationScanner(self._config_parser)
        self._configuration_analyzer = ConfigurationAnalyzer(self._config_scanner)

        # Initialize project analyzer
        self._project_analyzer = ProjectAnalyzer(
            self._session_parser,
            self._tool_parser,
            self._skills_parser,
            self._config_scanner,
        )

    def _build_project_map(self) -> Dict[str, str]:
        """
        Build a map of session IDs to project names.

        Returns:
            Dict mapping session_id to project_name
        """
        project_paths = {}
        if self.paths.projects_dir.exists():
            all_dirs = sorted(
                [d for d in self.paths.projects_dir.iterdir() if d.is_dir()],
                key=lambda x: len(x.name),
                reverse=True,
            )

            for project_dir in all_dirs:
                encoded_name = project_dir.name
                for session_file in project_dir.glob("*.jsonl"):
                    session_id = session_file.stem
                    if session_id not in project_paths or len(encoded_name) > len(
                        project_paths[session_id]
                    ):
                        # Extract project name from encoded path
                        if encoded_name.startswith("-Users-allannapier-code-"):
                            project_name = encoded_name.replace(
                                "-Users-allannapier-code-", "", 1
                            )
                        elif encoded_name.startswith("-"):
                            project_name = encoded_name[1:]
                        else:
                            project_name = encoded_name
                        project_paths[session_id] = project_name

        return project_paths

    def _create_time_filtered_service(
        self, time_filter: TimeFilter
    ) -> "DashboardService":
        """
        Create a new service instance with time filter applied.

        Args:
            time_filter: TimeFilter to apply

        Returns:
            DashboardService instance with time-filtered analyzers
        """
        service = DashboardService.__new__(DashboardService)
        service.paths = self.paths
        service._pricing_settings = self._pricing_settings
        service._history_parser = self._history_parser
        service._session_parser = self._session_parser
        service._file_parser = self._file_parser
        service._tool_parser = self._tool_parser
        service._skills_parser = self._skills_parser
        service._config_parser = self._config_parser
        service._debug_parser = self._debug_parser

        # Create time-filtered analyzers
        service._usage_analyzer = UsageAnalyzer(
            self._history_parser, self._session_parser, time_filter=time_filter
        )
        service._token_analyzer = TokenAnalyzer(
            self._session_parser,
            time_filter=time_filter,
            pricing_settings=self._pricing_settings,
        )
        service._integration_analyzer = IntegrationAnalyzer(self._debug_parser)
        service._features_analyzer = FeaturesAnalyzer(
            self._tool_parser,
            self._skills_parser,
            self._debug_parser,
            self._config_parser,
            time_filter=time_filter,
        )

        return service

    def _get_analyzer(
        self, analyzer_name: str, time_filter: Optional[TimeFilter] = None
    ):
        """
        Get an analyzer, optionally applying a time filter.

        Args:
            analyzer_name: Name of the analyzer attribute (e.g., 'usage_analyzer', 'token_analyzer')
            time_filter: Optional time filter to apply

        Returns:
            The requested analyzer, either from self or from a time-filtered service
        """
        if not time_filter:
            return getattr(self, f'_{analyzer_name}')
        return getattr(self._create_time_filtered_service(time_filter), f'_{analyzer_name}')

    def _build_project_dict(
        self, project_path: str, project_data: dict, total_tokens: int, cost: float
    ) -> dict:
        """
        Build standardized project data dictionary.

        Args:
            project_path: Path to the project
            project_data: Project data from analyzer
            total_tokens: Total tokens used
            cost: Total cost

        Returns:
            Standardized dict with all project fields
        """
        return {
            "path": project_path,
            "name": project_data["name"],
            "full_path": project_data["full_path"],
            "command_count": project_data["commands"],
            "session_count": project_data["sessions"],
            "message_count": project_data["messages"],
            "total_tokens": total_tokens,
            "total_cost": cost,
            "commands": project_data["commands"],  # backward compatibility
            "sessions": project_data["sessions"],
            "messages": project_data["messages"],
        }

    def _calculate_total_tokens(self, tokens) -> int:
        """
        Calculate total tokens from a token object.

        Args:
            tokens: Token object with input_tokens, output_tokens, cache_creation_input_tokens,
                   and cache_read_input_tokens attributes

        Returns:
            Sum of all token types
        """
        return (
            tokens.input_tokens
            + tokens.output_tokens
            + tokens.cache_creation_input_tokens
            + tokens.cache_read_input_tokens
        )

    def _extract_mcp_server_name(self, tool_name: str) -> Optional[str]:
        """
        Extract MCP server name from tool name.

        Args:
            tool_name: Tool name in format "mcp__server_name__function_name"

        Returns:
            Server name if valid format, None otherwise
        """
        parts = tool_name.split("__")
        return parts[1] if len(parts) >= 2 else None

    # ========== Public Methods for Usage Data ==========

    def get_usage_summary(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get usage summary statistics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with usage statistics suitable for web rendering
        """
        analyzer = self._get_analyzer('usage_analyzer', time_filter)

        summary: UsageSummary = analyzer.get_summary()

        # Get actual project count from project breakdown
        project_data = self.get_project_breakdown(time_filter)
        actual_project_count = project_data.get("total_projects", 0)

        return {
            "total_commands": summary.total_commands,
            "total_sessions": summary.total_sessions,
            "total_messages": summary.total_messages,
            "total_projects": actual_project_count,  # Use actual count from projects
            "earliest_activity": summary.earliest_activity.isoformat()
            if summary.earliest_activity
            else None,
            "latest_activity": summary.latest_activity.isoformat()
            if summary.latest_activity
            else None,
            "time_range_description": summary.time_range_description,
            "date_range_days": summary.date_range_days,
            "avg_commands_per_day": round(summary.avg_commands_per_day, 2)
            if summary.avg_commands_per_day
            else None,
            "avg_sessions_per_day": round(summary.avg_sessions_per_day, 2)
            if summary.avg_sessions_per_day
            else None,
        }

    def get_project_breakdown(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get per-project activity breakdown.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict mapping project paths to activity metrics
        """
        analyzer = self._get_analyzer('usage_analyzer', time_filter)

        breakdown = analyzer.get_project_breakdown()

        # Get token breakdown for cost and token data
        token_analyzer = self._get_analyzer('token_analyzer', time_filter)
        token_breakdown = token_analyzer.get_project_breakdown()

        # Convert to list of dicts with field names expected by template
        projects_data = []
        total_cost = 0.0
        most_active_project = None
        max_commands = 0

        for project_path, project_data in breakdown.items():
            # Get token data for this project
            token_summary = token_breakdown.get(project_path)
            total_tokens = 0
            cost = 0.0

            if token_summary:
                total_tokens = self._calculate_total_tokens(token_summary.total_tokens)
                cost = token_summary.total_cost
                total_cost += cost

            # Determine most active project (by commands)
            if project_data["commands"] > max_commands:
                max_commands = project_data["commands"]
                most_active_project = project_data["name"]

            projects_data.append(
                self._build_project_dict(project_path, project_data, total_tokens, cost)
            )

        # Sort by command count (descending)
        projects_data.sort(key=lambda x: x["command_count"], reverse=True)

        # Calculate average commands per project
        avg_commands = (
            sum(p["command_count"] for p in projects_data) / len(projects_data)
            if projects_data
            else 0
        )

        return {
            "projects": projects_data,
            "total_projects": len(projects_data),
            "total_cost": round(total_cost, 2),
            "avg_commands_per_project": round(avg_commands, 1),
            "most_active_project": most_active_project or "N/A",
        }

    def get_project_breakdown_by_model(
        self, model_id: str, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get per-project activity breakdown filtered by a specific model.

        Args:
            model_id: Model ID to filter by
            time_filter: Optional time filter to apply

        Returns:
            Dict with project data filtered to only show usage of the specified model
        """
        analyzer = self._get_analyzer('usage_analyzer', time_filter)
        token_analyzer = self._get_analyzer('token_analyzer', time_filter)

        # Get base project breakdown for structure (commands, sessions, etc.)
        base_breakdown = analyzer.get_project_breakdown()

        # Get model-by-project breakdown for token/cost data
        model_by_project = token_analyzer.get_model_by_project_breakdown()

        projects_data = []
        total_cost = 0.0
        most_active_project = None
        max_cost = 0.0

        for project_path, project_data in base_breakdown.items():
            # Get model-specific data for this project
            project_models = model_by_project.get(project_path, {})

            if model_id in project_models:
                tokens, cost = project_models[model_id]
                total_tokens = self._calculate_total_tokens(tokens)
                total_cost += cost

                # Track most active by cost for model filter
                if cost > max_cost:
                    max_cost = cost
                    most_active_project = project_data["name"]

                projects_data.append(
                    self._build_project_dict(project_path, project_data, total_tokens, cost)
                )

        # Sort by cost (descending) for model filter
        projects_data.sort(key=lambda x: x["total_cost"], reverse=True)

        avg_commands = (
            sum(p["command_count"] for p in projects_data) / len(projects_data)
            if projects_data
            else 0
        )

        return {
            "projects": projects_data,
            "total_projects": len(projects_data),
            "total_cost": round(total_cost, 2),
            "avg_commands_per_project": round(avg_commands, 1),
            "most_active_project": most_active_project or "N/A",
        }

    # ========== Public Methods for Token Data ==========

    def get_token_summary(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get token usage summary with cost calculations.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with token statistics and costs (flattened for template use)
        """
        analyzer = self._get_analyzer('token_analyzer', time_filter)

        summary: TokenSummary = analyzer.get_summary()

        return {
            # Flat structure for templates
            "input_tokens": summary.total_tokens.input_tokens,
            "output_tokens": summary.total_tokens.output_tokens,
            "cache_creation_tokens": summary.total_tokens.cache_creation_input_tokens,
            "cache_read_tokens": summary.total_tokens.cache_read_input_tokens,
            "input_cost": round(summary.cost_breakdown.input_cost, 2),
            "output_cost": round(summary.cost_breakdown.output_cost, 2),
            "cache_creation_cost": round(summary.cost_breakdown.cache_write_cost, 2),
            "cache_read_cost": round(summary.cost_breakdown.cache_read_cost, 2),
            "total_cost": round(summary.cost_breakdown.total_cost, 2),
            "cache_savings": round(summary.cost_breakdown.cache_savings, 2),
            "cache_hit_rate": round(summary.cache_efficiency_pct, 1),
        }

    def get_model_breakdown(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get per-model token usage and costs.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict mapping model names to token usage and costs
        """
        analyzer = self._get_analyzer('token_analyzer', time_filter)

        breakdown = analyzer.get_model_breakdown()

        models_data = []
        total_cost = sum(cost for _, cost in breakdown.values())

        for model_id, (tokens, cost) in breakdown.items():
            total_tokens = self._calculate_total_tokens(tokens)
            models_data.append(
                {
                    "model_id": model_id,
                    "model_name": get_model_display_name(model_id),
                    "input_tokens": tokens.input_tokens,
                    "output_tokens": tokens.output_tokens,
                    "cache_creation_tokens": tokens.cache_creation_input_tokens,
                    "cache_read_tokens": tokens.cache_read_input_tokens,
                    "total_tokens": total_tokens,
                    "cost": round(cost, 4),
                    "cost_percentage": round(
                        (cost / total_cost * 100) if total_cost > 0 else 0, 1
                    ),
                }
            )

        return {
            "models": models_data,
            "total_models": len(models_data),
            "total_cost": round(total_cost, 2),
        }

    def get_available_models(
        self, time_filter: Optional[TimeFilter] = None
    ) -> List[Dict[str, str]]:
        """
        Get list of available models for filtering.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            List of dicts with model id and display name
        """
        breakdown = self.get_model_breakdown(time_filter)
        return [
            {"id": model["model_id"], "name": model["model_name"]}
            for model in breakdown["models"]
        ]

    def get_daily_token_trend(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get daily token usage for trend chart.

        Args:
            time_filter: Optional time filter to apply (default 7 days if not provided)

        Returns:
            Dict with labels and data arrays for charting
        """
        # Calculate days from time filter or use default
        days = 7
        if time_filter and time_filter.start_time:
            now = datetime.now()
            delta = now - time_filter.start_time
            days = max(1, delta.days + 1)  # Include current partial day, at least 1 day

        daily_stats = self._session_parser.get_daily_stats(
            days=days, time_filter=time_filter
        )

        labels = []
        data = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for date_str in sorted(daily_stats.keys()):
            stats = daily_stats[date_str]
            # Calculate total tokens for this day
            total = self._calculate_total_tokens(stats.total_tokens)
            data.append(total)

            # Create human-readable label based on how many days we're showing
            date = datetime.strptime(date_str, "%Y-%m-%d")
            days_ago = (today - date).days

            if days <= 7:
                # For week or less, show relative labels
                if days_ago == 0:
                    labels.append("Today")
                elif days_ago == 1:
                    labels.append("Yesterday")
                else:
                    labels.append(f"{days_ago}d ago")
            else:
                # For longer periods, show date labels (Dec 15 format)
                labels.append(date.strftime("%b %d"))

        return {
            "labels": labels,
            "data": data,
        }

    def get_daily_cost_trend(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get daily cost trend for charts.

        Args:
            time_filter: Optional time filter to apply (default 7 days if not provided)

        Returns:
            Dict with labels and datasets for Chart.js
        """
        # Calculate days from time filter or use default
        days = 7
        if time_filter and time_filter.start_time:
            now = datetime.now()
            delta = now - time_filter.start_time
            days = max(1, delta.days + 1)

        daily_costs = self._session_parser.get_daily_cost_trend(
            days=days, time_filter=time_filter
        )

        labels = []
        total_costs = []
        input_costs = []
        output_costs = []
        cache_costs = []

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for date_str in sorted(daily_costs.keys()):
            costs = daily_costs[date_str]
            total_costs.append(costs["total_cost"])
            input_costs.append(costs["input_cost"])
            output_costs.append(costs["output_cost"])
            cache_costs.append(costs["cache_write_cost"] + costs["cache_read_cost"])

            # Create human-readable label based on how many days we're showing
            date = datetime.strptime(date_str, "%Y-%m-%d")
            days_ago = (today - date).days

            if days <= 7:
                # For week or less, show relative labels
                if days_ago == 0:
                    labels.append("Today")
                elif days_ago == 1:
                    labels.append("Yesterday")
                else:
                    labels.append(f"{days_ago}d ago")
            else:
                # For longer periods, show date labels (Dec 15 format)
                labels.append(date.strftime("%b %d"))

        return {
            "labels": labels,
            "datasets": {
                "total": total_costs,
                "input": input_costs,
                "output": output_costs,
                "cache": cache_costs,
            },
        }

    def get_project_cost_trend(
        self,
        time_filter: Optional[TimeFilter] = None,
        max_projects: int = 8,
    ) -> Dict[str, Any]:
        """
        Get daily cost trend per project for charts.

        Args:
            time_filter: Optional time filter to apply (default 7 days if not provided)
            max_projects: Maximum number of projects to include

        Returns:
            Dict with labels and datasets for Chart.js line chart
        """
        # Calculate days from time filter or use default
        days = 7
        if time_filter and time_filter.start_time:
            now = datetime.now()
            delta = now - time_filter.start_time
            days = max(1, delta.days + 1)

        project_daily_stats = self._session_parser.get_project_daily_stats(
            days=days, time_filter=time_filter, max_projects=max_projects
        )

        # Get date labels from the first project
        labels = []
        if project_daily_stats:
            first_project = list(project_daily_stats.values())[0]
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            for date_str in sorted(first_project.keys()):
                # Create human-readable label
                date = datetime.strptime(date_str, "%Y-%m-%d")
                days_ago = (today - date).days

                if days <= 7:
                    if days_ago == 0:
                        labels.append("Today")
                    elif days_ago == 1:
                        labels.append("Yesterday")
                    else:
                        labels.append(f"{days_ago}d ago")
                else:
                    labels.append(date.strftime("%b %d"))

        # Build datasets for each project
        datasets = []
        colors = [
            "#0770E3",  # Blue
            "#34D399",  # Green
            "#F59E0B",  # Amber
            "#8B5CF6",  # Purple
            "#EC4899",  # Pink
            "#14B8A6",  # Teal
            "#F97316",  # Orange
            "#6366F1",  # Indigo
        ]

        for idx, (project_name, daily_data) in enumerate(project_daily_stats.items()):
            color = colors[idx % len(colors)]
            data = [
                daily_data[date]["total_cost"] for date in sorted(daily_data.keys())
            ]

            datasets.append(
                {
                    "label": project_name,
                    "data": data,
                    "borderColor": color,
                    "color": color,
                }
            )

        return {
            "labels": labels,
            "datasets": datasets,
        }

    def get_token_summary_by_model(
        self, model_id: str, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get token usage summary filtered by a specific model.

        Args:
            model_id: Model ID to filter by
            time_filter: Optional time filter to apply

        Returns:
            Dict with token statistics for the specified model
        """
        analyzer = self._get_analyzer('token_analyzer', time_filter)

        breakdown = analyzer.get_model_breakdown()

        if model_id not in breakdown:
            # Return empty summary if model not found
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "cache_creation_cost": 0.0,
                "cache_read_cost": 0.0,
                "total_cost": 0.0,
                "cache_savings": 0.0,
                "cache_hit_rate": 0.0,
            }

        tokens, total_cost = breakdown[model_id]
        cost_breakdown = analyzer.calculate_cost(tokens, model_id)

        # Calculate cache efficiency for this model
        total_input = (
            tokens.input_tokens
            + tokens.cache_creation_input_tokens
            + tokens.cache_read_input_tokens
        )
        cache_hit_rate = (
            (tokens.cache_read_input_tokens / total_input * 100)
            if total_input > 0
            else 0
        )

        return {
            "input_tokens": tokens.input_tokens,
            "output_tokens": tokens.output_tokens,
            "cache_creation_tokens": tokens.cache_creation_input_tokens,
            "cache_read_tokens": tokens.cache_read_input_tokens,
            "input_cost": round(cost_breakdown.input_cost, 2),
            "output_cost": round(cost_breakdown.output_cost, 2),
            "cache_creation_cost": round(cost_breakdown.cache_write_cost, 2),
            "cache_read_cost": round(cost_breakdown.cache_read_cost, 2),
            "total_cost": round(cost_breakdown.total_cost, 2),
            "cache_savings": round(cost_breakdown.cache_savings, 2),
            "cache_hit_rate": round(cache_hit_rate, 1),
        }

    def get_project_token_breakdown(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get token usage breakdown per project.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict mapping project names to token summaries
        """
        analyzer = self._get_analyzer('token_analyzer', time_filter)

        breakdown = analyzer.get_project_breakdown()

        projects_data = []
        total_cost = 0.0

        for project_path, summary in breakdown.items():
            project_data = {
                "project_path": project_path,
                "total_tokens": {
                    "input_tokens": summary.total_tokens.input_tokens,
                    "output_tokens": summary.total_tokens.output_tokens,
                    "cache_creation_input_tokens": summary.total_tokens.cache_creation_input_tokens,
                    "cache_read_input_tokens": summary.total_tokens.cache_read_input_tokens,
                },
                "cost_breakdown": {
                    "input_cost": round(summary.cost_breakdown.input_cost, 4),
                    "output_cost": round(summary.cost_breakdown.output_cost, 4),
                    "cache_write_cost": round(
                        summary.cost_breakdown.cache_write_cost, 4
                    ),
                    "cache_read_cost": round(summary.cost_breakdown.cache_read_cost, 4),
                    "total_cost": round(summary.cost_breakdown.total_cost, 4),
                },
                "cache_efficiency_pct": round(summary.cache_efficiency_pct, 2),
            }
            projects_data.append(project_data)
            total_cost += summary.total_cost

        return {
            "projects": projects_data,
            "total_projects": len(projects_data),
            "total_cost": round(total_cost, 4),
        }

    # ========== Public Methods for Integration Data ==========

    def get_integration_summary(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get MCP integration usage summary.

        Args:
            time_filter: Optional time filter (not used by integration analyzer)

        Returns:
            Dict with integration statistics
        """
        summary: IntegrationSummary = self._integration_analyzer.get_summary()

        top_servers_data = [
            {
                "server_name": name,
                "tool_call_count": stats.tool_call_count,
                "connection_count": stats.connection_count,
                "error_count": stats.error_count,
                "error_rate": round(
                    (stats.error_count / stats.tool_call_count * 100)
                    if stats.tool_call_count > 0
                    else 0,
                    2,
                ),
            }
            for name, stats in summary.top_servers
        ]

        return {
            "total_servers": summary.total_servers,
            "total_tool_calls": summary.total_tool_calls,
            "total_connections": summary.total_connections,
            "total_errors": summary.total_errors,
            "error_rate": round(
                (summary.total_errors / summary.total_tool_calls * 100)
                if summary.total_tool_calls > 0
                else 0,
                2,
            ),
            "has_integrations": summary.has_integrations,
            "top_servers": top_servers_data,
        }

    def get_server_details(self) -> Dict[str, Any]:
        """
        Get detailed statistics for all MCP servers.

        Returns:
            Dict mapping server names to detailed statistics
        """
        all_servers = self._integration_analyzer.get_server_details()

        servers_data = {}
        for server_name, stats in all_servers.items():
            servers_data[server_name] = {
                "tool_call_count": stats.tool_call_count,
                "connection_count": stats.connection_count,
                "error_count": stats.error_count,
                "error_rate": round(
                    (stats.error_count / stats.tool_call_count * 100)
                    if stats.tool_call_count > 0
                    else 0,
                    2,
                ),
            }

        return {
            "servers": servers_data,
            "total_servers": len(servers_data),
        }

    # ========== Public Methods for Features Data ==========

    def get_features_summary(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive features usage summary.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with all feature data
        """
        analyzer = self._get_analyzer('features_analyzer', time_filter)

        summary: FeaturesSummary = analyzer.get_summary()

        return {
            "subagents": {
                "total_calls": summary.total_subagent_calls,
                "unique_used": summary.unique_subagents_used,
                "stats": {
                    name: {
                        "invocation_count": stats.invocation_count,
                        "total_tokens": stats.total_tokens,
                        "session_count": stats.session_count,
                    }
                    for name, stats in summary.subagent_stats.items()
                },
            },
            "skills": {
                "total_installed": summary.total_skills,
                "installed": [
                    {
                        "name": skill.name,
                        "description": skill.description,
                        "has_skills_md": skill.has_skills_md,
                    }
                    for skill in summary.installed_skills
                ],
            },
            "mcps": {
                "total_servers": summary.total_mcp_servers,
                "enabled": summary.enabled_mcps,
                "stats": {
                    name: {
                        "tool_call_count": stats.tool_call_count,
                        "connection_count": stats.connection_count,
                        "error_count": stats.error_count,
                    }
                    for name, stats in summary.mcp_stats.items()
                },
            },
            "tools": {
                "total_calls": summary.total_tool_calls,
                "stats": {
                    name: {
                        "invocation_count": stats.invocation_count,
                        "total_tokens": stats.total_tokens,
                        "session_count": stats.session_count,
                    }
                    for name, stats in summary.tool_stats.items()
                },
            },
            "configuration": {
                "always_thinking_enabled": summary.always_thinking_enabled,
                "enabled_plugins": summary.enabled_plugins,
            },
        }

    def get_top_tools(
        self, limit: int = 10, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get most frequently used tools.

        Args:
            limit: Maximum number of tools to return
            time_filter: Optional time filter to apply

        Returns:
            List of top tools with usage stats
        """
        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(
                time_filter
            )._features_analyzer

        top_tools = analyzer.get_top_tools(limit=limit)

        total_tokens = 0
        total_cost = 0.0
        total_calls = 0
        tools_data = []

        for name, stats in top_tools:
            # Calculate cost using default Sonnet pricing
            # Note: Tool-level token tracking doesn't include model info, so we use default rates
            pricing = DEFAULT_PRICING  # Tools don't track which model was used
            input_cost = (stats.total_input_tokens / 1_000_000) * pricing[
                "input_per_mtok"
            ]
            output_cost = (stats.total_output_tokens / 1_000_000) * pricing[
                "output_per_mtok"
            ]
            cache_read_cost = (stats.total_cache_read_tokens / 1_000_000) * pricing[
                "cache_read_per_mtok"
            ]
            cache_write_cost = (stats.total_cache_write_tokens / 1_000_000) * pricing[
                "cache_write_per_mtok"
            ]
            tool_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

            total_tokens += stats.total_tokens
            total_cost += tool_cost
            total_calls += stats.invocation_count

            tools_data.append(
                {
                    "tool_name": name,
                    "invocation_count": stats.invocation_count,
                    "total_tokens": stats.total_tokens,
                    "input_tokens": stats.total_input_tokens,
                    "output_tokens": stats.total_output_tokens,
                    "total_cost": round(tool_cost, 4),
                    "cost_per_call": round(tool_cost / stats.invocation_count, 6)
                    if stats.invocation_count > 0
                    else 0,
                    "avg_tokens_per_call": round(
                        stats.total_tokens / stats.invocation_count
                        if stats.invocation_count > 0
                        else 0,
                        1,
                    ),
                    "session_count": stats.session_count,
                }
            )

        return {
            "tools": tools_data,
            "total_shown": len(tools_data),
            "total_tokens": total_tokens,
            "total_calls": total_calls,
            "total_cost": round(total_cost, 2),
        }

    def get_top_subagents(
        self, limit: int = 10, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get most frequently used sub-agents.

        Args:
            limit: Maximum number of sub-agents to return
            time_filter: Optional time filter to apply

        Returns:
            List of top sub-agents with usage stats
        """
        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(
                time_filter
            )._features_analyzer

        top_subagents = analyzer.get_top_subagents(limit=limit)

        subagents_data = [
            {
                "type": name,
                "invocation_count": stats.invocation_count,
                "success_count": stats.success_count,
                "error_count": stats.error_count,
                "success_rate": round(
                    (stats.success_count / stats.invocation_count * 100)
                    if stats.invocation_count > 0
                    else 0,
                    2,
                ),
            }
            for name, stats in top_subagents
        ]

        return {
            "subagents": subagents_data,
            "total_shown": len(subagents_data),
        }

    def get_mcp_integrations(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get MCP server integration statistics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with MCP server usage data
        """
        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(
                time_filter
            )._features_analyzer

        # Get all tool stats from the tool parser
        all_tools = analyzer.tool_parser.get_tool_stats(time_filter=time_filter)

        # Filter for MCP tools (tools that start with mcp__)
        mcp_tools = {
            name: stats for name, stats in all_tools.items() if name.startswith("mcp__")
        }

        # Group by MCP server
        servers = {}
        for tool_name, stats in mcp_tools.items():
            # Extract server name from tool name: mcp__server_name__function_name
            server_name = self._extract_mcp_server_name(tool_name)
            if server_name:
                if server_name not in servers:
                    servers[server_name] = {
                        "server_name": server_name,
                        "tool_count": 0,
                        "total_calls": 0,
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                        "total_cost": 0.0,
                        "tools": [],
                    }

                servers[server_name]["tool_count"] += 1
                servers[server_name]["total_calls"] += stats.invocation_count
                servers[server_name]["total_tokens"] += stats.total_tokens
                servers[server_name]["input_tokens"] += stats.total_input_tokens
                servers[server_name]["output_tokens"] += stats.total_output_tokens
                servers[server_name]["cache_read_tokens"] += (
                    stats.total_cache_read_tokens
                )
                servers[server_name]["cache_write_tokens"] += (
                    stats.total_cache_write_tokens
                )
                servers[server_name]["tools"].append(
                    {
                        "tool_name": tool_name,
                        "invocation_count": stats.invocation_count,
                        "total_tokens": stats.total_tokens,
                    }
                )

        # Calculate costs per server using default pricing (Sonnet 4.5)
        # Note: MCP-level token tracking doesn't include model info, so we use default rates
        pricing = DEFAULT_PRICING
        for server in servers.values():
            input_cost = (server["input_tokens"] / 1_000_000) * pricing[
                "input_per_mtok"
            ]
            output_cost = (server["output_tokens"] / 1_000_000) * pricing[
                "output_per_mtok"
            ]
            cache_read_cost = (server["cache_read_tokens"] / 1_000_000) * pricing[
                "cache_read_per_mtok"
            ]
            cache_write_cost = (server["cache_write_tokens"] / 1_000_000) * pricing[
                "cache_write_per_mtok"
            ]
            server["total_cost"] = round(
                input_cost + output_cost + cache_read_cost + cache_write_cost, 4
            )

        # Sort servers by total calls
        sorted_servers = sorted(
            servers.values(), key=lambda x: x["total_calls"], reverse=True
        )

        # Calculate totals
        total_mcp_servers = len(servers)
        total_mcp_tools = len(mcp_tools)
        total_mcp_calls = sum(stats.invocation_count for stats in mcp_tools.values())
        total_mcp_tokens = sum(s["total_tokens"] for s in servers.values())
        total_mcp_cost = sum(s["total_cost"] for s in servers.values())
        most_used_server = (
            sorted_servers[0]["server_name"] if sorted_servers else "None"
        )

        return {
            "total_servers": total_mcp_servers,
            "total_tools": total_mcp_tools,
            "total_calls": total_mcp_calls,
            "total_tokens": total_mcp_tokens,
            "total_cost": round(total_mcp_cost, 2),
            "most_used_server": most_used_server,
            "servers": sorted_servers,
        }

    def get_file_statistics(
        self, limit: int = 20, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get file operation statistics from Read, Write, and Edit tool usage.

        Args:
            limit: Maximum number of files to return
            time_filter: Optional time filter to apply

        Returns:
            Dict with file operation statistics
        """
        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(
                time_filter
            )._features_analyzer

        # Get all tool invocations
        file_ops = {}

        for invocation in analyzer.tool_parser.parse_all(time_filter=time_filter):
            if invocation.tool_name in ["Read", "Write", "Edit"]:
                file_path = invocation.input_params.get("file_path")
                if file_path:
                    if file_path not in file_ops:
                        file_ops[file_path] = {
                            "file_path": file_path,
                            "read_count": 0,
                            "write_count": 0,
                            "edit_count": 0,
                            "total_operations": 0,
                            "projects": set(),
                        }

                    tool_type = invocation.tool_name.lower()
                    file_ops[file_path][f"{tool_type}_count"] += 1
                    file_ops[file_path]["total_operations"] += 1

                    if invocation.project:
                        file_ops[file_path]["projects"].add(invocation.project)

        # Convert sets to counts and create list
        files_list = []
        for file_path, ops in file_ops.items():
            ops["project_count"] = len(ops["projects"])
            ops.pop("projects")  # Remove set before returning
            files_list.append(ops)

        # Sort by total operations
        sorted_files = sorted(
            files_list, key=lambda x: x["total_operations"], reverse=True
        )[:limit]

        # Calculate statistics
        total_files = len(file_ops)
        total_operations = sum(f["total_operations"] for f in files_list)
        total_edits = sum(f["edit_count"] for f in files_list)
        total_reads = sum(f["read_count"] for f in files_list)
        total_writes = sum(f["write_count"] for f in files_list)

        most_edited_file = sorted_files[0]["file_path"] if sorted_files else "N/A"
        if most_edited_file != "N/A":
            # Shorten path for display
            import os

            most_edited_file = os.path.basename(most_edited_file)

        avg_edits_per_file = total_edits / total_files if total_files > 0 else 0

        return {
            "total_files": total_files,
            "most_edited_file": most_edited_file,
            "avg_edits_per_file": round(avg_edits_per_file, 1),
            "total_operations": total_operations,
            "total_edits": total_edits,
            "total_reads": total_reads,
            "total_writes": total_writes,
            "file_operations": sorted_files,
        }

    # ========== Public Methods for Full Dashboard ==========

    def get_full_dashboard(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get complete dashboard data combining all metrics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with all dashboard data
        """
        return {
            "usage": self.get_usage_summary(time_filter=time_filter),
            "project_breakdown": self.get_project_breakdown(time_filter=time_filter),
            "tokens": self.get_token_summary(time_filter=time_filter),
            "models": self.get_model_breakdown(time_filter=time_filter),
            "integrations": self.get_integration_summary(time_filter=time_filter),
            "features": self.get_features_summary(time_filter=time_filter),
            "top_tools": self.get_top_tools(limit=5, time_filter=time_filter),
            "top_subagents": self.get_top_subagents(limit=5, time_filter=time_filter),
        }

    # ========== Configuration Methods ==========

    def get_discovered_repositories(self) -> List[Dict]:
        """
        Get all discovered repositories with .claude/ directories.

        Returns:
            List of repository dicts
        """
        repos = self._config_scanner.scan_for_repositories()
        return [repo.to_dict() for repo in repos]

    def get_configuration_features(self, repo_path: str) -> Dict[str, Any]:
        """
        Get all configuration features for a repository.

        Args:
            repo_path: Path to repository as string

        Returns:
            Dict with all features categorized
        """
        path = Path(repo_path)
        return self._configuration_analyzer.get_feature_breakdown(path)

    def get_feature_detail(
        self, repo_path: str, feature_type: str, feature_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific feature.

        Args:
            repo_path: Path to repository as string
            feature_type: Type of feature (skill, command, etc.)
            feature_id: ID/name of the feature

        Returns:
            Dict with feature details
        """
        path = Path(repo_path)
        return self._configuration_analyzer.get_feature_detail(
            feature_type, feature_id, path
        )

    def get_inheritance_chain(
        self, repo_path: str, feature_type: str, feature_id: str
    ) -> List[Dict]:
        """
        Get inheritance chain for a feature.

        Args:
            repo_path: Path to repository as string
            feature_type: Type of feature
            feature_id: ID/name of the feature

        Returns:
            List of inheritance levels
        """
        path = Path(repo_path)
        inheritance_tree = self._configuration_analyzer.get_inheritance_tree(
            feature_type, feature_id, path
        )
        return inheritance_tree.get("levels", [])

    # ========== Project Analysis Methods ==========

    def get_project_analysis(
        self,
        project_path: str,
        project_name: str,
        time_filter: Optional[TimeFilter] = None,
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for a specific project.

        Args:
            project_path: Path to the project
            project_name: Display name of the project
            time_filter: Optional time filter for analysis period

        Returns:
            Dict with project analysis including recommendations and metrics
        """
        analysis = self._project_analyzer.analyze_project(
            project_path=project_path,
            project_name=project_name,
            time_filter=time_filter,
        )
        return analysis.to_dict()

    def analyze_all_projects(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Analyze all projects for optimization opportunities.

        Args:
            time_filter: Optional time filter for analysis period

        Returns:
            Dict with all project analyses and summary
        """
        # Get all projects from project breakdown
        project_data = self.get_project_breakdown(time_filter)
        all_analyses = []

        for project in project_data.get("projects", []):
            # Skip projects with very low usage
            if project.get("command_count", 0) < 2:
                continue

            try:
                analysis = self._project_analyzer.analyze_project(
                    project_path=project.get("full_path", project.get("path", "")),
                    project_name=project.get("name", "Unknown"),
                    time_filter=time_filter,
                )
                all_analyses.append(analysis.to_dict())
            except Exception:
                # Skip projects that can't be analyzed
                continue

        # Sort by total recommendations (high severity first)
        all_analyses.sort(
            key=lambda x: (
                x.get("summary", {}).get("high_severity", 0),
                x.get("summary", {}).get("medium_severity", 0),
                x.get("summary", {}).get("low_severity", 0),
            ),
            reverse=True,
        )

        # Calculate summary stats
        total_high = sum(
            a.get("summary", {}).get("high_severity", 0) for a in all_analyses
        )
        total_medium = sum(
            a.get("summary", {}).get("medium_severity", 0) for a in all_analyses
        )
        total_low = sum(
            a.get("summary", {}).get("low_severity", 0) for a in all_analyses
        )

        return {
            "projects": all_analyses,
            "total_projects_analyzed": len(all_analyses),
            "summary": {
                "total_high_severity": total_high,
                "total_medium_severity": total_medium,
                "total_low_severity": total_low,
                "total_recommendations": total_high + total_medium + total_low,
            },
        }

    # ========== Pricing Settings Methods ==========

    def get_pricing_settings(
        self, additional_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get pricing settings for all models.

        Args:
            additional_models: Optional list of additional model IDs to include
                              (e.g., models discovered from session data)

        Returns:
            Dict with default pricing and custom overrides.
        """
        all_pricing = self._pricing_settings.get_all_pricing(
            additional_models=additional_models
        )
        custom_pricing = self._pricing_settings.get_custom_pricing_summary()

        return {
            "all_pricing": all_pricing,
            "custom_pricing": custom_pricing,
            "has_custom_pricing": len(custom_pricing) > 0,
        }

    def update_model_pricing(
        self,
        model: str,
        input_per_mtok: float,
        output_per_mtok: float,
        cache_write_per_mtok: float,
        cache_read_per_mtok: float,
    ) -> Dict[str, Any]:
        """
        Update pricing for a specific model.

        Args:
            model: Model identifier
            input_per_mtok: Price per million input tokens
            output_per_mtok: Price per million output tokens
            cache_write_per_mtok: Price per million cache write tokens
            cache_read_per_mtok: Price per million cache read tokens

        Returns:
            Dict with success status and updated pricing.
        """
        success = self._pricing_settings.set_pricing_for_model(
            model,
            input_per_mtok,
            output_per_mtok,
            cache_write_per_mtok,
            cache_read_per_mtok,
        )

        if success:
            # Return updated pricing
            updated_pricing = self._pricing_settings.get_pricing_for_model(model)
            return {"success": True, "model": model, "pricing": updated_pricing}
        else:
            return {"success": False, "error": "Failed to save pricing settings"}

    def reset_model_pricing(self, model: str) -> Dict[str, Any]:
        """
        Reset pricing for a specific model to default.

        Args:
            model: Model identifier

        Returns:
            Dict with success status and default pricing.
        """
        success = self._pricing_settings.reset_pricing_for_model(model)

        if success:
            from ...analyzers.tokens import MODEL_PRICING, DEFAULT_PRICING

            default_pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
            return {"success": True, "model": model, "pricing": default_pricing}
        else:
            return {"success": False, "error": "Failed to reset pricing settings"}

    def get_all_models_from_sessions(self) -> List[str]:
        """
        Extract all unique model IDs from all parsed sessions.

        This dynamically discovers which models have been used on this machine,
        allowing the pricing settings page to show any custom models that aren't
        in the default MODEL_PRICING dictionary.

        Returns:
            List of unique model IDs
        """
        all_models = set()

        # Get model-by-project breakdown which contains all models used
        model_by_project = self._token_analyzer.get_model_by_project_breakdown()

        # Extract unique model IDs from all projects
        for project_models in model_by_project.values():
            all_models.update(project_models.keys())

        return sorted(all_models)

    # ========== Sub-Agent Exchange Methods ==========

    def get_subagent_exchanges(
        self,
        time_filter: Optional[TimeFilter] = None,
        project_filter: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get sub-agent exchange data for visualization.

        Args:
            time_filter: Optional time filter to apply
            project_filter: Optional project path to filter by
            limit: Maximum number of exchanges to return

        Returns:
            Dict with exchange data suitable for charting and display
        """
        exchanges = self._subagent_parser.parse_exchanges(
            time_filter=time_filter, project_filter=project_filter
        )

        # Limit results
        exchanges = exchanges[:limit]

        # Convert to serializable format
        exchanges_data = []
        for e in exchanges:
            exchanges_data.append(
                {
                    "agent_id": e.agent_id,
                    "session_id": e.session_id,
                    "project": e.project,
                    "project_name": Path(e.project).name if e.project else "Unknown",
                    "timestamp": e.timestamp,
                    "date": e.timestamp[:10] if e.timestamp else "",
                    "time": e.timestamp[11:19] if len(e.timestamp) > 19 else "",
                    "duration_ms": e.duration_ms,
                    "duration_seconds": round(e.duration_seconds, 1),
                    "subagent_type": e.subagent_type or "unknown",
                    "description": e.description,
                    "prompt": e.prompt,
                    "prompt_preview": e.prompt[:200] + "..."
                    if len(e.prompt) > 200
                    else e.prompt,
                    "result_text": e.result_text,
                    "result_preview": e.result_text[:300] + "..."
                    if len(e.result_text) > 300
                    else e.result_text,
                    "total_tokens": e.total_tokens,
                    "total_tool_use_count": e.total_tool_use_count,
                    "cost": round(e.subagent_cost, 4),
                    "status": e.status,
                    "subagent_usage": {
                        "input_tokens": e.subagent_usage.input_tokens
                        if e.subagent_usage
                        else 0,
                        "output_tokens": e.subagent_usage.output_tokens
                        if e.subagent_usage
                        else 0,
                        "cache_creation_tokens": e.subagent_usage.cache_creation_input_tokens
                        if e.subagent_usage
                        else 0,
                        "cache_read_tokens": e.subagent_usage.cache_read_input_tokens
                        if e.subagent_usage
                        else 0,
                    },
                    "parent_usage": {
                        "input_tokens": e.parent_usage.input_tokens
                        if e.parent_usage
                        else 0,
                        "output_tokens": e.parent_usage.output_tokens
                        if e.parent_usage
                        else 0,
                        "cache_creation_tokens": e.parent_usage.cache_creation_input_tokens
                        if e.parent_usage
                        else 0,
                        "cache_read_tokens": e.parent_usage.cache_read_input_tokens
                        if e.parent_usage
                        else 0,
                    },
                }
            )

        return {
            "exchanges": exchanges_data,
            "total_count": len(exchanges_data),
        }

    def get_subagent_summary(
        self, time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated sub-agent exchange statistics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with summary statistics
        """
        stats = self._subagent_parser.get_exchange_stats(time_filter=time_filter)

        # Get type breakdown as list for charts
        type_breakdown = []
        for type_name, type_stats in stats.get("by_type", {}).items():
            type_breakdown.append(
                {
                    "type": type_name,
                    "count": type_stats["count"],
                    "total_tokens": type_stats["total_tokens"],
                    "total_cost": round(type_stats["total_cost"], 2),
                }
            )

        # Sort by count descending
        type_breakdown.sort(key=lambda x: x["count"], reverse=True)

        return {
            "total_exchanges": stats["total_exchanges"],
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
            "avg_tokens_per_exchange": stats["avg_tokens_per_exchange"],
            "avg_duration_seconds": stats["avg_duration_seconds"],
            "type_breakdown": type_breakdown,
            "unique_types": len(type_breakdown),
        }

    def get_subagent_exchange_detail(
        self, agent_id: str, time_filter: Optional[TimeFilter] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific sub-agent exchange.

        Args:
            agent_id: The agent ID to look up
            time_filter: Optional time filter

        Returns:
            Dict with exchange details or None if not found
        """
        exchanges = self._subagent_parser.parse_exchanges(time_filter=time_filter)

        for e in exchanges:
            if e.agent_id == agent_id:
                return {
                    "agent_id": e.agent_id,
                    "session_id": e.session_id,
                    "project": e.project,
                    "project_name": Path(e.project).name if e.project else "Unknown",
                    "timestamp": e.timestamp,
                    "duration_ms": e.duration_ms,
                    "duration_seconds": round(e.duration_seconds, 1),
                    "subagent_type": e.subagent_type or "unknown",
                    "description": e.description,
                    "prompt": e.prompt,
                    "result_text": e.result_text,
                    "total_tokens": e.total_tokens,
                    "total_tool_use_count": e.total_tool_use_count,
                    "cost": round(e.subagent_cost, 4),
                    "status": e.status,
                    "subagent_usage": {
                        "input_tokens": e.subagent_usage.input_tokens
                        if e.subagent_usage
                        else 0,
                        "output_tokens": e.subagent_usage.output_tokens
                        if e.subagent_usage
                        else 0,
                        "cache_creation_tokens": e.subagent_usage.cache_creation_input_tokens
                        if e.subagent_usage
                        else 0,
                        "cache_read_tokens": e.subagent_usage.cache_read_input_tokens
                        if e.subagent_usage
                        else 0,
                    },
                    "parent_usage": {
                        "input_tokens": e.parent_usage.input_tokens
                        if e.parent_usage
                        else 0,
                        "output_tokens": e.parent_usage.output_tokens
                        if e.parent_usage
                        else 0,
                        "cache_creation_tokens": e.parent_usage.cache_creation_input_tokens
                        if e.parent_usage
                        else 0,
                        "cache_read_tokens": e.parent_usage.cache_read_input_tokens
                        if e.parent_usage
                        else 0,
                    },
                }

        return None

    def get_subagent_chart_data(
        self, time_filter: Optional[TimeFilter] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get sub-agent exchange data formatted for Chart.js scatter/bubble chart.

        Args:
            time_filter: Optional time filter to apply
            limit: Maximum number of exchanges to return

        Returns:
            Dict with chart-ready data
        """
        exchanges = self._subagent_parser.parse_exchanges(time_filter=time_filter)
        exchanges = exchanges[:limit]

        # Color map for different subagent types
        type_colors = {
            "explore": "#0770E3",  # Blue
            "Explore": "#0770E3",
            "general": "#34D399",  # Green
            "research": "#F59E0B",  # Amber
            "code": "#8B5CF6",  # Purple
            "unknown": "#6B7280",  # Gray
        }

        # Group exchanges by subagent_type for datasets
        type_data: Dict[str, list] = {}
        for e in exchanges:
            type_name = e.subagent_type or "unknown"
            if type_name not in type_data:
                type_data[type_name] = []

            # Create data point
            type_data[type_name].append(
                {
                    "x": e.timestamp,
                    "y": e.total_tokens,
                    "r": min(20, max(5, e.duration_seconds / 10))
                    if e.duration_seconds
                    else 5,
                    "agent_id": e.agent_id,
                    "cost": round(e.subagent_cost, 4),
                    "duration": round(e.duration_seconds, 1),
                    "description": e.description[:50] if e.description else "",
                }
            )

        # Build datasets for Chart.js
        datasets = []
        for type_name, points in type_data.items():
            color = type_colors.get(type_name, type_colors["unknown"])
            datasets.append(
                {
                    "label": type_name,
                    "data": points,
                    "backgroundColor": color + "80",  # Add alpha
                    "borderColor": color,
                }
            )

        return {
            "datasets": datasets,
            "total_points": sum(len(d["data"]) for d in datasets),
        }

    # ========== Tool Invocation Methods ==========

    def get_tool_invocations(
        self,
        tool_name: str,
        time_filter: Optional[TimeFilter] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get individual invocations for a specific tool grouped by session.

        Args:
            tool_name: The name of the tool to get invocations for
            time_filter: Optional time filter to apply
            limit: Maximum number of invocations to return

        Returns:
            Dict with tool details and invocations grouped by session
        """
        from ...analyzers.tokens import DEFAULT_PRICING

        tool_parser = self._features_analyzer.tool_parser

        # Collect all invocations for this tool
        invocations = []
        for inv in tool_parser.parse_all(time_filter=time_filter):
            if inv.tool_name == tool_name:
                invocations.append(inv)

        # Sort by timestamp descending
        invocations.sort(key=lambda x: x.timestamp, reverse=True)
        invocations = invocations[:limit]

        # Group by session
        sessions_map: Dict[str, Dict[str, Any]] = {}
        for inv in invocations:
            session_id = inv.session_id
            if session_id not in sessions_map:
                sessions_map[session_id] = {
                    "session_id": session_id,
                    "project": inv.project,
                    "project_name": Path(inv.project).name
                    if inv.project
                    else "Unknown",
                    "invocations": [],
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "invocation_count": 0,
                    "first_timestamp": inv.timestamp,
                    "last_timestamp": inv.timestamp,
                }

            session = sessions_map[session_id]
            session["invocations"].append(
                {
                    "timestamp": inv.timestamp,
                    "date": inv.timestamp[:10] if inv.timestamp else "",
                    "time": inv.timestamp[11:19] if len(inv.timestamp) > 19 else "",
                    "input_params": inv.input_params,
                    "input_tokens": inv.input_tokens,
                    "output_tokens": inv.output_tokens,
                    "cache_read_tokens": inv.cache_read_tokens,
                    "cache_write_tokens": inv.cache_write_tokens,
                    "total_tokens": inv.total_tokens,
                }
            )
            session["total_tokens"] += inv.total_tokens
            session["input_tokens"] += inv.input_tokens
            session["output_tokens"] += inv.output_tokens
            session["cache_read_tokens"] += inv.cache_read_tokens
            session["cache_write_tokens"] += inv.cache_write_tokens
            session["invocation_count"] += 1

            # Track first/last timestamps
            if inv.timestamp < session["first_timestamp"]:
                session["first_timestamp"] = inv.timestamp
            if inv.timestamp > session["last_timestamp"]:
                session["last_timestamp"] = inv.timestamp

        # Calculate costs for each session
        pricing = DEFAULT_PRICING
        for session in sessions_map.values():
            input_cost = (session["input_tokens"] / 1_000_000) * pricing[
                "input_per_mtok"
            ]
            output_cost = (session["output_tokens"] / 1_000_000) * pricing[
                "output_per_mtok"
            ]
            cache_read_cost = (session["cache_read_tokens"] / 1_000_000) * pricing[
                "cache_read_per_mtok"
            ]
            cache_write_cost = (session["cache_write_tokens"] / 1_000_000) * pricing[
                "cache_write_per_mtok"
            ]
            session["cost"] = round(
                input_cost + output_cost + cache_read_cost + cache_write_cost, 4
            )

        # Convert to list and sort by last_timestamp
        sessions = list(sessions_map.values())
        sessions.sort(key=lambda x: x["last_timestamp"], reverse=True)

        # Calculate totals
        total_tokens = sum(s["total_tokens"] for s in sessions)
        total_invocations = sum(s["invocation_count"] for s in sessions)
        total_cost = sum(s["cost"] for s in sessions)
        total_input = sum(s["input_tokens"] for s in sessions)
        total_output = sum(s["output_tokens"] for s in sessions)
        total_cache_read = sum(s["cache_read_tokens"] for s in sessions)
        total_cache_write = sum(s["cache_write_tokens"] for s in sessions)

        return {
            "tool_name": tool_name,
            "sessions": sessions,
            "session_count": len(sessions),
            "total_invocations": total_invocations,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "avg_tokens_per_call": round(total_tokens / total_invocations, 1)
            if total_invocations > 0
            else 0,
        }

    def get_tool_chart_data(
        self,
        tool_name: str,
        time_filter: Optional[TimeFilter] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get tool invocation data formatted for Chart.js scatter/bubble chart.

        Args:
            tool_name: The name of the tool
            time_filter: Optional time filter to apply
            limit: Maximum number of invocations to return

        Returns:
            Dict with chart-ready data
        """
        from ...analyzers.tokens import DEFAULT_PRICING

        tool_parser = self._features_analyzer.tool_parser
        pricing = DEFAULT_PRICING

        # Collect all invocations for this tool
        invocations = []
        for inv in tool_parser.parse_all(time_filter=time_filter):
            if inv.tool_name == tool_name:
                invocations.append(inv)

        # Sort by timestamp descending and limit
        invocations.sort(key=lambda x: x.timestamp, reverse=True)
        invocations = invocations[:limit]

        # Build chart data points
        data_points = []
        for idx, inv in enumerate(invocations):
            # Calculate cost for this invocation
            input_cost = (inv.input_tokens / 1_000_000) * pricing["input_per_mtok"]
            output_cost = (inv.output_tokens / 1_000_000) * pricing["output_per_mtok"]
            cache_read_cost = (inv.cache_read_tokens / 1_000_000) * pricing[
                "cache_read_per_mtok"
            ]
            cache_write_cost = (inv.cache_write_tokens / 1_000_000) * pricing[
                "cache_write_per_mtok"
            ]
            cost = input_cost + output_cost + cache_read_cost + cache_write_cost

            # Generate a unique ID for this invocation
            invocation_id = f"{inv.session_id}_{inv.timestamp}_{idx}"

            # Get a preview of what the tool did
            params_preview = self._get_tool_params_preview(
                inv.tool_name, inv.input_params
            )

            data_points.append(
                {
                    "x": inv.timestamp,
                    "y": inv.total_tokens,
                    "r": min(15, max(4, inv.total_tokens / 1000)),  # Scale bubble size
                    "invocation_id": invocation_id,
                    "session_id": inv.session_id,
                    "project": Path(inv.project).name if inv.project else "Unknown",
                    "cost": round(cost, 6),
                    "input_tokens": inv.input_tokens,
                    "output_tokens": inv.output_tokens,
                    "cache_read_tokens": inv.cache_read_tokens,
                    "cache_write_tokens": inv.cache_write_tokens,
                    "params_preview": params_preview,
                }
            )

        return {
            "tool_name": tool_name,
            "datasets": [
                {
                    "label": tool_name,
                    "data": data_points,
                    "backgroundColor": "#0770E380",  # Brand blue with alpha
                    "borderColor": "#0770E3",
                }
            ],
            "total_points": len(data_points),
        }

    def _get_tool_params_preview(self, tool_name: str, params: dict) -> str:
        """Get a human-readable preview of tool parameters."""
        if not params:
            return ""

        # Tool-specific previews
        if tool_name == "Read":
            file_path = params.get("filePath", params.get("file_path", ""))
            if file_path:
                return Path(file_path).name
        elif tool_name == "Write":
            file_path = params.get("filePath", params.get("file_path", ""))
            if file_path:
                return f"Write: {Path(file_path).name}"
        elif tool_name == "Edit":
            file_path = params.get("filePath", params.get("file_path", ""))
            if file_path:
                return f"Edit: {Path(file_path).name}"
        elif tool_name == "Bash":
            command = params.get("command", "")
            if command:
                return command[:50] + ("..." if len(command) > 50 else "")
        elif tool_name == "Glob":
            pattern = params.get("pattern", "")
            return pattern[:40] if pattern else ""
        elif tool_name == "Grep":
            pattern = params.get("pattern", "")
            return f"/{pattern[:30]}/" if pattern else ""
        elif tool_name == "Task":
            desc = params.get("description", "")
            return desc[:40] + ("..." if len(desc) > 40 else "")
        elif tool_name == "WebFetch":
            url = params.get("url", "")
            if url:
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    return parsed.netloc[:30]
                except Exception:
                    return url[:30]

        # Generic fallback - show first key-value
        for key, value in params.items():
            if isinstance(value, str) and value:
                return f"{key}: {value[:30]}..."
            break

        return ""

    def get_tool_invocation_detail(
        self,
        tool_name: str,
        invocation_id: str,
        time_filter: Optional[TimeFilter] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool invocation.

        Args:
            tool_name: The name of the tool
            invocation_id: The invocation ID (session_id_timestamp_idx format)
            time_filter: Optional time filter

        Returns:
            Dict with invocation details or None if not found
        """
        from ...analyzers.tokens import DEFAULT_PRICING

        tool_parser = self._features_analyzer.tool_parser
        pricing = DEFAULT_PRICING

        # Parse invocation_id to extract session_id and timestamp
        parts = invocation_id.rsplit("_", 2)
        if len(parts) < 2:
            return None

        target_session = parts[0]
        target_timestamp = parts[1] if len(parts) >= 2 else ""

        # Find the matching invocation
        for inv in tool_parser.parse_all(time_filter=time_filter):
            if inv.tool_name != tool_name:
                continue

            if inv.session_id == target_session and inv.timestamp == target_timestamp:
                # Calculate cost
                input_cost = (inv.input_tokens / 1_000_000) * pricing["input_per_mtok"]
                output_cost = (inv.output_tokens / 1_000_000) * pricing[
                    "output_per_mtok"
                ]
                cache_read_cost = (inv.cache_read_tokens / 1_000_000) * pricing[
                    "cache_read_per_mtok"
                ]
                cache_write_cost = (inv.cache_write_tokens / 1_000_000) * pricing[
                    "cache_write_per_mtok"
                ]
                cost = input_cost + output_cost + cache_read_cost + cache_write_cost

                return {
                    "tool_name": inv.tool_name,
                    "timestamp": inv.timestamp,
                    "date": inv.timestamp[:10] if inv.timestamp else "",
                    "time": inv.timestamp[11:19] if len(inv.timestamp) > 19 else "",
                    "session_id": inv.session_id,
                    "project": inv.project,
                    "project_name": Path(inv.project).name
                    if inv.project
                    else "Unknown",
                    "input_params": inv.input_params,
                    "input_tokens": inv.input_tokens,
                    "output_tokens": inv.output_tokens,
                    "cache_read_tokens": inv.cache_read_tokens,
                    "cache_write_tokens": inv.cache_write_tokens,
                    "total_tokens": inv.total_tokens,
                    "cost": round(cost, 6),
                    "params_preview": self._get_tool_params_preview(
                        inv.tool_name, inv.input_params
                    ),
                }

        return None

    def get_unified_timeline_data(
        self,
        time_filter: Optional[TimeFilter] = None,
        session_id: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """
        Get all tool invocations across all tools for a unified timeline view.

        Args:
            time_filter: Optional time filter to apply
            session_id: Optional session ID to filter to a specific conversation
            limit: Maximum number of invocations to return

        Returns:
            Dict with timeline data including chart data and session list
        """
        from ...analyzers.tokens import DEFAULT_PRICING

        tool_parser = self._features_analyzer.tool_parser
        pricing = DEFAULT_PRICING

        # Collect all invocations across all tools
        invocations = []
        sessions_set = set()

        for inv in tool_parser.parse_all(time_filter=time_filter):
            # Filter by session if specified
            if session_id and inv.session_id != session_id:
                continue

            invocations.append(inv)
            sessions_set.add(
                (
                    inv.session_id,
                    Path(inv.project).name if inv.project else "Unknown",
                )
            )

        # Sort by timestamp ascending for timeline view
        invocations.sort(key=lambda x: x.timestamp)
        invocations = invocations[-limit:]  # Keep most recent

        # Define colors for different tools
        tool_colors = {
            "Read": {"bg": "#3B82F680", "border": "#3B82F6"},  # Blue
            "Write": {"bg": "#10B98180", "border": "#10B981"},  # Green
            "Edit": {"bg": "#F59E0B80", "border": "#F59E0B"},  # Amber
            "Bash": {"bg": "#6366F180", "border": "#6366F1"},  # Indigo
            "Glob": {"bg": "#8B5CF680", "border": "#8B5CF6"},  # Violet
            "Grep": {"bg": "#EC489980", "border": "#EC4899"},  # Pink
            "Task": {"bg": "#EF444480", "border": "#EF4444"},  # Red
            "WebFetch": {"bg": "#06B6D480", "border": "#06B6D4"},  # Cyan
            "TodoWrite": {"bg": "#84CC1680", "border": "#84CC16"},  # Lime
            "TodoRead": {"bg": "#22C55E80", "border": "#22C55E"},  # Green
        }
        default_color = {"bg": "#6B728080", "border": "#6B7280"}  # Gray

        # Group invocations by tool for datasets
        tools_data: Dict[str, list] = {}
        all_points = []

        for idx, inv in enumerate(invocations):
            tool_name = inv.tool_name

            if tool_name not in tools_data:
                tools_data[tool_name] = []

            # Calculate cost
            input_cost = (inv.input_tokens / 1_000_000) * pricing["input_per_mtok"]
            output_cost = (inv.output_tokens / 1_000_000) * pricing["output_per_mtok"]
            cache_read_cost = (inv.cache_read_tokens / 1_000_000) * pricing[
                "cache_read_per_mtok"
            ]
            cache_write_cost = (inv.cache_write_tokens / 1_000_000) * pricing[
                "cache_write_per_mtok"
            ]
            cost = input_cost + output_cost + cache_read_cost + cache_write_cost

            # Generate invocation ID
            invocation_id = f"{inv.session_id}_{inv.timestamp}_{idx}"

            # Get preview
            params_preview = self._get_tool_params_preview(
                inv.tool_name, inv.input_params
            )

            point = {
                "x": inv.timestamp,
                "y": inv.total_tokens,
                "r": min(15, max(4, inv.total_tokens / 1000)),
                "invocation_id": invocation_id,
                "tool_name": tool_name,
                "session_id": inv.session_id,
                "project": Path(inv.project).name if inv.project else "Unknown",
                "cost": round(cost, 6),
                "input_tokens": inv.input_tokens,
                "output_tokens": inv.output_tokens,
                "cache_read_tokens": inv.cache_read_tokens,
                "cache_write_tokens": inv.cache_write_tokens,
                "params_preview": params_preview,
            }

            tools_data[tool_name].append(point)
            all_points.append(point)

        # Build datasets for Chart.js
        datasets = []
        for tool_name, points in sorted(tools_data.items()):
            colors = tool_colors.get(tool_name, default_color)
            datasets.append(
                {
                    "label": tool_name,
                    "data": points,
                    "backgroundColor": colors["bg"],
                    "borderColor": colors["border"],
                }
            )

        # Build sessions list for filter dropdown
        sessions_list = [
            {"session_id": sid, "project_name": proj}
            for sid, proj in sorted(sessions_set, key=lambda x: x[1])
        ]

        # Calculate totals
        total_tokens = sum(p["y"] for p in all_points)
        total_cost = sum(p["cost"] for p in all_points)
        tool_counts = {tool: len(points) for tool, points in tools_data.items()}

        return {
            "datasets": datasets,
            "sessions": sessions_list,
            "total_invocations": len(all_points),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "tool_counts": tool_counts,
            "selected_session": session_id,
            "invocations": all_points,  # For table view
        }
