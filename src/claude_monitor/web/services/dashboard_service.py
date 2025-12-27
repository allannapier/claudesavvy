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
from ...parsers.history import HistoryParser
from ...parsers.sessions import SessionParser
from ...parsers.debug import DebugLogParser
from ...parsers.files import FileHistoryParser
from ...parsers.tools import ToolUsageParser
from ...parsers.skills import SkillsParser, ConfigurationParser
from ...parsers.configuration_scanner import ConfigurationScanner
from ...analyzers.usage import UsageAnalyzer, UsageSummary
from ...analyzers.tokens import TokenAnalyzer, TokenSummary, get_model_display_name
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
            self.paths.file_history_dir,
            self.paths.get_project_session_files()
        )
        self._tool_parser = ToolUsageParser(self.paths.get_project_session_files())
        self._skills_parser = SkillsParser(self.paths.base_dir / "skills")
        self._config_parser = ConfigurationParser(self.paths.base_dir)

        # Build project map for debug logs
        project_map = self._build_project_map()
        self._debug_parser = DebugLogParser(self.paths.get_debug_log_files(), project_map)

        # Initialize analyzers (no time filter by default)
        self._usage_analyzer = UsageAnalyzer(
            self._history_parser,
            self._session_parser,
            time_filter=None
        )
        self._token_analyzer = TokenAnalyzer(
            self._session_parser,
            time_filter=None
        )
        self._integration_analyzer = IntegrationAnalyzer(self._debug_parser)
        self._features_analyzer = FeaturesAnalyzer(
            self._tool_parser,
            self._skills_parser,
            self._debug_parser,
            self._config_parser,
            time_filter=None
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
                reverse=True
            )

            for project_dir in all_dirs:
                encoded_name = project_dir.name
                for session_file in project_dir.glob("*.jsonl"):
                    session_id = session_file.stem
                    if session_id not in project_paths or len(encoded_name) > len(project_paths[session_id]):
                        # Extract project name from encoded path
                        if encoded_name.startswith('-Users-allannapier-code-'):
                            project_name = encoded_name.replace('-Users-allannapier-code-', '', 1)
                        elif encoded_name.startswith('-'):
                            project_name = encoded_name[1:]
                        else:
                            project_name = encoded_name
                        project_paths[session_id] = project_name

        return project_paths

    def _create_time_filtered_service(self, time_filter: TimeFilter) -> 'DashboardService':
        """
        Create a new service instance with time filter applied.

        Args:
            time_filter: TimeFilter to apply

        Returns:
            DashboardService instance with time-filtered analyzers
        """
        service = DashboardService.__new__(DashboardService)
        service.paths = self.paths
        service._history_parser = self._history_parser
        service._session_parser = self._session_parser
        service._file_parser = self._file_parser
        service._tool_parser = self._tool_parser
        service._skills_parser = self._skills_parser
        service._config_parser = self._config_parser
        service._debug_parser = self._debug_parser

        # Create time-filtered analyzers
        service._usage_analyzer = UsageAnalyzer(
            self._history_parser,
            self._session_parser,
            time_filter=time_filter
        )
        service._token_analyzer = TokenAnalyzer(
            self._session_parser,
            time_filter=time_filter
        )
        service._integration_analyzer = IntegrationAnalyzer(self._debug_parser)
        service._features_analyzer = FeaturesAnalyzer(
            self._tool_parser,
            self._skills_parser,
            self._debug_parser,
            self._config_parser,
            time_filter=time_filter
        )

        return service

    # ========== Public Methods for Usage Data ==========

    def get_usage_summary(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get usage summary statistics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with usage statistics suitable for web rendering
        """
        analyzer = self._usage_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._usage_analyzer

        summary: UsageSummary = analyzer.get_summary()

        # Get actual project count from project breakdown
        project_data = self.get_project_breakdown(time_filter)
        actual_project_count = project_data.get('total_projects', 0)

        return {
            'total_commands': summary.total_commands,
            'total_sessions': summary.total_sessions,
            'total_messages': summary.total_messages,
            'total_projects': actual_project_count,  # Use actual count from projects
            'earliest_activity': summary.earliest_activity.isoformat() if summary.earliest_activity else None,
            'latest_activity': summary.latest_activity.isoformat() if summary.latest_activity else None,
            'time_range_description': summary.time_range_description,
            'date_range_days': summary.date_range_days,
            'avg_commands_per_day': round(summary.avg_commands_per_day, 2) if summary.avg_commands_per_day else None,
            'avg_sessions_per_day': round(summary.avg_sessions_per_day, 2) if summary.avg_sessions_per_day else None,
        }

    def get_project_breakdown(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get per-project activity breakdown.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict mapping project paths to activity metrics
        """
        analyzer = self._usage_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._usage_analyzer

        breakdown = analyzer.get_project_breakdown()

        # Get token breakdown for cost and token data
        token_analyzer = self._token_analyzer
        if time_filter:
            token_analyzer = self._create_time_filtered_service(time_filter)._token_analyzer
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
                total_tokens = (
                    token_summary.total_tokens.input_tokens +
                    token_summary.total_tokens.output_tokens +
                    token_summary.total_tokens.cache_creation_input_tokens +
                    token_summary.total_tokens.cache_read_input_tokens
                )
                cost = token_summary.total_cost
                total_cost += cost

            # Determine most active project (by commands)
            if project_data['commands'] > max_commands:
                max_commands = project_data['commands']
                most_active_project = project_data['name']

            projects_data.append({
                'path': project_path,
                'name': project_data['name'],
                'full_path': project_data['full_path'],
                # Field names expected by template
                'command_count': project_data['commands'],
                'session_count': project_data['sessions'],
                'message_count': project_data['messages'],
                'total_tokens': total_tokens,
                'total_cost': cost,
                # Keep original names for backward compatibility
                'commands': project_data['commands'],
                'sessions': project_data['sessions'],
                'messages': project_data['messages'],
            })

        # Sort by command count (descending)
        projects_data.sort(key=lambda x: x['command_count'], reverse=True)

        # Calculate average commands per project
        avg_commands = sum(p['command_count'] for p in projects_data) / len(projects_data) if projects_data else 0

        return {
            'projects': projects_data,
            'total_projects': len(projects_data),
            'total_cost': round(total_cost, 2),
            'avg_commands_per_project': round(avg_commands, 1),
            'most_active_project': most_active_project or 'N/A'
        }

    def get_project_breakdown_by_model(
        self,
        model_id: str,
        time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get per-project activity breakdown filtered by a specific model.

        Args:
            model_id: Model ID to filter by
            time_filter: Optional time filter to apply

        Returns:
            Dict with project data filtered to only show usage of the specified model
        """
        analyzer = self._usage_analyzer
        token_analyzer = self._token_analyzer
        if time_filter:
            filtered_service = self._create_time_filtered_service(time_filter)
            analyzer = filtered_service._usage_analyzer
            token_analyzer = filtered_service._token_analyzer

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
                total_tokens = (
                    tokens.input_tokens +
                    tokens.output_tokens +
                    tokens.cache_creation_input_tokens +
                    tokens.cache_read_input_tokens
                )
                total_cost += cost

                # Track most active by cost for model filter
                if cost > max_cost:
                    max_cost = cost
                    most_active_project = project_data['name']

                projects_data.append({
                    'path': project_path,
                    'name': project_data['name'],
                    'full_path': project_data['full_path'],
                    'command_count': project_data['commands'],
                    'session_count': project_data['sessions'],
                    'message_count': project_data['messages'],
                    'total_tokens': total_tokens,
                    'total_cost': cost,
                    'commands': project_data['commands'],
                    'sessions': project_data['sessions'],
                    'messages': project_data['messages'],
                })

        # Sort by cost (descending) for model filter
        projects_data.sort(key=lambda x: x['total_cost'], reverse=True)

        avg_commands = sum(p['command_count'] for p in projects_data) / len(projects_data) if projects_data else 0

        return {
            'projects': projects_data,
            'total_projects': len(projects_data),
            'total_cost': round(total_cost, 2),
            'avg_commands_per_project': round(avg_commands, 1),
            'most_active_project': most_active_project or 'N/A'
        }

    # ========== Public Methods for Token Data ==========

    def get_token_summary(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get token usage summary with cost calculations.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with token statistics and costs (flattened for template use)
        """
        analyzer = self._token_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._token_analyzer

        summary: TokenSummary = analyzer.get_summary()

        return {
            # Flat structure for templates
            'input_tokens': summary.total_tokens.input_tokens,
            'output_tokens': summary.total_tokens.output_tokens,
            'cache_creation_tokens': summary.total_tokens.cache_creation_input_tokens,
            'cache_read_tokens': summary.total_tokens.cache_read_input_tokens,
            'input_cost': round(summary.cost_breakdown.input_cost, 2),
            'output_cost': round(summary.cost_breakdown.output_cost, 2),
            'cache_creation_cost': round(summary.cost_breakdown.cache_write_cost, 2),
            'cache_read_cost': round(summary.cost_breakdown.cache_read_cost, 2),
            'total_cost': round(summary.cost_breakdown.total_cost, 2),
            'cache_savings': round(summary.cost_breakdown.cache_savings, 2),
            'cache_hit_rate': round(summary.cache_efficiency_pct, 1),
        }

    def get_model_breakdown(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get per-model token usage and costs.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict mapping model names to token usage and costs
        """
        analyzer = self._token_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._token_analyzer

        breakdown = analyzer.get_model_breakdown()

        models_data = []
        total_cost = sum(cost for _, cost in breakdown.values())

        for model_id, (tokens, cost) in breakdown.items():
            total_tokens = (
                tokens.input_tokens +
                tokens.output_tokens +
                tokens.cache_creation_input_tokens +
                tokens.cache_read_input_tokens
            )
            models_data.append({
                'model_id': model_id,
                'model_name': get_model_display_name(model_id),
                'input_tokens': tokens.input_tokens,
                'output_tokens': tokens.output_tokens,
                'cache_creation_tokens': tokens.cache_creation_input_tokens,
                'cache_read_tokens': tokens.cache_read_input_tokens,
                'total_tokens': total_tokens,
                'cost': round(cost, 4),
                'cost_percentage': round((cost / total_cost * 100) if total_cost > 0 else 0, 1),
            })

        return {
            'models': models_data,
            'total_models': len(models_data),
            'total_cost': round(total_cost, 2),
        }

    def get_available_models(self, time_filter: Optional[TimeFilter] = None) -> List[Dict[str, str]]:
        """
        Get list of available models for filtering.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            List of dicts with model id and display name
        """
        breakdown = self.get_model_breakdown(time_filter)
        return [
            {'id': model['model_id'], 'name': model['model_name']}
            for model in breakdown['models']
        ]

    def get_daily_token_trend(
        self,
        days: int = 7,
        time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get daily token usage for trend chart.

        Args:
            days: Number of days to include (default 7). Note: when time_filter
                  is provided, days is calculated from the filter's start_time
                  and this parameter is ignored.
            time_filter: Optional time filter - if provided, overrides days parameter

        Returns:
            Dict with labels and data arrays for charting
        """
        # If time filter is provided, calculate days from it (overrides days param)
        if time_filter and time_filter.start_time:
            now = datetime.now()
            delta = now - time_filter.start_time
            days = max(1, delta.days + 1)  # Include current partial day, at least 1 day

        daily_stats = self._session_parser.get_daily_stats(days=days, time_filter=time_filter)

        labels = []
        data = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for date_str in sorted(daily_stats.keys()):
            stats = daily_stats[date_str]
            # Calculate total tokens for this day
            total = (
                stats.total_tokens.input_tokens +
                stats.total_tokens.output_tokens +
                stats.total_tokens.cache_creation_input_tokens +
                stats.total_tokens.cache_read_input_tokens
            )
            data.append(total)

            # Create human-readable label based on how many days we're showing
            date = datetime.strptime(date_str, '%Y-%m-%d')
            days_ago = (today - date).days

            if days <= 7:
                # For week or less, show relative labels
                if days_ago == 0:
                    labels.append('Today')
                elif days_ago == 1:
                    labels.append('Yesterday')
                else:
                    labels.append(f'{days_ago}d ago')
            else:
                # For longer periods, show date labels (Dec 15 format)
                labels.append(date.strftime('%b %d'))

        return {
            'labels': labels,
            'data': data,
        }

    def get_daily_cost_trend(
        self,
        days: int = 7,
        time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get daily cost trend for charts.

        Args:
            days: Number of days to include (default 7)
            time_filter: Optional time filter - if provided, overrides days parameter

        Returns:
            Dict with labels and datasets for Chart.js
        """
        # If time filter is provided, calculate days from it
        if time_filter and time_filter.start_time:
            now = datetime.now()
            delta = now - time_filter.start_time
            days = max(1, delta.days + 1)

        daily_costs = self._session_parser.get_daily_cost_trend(days=days, time_filter=time_filter)

        labels = []
        total_costs = []
        input_costs = []
        output_costs = []
        cache_costs = []

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for date_str in sorted(daily_costs.keys()):
            costs = daily_costs[date_str]
            total_costs.append(costs['total_cost'])
            input_costs.append(costs['input_cost'])
            output_costs.append(costs['output_cost'])
            cache_costs.append(costs['cache_write_cost'] + costs['cache_read_cost'])

            # Create human-readable label based on how many days we're showing
            date = datetime.strptime(date_str, '%Y-%m-%d')
            days_ago = (today - date).days

            if days <= 7:
                # For week or less, show relative labels
                if days_ago == 0:
                    labels.append('Today')
                elif days_ago == 1:
                    labels.append('Yesterday')
                else:
                    labels.append(f'{days_ago}d ago')
            else:
                # For longer periods, show date labels (Dec 15 format)
                labels.append(date.strftime('%b %d'))

        return {
            'labels': labels,
            'datasets': {
                'total': total_costs,
                'input': input_costs,
                'output': output_costs,
                'cache': cache_costs,
            }
        }

    def get_project_cost_trend(
        self,
        days: int = 7,
        time_filter: Optional[TimeFilter] = None,
        max_projects: int = 8
    ) -> Dict[str, Any]:
        """
        Get daily cost trend per project for charts.

        Args:
            days: Number of days to include (default 7)
            time_filter: Optional time filter - if provided, overrides days parameter
            max_projects: Maximum number of projects to include

        Returns:
            Dict with labels and datasets for Chart.js line chart
        """
        # If time filter is provided, calculate days from it
        if time_filter and time_filter.start_time:
            now = datetime.now()
            delta = now - time_filter.start_time
            days = max(1, delta.days + 1)

        project_daily_stats = self._session_parser.get_project_daily_stats(
            days=days,
            time_filter=time_filter,
            max_projects=max_projects
        )

        # Get date labels from the first project
        labels = []
        if project_daily_stats:
            first_project = list(project_daily_stats.values())[0]
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            for date_str in sorted(first_project.keys()):
                # Create human-readable label
                date = datetime.strptime(date_str, '%Y-%m-%d')
                days_ago = (today - date).days

                if days <= 7:
                    if days_ago == 0:
                        labels.append('Today')
                    elif days_ago == 1:
                        labels.append('Yesterday')
                    else:
                        labels.append(f'{days_ago}d ago')
                else:
                    labels.append(date.strftime('%b %d'))

        # Build datasets for each project
        datasets = []
        colors = [
            '#0770E3',  # Blue
            '#34D399',  # Green
            '#F59E0B',  # Amber
            '#8B5CF6',  # Purple
            '#EC4899',  # Pink
            '#14B8A6',  # Teal
            '#F97316',  # Orange
            '#6366F1',  # Indigo
        ]

        for idx, (project_name, daily_data) in enumerate(project_daily_stats.items()):
            color = colors[idx % len(colors)]
            data = [daily_data[date]['total_cost'] for date in sorted(daily_data.keys())]

            datasets.append({
                'label': project_name,
                'data': data,
                'borderColor': color,
                'color': color,
            })

        return {
            'labels': labels,
            'datasets': datasets,
        }

    def get_token_summary_by_model(
        self,
        model_id: str,
        time_filter: Optional[TimeFilter] = None
    ) -> Dict[str, Any]:
        """
        Get token usage summary filtered by a specific model.

        Args:
            model_id: Model ID to filter by
            time_filter: Optional time filter to apply

        Returns:
            Dict with token statistics for the specified model
        """
        analyzer = self._token_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._token_analyzer

        breakdown = analyzer.get_model_breakdown()

        if model_id not in breakdown:
            # Return empty summary if model not found
            return {
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_creation_tokens': 0,
                'cache_read_tokens': 0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'cache_creation_cost': 0.0,
                'cache_read_cost': 0.0,
                'total_cost': 0.0,
                'cache_savings': 0.0,
                'cache_hit_rate': 0.0,
            }

        tokens, total_cost = breakdown[model_id]
        cost_breakdown = analyzer.calculate_cost(tokens, model_id)

        # Calculate cache efficiency for this model
        total_input = tokens.input_tokens + tokens.cache_creation_input_tokens + tokens.cache_read_input_tokens
        cache_hit_rate = (tokens.cache_read_input_tokens / total_input * 100) if total_input > 0 else 0

        return {
            'input_tokens': tokens.input_tokens,
            'output_tokens': tokens.output_tokens,
            'cache_creation_tokens': tokens.cache_creation_input_tokens,
            'cache_read_tokens': tokens.cache_read_input_tokens,
            'input_cost': round(cost_breakdown.input_cost, 2),
            'output_cost': round(cost_breakdown.output_cost, 2),
            'cache_creation_cost': round(cost_breakdown.cache_write_cost, 2),
            'cache_read_cost': round(cost_breakdown.cache_read_cost, 2),
            'total_cost': round(cost_breakdown.total_cost, 2),
            'cache_savings': round(cost_breakdown.cache_savings, 2),
            'cache_hit_rate': round(cache_hit_rate, 1),
        }

    def get_project_token_breakdown(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get token usage breakdown per project.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict mapping project names to token summaries
        """
        analyzer = self._token_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._token_analyzer

        breakdown = analyzer.get_project_breakdown()

        projects_data = []
        total_cost = 0.0

        for project_path, summary in breakdown.items():
            project_data = {
                'project_path': project_path,
                'total_tokens': {
                    'input_tokens': summary.total_tokens.input_tokens,
                    'output_tokens': summary.total_tokens.output_tokens,
                    'cache_creation_input_tokens': summary.total_tokens.cache_creation_input_tokens,
                    'cache_read_input_tokens': summary.total_tokens.cache_read_input_tokens,
                },
                'cost_breakdown': {
                    'input_cost': round(summary.cost_breakdown.input_cost, 4),
                    'output_cost': round(summary.cost_breakdown.output_cost, 4),
                    'cache_write_cost': round(summary.cost_breakdown.cache_write_cost, 4),
                    'cache_read_cost': round(summary.cost_breakdown.cache_read_cost, 4),
                    'total_cost': round(summary.cost_breakdown.total_cost, 4),
                },
                'cache_efficiency_pct': round(summary.cache_efficiency_pct, 2),
            }
            projects_data.append(project_data)
            total_cost += summary.total_cost

        return {
            'projects': projects_data,
            'total_projects': len(projects_data),
            'total_cost': round(total_cost, 4),
        }

    # ========== Public Methods for Integration Data ==========

    def get_integration_summary(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
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
                'server_name': name,
                'tool_call_count': stats.tool_call_count,
                'connection_count': stats.connection_count,
                'error_count': stats.error_count,
                'error_rate': round(
                    (stats.error_count / stats.tool_call_count * 100)
                    if stats.tool_call_count > 0 else 0,
                    2
                ),
            }
            for name, stats in summary.top_servers
        ]

        return {
            'total_servers': summary.total_servers,
            'total_tool_calls': summary.total_tool_calls,
            'total_connections': summary.total_connections,
            'total_errors': summary.total_errors,
            'error_rate': round(
                (summary.total_errors / summary.total_tool_calls * 100)
                if summary.total_tool_calls > 0 else 0,
                2
            ),
            'has_integrations': summary.has_integrations,
            'top_servers': top_servers_data,
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
                'tool_call_count': stats.tool_call_count,
                'connection_count': stats.connection_count,
                'error_count': stats.error_count,
                'error_rate': round(
                    (stats.error_count / stats.tool_call_count * 100)
                    if stats.tool_call_count > 0 else 0,
                    2
                ),
            }

        return {
            'servers': servers_data,
            'total_servers': len(servers_data),
        }

    # ========== Public Methods for Features Data ==========

    def get_features_summary(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get comprehensive features usage summary.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with all feature data
        """
        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._features_analyzer

        summary: FeaturesSummary = analyzer.get_summary()

        return {
            'subagents': {
                'total_calls': summary.total_subagent_calls,
                'unique_used': summary.unique_subagents_used,
                'stats': {
                    name: {
                        'invocation_count': stats.invocation_count,
                        'total_tokens': stats.total_tokens,
                        'session_count': stats.session_count,
                    }
                    for name, stats in summary.subagent_stats.items()
                },
            },
            'skills': {
                'total_installed': summary.total_skills,
                'installed': [
                    {
                        'name': skill.name,
                        'description': skill.description,
                        'has_skills_md': skill.has_skills_md,
                    }
                    for skill in summary.installed_skills
                ],
            },
            'mcps': {
                'total_servers': summary.total_mcp_servers,
                'enabled': summary.enabled_mcps,
                'stats': {
                    name: {
                        'tool_call_count': stats.tool_call_count,
                        'connection_count': stats.connection_count,
                        'error_count': stats.error_count,
                    }
                    for name, stats in summary.mcp_stats.items()
                },
            },
            'tools': {
                'total_calls': summary.total_tool_calls,
                'stats': {
                    name: {
                        'invocation_count': stats.invocation_count,
                        'total_tokens': stats.total_tokens,
                        'session_count': stats.session_count,
                    }
                    for name, stats in summary.tool_stats.items()
                },
            },
            'configuration': {
                'always_thinking_enabled': summary.always_thinking_enabled,
                'enabled_plugins': summary.enabled_plugins,
            },
        }

    def get_top_tools(self, limit: int = 10, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get most frequently used tools.

        Args:
            limit: Maximum number of tools to return
            time_filter: Optional time filter to apply

        Returns:
            List of top tools with usage stats
        """
        from ...analyzers.tokens import DEFAULT_PRICING

        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._features_analyzer

        top_tools = analyzer.get_top_tools(limit=limit)

        total_tokens = 0
        total_cost = 0.0
        total_calls = 0
        tools_data = []

        for name, stats in top_tools:
            # Calculate cost using default Sonnet pricing
            input_cost = (stats.total_input_tokens / 1_000_000) * DEFAULT_PRICING['input_per_mtok']
            output_cost = (stats.total_output_tokens / 1_000_000) * DEFAULT_PRICING['output_per_mtok']
            cache_read_cost = (stats.total_cache_read_tokens / 1_000_000) * DEFAULT_PRICING['cache_read_per_mtok']
            cache_write_cost = (stats.total_cache_write_tokens / 1_000_000) * DEFAULT_PRICING['cache_write_per_mtok']
            tool_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

            total_tokens += stats.total_tokens
            total_cost += tool_cost
            total_calls += stats.invocation_count

            tools_data.append({
                'tool_name': name,
                'invocation_count': stats.invocation_count,
                'total_tokens': stats.total_tokens,
                'input_tokens': stats.total_input_tokens,
                'output_tokens': stats.total_output_tokens,
                'total_cost': round(tool_cost, 4),
                'avg_tokens_per_call': round(
                    stats.total_tokens / stats.invocation_count
                    if stats.invocation_count > 0 else 0,
                    1
                ),
                'session_count': stats.session_count,
            })

        return {
            'tools': tools_data,
            'total_shown': len(tools_data),
            'total_tokens': total_tokens,
            'total_calls': total_calls,
            'total_cost': round(total_cost, 2),
        }

    def get_top_subagents(self, limit: int = 10, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
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
            analyzer = self._create_time_filtered_service(time_filter)._features_analyzer

        top_subagents = analyzer.get_top_subagents(limit=limit)

        subagents_data = [
            {
                'type': name,
                'invocation_count': stats.invocation_count,
                'success_count': stats.success_count,
                'error_count': stats.error_count,
                'success_rate': round(
                    (stats.success_count / stats.invocation_count * 100)
                    if stats.invocation_count > 0 else 0,
                    2
                ),
            }
            for name, stats in top_subagents
        ]

        return {
            'subagents': subagents_data,
            'total_shown': len(subagents_data),
        }

    def get_mcp_integrations(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get MCP server integration statistics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with MCP server usage data
        """
        from ...analyzers.tokens import DEFAULT_PRICING

        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._features_analyzer

        # Get all tool stats from the tool parser
        all_tools = analyzer.tool_parser.get_tool_stats(time_filter=time_filter)

        # Filter for MCP tools (tools that start with mcp__)
        mcp_tools = {name: stats for name, stats in all_tools.items() if name.startswith('mcp__')}

        # Group by MCP server
        servers = {}
        for tool_name, stats in mcp_tools.items():
            # Extract server name from tool name: mcp__server_name__function_name
            parts = tool_name.split('__')
            if len(parts) >= 2:
                server_name = parts[1]

                if server_name not in servers:
                    servers[server_name] = {
                        'server_name': server_name,
                        'tool_count': 0,
                        'total_calls': 0,
                        'total_tokens': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'cache_read_tokens': 0,
                        'cache_write_tokens': 0,
                        'total_cost': 0.0,
                        'tools': []
                    }

                servers[server_name]['tool_count'] += 1
                servers[server_name]['total_calls'] += stats.invocation_count
                servers[server_name]['total_tokens'] += stats.total_tokens
                servers[server_name]['input_tokens'] += stats.total_input_tokens
                servers[server_name]['output_tokens'] += stats.total_output_tokens
                servers[server_name]['cache_read_tokens'] += stats.total_cache_read_tokens
                servers[server_name]['cache_write_tokens'] += stats.total_cache_write_tokens
                servers[server_name]['tools'].append({
                    'tool_name': tool_name,
                    'invocation_count': stats.invocation_count,
                    'total_tokens': stats.total_tokens
                })

        # Calculate costs per server using default pricing (Sonnet 4.5)
        # Note: Tool-level token tracking doesn't include model info, so we use default rates
        for server in servers.values():
            input_cost = (server['input_tokens'] / 1_000_000) * DEFAULT_PRICING['input_per_mtok']
            output_cost = (server['output_tokens'] / 1_000_000) * DEFAULT_PRICING['output_per_mtok']
            cache_read_cost = (server['cache_read_tokens'] / 1_000_000) * DEFAULT_PRICING['cache_read_per_mtok']
            cache_write_cost = (server['cache_write_tokens'] / 1_000_000) * DEFAULT_PRICING['cache_write_per_mtok']
            server['total_cost'] = round(input_cost + output_cost + cache_read_cost + cache_write_cost, 4)

        # Sort servers by total calls
        sorted_servers = sorted(servers.values(), key=lambda x: x['total_calls'], reverse=True)

        # Calculate totals
        total_mcp_servers = len(servers)
        total_mcp_tools = len(mcp_tools)
        total_mcp_calls = sum(stats.invocation_count for stats in mcp_tools.values())
        total_mcp_tokens = sum(s['total_tokens'] for s in servers.values())
        total_mcp_cost = sum(s['total_cost'] for s in servers.values())
        most_used_server = sorted_servers[0]['server_name'] if sorted_servers else 'None'

        return {
            'total_servers': total_mcp_servers,
            'total_tools': total_mcp_tools,
            'total_calls': total_mcp_calls,
            'total_tokens': total_mcp_tokens,
            'total_cost': round(total_mcp_cost, 2),
            'most_used_server': most_used_server,
            'servers': sorted_servers
        }

    def get_file_statistics(self, limit: int = 20, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
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
            analyzer = self._create_time_filtered_service(time_filter)._features_analyzer

        # Get all tool invocations
        file_ops = {}

        for invocation in analyzer.tool_parser.parse_all(time_filter=time_filter):
            if invocation.tool_name in ['Read', 'Write', 'Edit']:
                file_path = invocation.input_params.get('file_path')
                if file_path:
                    if file_path not in file_ops:
                        file_ops[file_path] = {
                            'file_path': file_path,
                            'read_count': 0,
                            'write_count': 0,
                            'edit_count': 0,
                            'total_operations': 0,
                            'projects': set()
                        }

                    tool_type = invocation.tool_name.lower()
                    file_ops[file_path][f'{tool_type}_count'] += 1
                    file_ops[file_path]['total_operations'] += 1

                    if invocation.project:
                        file_ops[file_path]['projects'].add(invocation.project)

        # Convert sets to counts and create list
        files_list = []
        for file_path, ops in file_ops.items():
            ops['project_count'] = len(ops['projects'])
            ops.pop('projects')  # Remove set before returning
            files_list.append(ops)

        # Sort by total operations
        sorted_files = sorted(files_list, key=lambda x: x['total_operations'], reverse=True)[:limit]

        # Calculate statistics
        total_files = len(file_ops)
        total_operations = sum(f['total_operations'] for f in files_list)
        total_edits = sum(f['edit_count'] for f in files_list)
        total_reads = sum(f['read_count'] for f in files_list)
        total_writes = sum(f['write_count'] for f in files_list)

        most_edited_file = sorted_files[0]['file_path'] if sorted_files else 'N/A'
        if most_edited_file != 'N/A':
            # Shorten path for display
            import os
            most_edited_file = os.path.basename(most_edited_file)

        avg_edits_per_file = total_edits / total_files if total_files > 0 else 0

        return {
            'total_files': total_files,
            'most_edited_file': most_edited_file,
            'avg_edits_per_file': round(avg_edits_per_file, 1),
            'total_operations': total_operations,
            'total_edits': total_edits,
            'total_reads': total_reads,
            'total_writes': total_writes,
            'file_operations': sorted_files
        }

    # ========== Public Methods for Full Dashboard ==========

    def get_full_dashboard(self, time_filter: Optional[TimeFilter] = None) -> Dict[str, Any]:
        """
        Get complete dashboard data combining all metrics.

        Args:
            time_filter: Optional time filter to apply

        Returns:
            Dict with all dashboard data
        """
        return {
            'usage': self.get_usage_summary(time_filter=time_filter),
            'project_breakdown': self.get_project_breakdown(time_filter=time_filter),
            'tokens': self.get_token_summary(time_filter=time_filter),
            'models': self.get_model_breakdown(time_filter=time_filter),
            'integrations': self.get_integration_summary(time_filter=time_filter),
            'features': self.get_features_summary(time_filter=time_filter),
            'top_tools': self.get_top_tools(limit=5, time_filter=time_filter),
            'top_subagents': self.get_top_subagents(limit=5, time_filter=time_filter),
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
        return inheritance_tree.get('levels', [])

    # ========== Project Analysis Methods ==========

    def get_project_analysis(
        self,
        project_path: str,
        project_name: str,
        time_filter: Optional[TimeFilter] = None
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
            time_filter=time_filter
        )
        return analysis.to_dict()

    def analyze_all_projects(
        self,
        time_filter: Optional[TimeFilter] = None
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

        for project in project_data.get('projects', []):
            # Skip projects with very low usage
            if project.get('command_count', 0) < 2:
                continue

            try:
                analysis = self._project_analyzer.analyze_project(
                    project_path=project.get('full_path', project.get('path', '')),
                    project_name=project.get('name', 'Unknown'),
                    time_filter=time_filter
                )
                all_analyses.append(analysis.to_dict())
            except Exception:
                # Skip projects that can't be analyzed
                continue

        # Sort by total recommendations (high severity first)
        all_analyses.sort(
            key=lambda x: (
                x.get('summary', {}).get('high_severity', 0),
                x.get('summary', {}).get('medium_severity', 0),
                x.get('summary', {}).get('low_severity', 0)
            ),
            reverse=True
        )

        # Calculate summary stats
        total_high = sum(a.get('summary', {}).get('high_severity', 0) for a in all_analyses)
        total_medium = sum(a.get('summary', {}).get('medium_severity', 0) for a in all_analyses)
        total_low = sum(a.get('summary', {}).get('low_severity', 0) for a in all_analyses)

        return {
            'projects': all_analyses,
            'total_projects_analyzed': len(all_analyses),
            'summary': {
                'total_high_severity': total_high,
                'total_medium_severity': total_medium,
                'total_low_severity': total_low,
                'total_recommendations': total_high + total_medium + total_low,
            }
        }
