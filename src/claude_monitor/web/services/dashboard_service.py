"""Dashboard service for Claude Monitor web interface.

This service provides a clean interface for web routes to access dashboard data
by wrapping existing parsers and analyzers. It reuses all existing business logic
while returning data as dicts/dataclasses suitable for web rendering.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import asdict

from ...utils.paths import get_claude_paths, ClaudeDataPaths
from ...utils.time_filter import TimeFilter
from ...parsers.history import HistoryParser
from ...parsers.sessions import SessionParser
from ...parsers.debug import DebugLogParser
from ...parsers.files import FileHistoryParser
from ...parsers.tools import ToolUsageParser, ToolStats
from ...parsers.skills import SkillsParser, ConfigurationParser, SkillInfo
from ...analyzers.usage import UsageAnalyzer, UsageSummary
from ...analyzers.tokens import TokenAnalyzer, TokenSummary
from ...analyzers.integrations import IntegrationAnalyzer, IntegrationSummary
from ...analyzers.features import FeaturesAnalyzer, FeaturesSummary


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

        # Convert to list of dicts for easier JSON serialization
        return {
            'projects': [
                {
                    'path': project_path,
                    **project_data
                }
                for project_path, project_data in breakdown.items()
            ],
            'total_projects': len(breakdown)
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
        for model_name, (tokens, cost) in breakdown.items():
            models_data.append({
                'model_name': model_name,
                'tokens': {
                    'input_tokens': tokens.input_tokens,
                    'output_tokens': tokens.output_tokens,
                    'cache_creation_input_tokens': tokens.cache_creation_input_tokens,
                    'cache_read_input_tokens': tokens.cache_read_input_tokens,
                },
                'cost': round(cost, 4),
            })

        return {
            'models': models_data,
            'total_models': len(models_data)
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
                        'success_count': stats.success_count,
                        'error_count': stats.error_count,
                        'success_rate': round(
                            (stats.success_count / stats.invocation_count * 100)
                            if stats.invocation_count > 0 else 0,
                            2
                        ),
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
                        'enabled': skill.enabled,
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
                        'success_count': stats.success_count,
                        'error_count': stats.error_count,
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
        analyzer = self._features_analyzer
        if time_filter:
            analyzer = self._create_time_filtered_service(time_filter)._features_analyzer

        top_tools = analyzer.get_top_tools(limit=limit)

        tools_data = [
            {
                'tool_name': name,
                'invocation_count': stats.invocation_count,
                'total_tokens': stats.total_tokens,
                'avg_tokens_per_call': round(
                    stats.total_tokens / stats.invocation_count
                    if stats.invocation_count > 0 else 0,
                    1
                ),
                'session_count': stats.session_count,
            }
            for name, stats in top_tools
        ]

        return {
            'tools': tools_data,
            'total_shown': len(tools_data),
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
                        'tools': []
                    }

                servers[server_name]['tool_count'] += 1
                servers[server_name]['total_calls'] += stats.invocation_count
                servers[server_name]['total_tokens'] += stats.total_tokens
                servers[server_name]['tools'].append({
                    'tool_name': tool_name,
                    'invocation_count': stats.invocation_count,
                    'total_tokens': stats.total_tokens
                })

        # Sort servers by total calls
        sorted_servers = sorted(servers.values(), key=lambda x: x['total_calls'], reverse=True)

        # Calculate totals
        total_mcp_servers = len(servers)
        total_mcp_tools = len(mcp_tools)
        total_mcp_calls = sum(stats.invocation_count for stats in mcp_tools.values())
        most_used_server = sorted_servers[0]['server_name'] if sorted_servers else 'None'

        return {
            'total_servers': total_mcp_servers,
            'total_tools': total_mcp_tools,
            'total_calls': total_mcp_calls,
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
            'files': sorted_files
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
