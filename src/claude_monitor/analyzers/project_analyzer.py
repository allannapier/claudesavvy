"""Project analyzer for optimization recommendations."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from ..parsers.sessions import SessionParser
from ..parsers.tools import ToolUsageParser
from ..parsers.skills import SkillsParser
from ..parsers.configuration_scanner import ConfigurationScanner
from ..utils.time_filter import TimeFilter
from ..models import (
    Recommendation,
    ProjectAnalysis,
    ConfigSource,
)


class ProjectAnalyzer:
    """Analyzer for providing optimization recommendations for a project."""

    # Thresholds for recommendations
    MCP_SERVER_COUNT_HIGH = 8
    MCP_SERVER_COUNT_WARN = 5
    MCP_TOOL_COUNT_HIGH = 50
    MCP_TOOL_COUNT_WARN = 30

    CLAUDE_MD_SIZE_HIGH = 10_000  # bytes
    CLAUDE_MD_SIZE_WARN = 5_000   # bytes

    CACHE_HIT_RATE_LOW = 30
    CACHE_HIT_RATE_WARN = 50

    OPUS_USAGE_RATIO_HIGH = 0.80  # 80%+ usage is high
    OPUS_USAGE_RATIO_WARN = 0.60  # 60%+ usage is warning

    MODEL_PRICING_TIER = {
        # Opus - premium tier
        'claude-opus-4-5-20251101': 'premium',
        'claude-opus-4-20250514': 'premium',
        # Sonnet - mid tier
        'claude-sonnet-4-5-20250929': 'mid',
        'claude-sonnet-4-20250514': 'mid',
        # Haiku - economy tier
        'claude-haiku-4-5-20251001': 'economy',
    }

    def __init__(
        self,
        session_parser: SessionParser,
        tool_parser: ToolUsageParser,
        skills_parser: SkillsParser,
        config_scanner: ConfigurationScanner,
    ):
        """
        Initialize project analyzer.

        Args:
            session_parser: Parser for session/token data
            tool_parser: Parser for tool usage data
            skills_parser: Parser for skills configuration
            config_scanner: Scanner for project configuration
        """
        self.session_parser = session_parser
        self.tool_parser = tool_parser
        self.skills_parser = skills_parser
        self.config_scanner = config_scanner

    def analyze_project(
        self,
        project_path: str,
        project_name: str,
        time_filter: Optional[TimeFilter] = None
    ) -> ProjectAnalysis:
        """
        Analyze a project for optimization opportunities.

        Args:
            project_path: Path to the project
            project_name: Display name of the project
            time_filter: Optional time filter for analysis period

        Returns:
            ProjectAnalysis with recommendations and metrics
        """
        recommendations: List[Recommendation] = []
        metrics: Dict[str, Any] = {}

        # Run all analysis checks
        mcp_recs, mcp_metrics = self._check_mcp_usage(project_path, time_filter)
        recommendations.extend(mcp_recs)
        metrics.update(mcp_metrics)

        model_recs, model_metrics = self._check_model_usage(project_path, time_filter)
        recommendations.extend(model_recs)
        metrics.update(model_metrics)

        config_recs, config_metrics = self._check_config_size(project_path)
        recommendations.extend(config_recs)
        metrics.update(config_metrics)

        cache_recs, cache_metrics = self._check_cache_efficiency(project_path, time_filter)
        recommendations.extend(cache_recs)
        metrics.update(cache_metrics)

        agent_recs, agent_metrics = self._check_agent_configuration(project_path)
        recommendations.extend(agent_recs)
        metrics.update(agent_metrics)

        return ProjectAnalysis(
            project_path=project_path,
            project_name=project_name,
            recommendations=recommendations,
            metrics=metrics,
            analyzed_at=datetime.now()
        )

    def _resolve_project_path(self, project_path: str) -> Optional[Path]:
        """
        Resolve the actual filesystem path from a project path string.

        Handles encoded paths from session files and returns the actual Path.
        """
        from urllib.parse import unquote

        # URL decode if needed
        decoded_path = unquote(project_path)

        # If it's an absolute path that exists, use it
        path_obj = Path(decoded_path)
        if path_obj.exists():
            return path_obj

        # Handle encoded format like: -Users-allannapier-code-claude_monitor
        if decoded_path.startswith('-'):
            # Try common base directories
            base_dirs = [
                Path.home() / 'code',
                Path.home() / 'projects',
                Path.home() / 'repos',
                Path.home() / 'workspace',
                Path.home() / 'dev',
            ]

            # Extract the project name from the encoded path
            parts = decoded_path.lstrip('-').split('-')
            if len(parts) >= 2:
                # Format might be -Users-username-code-project_name
                # or just -code-project_name
                for base_dir in base_dirs:
                    if base_dir.exists():
                        # Try to find matching project
                        for potential_path in base_dir.rglob(parts[-1]):
                            if potential_path.is_dir():
                                # Check if it has .claude directory or looks like a code project
                                if (potential_path / '.claude').exists() or \
                                   (potential_path / '.git').exists() or \
                                   list(potential_path.glob('*.py')) or \
                                   list(potential_path.glob('*.ts')) or \
                                   list(potential_path.glob('*.js')):
                                    return potential_path

        # If no match found, check common directories for the project name
        project_name = Path(decoded_path).name
        for base_dir in [Path.home() / 'code', Path.home() / 'projects']:
            if base_dir.exists():
                potential = base_dir / project_name
                if potential.exists():
                    return potential

        return None  # Couldn't resolve

    def _tool_used_in_project(self, stats, project_path: str, project_path_obj: Optional[Path]) -> bool:
        """
        Check if a tool was used in the specified project.

        Args:
            stats: ToolStats object with projects set
            project_path: Original project path string (may be encoded)
            project_path_obj: Resolved Path object for the project

        Returns:
            True if the tool was used in this project
        """
        from urllib.parse import unquote

        if not stats.projects:
            return False

        # Normalize the project path for comparison
        # Try various normalizations that might match the cwd in sessions
        project_variants = []

        # Original path
        project_variants.append(project_path)

        # URL decoded
        project_variants.append(unquote(project_path))

        # If we have a resolved path object
        if project_path_obj:
            project_variants.append(str(project_path_obj))
            project_variants.append(str(project_path_obj.resolve()))

        # Extract just the project name
        project_name = Path(project_path).name
        project_variants.append(project_name)

        # Check if any of the tool's project paths match our variants
        for tool_project in stats.projects:
            for variant in project_variants:
                # Try exact match
                if tool_project == variant:
                    return True
                # Try substring match (handle both directions)
                if variant in tool_project or tool_project in variant:
                    return True
                # Try matching just the project name
                if project_name in tool_project or tool_project.endswith(project_name):
                    return True

        return False

    def _check_mcp_usage(
        self,
        project_path: str,
        time_filter: Optional[TimeFilter]
    ) -> tuple[List[Recommendation], Dict[str, Any]]:
        """Check MCP server usage for optimization opportunities."""
        recommendations = []
        metrics = {}

        # First, get all configured MCP servers from all sources
        # Try to resolve the actual project path from the encoded path
        project_path_obj = self._resolve_project_path(project_path)

        # Scan for repositories to find all MCPs
        repos = self.config_scanner.scan_for_repositories()

        # Collect all MCPs from user, project, and plugins
        all_configured_mcps = {}  # name -> source
        user_mcps = []
        project_mcps = []
        plugin_mcps = []

        # User-level MCPs (from ~/.claude/)
        for repo in repos:
            if repo.name == "User Configuration":
                try:
                    features = self.config_scanner.get_all_features(repo.claude_dir.parent)
                    for mcp in features.mcps:
                        if mcp.name not in all_configured_mcps:
                            all_configured_mcps[mcp.name] = mcp.source
                            user_mcps.append(mcp.name)
                except Exception:
                    # Failed to read user MCP config; skip
                    pass

        # Project-level MCPs
        if project_path_obj:
            try:
                project_claude_dir = project_path_obj / '.claude'
                if project_claude_dir.exists():
                    features = self.config_scanner.get_all_features(project_path_obj)
                    for mcp in features.mcps:
                        if mcp.source.value == 'project':
                            if mcp.name not in all_configured_mcps:
                                all_configured_mcps[mcp.name] = mcp.source
                                project_mcps.append(mcp.name)
                        elif mcp.source.value == 'plugin' and mcp.name not in all_configured_mcps:
                            all_configured_mcps[mcp.name] = mcp.source
                            plugin_mcps.append(mcp.name)
            except Exception:
                # Failed to read project MCP config; skip
                pass

        # Also get plugin MCPs from user config
        for repo in repos:
            if repo.name == "User Configuration":
                try:
                    features = self.config_scanner.get_all_features(repo.claude_dir.parent)
                    for mcp in features.mcps:
                        if mcp.source.value == 'plugin' and mcp.name not in all_configured_mcps:
                            all_configured_mcps[mcp.name] = mcp.source
                            if mcp.name not in plugin_mcps:
                                plugin_mcps.append(mcp.name)
                except Exception:
                    # Failed to read plugin MCP config; skip
                    pass

        # Total configured MCP count
        mcp_count = len(all_configured_mcps)
        metrics['mcp_server_count'] = mcp_count
        metrics['mcp_user_count'] = len(user_mcps)
        metrics['mcp_project_count'] = len(project_mcps)
        metrics['mcp_plugin_count'] = len(plugin_mcps)

        # Get ALL tool usage stats from session data (we'll filter by project)
        all_tools = self.tool_parser.get_tool_stats(time_filter=time_filter)

        # Filter tools to only those used in THIS project
        # The stats.projects set contains the project paths where each tool was used
        project_mcp_tools = {}
        for tool_name, stats in all_tools.items():
            if not tool_name.startswith('mcp__'):
                continue
            # Check if this tool was used in the current project
            # stats.projects is a set of project paths (cwd values from sessions)
            if self._tool_used_in_project(stats, project_path, project_path_obj):
                project_mcp_tools[tool_name] = stats

        # Group by server and count calls (only for this project's usage)
        server_usage = {}
        for tool_name, stats in project_mcp_tools.items():
            parts = tool_name.split('__')
            if len(parts) >= 2:
                server_name = parts[1]
                if server_name not in server_usage:
                    server_usage[server_name] = {'calls': 0, 'tokens': 0, 'tools': set()}
                server_usage[server_name]['calls'] += stats.invocation_count
                server_usage[server_name]['tokens'] += stats.total_tokens
                server_usage[server_name]['tools'].add(tool_name)

        # Count tools per server
        for server_name, data in server_usage.items():
            data['tool_count'] = len(data['tools'])

        # MCPs that are configured but never used in this project
        unused_mcps = [name for name in all_configured_mcps.keys() if name not in server_usage]
        metrics['mcp_unused_count'] = len(unused_mcps)
        metrics['mcp_used_count'] = len(server_usage)
        metrics['mcp_total_tools'] = len(project_mcp_tools)
        metrics['mcp_total_calls'] = sum(s['calls'] for s in server_usage.values())

        # Get the tool count for use in recommendations
        tool_count = len(project_mcp_tools)

        # Check for too many MCP servers
        if mcp_count >= self.MCP_SERVER_COUNT_HIGH:
            recommendations.append(Recommendation(
                category='mcp',
                severity='high',
                title='Too many MCP servers enabled',
                description=(
                    f'This project has {mcp_count} MCP servers enabled with {tool_count} tools used. '
                    f'Each MCP server and its tools are included in the context, increasing token usage.'
                ),
                impact='Estimated 15-25% token overhead per request',
                action_items=[
                    f'Remove {mcp_count - self.MCP_SERVER_COUNT_WARN} or more unused MCP servers',
                    'Consider using project-specific MCP configuration instead of global',
                    'Review if all servers are necessary for this project',
                ],
                details={
                    'mcp_count': mcp_count,
                    'tool_count': tool_count,
                    'threshold': self.MCP_SERVER_COUNT_HIGH,
                }
            ))
        elif mcp_count >= self.MCP_SERVER_COUNT_WARN:
            recommendations.append(Recommendation(
                category='mcp',
                severity='medium',
                title='Moderate number of MCP servers',
                description=(
                    f'This project has {mcp_count} MCP servers with {tool_count} tools used. '
                    'Consider reviewing if all are actively used.'
                ),
                impact='Potential 5-10% token overhead',
                action_items=[
                    'Review MCP server usage below',
                    'Remove any servers that are not actively used',
                ],
                details={
                    'mcp_count': mcp_count,
                    'tool_count': tool_count,
                }
            ))

        # Check for unused MCPs
        if unused_mcps:
            recommendations.append(Recommendation(
                category='mcp',
                severity='medium',
                title=f'{len(unused_mcps)} unused MCP server(s)',
                description=(
                    f'The following MCP servers have no recorded tool calls: '
                    f'{", ".join(unused_mcps[:5])}' +
                    (f' and {len(unused_mcps) - 5} more' if len(unused_mcps) > 5 else '')
                ),
                impact='Reducing MCP servers decreases context size on every request',
                action_items=[
                    f'Remove unused servers: {", ".join(unused_mcps[:3])}',
                    'Move rarely-used servers to project-specific configuration',
                ],
                details={
                    'unused_servers': unused_mcps,
                }
            ))

        # Check for excessive tool count
        if tool_count >= self.MCP_TOOL_COUNT_HIGH:
            recommendations.append(Recommendation(
                category='mcp',
                severity='high',
                title='Very high number of MCP tools available',
                description=(
                    f'This project has access to {tool_count} MCP tools. '
                    'Each tool description adds to the system prompt context.'
                ),
                impact='High tool count significantly increases per-request token usage',
                action_items=[
                    'Consider disabling MCP servers with many unused tools',
                    'Use project-level permissions to limit available tools',
                ],
                details={
                    'tool_count': tool_count,
                }
            ))

        return recommendations, metrics

    def _check_model_usage(
        self,
        project_path: str,
        time_filter: Optional[TimeFilter]
    ) -> tuple[List[Recommendation], Dict[str, Any]]:
        """Check model usage patterns for optimization opportunities."""
        recommendations = []
        metrics = {}

        # Get project stats
        project_stats = self.session_parser.get_project_stats(time_filter=time_filter)

        # Find matching project (need to match by path pattern)
        matching_stats = None
        for proj_key, stats in project_stats.items():
            # Simple matching - check if project name is contained in the key
            if project_path.lower() in proj_key.lower() or proj_key.lower() in project_path.lower():
                matching_stats = stats
                break

        if not matching_stats:
            # No matching stats found
            return recommendations, metrics

        # Analyze model usage
        total_calls = 0
        total_cost = 0.0
        premium_calls = 0
        mid_calls = 0
        economy_calls = 0

        model_breakdown = {}

        for model_id, tokens in matching_stats.model_usage.items():
            from ..analyzers.tokens import TokenAnalyzer
            cost = TokenAnalyzer.calculate_cost(tokens, model_id)
            total_cost += cost.total_cost
            total_calls += 1

            tier = self.MODEL_PRICING_TIER.get(model_id, 'mid')
            if tier == 'premium':
                premium_calls += 1
            elif tier == 'economy':
                economy_calls += 1
            else:
                mid_calls += 1

            model_breakdown[model_id] = {
                'tokens': tokens.input_tokens + tokens.output_tokens,
                'cost': cost.total_cost,
                'tier': tier,
            }

        metrics['model_total_cost'] = round(total_cost, 4)
        metrics['model_breakdown'] = model_breakdown
        metrics['haiku_used'] = economy_calls > 0

        # Calculate ratios
        if total_calls > 0:
            premium_ratio = premium_calls / total_calls
            metrics['premium_model_ratio'] = round(premium_ratio, 2)
        else:
            premium_ratio = 0
            metrics['premium_model_ratio'] = 0.0

        # Check for excessive Opus/premium usage
        if premium_ratio >= self.OPUS_USAGE_RATIO_HIGH and economy_calls == 0:
            recommendations.append(Recommendation(
                category='model',
                severity='high',
                title='Exclusively using premium models without Haiku',
                description=(
                    f'{premium_ratio * 100:.0f}% of your usage is premium models (Opus). '
                    'Claude Haiku 4.5 is ~15x cheaper and suitable for many subtasks.'
                ),
                impact='Potential 40-60% cost reduction by using Haiku for appropriate tasks',
                action_items=[
                    'Configure Haiku-based agents for: file reading, simple edits, tool operations',
                    'Reserve Opus for: complex reasoning, architecture decisions, code review',
                    'Consider creating agents with different model tiers for different tasks',
                ],
                details={
                    'premium_ratio': premium_ratio,
                    'haiku_available': False,
                }
            ))
        elif premium_ratio >= self.OPUS_USAGE_RATIO_HIGH:
            recommendations.append(Recommendation(
                category='model',
                severity='medium',
                title='High premium model usage despite Haiku availability',
                description=(
                    f'{premium_ratio * 100:.0f}% of your usage is premium models. '
                    'While Haiku is available, consider using it more for subtasks.'
                ),
                impact='Potential 20-40% cost reduction by increasing Haiku usage',
                action_items=[
                    'Review which tasks actually require Opus',
                    'Create Haiku agents for routine file operations',
                ],
                details={
                    'premium_ratio': premium_ratio,
                }
            ))
        elif premium_ratio >= self.OPUS_USAGE_RATIO_WARN:
            recommendations.append(Recommendation(
                category='model',
                severity='low',
                title='Consider using Haiku for more tasks',
                description=(
                    f'{premium_ratio * 100:.0f}% of your usage is premium models. '
                    'Some tasks may be suitable for Haiku 4.5.'
                ),
                impact='Potential 10-20% cost reduction',
                action_items=[
                    'Evaluate if simple tasks could use Haiku instead',
                    'Create task-specific agents with appropriate model tiers',
                ],
                details={
                    'premium_ratio': premium_ratio,
                }
            ))

        # Check if Haiku is ever used
        if economy_calls == 0:
            recommendations.append(Recommendation(
                category='agent',
                severity='info',
                title='No Haiku usage detected',
                description=(
                    'This project has not used Claude Haiku 4.5 during the analysis period. '
                    'Haiku is 15x cheaper than Opus and great for subtasks.'
                ),
                impact='High potential for cost savings',
                action_items=[
                    'Create an agent with `model: haiku` for file operations',
                    'Use Haiku agents for: reading files, simple edits, running tools',
                ],
                details={
                    'haiku_used': False,
                }
            ))

        return recommendations, metrics

    def _check_config_size(
        self,
        project_path: str
    ) -> tuple[List[Recommendation], Dict[str, Any]]:
        """Check configuration file sizes for optimization opportunities."""
        recommendations = []
        metrics = {}

        project_path_obj = Path(project_path)
        claude_dir = project_path_obj / '.claude'

        if not claude_dir.exists():
            # Try user config
            claude_dir = Path.home() / '.claude'

        # Check CLAUDE.md size
        claude_md = claude_dir / 'CLAUDE.md'
        if claude_md.exists():
            size = claude_md.stat().st_size
            metrics['claude_md_size_bytes'] = size
            metrics['claude_md_size_kb'] = round(size / 1024, 1)

            if size >= self.CLAUDE_MD_SIZE_HIGH:
                recommendations.append(Recommendation(
                    category='config',
                    severity='high',
                    title='CLAUDE.md is very large',
                    description=(
                        f'The CLAUDE.md file is {round(size / 1024, 1)}KB. '
                        f'This content is included in every conversation context.'
                    ),
                    impact=f'~{size // 4} tokens per request overhead',
                    action_items=[
                        'Move examples to a separate docs/ folder',
                        'Reference external files with Read tool instead of inline',
                        'Consider splitting into focused agents for different tasks',
                        'Keep CLAUDE.md to essential project context only',
                    ],
                    details={
                        'size_bytes': size,
                        'size_kb': round(size / 1024, 1),
                        'estimated_tokens': size // 4,
                    }
                ))
            elif size >= self.CLAUDE_MD_SIZE_WARN:
                recommendations.append(Recommendation(
                    category='config',
                    severity='low',
                    title='CLAUDE.md is moderately large',
                    description=(
                        f'The CLAUDE.md file is {round(size / 1024, 1)}KB. '
                        'Consider whether all content needs to be in every conversation.'
                    ),
                    impact=f'~{size // 4} tokens per request overhead',
                    action_items=[
                        'Review if examples can be moved to separate files',
                        'Consider using task-specific agents',
                    ],
                    details={
                        'size_bytes': size,
                    }
                ))
        else:
            metrics['claude_md_size_bytes'] = 0

        # Count skills, commands, agents
        features = self.config_scanner.get_all_features(project_path_obj)

        skills_count = len([s for s in features.skills if s.source == ConfigSource.PROJECT])
        commands_count = len([c for c in features.commands if c.source == ConfigSource.PROJECT])
        agents_count = len([a for a in features.agents if a.source == ConfigSource.PROJECT])

        metrics['project_skills_count'] = skills_count
        metrics['project_commands_count'] = commands_count
        metrics['project_agents_count'] = agents_count

        if skills_count > 10:
            recommendations.append(Recommendation(
                category='config',
                severity='low',
                title=f'Many project-specific skills ({skills_count})',
                description=(
                    f'This project defines {skills_count} skills. '
                    'Each skill adds to the available context.'
                ),
                impact='Moderate impact on context size',
                action_items=[
                    'Consolidate related skills',
                    'Move rarely-used skills to user-level configuration',
                ],
                details={
                    'skills_count': skills_count,
                }
            ))

        return recommendations, metrics

    def _check_cache_efficiency(
        self,
        project_path: str,
        time_filter: Optional[TimeFilter]
    ) -> tuple[List[Recommendation], Dict[str, Any]]:
        """Check cache efficiency for optimization opportunities."""
        recommendations = []
        metrics = {}

        # Get project stats
        project_stats = self.session_parser.get_project_stats(time_filter=time_filter)

        # Find matching project
        matching_stats = None
        for proj_key, stats in project_stats.items():
            if project_path.lower() in proj_key.lower() or proj_key.lower() in project_path.lower():
                matching_stats = stats
                break

        if not matching_stats:
            return recommendations, metrics

        total_tokens = matching_stats.total_tokens
        cache_read = total_tokens.cache_read_input_tokens
        cache_creation = total_tokens.cache_creation_input_tokens
        regular_input = total_tokens.input_tokens

        total_cacheable = cache_read + cache_creation + regular_input
        if total_cacheable > 0:
            cache_hit_rate = (cache_read / total_cacheable) * 100
            cache_write_rate = (cache_creation / total_cacheable) * 100
        else:
            cache_hit_rate = 0
            cache_write_rate = 0

        metrics['cache_hit_rate'] = round(cache_hit_rate, 1)
        metrics['cache_write_rate'] = round(cache_write_rate, 1)
        metrics['cache_read_tokens'] = cache_read
        metrics['cache_write_tokens'] = cache_creation

        # Check cache efficiency
        if cache_hit_rate < self.CACHE_HIT_RATE_LOW and total_cacheable > 1000:
            recommendations.append(Recommendation(
                category='cache',
                severity='medium',
                title='Low cache hit rate',
                description=(
                    f'Cache hit rate is {cache_hit_rate:.1f}%. '
                    f'Only {cache_read:,} cache read tokens vs {regular_input:,} regular input tokens. '
                    'Prompt caching can significantly reduce costs for repeated context.'
                ),
                impact='Higher cache hit rate = lower costs',
                action_items=[
                    'Use system prompts for consistent instructions',
                    'Keep CLAUDE.md stable to benefit from caching',
                    'Reusing the same context across sessions enables cache hits',
                ],
                details={
                    'cache_hit_rate': cache_hit_rate,
                    'cache_read_tokens': cache_read,
                    'regular_input_tokens': regular_input,
                }
            ))
        elif cache_hit_rate < self.CACHE_HIT_RATE_WARN and total_cacheable > 1000:
            recommendations.append(Recommendation(
                category='cache',
                severity='low',
                title='Moderate cache hit rate',
                description=(
                    f'Cache hit rate is {cache_hit_rate:.1f}%. '
                    'There may be room for improvement.'
                ),
                impact='Potential 5-15% savings with better cache utilization',
                action_items=[
                    'Minimize changes to system prompts',
                    'Keep project context consistent across sessions',
                ],
                details={
                    'cache_hit_rate': cache_hit_rate,
                }
            ))

        return recommendations, metrics

    def _check_agent_configuration(
        self,
        project_path: str
    ) -> tuple[List[Recommendation], Dict[str, Any]]:
        """Check agent configuration for optimization opportunities."""
        recommendations = []
        metrics = {}

        project_path_obj = Path(project_path)
        features = self.config_scanner.get_all_features(project_path_obj)

        # Filter for project-level agents
        project_agents = [a for a in features.agents if a.source == ConfigSource.PROJECT]
        metrics['project_agents_count'] = len(project_agents)

        # Check agent model configuration
        haiku_agents = [a for a in project_agents if a.model and 'haiku' in a.model.lower()]
        opus_agents = [a for a in project_agents if a.model and 'opus' in a.model.lower()]
        sonnet_agents = [a for a in project_agents if a.model and 'sonnet' in a.model.lower()]
        inherit_agents = [a for a in project_agents if not a.model or a.model == 'inherit']

        metrics['haiku_agents'] = len(haiku_agents)
        metrics['opus_agents'] = len(opus_agents)
        metrics['sonnet_agents'] = len(sonnet_agents)
        metrics['inherit_agents'] = len(inherit_agents)

        if len(project_agents) > 0 and len(haiku_agents) == 0:
            recommendations.append(Recommendation(
                category='agent',
                severity='info',
                title='No Haiku-based agents configured',
                description=(
                    f'This project has {len(project_agents)} agent(s), but none are configured '
                    'to use Haiku. Haiku agents are ideal for routine tasks.'
                ),
                impact='Potential savings on routine operations',
                action_items=[
                    'Create an agent with `model: haiku` for file operations',
                    'Example: "file-reader" agent with model: haiku for Read/Edit/Write tools',
                ],
                details={
                    'total_agents': len(project_agents),
                    'haiku_agents': 0,
                }
            ))

        # Check if agents have no model specified (inherit = uses user default, often Opus)
        if len(inherit_agents) > 0:
            recommendations.append(Recommendation(
                category='agent',
                severity='low',
                title=f'{len(inherit_agents)} agent(s) inherit model (no explicit model set)',
                description=(
                    f'{len(inherit_agents)} agent(s) use `model: inherit` which means they use '
                    'your default model. If your default is Opus, consider setting cheaper models '
                    'for routine tasks.'
                ),
                impact='Agents without explicit model may use expensive default',
                action_items=[
                    'Set `model: haiku` for agents doing simple file operations',
                    'Set `model: sonnet` for general-purpose coding agents',
                    'Reserve `model: opus` for complex reasoning agents',
                ],
                details={
                    'inherit_agents': [a.name for a in inherit_agents],
                }
            ))

        return recommendations, metrics
