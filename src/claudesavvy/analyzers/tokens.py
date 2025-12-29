"""Token usage analyzer with cost calculations."""

from dataclasses import dataclass
from typing import Optional

from ..parsers.sessions import SessionParser, TokenUsage
from ..utils.time_filter import TimeFilter


# Friendly display names for models
MODEL_DISPLAY_NAMES = {
    'claude-opus-4-5-20251101': 'Claude Opus 4.5',
    'claude-opus-4-20250514': 'Claude Opus 4',
    'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
    # AWS Bedrock model IDs - Sonnet
    'eu.anthropic.claude-sonnet-4-5-20250929-v1:0': 'Claude Sonnet 4.5 (Bedrock EU)',
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0': 'Claude Sonnet 4.5 (Bedrock US)',
    'anthropic.claude-sonnet-4-5-20250929-v1:0': 'Claude Sonnet 4.5 (Bedrock)',
    # AWS Bedrock model IDs - Opus
    'eu.anthropic.claude-opus-4-20250514-v1:0': 'Claude Opus 4 (Bedrock EU)',
    'us.anthropic.claude-opus-4-20250514-v1:0': 'Claude Opus 4 (Bedrock US)',
    'anthropic.claude-opus-4-20250514-v1:0': 'Claude Opus 4 (Bedrock)',
    'eu.anthropic.claude-opus-4-5-20251101-v1:0': 'Claude Opus 4.5 (Bedrock EU)',
    'us.anthropic.claude-opus-4-5-20251101-v1:0': 'Claude Opus 4.5 (Bedrock US)',
    'anthropic.claude-opus-4-5-20251101-v1:0': 'Claude Opus 4.5 (Bedrock)',
    # AWS Bedrock model IDs - Haiku
    'eu.anthropic.claude-haiku-4-5-20251001-v1:0': 'Claude Haiku 4.5 (Bedrock EU)',
    'us.anthropic.claude-haiku-4-5-20251001-v1:0': 'Claude Haiku 4.5 (Bedrock US)',
    'anthropic.claude-haiku-4-5-20251001-v1:0': 'Claude Haiku 4.5 (Bedrock)',
}


def get_model_display_name(model_id: str) -> str:
    """Get friendly display name for a model ID."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


# Claude API pricing by model (as of 2025)
# https://www.anthropic.com/pricing
MODEL_PRICING = {
    # Opus 4.5
    'claude-opus-4-5-20251101': {
        'input_per_mtok': 15.00,
        'output_per_mtok': 75.00,
        'cache_write_per_mtok': 18.75,
        'cache_read_per_mtok': 1.50,
    },
    # Sonnet 4.5
    'claude-sonnet-4-5-20250929': {
        'input_per_mtok': 3.00,
        'output_per_mtok': 15.00,
        'cache_write_per_mtok': 3.75,
        'cache_read_per_mtok': 0.30,
    },
    # Haiku 4.5
    'claude-haiku-4-5-20251001': {
        'input_per_mtok': 1.00,
        'output_per_mtok': 5.00,
        'cache_write_per_mtok': 1.25,
        'cache_read_per_mtok': 0.10,
    },
    # Opus 4
    'claude-opus-4-20250514': {
        'input_per_mtok': 15.00,
        'output_per_mtok': 75.00,
        'cache_write_per_mtok': 18.75,
        'cache_read_per_mtok': 1.50,
    },
}

# Bedrock pricing aliases - Sonnet
for region in ['eu', 'us', '']:
    prefix = f'{region}.anthropic' if region else 'anthropic'
    MODEL_PRICING[f'{prefix}.claude-sonnet-4-5-20250929-v1:0'] = MODEL_PRICING['claude-sonnet-4-5-20250929']

# Bedrock pricing aliases - Opus 4.5
for region in ['eu', 'us', '']:
    prefix = f'{region}.anthropic' if region else 'anthropic'
    MODEL_PRICING[f'{prefix}.claude-opus-4-5-20251101-v1:0'] = MODEL_PRICING['claude-opus-4-5-20251101']

# Bedrock pricing aliases - Opus 4
for region in ['eu', 'us', '']:
    prefix = f'{region}.anthropic' if region else 'anthropic'
    MODEL_PRICING[f'{prefix}.claude-opus-4-20250514-v1:0'] = MODEL_PRICING['claude-opus-4-20250514']

# Bedrock pricing aliases - Haiku 4.5
for region in ['eu', 'us', '']:
    prefix = f'{region}.anthropic' if region else 'anthropic'
    MODEL_PRICING[f'{prefix}.claude-haiku-4-5-20251001-v1:0'] = MODEL_PRICING['claude-haiku-4-5-20251001']

# Default pricing (Sonnet 4.5)
DEFAULT_PRICING = MODEL_PRICING['claude-sonnet-4-5-20250929']


@dataclass
class CostBreakdown:
    """Breakdown of costs by token type."""

    input_cost: float
    output_cost: float
    cache_write_cost: float
    cache_read_cost: float

    @property
    def total_cost(self) -> float:
        """Total cost across all token types."""
        return self.input_cost + self.output_cost + self.cache_write_cost + self.cache_read_cost

    @property
    def cache_savings(self) -> float:
        """
        Calculate savings from cache hits.
        Savings = what cache reads would have cost as regular input - actual cache read cost
        """
        # If cache reads were regular input tokens instead
        regular_cost = (self.cache_read_cost / DEFAULT_PRICING['cache_read_per_mtok']) * DEFAULT_PRICING['input_per_mtok']
        return regular_cost - self.cache_read_cost


@dataclass
class TokenSummary:
    """Summary of token usage and costs."""

    total_tokens: TokenUsage
    cost_breakdown: CostBreakdown
    cache_efficiency_pct: float

    @property
    def total_cost(self) -> float:
        """Total estimated cost."""
        return self.cost_breakdown.total_cost

    @property
    def net_cost(self) -> float:
        """Net cost after cache savings."""
        return self.cost_breakdown.total_cost

    @property
    def cache_savings(self) -> float:
        """Savings from cache efficiency."""
        return self.cost_breakdown.cache_savings

    def format_cost(self, cost: float) -> str:
        """Format cost as currency string."""
        return f"${cost:.2f}"


class TokenAnalyzer:
    """Analyzer for token usage and cost calculations."""

    def __init__(
        self,
        session_parser: SessionParser,
        time_filter: Optional[TimeFilter] = None
    ):
        """
        Initialize token analyzer.

        Args:
            session_parser: Parser for session data
            time_filter: Optional time filter
        """
        self.session_parser = session_parser
        self.time_filter = time_filter

    @staticmethod
    def calculate_cost(tokens: TokenUsage, model: Optional[str] = None) -> CostBreakdown:
        """
        Calculate costs for token usage based on model pricing.

        Args:
            tokens: TokenUsage instance
            model: Optional model name to get specific pricing

        Returns:
            CostBreakdown with calculated costs
        """
        # Get pricing for the specific model, or use default
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

        input_cost = (tokens.input_tokens / 1_000_000) * pricing['input_per_mtok']
        output_cost = (tokens.output_tokens / 1_000_000) * pricing['output_per_mtok']
        cache_write_cost = (tokens.cache_creation_input_tokens / 1_000_000) * pricing['cache_write_per_mtok']
        cache_read_cost = (tokens.cache_read_input_tokens / 1_000_000) * pricing['cache_read_per_mtok']

        return CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            cache_write_cost=cache_write_cost,
            cache_read_cost=cache_read_cost
        )

    def get_summary(self) -> TokenSummary:
        """
        Get token usage summary with costs calculated per model.

        Returns:
            TokenSummary with aggregated data
        """
        # Get session stats
        stats = self.session_parser.get_stats(time_filter=self.time_filter)

        # Calculate costs per model and aggregate
        total_cost = CostBreakdown(
            input_cost=0.0,
            output_cost=0.0,
            cache_write_cost=0.0,
            cache_read_cost=0.0
        )

        for model, tokens in stats.model_usage.items():
            model_cost = self.calculate_cost(tokens, model)
            total_cost.input_cost += model_cost.input_cost
            total_cost.output_cost += model_cost.output_cost
            total_cost.cache_write_cost += model_cost.cache_write_cost
            total_cost.cache_read_cost += model_cost.cache_read_cost

        return TokenSummary(
            total_tokens=stats.total_tokens,
            cost_breakdown=total_cost,
            cache_efficiency_pct=stats.total_tokens.cache_efficiency_percentage
        )

    def get_model_breakdown(self) -> dict[str, tuple[TokenUsage, float]]:
        """
        Get per-model token usage and costs.

        Returns:
            Dict mapping model names to (TokenUsage, cost) tuples
        """
        stats = self.session_parser.get_stats(time_filter=self.time_filter)
        breakdown = {}

        for model, tokens in stats.model_usage.items():
            cost = self.calculate_cost(tokens, model)
            breakdown[model] = (tokens, cost.total_cost)

        # Sort by cost descending
        return dict(sorted(breakdown.items(), key=lambda x: x[1][1], reverse=True))

    def get_model_by_project_breakdown(self) -> dict[str, dict[str, tuple[TokenUsage, float]]]:
        """
        Get model usage breakdown per project.

        Returns:
            Dict mapping project names to dict of (model -> (TokenUsage, cost))
        """
        project_stats = self.session_parser.get_project_stats(time_filter=self.time_filter)
        result = {}

        for project, stats in project_stats.items():
            project_models = {}
            for model, tokens in stats.model_usage.items():
                cost = self.calculate_cost(tokens, model)
                project_models[model] = (tokens, cost.total_cost)

            if project_models:
                # Sort by cost descending
                result[project] = dict(sorted(project_models.items(), key=lambda x: x[1][1], reverse=True))

        # Sort projects by total cost descending
        return dict(sorted(result.items(),
                          key=lambda x: sum(c for _, c in x[1].values()),
                          reverse=True))

    def get_project_breakdown(self) -> dict[str, TokenSummary]:
        """
        Get per-project token usage and costs.

        Calculates costs per model within each project for accurate pricing.

        Returns:
            Dict mapping project paths to TokenSummary
        """
        project_stats = self.session_parser.get_project_stats(time_filter=self.time_filter)

        breakdown = {}
        for project, stats in project_stats.items():
            # Calculate costs per model and aggregate for accurate pricing
            total_cost = CostBreakdown(
                input_cost=0.0,
                output_cost=0.0,
                cache_write_cost=0.0,
                cache_read_cost=0.0
            )

            for model, tokens in stats.model_usage.items():
                model_cost = self.calculate_cost(tokens, model)
                total_cost.input_cost += model_cost.input_cost
                total_cost.output_cost += model_cost.output_cost
                total_cost.cache_write_cost += model_cost.cache_write_cost
                total_cost.cache_read_cost += model_cost.cache_read_cost

            breakdown[project] = TokenSummary(
                total_tokens=stats.total_tokens,
                cost_breakdown=total_cost,
                cache_efficiency_pct=stats.total_tokens.cache_efficiency_percentage
            )

        return breakdown

    @staticmethod
    def format_token_count(count: int) -> str:
        """
        Format token count in human-readable form.

        Args:
            count: Token count

        Returns:
            Formatted string (e.g., "1.2M", "458K", "1234")
        """
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.0f}K"
        else:
            return str(count)
