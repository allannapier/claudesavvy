"""Token usage analyzer with cost calculations."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Dict

from ..parsers.sessions import SessionParser, TokenUsage
from ..utils.time_filter import TimeFilter
from ..utils.pricing import (  # noqa: F401 - re-exported for existing importers
    MODEL_PRICING,
    DEFAULT_PRICING,
    resolve_model_pricing,
)

if TYPE_CHECKING:
    from ..utils.pricing import PricingSettings


# Friendly display names for models
MODEL_DISPLAY_NAMES = {
    'claude-fable-5': 'Claude Fable 5',
    'claude-opus-4-8': 'Claude Opus 4.8',
    'claude-opus-4-7': 'Claude Opus 4.7',
    'claude-opus-4-6': 'Claude Opus 4.6',
    'claude-sonnet-4-6': 'Claude Sonnet 4.6',
    'claude-haiku-4-6': 'Claude Haiku 4.6',
    'claude-opus-4-5-20251101': 'Claude Opus 4.5',
    'claude-opus-4-5': 'Claude Opus 4.5',
    'claude-opus-4-1-20250805': 'Claude Opus 4.1',
    'claude-opus-4-20250514': 'Claude Opus 4',
    'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5',
    'claude-sonnet-4-5': 'Claude Sonnet 4.5',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
    'claude-haiku-4-5': 'Claude Haiku 4.5',
    'claude-3-5-haiku-20241022': 'Claude Haiku 3.5',
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
    if model_id in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_id]

    from ..utils.pricing import _normalize_model_id
    return MODEL_DISPLAY_NAMES.get(_normalize_model_id(model_id), model_id)


# Pricing lives in utils.pricing (single source of truth, synced with
# https://platform.claude.com/docs/en/about-claude/pricing). MODEL_PRICING,
# DEFAULT_PRICING, and resolve_model_pricing are re-exported above for
# backward compatibility. resolve_model_pricing() handles dated IDs,
# Bedrock/Vertex prefixes, and model-family fallbacks.


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
        time_filter: Optional[TimeFilter] = None,
        pricing_settings: Optional['PricingSettings'] = None
    ):
        """
        Initialize token analyzer.

        Args:
            session_parser: Parser for session data
            time_filter: Optional time filter
            pricing_settings: Optional pricing settings for custom model pricing
        """
        self.session_parser = session_parser
        self.time_filter = time_filter
        self.pricing_settings = pricing_settings

    def _get_pricing_for_model(self, model: Optional[str]) -> Dict[str, float]:
        """
        Get pricing configuration for a model.

        Args:
            model: Model identifier

        Returns:
            Pricing dictionary with pricing per token type.
            Uses custom pricing if available, otherwise default pricing.
        """
        if self.pricing_settings and model:
            return self.pricing_settings.get_pricing_for_model(model)
        return resolve_model_pricing(model)

    def calculate_cost(self, tokens: TokenUsage, model: Optional[str] = None) -> CostBreakdown:
        """
        Calculate costs for token usage based on model pricing.

        Args:
            tokens: TokenUsage instance
            model: Optional model name to get specific pricing

        Returns:
            CostBreakdown with calculated costs
        """
        # Get pricing for the specific model (with custom override if set)
        pricing = self._get_pricing_for_model(model)

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
