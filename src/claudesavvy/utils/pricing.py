"""Pricing configuration management for Claude Code models.

This module handles loading, saving, and retrieving custom pricing configurations
for different Claude models. Pricing is stored in the Claude Monitor settings file.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional


def _pricing(input_per_mtok: float, output_per_mtok: float,
             cache_write_per_mtok: float, cache_read_per_mtok: float) -> Dict[str, float]:
    return {
        'input_per_mtok': input_per_mtok,
        'output_per_mtok': output_per_mtok,
        'cache_write_per_mtok': cache_write_per_mtok,
        'cache_read_per_mtok': cache_read_per_mtok,
    }


# Published Claude API pricing per million tokens (input, output,
# 5-minute cache write, cache read). Source:
# https://platform.claude.com/docs/en/about-claude/pricing
_FABLE_PRICING = _pricing(10.00, 50.00, 12.50, 1.00)
_OPUS_PRICING = _pricing(5.00, 25.00, 6.25, 0.50)        # Opus 4.5 and later
_OPUS_LEGACY_PRICING = _pricing(15.00, 75.00, 18.75, 1.50)  # Opus 4.1 and earlier
_SONNET_PRICING = _pricing(3.00, 15.00, 3.75, 0.30)
_HAIKU_PRICING = _pricing(1.00, 5.00, 1.25, 0.10)        # Haiku 4.5 and later
_HAIKU_35_PRICING = _pricing(0.80, 4.00, 1.00, 0.08)
_HAIKU_3_PRICING = _pricing(0.25, 1.25, 0.30, 0.03)

# Built-in pricing keyed by normalized model ID (lowercase, no date suffix,
# no cloud-provider prefixes). See _normalize_model_id().
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    'claude-fable-5': _FABLE_PRICING,
    'claude-mythos-5': _FABLE_PRICING,
    'claude-opus-4-8': _OPUS_PRICING,
    'claude-opus-4-7': _OPUS_PRICING,
    'claude-opus-4-6': _OPUS_PRICING,
    'claude-opus-4-5': _OPUS_PRICING,
    'claude-opus-4-1': _OPUS_LEGACY_PRICING,
    'claude-opus-4-0': _OPUS_LEGACY_PRICING,
    'claude-opus-4': _OPUS_LEGACY_PRICING,
    'claude-3-opus': _OPUS_LEGACY_PRICING,
    'claude-sonnet-4-6': _SONNET_PRICING,
    'claude-sonnet-4-5': _SONNET_PRICING,
    'claude-sonnet-4-0': _SONNET_PRICING,
    'claude-sonnet-4': _SONNET_PRICING,
    'claude-3-7-sonnet': _SONNET_PRICING,
    'claude-3-5-sonnet': _SONNET_PRICING,
    'claude-3-sonnet': _SONNET_PRICING,
    'claude-haiku-4-6': _HAIKU_PRICING,
    'claude-haiku-4-5': _HAIKU_PRICING,
    'claude-3-5-haiku': _HAIKU_35_PRICING,
    'claude-3-haiku': _HAIKU_3_PRICING,
}

# Substring fallbacks for model IDs that don't match the table even after
# normalization (e.g. future releases). Order matters: most specific first.
_FAMILY_FALLBACKS = [
    ('fable', _FABLE_PRICING),
    ('mythos', _FABLE_PRICING),
    ('3-5-haiku', _HAIKU_35_PRICING),
    ('haiku-3-5', _HAIKU_35_PRICING),
    ('haiku', _HAIKU_PRICING),
    ('3-opus', _OPUS_LEGACY_PRICING),
    ('opus', _OPUS_PRICING),
    ('sonnet', _SONNET_PRICING),
]

# Default pricing for models that can't be matched to any known family
# (Sonnet rates, matching the historical behaviour of this module).
DEFAULT_PRICING = _SONNET_PRICING


def _normalize_model_id(model: str) -> str:
    """Reduce a raw model ID to its canonical form for pricing lookup.

    Handles dated IDs ("claude-sonnet-4-5-20250929"), Bedrock IDs
    ("eu.anthropic.claude-opus-4-5-20251101-v1:0"), Vertex IDs
    ("claude-sonnet-4-5@20250929"), and Claude Code context-window
    markers ("claude-sonnet-4-5-20250929[1m]").
    """
    normalized = model.lower().strip()
    normalized = re.sub(r'\[[^\]]*\]$', '', normalized)
    normalized = re.sub(r'^(?:[a-z0-9-]+\.)?anthropic\.', '', normalized)
    normalized = re.sub(r'-v\d+(?::\d+)?$', '', normalized)
    normalized = re.sub(r'[@-]\d{8}$', '', normalized)
    return normalized


def resolve_model_pricing(model: Optional[str]) -> Dict[str, float]:
    """Resolve built-in pricing for a model ID.

    Tries an exact match, then a normalized match, then a model-family
    substring fallback. Returns DEFAULT_PRICING for unrecognized models.
    """
    if not model:
        return DEFAULT_PRICING

    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    normalized = _normalize_model_id(model)
    if normalized in MODEL_PRICING:
        return MODEL_PRICING[normalized]

    for fragment, pricing in _FAMILY_FALLBACKS:
        if fragment in normalized:
            return pricing

    return DEFAULT_PRICING


class PricingSettings:
    """Manages custom pricing settings for Claude models."""

    def __init__(self, settings_dir: Path):
        """
        Initialize pricing settings manager.

        Args:
            settings_dir: Directory containing settings files (typically ~/.claude)
        """
        self.settings_dir = settings_dir
        self.pricing_file = settings_dir / "pricing.json"
        self._custom_pricing: Optional[Dict[str, Dict[str, float]]] = None

    def load_custom_pricing(self) -> Dict[str, Dict[str, float]]:
        """
        Load custom pricing from pricing.json file.

        Returns:
            Dictionary mapping model IDs to pricing configurations.
            Empty dict if no custom pricing exists.
        """
        if self._custom_pricing is not None:
            return self._custom_pricing

        if not self.pricing_file.exists():
            self._custom_pricing = {}
            return self._custom_pricing

        try:
            with open(self.pricing_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._custom_pricing = data.get('models', {})
                return self._custom_pricing
        except (OSError, json.JSONDecodeError):
            # If file is corrupted or unreadable, start fresh
            self._custom_pricing = {}
            return self._custom_pricing

    def save_custom_pricing(self, pricing: Dict[str, Dict[str, float]]) -> bool:
        """
        Save custom pricing to pricing.json file.

        Args:
            pricing: Dictionary mapping model IDs to pricing configurations.

        Returns:
            True if save was successful, False otherwise.
        """
        try:
            # Ensure settings directory exists
            self.settings_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data structure
            data = {
                'models': pricing,
                'version': '1.0'
            }

            # Write to file with atomic operation
            temp_file = self.pricing_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_file.replace(self.pricing_file)

            # Update cached value
            self._custom_pricing = pricing
            return True

        except (OSError, json.JSONDecodeError):
            return False

    def get_pricing_for_model(self, model: str) -> Dict[str, float]:
        """
        Get pricing configuration for a specific model.

        Args:
            model: Model identifier (e.g., 'claude-sonnet-4-5-20250929')

        Returns:
            Pricing dictionary with keys: input_per_mtok, output_per_mtok,
            cache_write_per_mtok, cache_read_per_mtok.
            Returns custom pricing if set, otherwise built-in pricing for
            the model's family, otherwise default pricing.
        """
        custom_pricing = self.load_custom_pricing()
        if model in custom_pricing:
            return custom_pricing[model]

        # Fall back to built-in published pricing for the model
        return resolve_model_pricing(model)

    def set_pricing_for_model(
        self,
        model: str,
        input_per_mtok: float,
        output_per_mtok: float,
        cache_write_per_mtok: float,
        cache_read_per_mtok: float
    ) -> bool:
        """
        Set custom pricing for a specific model.

        Args:
            model: Model identifier
            input_per_mtok: Price per million input tokens
            output_per_mtok: Price per million output tokens
            cache_write_per_mtok: Price per million cache write tokens
            cache_read_per_mtok: Price per million cache read tokens

        Returns:
            True if save was successful, False otherwise.
        """
        custom_pricing = self.load_custom_pricing()

        custom_pricing[model] = {
            'input_per_mtok': input_per_mtok,
            'output_per_mtok': output_per_mtok,
            'cache_write_per_mtok': cache_write_per_mtok,
            'cache_read_per_mtok': cache_read_per_mtok
        }

        return self.save_custom_pricing(custom_pricing)

    def reset_pricing_for_model(self, model: str) -> bool:
        """
        Reset pricing for a specific model to default.

        Args:
            model: Model identifier

        Returns:
            True if reset was successful, False otherwise.
        """
        custom_pricing = self.load_custom_pricing()

        if model in custom_pricing:
            del custom_pricing[model]
            return self.save_custom_pricing(custom_pricing)

        return True  # Already at default

    def get_all_pricing(self, additional_models: Optional[list[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get pricing for all known models, including custom overrides.

        Args:
            additional_models: List of model IDs to include
                              (e.g., models discovered from session data)

        Returns:
            Dictionary mapping all model IDs to their current pricing
            (custom if set, default otherwise).
        """
        custom_pricing = self.load_custom_pricing()
        result = {}

        # Include models with custom pricing first
        for model, pricing in custom_pricing.items():
            result[model] = pricing

        # Add additional models from session data
        if additional_models:
            for model in additional_models:
                # Only add if not already present (custom pricing takes precedence)
                if model not in result:
                    result[model] = resolve_model_pricing(model)

        return result

    def get_custom_pricing_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get only models that have custom pricing set.

        Returns:
            Dictionary of models with custom pricing overrides.
        """
        return self.load_custom_pricing()
