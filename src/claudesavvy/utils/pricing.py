"""Pricing configuration management for Claude Code models.

This module handles loading, saving, and retrieving custom pricing configurations
for different Claude models. Pricing is stored in the Claude Monitor settings file.
"""

import json
from pathlib import Path
from typing import Dict, Optional

# Default pricing for models without custom pricing or models not in MODEL_PRICING
DEFAULT_PRICING = {
    'input_per_mtok': 3.00,
    'output_per_mtok': 15.00,
    'cache_write_per_mtok': 3.75,
    'cache_read_per_mtok': 0.30
}


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
            Returns custom pricing if set, otherwise default pricing.
        """
        custom_pricing = self.load_custom_pricing()
        if model in custom_pricing:
            return custom_pricing[model]

        # Fall back to default pricing
        return DEFAULT_PRICING

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
                    result[model] = DEFAULT_PRICING

        return result

    def get_custom_pricing_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get only models that have custom pricing set.

        Returns:
            Dictionary of models with custom pricing overrides.
        """
        return self.load_custom_pricing()
