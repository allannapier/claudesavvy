#!/usr/bin/env python3
"""Test script for pricing settings."""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from claudesavvy.utils.pricing import PricingSettings
from claudesavvy.analyzers.tokens import MODEL_PRICING, DEFAULT_PRICING

def test_pricing_settings():
    """Test pricing settings functionality."""
    print("Testing PricingSettings module...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pricing_settings = PricingSettings(tmpdir_path)

        # Test 1: Load custom pricing when file doesn't exist
        print("\n1. Testing load when no pricing file exists...")
        custom = pricing_settings.load_custom_pricing()
        assert custom == {}, f"Expected empty dict, got {custom}"
        print("   ✓ Returns empty dict")

        # Test 2: Get pricing for model (should return default)
        print("\n2. Testing get_pricing_for_model with defaults...")
        sonnet_pricing = pricing_settings.get_pricing_for_model('claude-sonnet-4-5-20250929')
        assert sonnet_pricing == DEFAULT_PRICING, "Expected default pricing"
        print(f"   ✓ Returns default pricing: {sonnet_pricing}")

        # Test 3: Set custom pricing
        print("\n3. Testing set_pricing_for_model...")
        success = pricing_settings.set_pricing_for_model(
            'claude-sonnet-4-5-20250929',
            input_per_mtok=5.0,
            output_per_mtok=20.0,
            cache_write_per_mtok=6.0,
            cache_read_per_mtok=0.50
        )
        assert success, "Failed to save pricing"
        print("   ✓ Custom pricing saved successfully")

        # Verify file was created
        pricing_file = tmpdir_path / "pricing.json"
        assert pricing_file.exists(), "Pricing file not created"
        print("   ✓ Pricing file created")

        # Test 4: Load custom pricing
        print("\n4. Testing load_custom_pricing with data...")
        loaded_custom = pricing_settings.load_custom_pricing()
        assert 'claude-sonnet-4-5-20250929' in loaded_custom, "Model not in custom pricing"
        assert loaded_custom['claude-sonnet-4-5-20250929']['input_per_mtok'] == 5.0
        print(f"   ✓ Custom pricing loaded: {loaded_custom}")

        # Test 5: Get custom pricing
        print("\n5. Testing get_pricing_for_model returns custom...")
        custom_pricing = pricing_settings.get_pricing_for_model('claude-sonnet-4-5-20250929')
        assert custom_pricing['input_per_mtok'] == 5.0, "Should return custom pricing"
        print(f"   ✓ Returns custom pricing: {custom_pricing}")

        # Test 6: Get all pricing
        print("\n6. Testing get_all_pricing...")
        all_pricing = pricing_settings.get_all_pricing()
        assert 'claude-sonnet-4-5-20250929' in all_pricing
        assert all_pricing['claude-sonnet-4-5-20250929']['input_per_mtok'] == 5.0
        print(f"   ✓ Returns all pricing with custom overrides")

        # Test 7: Reset pricing
        print("\n7. Testing reset_pricing_for_model...")
        reset_success = pricing_settings.reset_pricing_for_model('claude-sonnet-4-5-20250929')
        assert reset_success, "Failed to reset pricing"
        print("   ✓ Pricing reset successfully")

        # Verify it's back to default
        after_reset = pricing_settings.get_pricing_for_model('claude-sonnet-4-5-20250929')
        assert after_reset == DEFAULT_PRICING, "Should be back to default"
        print("   ✓ Pricing back to default")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_pricing_settings()
