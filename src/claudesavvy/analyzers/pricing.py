# Claude API pricing by model (as of 2025)
# https://www.anthropic.com/pricing
#
# This module has no imports from parsers or other analyzers so it can be
# imported by both without creating circular dependencies.

MODEL_PRICING: dict = {
    # Short-form model IDs used by Claude Code CLI (subagents, fast mode)
    'claude-opus-4-6': {
        'input_per_mtok': 5.00,
        'output_per_mtok': 25.00,
        'cache_write_per_mtok': 6.25,
        'cache_read_per_mtok': 0.50,
    },
    'claude-sonnet-4-6': {
        'input_per_mtok': 3.00,
        'output_per_mtok': 15.00,
        'cache_write_per_mtok': 3.75,
        'cache_read_per_mtok': 0.30,
    },
    'claude-haiku-4-6': {
        'input_per_mtok': 1.00,
        'output_per_mtok': 5.00,
        'cache_write_per_mtok': 1.25,
        'cache_read_per_mtok': 0.10,
    },
    # Opus 4.5
    'claude-opus-4-5-20251101': {
        'input_per_mtok': 5.00,
        'output_per_mtok': 25.00,
        'cache_write_per_mtok': 6.25,
        'cache_read_per_mtok': 0.50,
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

# Add Bedrock pricing aliases for models across regions
def _add_bedrock_aliases(model_id: str, bedrock_model_id_base: str) -> None:
    """Add regional Bedrock aliases (EU, US, default) for a model."""
    for region in ['eu', 'us', '']:
        prefix = f'{region}.anthropic' if region else 'anthropic'
        bedrock_id = f'{prefix}.{bedrock_model_id_base}'
        MODEL_PRICING[bedrock_id] = MODEL_PRICING[model_id]


_add_bedrock_aliases('claude-sonnet-4-5-20250929', 'claude-sonnet-4-5-20250929-v1:0')
_add_bedrock_aliases('claude-opus-4-5-20251101', 'claude-opus-4-5-20251101-v1:0')
_add_bedrock_aliases('claude-opus-4-20250514', 'claude-opus-4-20250514-v1:0')
_add_bedrock_aliases('claude-haiku-4-5-20251001', 'claude-haiku-4-5-20251001-v1:0')

# Default pricing (Sonnet 4.5)
DEFAULT_PRICING: dict = MODEL_PRICING['claude-sonnet-4-5-20250929']
