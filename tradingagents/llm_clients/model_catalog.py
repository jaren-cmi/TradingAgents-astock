"""Shared model catalog for CLI selections and validation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .model_discovery import DiscoveryResult, get_discovered_model_ids, get_discovery_result

ModelOption = Tuple[str, str]
ProviderModeOptions = Dict[str, Dict[str, List[ModelOption]]]


MODEL_OPTIONS: ProviderModeOptions = {
    "openai": {
        "quick": [
            ("GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"),
            ("GPT-5.4 Nano - Cheapest, high-volume tasks", "gpt-5.4-nano"),
            ("GPT-5.4 - Latest frontier, 1M context", "gpt-5.4"),
            ("GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"),
        ],
        "deep": [
            ("GPT-5.4 - Latest frontier, 1M context", "gpt-5.4"),
            ("GPT-5.2 - Strong reasoning, cost-effective", "gpt-5.2"),
            ("GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"),
            ("GPT-5.4 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.4-pro"),
        ],
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Haiku 4.5 - Fast, near-instant responses", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - Agents and coding", "claude-sonnet-4-5"),
        ],
        "deep": [
            ("Claude Opus 4.6 - Most intelligent, agents and coding", "claude-opus-4-6"),
            ("Claude Opus 4.5 - Premium, max intelligence", "claude-opus-4-5"),
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Sonnet 4.5 - Agents and coding", "claude-sonnet-4-5"),
        ],
    },
    "google": {
        "quick": [
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite-preview"),
            ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows", "gemini-3.1-pro-preview"),
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Pro - Stable pro model", "gemini-2.5-pro"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx", "grok-4-1-fast-non-reasoning"),
            ("Grok 4 Fast (Non-Reasoning) - Speed optimized", "grok-4-fast-non-reasoning"),
            ("Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx", "grok-4-1-fast-reasoning"),
        ],
        "deep": [
            ("Grok 4 - Flagship model", "grok-4-0709"),
            ("Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx", "grok-4-1-fast-reasoning"),
            ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
            ("Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx", "grok-4-1-fast-non-reasoning"),
        ],
    },
    "deepseek": {
        "quick": [
            ("DeepSeek V4 Flash - Latest V4 fast model", "deepseek-v4-flash"),
            ("DeepSeek V3.2", "deepseek-chat"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - Latest V4 flagship model", "deepseek-v4-pro"),
            ("DeepSeek V3.2 (thinking)", "deepseek-reasoner"),
            ("DeepSeek V3.2", "deepseek-chat"),
            ("Custom model ID", "custom"),
        ],
    },
    # DashScope model IDs iterate quickly; append newly released defaults here
    # when useful, but keep "Custom model ID" available so users can type any
    # newer model name without waiting for a catalog update.
    "qwen": {
        "quick": [
            ("Qwen 3.8 Flash", "qwen3.8-flash"),
            ("Qwen 3.5 Flash", "qwen3.5-flash"),
            ("Qwen Plus", "qwen-plus"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Qwen 3.8 Max", "qwen3.8-max-0902"),
            ("Qwen 3.8 Flash", "qwen3.8-flash"),
            ("Qwen 3.6 Plus", "qwen3.6-plus"),
            ("Qwen 3.5 Plus", "qwen3.5-plus"),
            ("Qwen 3 Max", "qwen3-max"),
            ("Custom model ID", "custom"),
        ],
    },
    "glm": {
        "quick": [
            ("GLM-4.7", "glm-4.7"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-5.1", "glm-5.1"),
            ("GLM-5", "glm-5"),
            ("Custom model ID", "custom"),
        ],
    },
    "kimi": {
        "quick": [
            ("Kimi K2 - Fast / latest discovered", "kimi-k2"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("Kimi K2 - Latest discovered", "kimi-k2"),
            ("Custom model ID", "custom"),
        ],
    },
    "minimax": {
        "quick": [
            ("MiniMax-M2.7-highspeed - Latest fast model", "MiniMax-M2.7-highspeed"),
            ("MiniMax-M2.5-highspeed - Stable fast model", "MiniMax-M2.5-highspeed"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("MiniMax-M2.7 - Latest flagship model", "MiniMax-M2.7"),
            ("MiniMax-M2.5 - Stable flagship model", "MiniMax-M2.5"),
            ("MiniMax-M2.7-highspeed - Fast alternative", "MiniMax-M2.7-highspeed"),
            ("Custom model ID", "custom"),
        ],
    },
    "openrouter": {
        "quick": [
            ("OpenRouter Auto", "openrouter/auto"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("OpenRouter Auto", "openrouter/auto"),
            ("Custom model ID", "custom"),
        ],
    },
    "openai_compatible": {
        "quick": [("Custom model ID", "custom")],
        "deep": [("Custom model ID", "custom")],
    },
    # OpenRouter: fetched dynamically. Azure: any deployed model name.
    "ollama": {
        "quick": [
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
        ],
        "deep": [
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
        ],
    },
}


def get_model_options(provider: str, mode: str) -> List[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    return MODEL_OPTIONS[provider.lower()][mode]


def get_known_models() -> Dict[str, List[str]]:
    """Build known model names from the shared CLI catalog."""
    return {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }


def _ensure_custom_option(options: List[ModelOption]) -> List[ModelOption]:
    """Ensure the custom model sentinel remains available at the end."""
    without_custom = [option for option in options if option[1] != "custom"]
    without_custom.append(("Custom model ID", "custom"))
    return without_custom


def get_dynamic_model_options(
    provider: str,
    mode: str,
    base_url: Optional[str] = None,
    discovery_result: Optional[DiscoveryResult] = None,
) -> List[ModelOption]:
    """Return dynamic model IDs plus fallback entries, always keeping custom last."""
    fallback = _ensure_custom_option(list(get_model_options(provider, mode)))
    discovered = (
        discovery_result.model_ids
        if discovery_result is not None
        else get_discovered_model_ids(provider, base_url=base_url)
    )
    if not discovered:
        return fallback

    merged: List[ModelOption] = []
    seen_values: set[str] = set()
    for model_id in discovered:
        if model_id in seen_values or model_id == "custom":
            continue
        merged.append((model_id, model_id))
        seen_values.add(model_id)

    for option in fallback:
        value = option[1]
        if value in seen_values or value == "custom":
            continue
        merged.append(option)
        seen_values.add(value)

    return _ensure_custom_option(merged)


def get_dynamic_model_selection(
    provider: str, mode: str, base_url: Optional[str] = None
) -> tuple[List[ModelOption], DiscoveryResult]:
    """Return merged model options and the cached discovery result behind them."""
    discovery_result = get_discovery_result(provider, base_url=base_url)
    return (
        get_dynamic_model_options(
            provider,
            mode,
            base_url=base_url,
            discovery_result=discovery_result,
        ),
        discovery_result,
    )
