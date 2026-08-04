"""Dynamic model discovery for OpenAI-compatible providers."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import requests

from .provider_config import PROVIDER_CONFIG

_DISCOVERY_TTL_SECONDS = 300
_DISCOVERY_TIMEOUT_SECONDS = 5
_KIMI_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
_DYNAMIC_PROVIDERS = {"deepseek", "glm", "qwen", "minimax", "kimi"}

_CACHE: Dict[tuple[str, str], tuple[float, List[str]]] = {}


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _models_endpoint(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    return normalized if normalized.endswith("/models") else f"{normalized}/models"


def _provider_settings(provider: str, base_url: Optional[str] = None) -> tuple[str, Optional[str], Optional[str]]:
    provider_key = provider.lower()
    if provider_key == "kimi":
        return (
            _normalize_base_url(base_url or _KIMI_DEFAULT_BASE_URL),
            os.getenv("MOONSHOT_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"),
            "MOONSHOT_API_KEY / ANTHROPIC_AUTH_TOKEN",
        )

    default_base, api_key_env = PROVIDER_CONFIG[provider_key]
    api_key = os.getenv(api_key_env) if api_key_env else None
    return _normalize_base_url(base_url or default_base), api_key, api_key_env


def _extract_model_ids(payload: dict) -> List[str]:
    ids: List[str] = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return sorted(set(ids), key=str.casefold)


def _fetch_models(base_url: str, api_key: str, timeout: int) -> List[str]:
    response = requests.get(
        _models_endpoint(base_url),
        headers={"Authorization": f"******"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid models payload")
    return _extract_model_ids(payload)


def _cache_key(provider: str, base_url: Optional[str]) -> tuple[str, str]:
    return provider.lower(), _normalize_base_url(base_url or "")


def clear_model_discovery_cache() -> None:
    _CACHE.clear()


def get_discovered_model_ids(
    provider: str,
    base_url: Optional[str] = None,
    timeout: int = _DISCOVERY_TIMEOUT_SECONDS,
) -> List[str]:
    provider_key = provider.lower()
    if provider_key not in _DYNAMIC_PROVIDERS:
        return []

    cache_key = _cache_key(provider_key, base_url)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] < _DISCOVERY_TTL_SECONDS:
        return list(cached[1])

    resolved_base_url, api_key, _ = _provider_settings(provider_key, base_url)
    if not api_key:
        return []

    try:
        model_ids = _fetch_models(resolved_base_url, api_key, timeout)
    except Exception:
        return []

    if model_ids:
        _CACHE[cache_key] = (now, model_ids)
    return list(model_ids)


def get_discovery_metadata(provider: str, base_url: Optional[str] = None) -> Dict[str, Optional[str]]:
    provider_key = provider.lower()
    if provider_key not in _DYNAMIC_PROVIDERS:
        return {"base_url": None, "api_key_env": None}
    resolved_base_url, _, api_key_env = _provider_settings(provider_key, base_url)
    return {"base_url": resolved_base_url, "api_key_env": api_key_env}
