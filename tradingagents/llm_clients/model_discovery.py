"""Dynamic model discovery for OpenAI-compatible providers."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Dict, List, Optional

import requests

from .provider_config import PROVIDER_CONFIG

logger = logging.getLogger(__name__)

_DISCOVERY_TTL_SECONDS = 300
_DISCOVERY_TIMEOUT_SECONDS = 5
_KIMI_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
_GLM_DOMESTIC_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
_GLM_GLOBAL_BASE_URL = "https://api.z.ai/api/paas/v4/"
_DYNAMIC_PROVIDERS = {
    "deepseek",
    "glm",
    "kimi",
    "minimax",
    "ollama",
    "openai",
    "openai_compatible",
    "openrouter",
    "qwen",
}

_VERSION_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)*)")
_DATE_SUFFIX_RE = re.compile(r"(?:^|[-_])\d{4,8}(?:$|[-_])")


@dataclass(frozen=True)
class DiscoveryResult:
    model_ids: list[str]
    status: str
    detail: str
    base_url: str | None


@dataclass(frozen=True)
class _ProviderSettings:
    base_urls: tuple[str, ...]
    api_key: str | None
    api_key_env: str | None
    requires_api_key: bool = True


_CACHE: Dict[tuple[str, str], tuple[float, DiscoveryResult]] = {}


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _models_endpoint(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    return normalized if normalized.endswith("/models") else f"{normalized}/models"


def _version_tuple(model_id: str) -> tuple[int, ...] | None:
    match = _VERSION_RE.search(model_id)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _stability_rank(model_id: str) -> int:
    lowered = model_id.casefold()
    if "preview" in lowered or "snapshot" in lowered:
        return 1
    match = _VERSION_RE.search(model_id)
    suffix = lowered[match.end():] if match else lowered
    if _DATE_SUFFIX_RE.search(suffix):
        return 1
    return 0


def _compare_model_ids(left: str, right: str) -> int:
    left_version = _version_tuple(left)
    right_version = _version_tuple(right)

    if left_version and right_version:
        for left_part, right_part in zip(left_version, right_version):
            if left_part != right_part:
                return -1 if left_part > right_part else 1
        if len(left_version) != len(right_version):
            return -1 if len(left_version) > len(right_version) else 1

        left_stability = _stability_rank(left)
        right_stability = _stability_rank(right)
        if left_stability != right_stability:
            return -1 if left_stability < right_stability else 1
    elif left_version and not right_version:
        return -1
    elif right_version and not left_version:
        return 1

    left_folded = left.casefold()
    right_folded = right.casefold()
    if left_folded < right_folded:
        return -1
    if left_folded > right_folded:
        return 1
    return 0


def _sorted_model_ids(ids: list[str]) -> list[str]:
    unique_ids = list(dict.fromkeys(ids))
    return sorted(unique_ids, key=cmp_to_key(_compare_model_ids))


def _provider_settings(provider: str, base_url: Optional[str] = None) -> _ProviderSettings:
    provider_key = provider.lower()
    if provider_key == "kimi":
        return _ProviderSettings(
            base_urls=(_normalize_base_url(base_url or _KIMI_DEFAULT_BASE_URL),),
            api_key=os.getenv("MOONSHOT_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"),
            api_key_env="MOONSHOT_API_KEY / ANTHROPIC_AUTH_TOKEN",
        )

    if provider_key == "openai_compatible":
        return _ProviderSettings(
            base_urls=((_normalize_base_url(base_url),) if base_url else ()),
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_key_env="OPENAI_COMPATIBLE_API_KEY / OPENAI_API_KEY",
        )

    default_base, api_key_env = PROVIDER_CONFIG[provider_key]
    if provider_key == "glm" and not base_url:
        # Zhipu's domestic and international keys are not reliably interchangeable.
        # If we only probe one endpoint, a 401 is silently indistinguishable from
        # "discovery is broken" for users on the other site.
        base_urls = (
            _normalize_base_url(_GLM_DOMESTIC_BASE_URL),
            _normalize_base_url(_GLM_GLOBAL_BASE_URL),
        )
    else:
        base_urls = (_normalize_base_url(base_url or default_base),)

    requires_api_key = api_key_env is not None
    api_key = os.getenv(api_key_env) if api_key_env else None
    return _ProviderSettings(
        base_urls=base_urls,
        api_key=api_key,
        api_key_env=api_key_env,
        requires_api_key=requires_api_key,
    )


def _extract_model_ids(payload: dict) -> List[str]:
    ids: List[str] = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return _sorted_model_ids(ids)


def _request_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": "Bearer " + api_key}


def _fetch_models(base_url: str, api_key: str | None, timeout: int) -> tuple[list[str], str]:
    endpoint = _models_endpoint(base_url)
    response = requests.get(
        endpoint,
        headers=_request_headers(api_key),
        timeout=timeout,
    )
    request_url = getattr(response, "url", endpoint)
    if not 200 <= response.status_code < 300:
        raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid models payload")
    return _extract_model_ids(payload), request_url


def _cache_key(provider: str, base_url: Optional[str]) -> tuple[str, str]:
    return provider.lower(), _normalize_base_url(base_url or "")


def _cache_result(cache_key: tuple[str, str], now: float, result: DiscoveryResult) -> DiscoveryResult:
    _CACHE[cache_key] = (now, result)
    return result


def clear_model_discovery_cache() -> None:
    _CACHE.clear()


def get_discovery_result(
    provider: str,
    base_url: Optional[str] = None,
    timeout: int = _DISCOVERY_TIMEOUT_SECONDS,
) -> DiscoveryResult:
    provider_key = provider.lower()
    cache_key = _cache_key(provider_key, base_url)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] < _DISCOVERY_TTL_SECONDS:
        return cached[1]

    if provider_key not in _DYNAMIC_PROVIDERS:
        return _cache_result(
            cache_key,
            now,
            DiscoveryResult(
                model_ids=[],
                status="unsupported",
                detail="该供应商不提供模型列表接口",
                base_url=None,
            ),
        )

    settings = _provider_settings(provider_key, base_url)
    if not settings.base_urls:
        return _cache_result(
            cache_key,
            now,
            DiscoveryResult(
                model_ids=[],
                status="unsupported",
                detail="请先在「API Base URL」填写兼容网关地址",
                base_url=None,
            ),
        )

    if settings.requires_api_key and not settings.api_key:
        return _cache_result(
            cache_key,
            now,
            DiscoveryResult(
                model_ids=[],
                status="no_api_key",
                detail=f"未配置 {settings.api_key_env}",
                base_url=settings.base_urls[0],
            ),
        )

    last_result: DiscoveryResult | None = None
    for candidate_base_url in settings.base_urls:
        try:
            model_ids, request_url = _fetch_models(candidate_base_url, settings.api_key, timeout)
        except requests.exceptions.Timeout as exc:
            last_result = DiscoveryResult(
                model_ids=[],
                status="network_error",
                detail=f"请求 {_models_endpoint(candidate_base_url)} 超时",
                base_url=candidate_base_url,
            )
            logger.debug("Model discovery timeout for %s via %s: %s", provider_key, candidate_base_url, exc)
        except requests.exceptions.ConnectionError as exc:
            last_result = DiscoveryResult(
                model_ids=[],
                status="network_error",
                detail=f"无法连接 {_models_endpoint(candidate_base_url)}",
                base_url=candidate_base_url,
            )
            logger.debug("Model discovery connection error for %s via %s: %s", provider_key, candidate_base_url, exc)
        except requests.HTTPError as exc:
            response = exc.response
            status_code = getattr(response, "status_code", "unknown")
            request_url = getattr(response, "url", _models_endpoint(candidate_base_url))
            last_result = DiscoveryResult(
                model_ids=[],
                status="http_error",
                detail=f"HTTP {status_code} from {request_url}",
                base_url=candidate_base_url,
            )
            logger.debug("Model discovery HTTP error for %s via %s: %s", provider_key, candidate_base_url, last_result.detail)
        except (ValueError, requests.exceptions.RequestException) as exc:
            last_result = DiscoveryResult(
                model_ids=[],
                status="empty",
                detail=f"无法解析 {_models_endpoint(candidate_base_url)} 返回的模型列表",
                base_url=candidate_base_url,
            )
            logger.debug("Model discovery parse/error for %s via %s: %s", provider_key, candidate_base_url, exc)
        else:
            if model_ids:
                return _cache_result(
                    cache_key,
                    now,
                    DiscoveryResult(
                        model_ids=model_ids,
                        status="ok",
                        detail=f"Fetched {len(model_ids)} models from {request_url}",
                        base_url=candidate_base_url,
                    ),
                )
            last_result = DiscoveryResult(
                model_ids=[],
                status="empty",
                detail=f"{request_url} 返回空模型列表",
                base_url=candidate_base_url,
            )
            logger.debug("Model discovery returned empty list for %s via %s", provider_key, candidate_base_url)

    return _cache_result(
        cache_key,
        now,
        last_result
        or DiscoveryResult(
            model_ids=[],
            status="empty",
            detail="实时模型列表为空",
            base_url=settings.base_urls[0],
        ),
    )


def get_discovered_model_ids(
    provider: str,
    base_url: Optional[str] = None,
    timeout: int = _DISCOVERY_TIMEOUT_SECONDS,
) -> List[str]:
    return list(get_discovery_result(provider, base_url=base_url, timeout=timeout).model_ids)


def get_discovery_metadata(provider: str, base_url: Optional[str] = None) -> Dict[str, Optional[str]]:
    provider_key = provider.lower()
    if provider_key not in _DYNAMIC_PROVIDERS:
        return {"base_url": None, "api_key_env": None}
    settings = _provider_settings(provider_key, base_url)
    return {
        "base_url": settings.base_urls[0] if settings.base_urls else None,
        "api_key_env": settings.api_key_env,
    }
