import pytest

from tradingagents.llm_clients.model_discovery import DiscoveryResult
from web.components import sidebar


@pytest.mark.unit
def test_sidebar_caption_for_success():
    caption = sidebar._discovery_caption(
        "glm",
        DiscoveryResult(
            ["glm-5.1", "glm-5"],
            "ok",
            "Fetched 2 models",
            "https://open.bigmodel.cn/api/paas/v4",
        ),
    )

    assert "✅ 已从 智谱 GLM 实时获取 2 个模型" in caption


@pytest.mark.unit
def test_sidebar_caption_for_http_error_hides_raw_url_suffix():
    caption = sidebar._discovery_caption(
        "openai",
        DiscoveryResult(
            [],
            "http_error",
            "HTTP 401 from https://api.openai.com/v1/models",
            "https://api.openai.com/v1",
        ),
    )

    assert "实时获取失败：HTTP 401" in caption
    assert "填写正确端点后重试" in caption


@pytest.mark.unit
def test_sidebar_caption_for_missing_key():
    caption = sidebar._discovery_caption(
        "qwen",
        DiscoveryResult([], "no_api_key", "未配置 DASHSCOPE_API_KEY", None),
    )

    assert "未配置 DASHSCOPE_API_KEY" in caption
    assert "配置后可自动获取最新模型" in caption
