import unittest
from unittest.mock import Mock, patch

import pytest
import requests

from tradingagents.llm_clients.model_catalog import get_dynamic_model_options, get_model_options
from tradingagents.llm_clients.model_discovery import (
    DiscoveryResult,
    clear_model_discovery_cache,
    get_discovered_model_ids,
    get_discovery_result,
)


def _ok_response(model_ids, url):
    response = Mock()
    response.status_code = 200
    response.url = url
    response.json.return_value = {"data": [{"id": model_id} for model_id in model_ids]}
    return response


@pytest.mark.unit
class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        clear_model_discovery_cache()

    def tearDown(self):
        clear_model_discovery_cache()

    @patch.dict("os.environ", {"ZHIPU_API_KEY": "zhipu-token"}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_get_discovered_model_ids_parses_and_sorts_latest_first(self, mock_get):
        mock_get.return_value = _ok_response(
            ["glm-4.7", "glm-5", "glm-5-preview", "glm-5"],
            "https://open.bigmodel.cn/api/paas/v4/models",
        )

        discovered = get_discovered_model_ids("glm")

        self.assertEqual(discovered, ["glm-5", "glm-5-preview", "glm-4.7"])
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args, ("https://open.bigmodel.cn/api/paas/v4/models",))
        self.assertEqual(kwargs["timeout"], 5)
        self.assertTrue(kwargs["headers"]["Authorization"].startswith("Bearer "))
        self.assertIn("zhipu-token", kwargs["headers"]["Authorization"])

    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "dashscope-token"}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_version_sort_handles_multi_digit_parts(self, mock_get):
        mock_get.return_value = _ok_response(
            ["qwen3.8-max", "qwen3.10-max"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        )

        discovered = get_discovered_model_ids("qwen")

        self.assertEqual(discovered, ["qwen3.10-max", "qwen3.8-max"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai-secret-key"}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_http_error_status_does_not_leak_api_key(self, mock_get):
        response = Mock()
        response.status_code = 401
        response.url = "https://api.openai.com/v1/models"
        mock_get.return_value = response

        result = get_discovery_result("openai")

        self.assertEqual(result.status, "http_error")
        self.assertIn("HTTP 401", result.detail)
        self.assertIn("https://api.openai.com/v1/models", result.detail)
        self.assertNotIn("openai-secret-key", result.detail)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertEqual(args, ("https://api.openai.com/v1/models",))
        self.assertEqual(kwargs["timeout"], 5)
        self.assertTrue(kwargs["headers"]["Authorization"].startswith("Bearer "))
        self.assertIn("openai-secret-key", kwargs["headers"]["Authorization"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai-secret-key"}, clear=False)
    @patch(
        "tradingagents.llm_clients.model_discovery.requests.get",
        side_effect=requests.exceptions.Timeout("timed out"),
    )
    def test_timeout_maps_to_network_error_without_key_leak(self, _mock_get):
        result = get_discovery_result("openai")

        self.assertEqual(result.status, "network_error")
        self.assertIn("超时", result.detail)
        self.assertNotIn("openai-secret-key", result.detail)

    @patch.dict("os.environ", {"ZHIPU_API_KEY": ""}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_missing_key_maps_to_no_api_key(self, mock_get):
        result = get_discovery_result("glm")

        self.assertEqual(result.status, "no_api_key")
        self.assertIn("ZHIPU_API_KEY", result.detail)
        mock_get.assert_not_called()

    @patch.dict("os.environ", {"MOONSHOT_API_KEY": "moonshot-token"}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_kimi_ok_status_and_custom_last(self, mock_get):
        mock_get.return_value = _ok_response(
            ["moonshot-v1-8k", "moonshot-v1-32k"],
            "https://api.moonshot.cn/v1/models",
        )

        result = get_discovery_result("kimi")
        options = get_dynamic_model_options("kimi", "deep", discovery_result=result)

        self.assertEqual(result.status, "ok")
        self.assertEqual(options[-1], ("Custom model ID", "custom"))
        self.assertEqual(
            options[:-1],
            [
                ("moonshot-v1-32k", "moonshot-v1-32k"),
                ("moonshot-v1-8k", "moonshot-v1-8k"),
                ("Kimi K2 - Latest discovered", "kimi-k2"),
            ],
        )

    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_ollama_discovery_does_not_require_api_key(self, mock_get):
        mock_get.return_value = _ok_response(
            ["qwen3:latest", "llama3.1"],
            "http://localhost:11434/v1/models",
        )

        result = get_discovery_result("ollama")

        self.assertEqual(result.status, "ok")
        mock_get.assert_called_once_with(
            "http://localhost:11434/v1/models",
            headers={},
            timeout=5,
        )

    @patch.dict("os.environ", {"ZHIPU_API_KEY": "zhipu-token"}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_glm_tries_alternate_endpoint_and_uses_successful_one(self, mock_get):
        def side_effect(url, headers, timeout):
            if url == "https://open.bigmodel.cn/api/paas/v4/models":
                return _ok_response(["glm-5.1", "glm-5"], url)
            response = Mock()
            response.status_code = 401
            response.url = url
            return response

        mock_get.side_effect = side_effect

        result = get_discovery_result("glm")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.base_url, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(result.model_ids[0], "glm-5.1")

    def test_dynamic_options_keep_custom_last_on_failure(self):
        options = get_dynamic_model_options(
            "qwen",
            "deep",
            discovery_result=DiscoveryResult([], "no_api_key", "未配置 DASHSCOPE_API_KEY", None),
        )

        self.assertEqual(options, get_model_options("qwen", "deep"))
        self.assertEqual(options[-1], ("Custom model ID", "custom"))

    def test_dynamic_options_merge_fallback_models_without_duplication(self):
        options = get_dynamic_model_options(
            "glm",
            "deep",
            discovery_result=DiscoveryResult(
                ["glm-5.1"],
                "ok",
                "Fetched 1 models",
                "https://open.bigmodel.cn/api/paas/v4",
            ),
        )

        self.assertEqual(
            options,
            [
                ("glm-5.1", "glm-5.1"),
                ("GLM-5", "glm-5"),
                ("Custom model ID", "custom"),
            ],
        )
