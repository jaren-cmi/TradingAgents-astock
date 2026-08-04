import unittest
from unittest.mock import ANY, Mock, patch

import pytest

from tradingagents.llm_clients.model_catalog import get_dynamic_model_options, get_model_options
from tradingagents.llm_clients.model_discovery import clear_model_discovery_cache, get_discovered_model_ids


@pytest.mark.unit
class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        clear_model_discovery_cache()

    def tearDown(self):
        clear_model_discovery_cache()

    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_get_discovered_model_ids_parses_openai_compatible_models(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "data": [
                {"id": "glm-5.1"},
                {"id": "glm-5"},
                {"id": "glm-5.1"},
                {"not_id": "ignored"},
            ]
        }
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        discovered = get_discovered_model_ids("glm")

        self.assertEqual(discovered, ["glm-5", "glm-5.1"])
        mock_get.assert_called_once_with(
            "https://api.z.ai/api/paas/v4/models",
            headers={"Authorization": ANY},
            timeout=5,
        )

    @patch("tradingagents.llm_clients.model_discovery.requests.get", side_effect=RuntimeError("boom"))
    def test_get_dynamic_model_options_falls_back_on_request_failure(self, _mock_get):
        options = get_dynamic_model_options("qwen", "quick")

        self.assertEqual(options, get_model_options("qwen", "quick"))

    @patch.dict("os.environ", {"MOONSHOT_API_KEY": "moonshot-token"}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_kimi_uses_moonshot_models_endpoint_and_keeps_custom_option_last(self, mock_get):
        response = Mock()
        response.json.return_value = {"data": [{"id": "moonshot-v1-8k"}, {"id": "moonshot-v1-32k"}]}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        options = get_dynamic_model_options("kimi", "deep")

        self.assertEqual(options[-1], ("Custom model ID", "custom"))
        self.assertEqual(options[:-1], [("moonshot-v1-32k", "moonshot-v1-32k"), ("moonshot-v1-8k", "moonshot-v1-8k")])
        mock_get.assert_called_once_with(
            "https://api.moonshot.cn/v1/models",
            headers={"Authorization": ANY},
            timeout=5,
        )
    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}, clear=False)
    @patch("tradingagents.llm_clients.model_discovery.requests.get")
    def test_get_dynamic_model_options_falls_back_without_api_key(self, mock_get):
        options = get_dynamic_model_options("qwen", "deep")

        self.assertEqual(options, get_model_options("qwen", "deep"))
        mock_get.assert_not_called()

