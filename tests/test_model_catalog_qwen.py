import warnings

import pytest

from tradingagents.llm_clients.base_client import BaseLLMClient
from tradingagents.llm_clients.capabilities import get_capabilities
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.validators import validate_model


class DummyQwenClient(BaseLLMClient):
    provider = "qwen"

    def get_llm(self):
        self.warn_if_unknown_model()
        return object()

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)


@pytest.mark.unit
def test_qwen_catalog_promotes_38_models_and_keeps_custom_last():
    quick_options = get_model_options("qwen", "quick")
    deep_options = get_model_options("qwen", "deep")

    assert quick_options[0] == ("Qwen 3.8 Flash", "qwen3.8-flash")
    assert deep_options[0] == ("Qwen 3.8 Max", "qwen3.8-max-0902")
    assert any(model_id.startswith("qwen3.8-") for _, model_id in quick_options)
    assert any(model_id.startswith("qwen3.8-") for _, model_id in deep_options)
    assert quick_options[-1] == ("Custom model ID", "custom")
    assert deep_options[-1] == ("Custom model ID", "custom")


@pytest.mark.unit
def test_qwen_38_model_ids_are_accepted_without_unknown_warning():
    client = DummyQwenClient("qwen3.8-max-0902")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.get_llm()

    assert not [w for w in caught if "not in the known model list" in str(w.message)]


@pytest.mark.unit
def test_qwen_38_family_keeps_permissive_structured_output_defaults():
    capabilities = get_capabilities("qwen3.8-max-0902")

    assert capabilities.supports_tool_choice is True
    assert capabilities.supports_json_mode is True
    assert capabilities.supports_json_schema is True
    assert capabilities.preferred_structured_method == "function_calling"
