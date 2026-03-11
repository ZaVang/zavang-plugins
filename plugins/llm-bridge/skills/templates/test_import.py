"""
LLM Bridge — Import & Structure Test

Validates that all modules can be imported correctly and that
the data models / provider registry are properly wired.
No API keys or network access required.
"""

import sys
import os

# Ensure llm_bridge package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_public_api_imports():
    """All public symbols should be importable from the top-level package."""
    from llm_bridge import (
        LLMBridge,
        ChatParameters,
        UnifiedResponse,
        UsageInfo,
        BridgeConfig,
        ModelConfig,
        RetryConfig,
    )
    print("  ✅ Public API imports OK")


def test_provider_registry():
    """PROVIDER_REGISTRY should contain all four provider keys."""
    from llm_bridge.providers import PROVIDER_REGISTRY

    expected = {"openai", "anthropic", "google", "vertexai"}
    actual = set(PROVIDER_REGISTRY.keys())
    assert expected == actual, f"Expected {expected}, got {actual}"
    print(f"  ✅ Provider registry OK: {sorted(actual)}")


def test_chat_parameters_fields():
    """ChatParameters should accept all documented fields."""
    from llm_bridge import ChatParameters

    p = ChatParameters(
        max_tokens=2048,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        frequency_penalty=0.3,
        presence_penalty=0.3,
        stop_sequences=["END", "STOP"],
        seed=42,
        system_prompt="You are a helpful assistant.",
    )
    assert p.max_tokens == 2048
    assert p.temperature == 0.5
    assert p.top_p == 0.9
    assert p.top_k == 40
    assert p.frequency_penalty == 0.3
    assert p.presence_penalty == 0.3
    assert p.stop_sequences == ["END", "STOP"]
    assert p.seed == 42
    assert p.system_prompt == "You are a helpful assistant."
    print("  ✅ ChatParameters all fields OK")


def test_chat_parameters_defaults():
    """ChatParameters with no args should have all fields as None."""
    from llm_bridge import ChatParameters

    p = ChatParameters()
    assert p.max_tokens is None
    assert p.temperature is None
    assert p.top_p is None
    assert p.top_k is None
    assert p.frequency_penalty is None
    assert p.presence_penalty is None
    assert p.stop_sequences is None
    assert p.seed is None
    assert p.system_prompt is None
    print("  ✅ ChatParameters defaults OK (all None)")


def test_unified_response_structure():
    """UnifiedResponse should accept all fields including new ones."""
    from llm_bridge import UnifiedResponse, UsageInfo

    r = UnifiedResponse(
        provider="openai",
        model="gpt-4o",
        content="Hello world",
        parsed=None,
        usage=UsageInfo(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            reasoning_tokens=3,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        stop_reason="stop",
        model_id="chatcmpl-abc123",
    )
    assert r.provider == "openai"
    assert r.stop_reason == "stop"
    assert r.model_id == "chatcmpl-abc123"
    assert r.usage.reasoning_tokens == 3
    assert r.usage.total_tokens == 15
    print("  ✅ UnifiedResponse structure OK")


def test_unified_response_defaults():
    """UnifiedResponse with minimal args should have sensible defaults."""
    from llm_bridge import UnifiedResponse

    r = UnifiedResponse(provider="test", model="test-model")
    assert r.content is None
    assert r.parsed is None
    assert r.stop_reason is None
    assert r.model_id is None
    assert r.usage.input_tokens == 0
    assert r.usage.output_tokens == 0
    assert r.usage.reasoning_tokens == 0
    print("  ✅ UnifiedResponse defaults OK")


def test_config_models():
    """BridgeConfig / ModelConfig / RetryConfig should be constructable."""
    from llm_bridge import BridgeConfig, ModelConfig, RetryConfig

    retry = RetryConfig(max_retries=5, base_delay=2.0)
    assert retry.max_retries == 5

    model = ModelConfig(
        provider="openai",
        model="gpt-4o",
        api_keys=["sk-test"],
        fallback_models=["gemini_flash"],
    )
    assert model.provider == "openai"
    assert len(model.api_keys) == 1

    config = BridgeConfig(
        models={"gpt4o": model},
        default_model="gpt4o",
        retry=retry,
        log_usage=True,
    )
    assert "gpt4o" in config.models
    assert config.default_model == "gpt4o"
    print("  ✅ Config models OK")


def test_key_rotator():
    """KeyRotator should cycle through keys round-robin."""
    from llm_bridge.router import KeyRotator

    rotator = KeyRotator(["key-a", "key-b", "key-c"])
    results = [rotator.next() for _ in range(6)]
    assert results == ["key-a", "key-b", "key-c", "key-a", "key-b", "key-c"]
    print("  ✅ KeyRotator round-robin OK")


def test_bridge_resolve_inline_model():
    """LLMBridge should resolve 'provider/model' strings without config."""
    from llm_bridge import LLMBridge, BridgeConfig

    bridge = LLMBridge(BridgeConfig())
    cfg = bridge._resolve_model("openai/gpt-4o")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"
    print("  ✅ Bridge inline model resolution OK")


def test_field_descriptions_present():
    """All ChatParameters fields should have non-empty descriptions."""
    from llm_bridge import ChatParameters

    for name, field_info in ChatParameters.model_fields.items():
        desc = field_info.description
        assert desc and len(desc) > 10, f"Field '{name}' missing description"
    print("  ✅ All ChatParameters fields have descriptions")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧪 LLM Bridge — Import & Structure Tests\n")

    tests = [
        test_public_api_imports,
        test_provider_registry,
        test_chat_parameters_fields,
        test_chat_parameters_defaults,
        test_unified_response_structure,
        test_unified_response_defaults,
        test_config_models,
        test_key_rotator,
        test_bridge_resolve_inline_model,
        test_field_descriptions_present,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed.")
    print()
