"""
LLM Bridge — Live API Test (Google Gemini)

Tests a real API call to Google Gemini using the LLM Bridge module.
Requires a valid GOOGLE_API_KEY in a .env file or environment variable.

Usage:
    1. Create a .env file in this directory:
       GOOGLE_API_KEY=AIza...your-key...

    2. Run:
       py test_live.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import certifi

# Fix SSL CA certificates issue in some Conda environments
os.environ["SSL_CERT_FILE"] = certifi.where()

from llm_bridge import ChatParameters, LLMBridge, UnifiedResponse
from llm_bridge.models import BridgeConfig, ModelConfig, RetryConfig, UsageInfo

# Load .env from current directory
load_dotenv()

# Set up logging so we can see the token usage log
logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")


# ── Pydantic model for structured output test ────────────────────
class CityInfo(BaseModel):
    """Structured output test schema."""

    city_name: str = Field(description="Name of the city")
    country: str = Field(description="Country the city belongs to")
    population_estimate: str = Field(description="Rough population estimate")
    famous_for: list[str] = Field(description="Top 3 things the city is famous for")


def build_config(api_key: str) -> BridgeConfig:
    """Build a BridgeConfig programmatically (no YAML file needed)."""
    return BridgeConfig(
        models={
            "gemini_flash": ModelConfig(
                provider="google",
                model="gemini-2.5-flash",
                api_keys=[api_key],
                max_tokens=1024,
                temperature=0.7,
            ),
        },
        default_model="gemini_flash",
        retry=RetryConfig(max_retries=2, base_delay=1.0),
        log_usage=True,
    )


async def test_simple_text(bridge: LLMBridge):
    """Test 1: Simple text completion."""
    print("\n" + "=" * 60)
    print("🧪 Test 1: Simple text completion")
    print("=" * 60)

    response = await bridge.chat(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Say 'Hello from LLM Bridge!' and nothing else."}],
        params=ChatParameters(
            max_tokens=50,
            temperature=0.1,
        ),
    )

    print(f"  Provider:     {response.provider}")
    print(f"  Model:        {response.model}")
    print(f"  Content:      {response.content}")
    print(f"  Stop reason:  {response.stop_reason}")
    print(f"  Model ID:     {response.model_id}")
    print(f"  Usage:        in={response.usage.input_tokens}, "
          f"out={response.usage.output_tokens}, "
          f"total={response.usage.total_tokens}")

    assert response.content is not None, "Content should not be None"
    assert response.usage.total_tokens > 0, "Total tokens should be > 0"
    print("  ✅ PASSED")
    return response


async def test_structured_output(bridge: LLMBridge):
    """Test 2: Structured output with Pydantic model."""
    print("\n" + "=" * 60)
    print("🧪 Test 2: Structured output (Pydantic)")
    print("=" * 60)

    response = await bridge.chat(
        model="gemini_flash",
        messages=[{"role": "user", "content": "Tell me about Tokyo, Japan."}],
        response_model=CityInfo,
        params=ChatParameters(
            max_tokens=512,
            temperature=0.3,
            system_prompt="You are a geography expert. Be concise.",
        ),
    )

    print(f"  Provider:     {response.provider}")
    print(f"  Model:        {response.model}")
    print(f"  Raw content:  {response.content[:100] if response.content else 'None'}...")
    print(f"  Stop reason:  {response.stop_reason}")

    if response.parsed:
        city: CityInfo = response.parsed
        print(f"  Parsed city:  {city.city_name}")
        print(f"  Country:      {city.country}")
        print(f"  Population:   {city.population_estimate}")
        print(f"  Famous for:   {city.famous_for}")
    else:
        print("  ⚠️  Parsed output is None")

    print(f"  Usage:        in={response.usage.input_tokens}, "
          f"out={response.usage.output_tokens}, "
          f"total={response.usage.total_tokens}")

    assert response.parsed is not None, "Parsed output should not be None"
    assert isinstance(response.parsed, CityInfo), "Parsed should be a CityInfo instance"
    print("  ✅ PASSED")
    return response


async def test_inline_model_string(bridge: LLMBridge):
    """Test 3: Using inline 'provider/model' string instead of alias."""
    print("\n" + "=" * 60)
    print("🧪 Test 3: Inline model string (google/gemini-2.5-flash)")
    print("=" * 60)

    response = await bridge.chat(
        model="google/gemini-2.0-flash",
        messages=[{"role": "user", "content": "What is the solution to x^2 + 2x + 1 = 0? Reply with just the number."}],
        params=ChatParameters(max_tokens=1000, temperature=0.0),
    )

    print(f"  Thinking:     {response.thinking}")
    print(f"  Content:      {response.content}")
    print(f"  Provider:     {response.provider}")
    print(f"  Model:        {response.model}")
    print(f"  Model ID:     {response.model_id}")
    print(f"  Stop reason:  {response.stop_reason}")
    print(f"  Usage:        in={response.usage.input_tokens}, "
          f"out={response.usage.output_tokens}"
          f"thinking={response.usage.reasoning_tokens}, "
          f"total={response.usage.total_tokens}")

    assert response.content is not None
    print("  ✅ PASSED")
    return response


async def test_chat_parameters_variety(bridge: LLMBridge):
    """Test 4: Various ChatParameters combinations."""
    print("\n" + "=" * 60)
    print("🧪 Test 4: ChatParameters variety")
    print("=" * 60)

    response = await bridge.chat(
        model="gemini_flash",
        messages=[{"role": "user", "content": "List 3 colors. One per line."}],
        params=ChatParameters(
            max_tokens=100,
            temperature=1.5,
            top_p=0.95,
            top_k=40,
            stop_sequences=["\n\n"],
            seed=123,
        ),
    )

    print(f"  Content:      {response.content}")
    print(f"  Stop reason:  {response.stop_reason}")
    print(f"  Usage:        total={response.usage.total_tokens}")

    assert response.content is not None
    print("  ✅ PASSED")
    return response


async def main():
    # ── Check API key ─────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found!")
        print("   Create a .env file in this directory with:")
        print("   GOOGLE_API_KEY=AIza...your-key...")
        sys.exit(1)

    print(f"🔑 API key loaded: {api_key[:8]}...{api_key[-4:]}")

    # ── Build bridge ──────────────────────────────────────────────
    config = build_config(api_key)
    bridge = LLMBridge(config)

    async def test_list_models(bridge: LLMBridge):
        print("=" * 60)
        print("🧪 Test 5: List Available Models")
        print("=" * 60)
        
        models_dict = await bridge.get_available_models()
        for provider, models in models_dict.items():
            print(f"  Provider: {provider} ({len(models)} models found)")
            # Print first 5 models as a sample
            sample = ", ".join(models[:20])
            print(f"  Sample:   {sample}{' ...' if len(models) > 5 else ''}")
        
        assert len(models_dict) > 0
        print("  ✅ PASSED")

    # ── Run tests ─────────────────────────────────────────────────
    tests = [
        test_simple_text,
        test_structured_output,
        test_inline_model_string,
        test_chat_parameters_variety,
        test_list_models,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            await test_fn(bridge)
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"🏁 Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("🎉 All live tests passed!")
    else:
        print("⚠️  Some tests failed — check output above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
