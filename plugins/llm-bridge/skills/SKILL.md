---
name: llm-bridge-python
description: Generate a production-ready LLM Bridge module in any Python project. Provides unified multi-provider LLM access (OpenAI, Anthropic, Google Gemini, Vertex AI) with structured output, retry, fallback, and load balancing. Use when the user wants to integrate LLM capabilities, add AI features, or connect to large language models.
---

# LLM Bridge Python Skill

Auto-generate a complete, production-ready LLM integration module in any Python project. The generated module unifies access to OpenAI, Anthropic, Google Gemini (AI Studio), and Vertex AI with:

- **Pydantic-based structured output** for all providers
- **Automatic retry with exponential backoff**
- **Dynamic fallback** across models/providers
- **API key rotation / load balancing**
- **Unified response format** with token usage tracking

## When to Use This Skill

Activate this skill when the user:
- Asks to "integrate LLM", "add AI features", "connect to GPT/Claude/Gemini"
- Needs a unified interface to call multiple LLM providers
- Wants structured output (JSON/Pydantic) from LLM calls
- Needs retry, fallback, or load balancing for LLM calls

## Workflow

### Step 1: Determine Target Location

Ask the user where the LLM Bridge module should be placed. Default recommendation: `infra/llm_bridge/` relative to the project root.

If the user provides a different path, use that instead. Record the chosen path as `$TARGET_DIR`.

### Step 2: Ask Which Providers to Include

Ask the user which providers they need. Options:
1. **OpenAI** (GPT-4o, o1, etc.)
2. **Anthropic** (Claude 3.5 Sonnet, Opus, etc.)
3. **Google Gemini** (via AI Studio)
4. **Vertex AI** (via Google Cloud)
5. **All of the above** (default)

Record the selection as `$PROVIDERS`.

### Step 3: Copy Template Files

Copy files from this skill's `templates/llm_bridge/` directory to `$TARGET_DIR` in the user's project. The files to copy are:

```
$TARGET_DIR/
├── __init__.py
├── models.py          # UnifiedResponse, BridgeConfig, etc.
├── config.py          # YAML config loader with env var interpolation
├── router.py          # Retry, fallback, key rotation logic
├── bridge.py          # Main LLMBridge class (entry point)
└── providers/
    ├── __init__.py
    ├── base.py               # Abstract base provider
    ├── openai_provider.py    # (if selected)
    ├── anthropic_provider.py # (if selected)
    └── google_provider.py    # (if selected)
```

If the user only selected specific providers, **skip** the provider files that are not needed and remove their imports from `providers/__init__.py`.

### Step 4: Generate Configuration File

Create a `llm_bridge_config.yaml` file in the project root (or the location preferred by the user). Use the example template from `templates/llm_bridge_config.example.yaml` as a base, keeping only the selected providers.

### Step 5: Update Dependencies

Check if a `requirements.txt` or `pyproject.toml` exists in the project root.

**For `requirements.txt`**, append these lines (skip if already present):
```
pydantic>=2.0
python-dotenv>=1.0
pyyaml>=6.0
```

And based on selected providers:
- OpenAI: `openai>=1.0`
- Anthropic: `anthropic>=0.40`
- Google Gemini / Vertex AI: `google-genai>=1.0`

**For `pyproject.toml`**, add to the `[project.dependencies]` section equivalently.

### Step 6: Display Usage Guide

After all files are created, display a usage guide to the user:

```python
# Quick Start — Using LLM Bridge

import asyncio
from infra.llm_bridge import LLMBridge

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")

    # Simple text response
    response = await bridge.chat(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content)
    print(f"Tokens used: {response.usage}")

    # Structured output with Pydantic
    from pydantic import BaseModel

    class Summary(BaseModel):
        title: str
        key_points: list[str]

    response = await bridge.chat(
        model="google/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Summarize quantum computing"}],
        response_model=Summary
    )
    print(response.parsed)  # Summary(title=..., key_points=[...])

asyncio.run(main())
```

Adjust the import path based on the actual `$TARGET_DIR` chosen in Step 1.

## Important Rules for AI

1. **All LLM calls in the project must go through `LLMBridge.chat()`** — never call provider SDKs directly.
2. **Always prefer structured output** (pass `response_model`) over raw text when the output has a known schema.
3. **Use the `google-genai` SDK** for Google models — never use the deprecated `google-generativeai`.
4. **Log token usage** after every call for cost tracking.
5. **Keep env vars in `.env`** — never hardcode API keys in source files.
