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

### Step 1: Determine Target Location and Module Name

Ask the user two things:

1. **Where** the LLM Bridge module should be placed (parent directory). Common choices:
   - `src/services/` — for service-oriented projects
   - `src/` — for flat layouts
   - `infra/` — for infrastructure-heavy projects
   - `app/` — for application packages
   - Custom path

2. **What to name** the module directory (this becomes the Python package name). Common choices:
   - `llm_bridge` (default)
   - `llm_client`
   - `ai_client`
   - `llm_gateway`
   - Custom name

Record the chosen parent as `$PARENT_DIR` and module name as `$MODULE_NAME`.
The full target path is `$TARGET_DIR = $PARENT_DIR/$MODULE_NAME`.

**Important**: the module name becomes the Python import name, e.g. if `$MODULE_NAME = ai_client` and `$PARENT_DIR = src/services`, then callers use `from src.services.ai_client import LLMBridge` (adjusted for the project's Python path). Record this as `$MODULE_IMPORT_PATH`.

### Step 2: Ask Which Providers to Include

Ask the user which providers they need. Options:
1. **OpenAI** (GPT-4o, o1, etc.)
2. **Anthropic** (Claude 3.5 Sonnet, Opus, etc.)
3. **Google Gemini** (via AI Studio)
4. **Vertex AI** (via Google Cloud — uses the same `google_provider.py` adapter, no extra file needed)
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
    ├── openai_provider.py    # (if OpenAI selected)
    ├── anthropic_provider.py # (if Anthropic selected)
    └── google_provider.py    # (if Google Gemini or Vertex AI selected)
```

If the user only selected specific providers, **skip** the provider files that are not needed and remove their imports from `providers/__init__.py`.

Note on Vertex AI: it reuses `google_provider.py` (same adapter, different init params). No separate file is needed — just keep `google_provider.py` and the `"vertexai": GoogleProvider` entry in `PROVIDER_REGISTRY`.

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

After all files are created, display a usage guide using the actual `$MODULE_IMPORT_PATH` determined in Step 1:

```python
# Quick Start — Using LLM Bridge

import asyncio
from $MODULE_IMPORT_PATH import LLMBridge

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

## Important Rules for AI

1. **All LLM calls in the project must go through `LLMBridge.chat()`** — never call provider SDKs directly.
2. **Always prefer structured output** (pass `response_model`) over raw text when the output has a known schema.
3. **Use the `google-genai` SDK** for Google models — never use the deprecated `google-generativeai`.
4. **Log token usage** after every call for cost tracking.
5. **Keep env vars in `.env`** — never hardcode API keys in source files.
6. **Always use the actual `$MODULE_IMPORT_PATH`** in import examples — never hardcode `infra.llm_bridge`.
