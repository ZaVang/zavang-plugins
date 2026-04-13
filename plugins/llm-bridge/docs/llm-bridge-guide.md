# LLM Bridge User Guide

LLM Bridge is a unified Python gateway for OpenAI, Anthropic, Google Gemini, and Vertex AI. A single `bridge.chat()` call routes to the right provider, handles API-key rotation, exponential-backoff retry, dynamic fallback, and structured Pydantic output — without touching provider SDKs directly.

---

## Table of Contents

- [Part A — LLM Gateway API](#part-a--llm-gateway-api)
  - [Quick Start](#quick-start)
  - [bridge.chat() Parameter Reference](#bridgechat-parameter-reference)
  - [Structured Output with Pydantic](#structured-output-with-pydantic)
  - [ChatParameters Reference](#chatparameters-reference)
  - [Config File Reference](#config-file-reference)
  - [Inline Model Syntax](#inline-model-syntax)
- [Part B — Skills](#part-b--skills)
  - [What Is a Skill](#what-is-a-skill)
  - [Directory Layout and SKILL.md Format](#directory-layout-and-skillmd-format)
  - [Jinja2 Templating](#jinja2-templating)
  - [Loading a Skill in bridge.chat()](#loading-a-skill-in-bridgechat)
  - [Loading Multiple Skills](#loading-multiple-skills)
  - [Configuring skills_dir in YAML](#configuring-skills_dir-in-yaml)
  - [Full End-to-End Example](#full-end-to-end-example)

---

## Part A — LLM Gateway API

### Quick Start

**1. Install dependencies**

Core library:

```bash
pip install pydantic>=2.0 pyyaml>=6.0 python-dotenv>=1.0 jinja2>=3.0 python-frontmatter>=1.0
```

Provider SDKs (install only those you need):

```bash
pip install openai>=1.0                # OpenAI
pip install anthropic>=0.40            # Anthropic
pip install google-genai>=1.0          # Google Gemini / Vertex AI
```

**2. Create a config file**

Create `llm_bridge_config.yaml` in your project root. API keys are read from environment variables (or a `.env` file):

```yaml
# llm_bridge_config.yaml

retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 60.0
  exponential_base: 2.0

log_usage: true
default_model: my-gpt4o

models:
  my-gpt4o:
    provider: openai
    model: gpt-4o
    api_keys:
      - ${OPENAI_API_KEY}
    max_tokens: 4096
    temperature: 0.7
```

Put your keys in `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

**3. Make your first call**

```python
import asyncio
from llm_bridge import LLMBridge

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")

    response = await bridge.chat(
        model="my-gpt4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    print(response.content)              # "Hello! How can I help you?"
    print(response.usage.total_tokens)   # 25
    print(response.stop_reason)          # "stop"

asyncio.run(main())
```

---

### bridge.chat() Parameter Reference

```python
async def chat(
    model: str,
    messages: list[dict],
    *,
    response_model: Optional[Type[BaseModel]] = None,
    params: Optional[ChatParameters] = None,
    skills: Optional[list[str]] = None,
    skill_vars: Optional[dict[str, Any]] = None,
) -> UnifiedResponse
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `str` | Yes | Config alias (e.g. `"my-gpt4o"`) or inline `"provider/model"` string (e.g. `"openai/gpt-4o"`). |
| `messages` | `list[dict]` | Yes | OpenAI-style message list. Each dict must have `"role"` and `"content"` keys. |
| `response_model` | `Type[BaseModel]` | No | Pydantic model class for structured output. When set, `response.parsed` holds the parsed instance. |
| `params` | `ChatParameters` | No | Generation parameters (temperature, max_tokens, etc.). All fields are optional. |
| `skills` | `list[str]` | No | Names of skills to load and inject as the system prompt. |
| `skill_vars` | `dict[str, Any]` | No | Jinja2 template variables passed to all loaded skills. |

**Return value — `UnifiedResponse`**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| None` | Raw text response from the model. |
| `parsed` | `Any \| None` | Parsed Pydantic instance when `response_model` was supplied. |
| `thinking` | `str \| None` | Chain-of-thought text when available (certain models). |
| `usage` | `UsageInfo` | Token counts: `input_tokens`, `output_tokens`, `total_tokens`. |
| `stop_reason` | `str \| None` | Why the model stopped (`"stop"`, `"end_turn"`, `"max_tokens"`, etc.). |
| `model_id` | `str \| None` | Provider-assigned completion ID (e.g. `"chatcmpl-abc123"`). |
| `provider` | `str` | Lowercase provider name: `"openai"`, `"anthropic"`, or `"google"`. |
| `model` | `str` | Model name as returned by the provider. |

**Example — all parameters in use**

```python
from llm_bridge import LLMBridge, ChatParameters
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")

    response = await bridge.chat(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Summarize quantum computing"}],
        response_model=Summary,
        params=ChatParameters(
            temperature=0.3,
            max_tokens=1024,
            system_prompt="Be concise and technical.",
        ),
        skills=["my-skill"],
        skill_vars={"audience": "engineers"},
    )

    print(response.parsed.title)       # "Quantum Computing"
    print(response.parsed.key_points)  # ["Superposition", ...]
```

---

### Structured Output with Pydantic

Pass a Pydantic `BaseModel` subclass as `response_model`. The bridge instructs the provider to return JSON matching that schema and populates `response.parsed` with the parsed instance.

```python
from pydantic import BaseModel
from llm_bridge import LLMBridge

class MovieReview(BaseModel):
    title: str
    rating: float          # 0.0 – 10.0
    pros: list[str]
    cons: list[str]
    verdict: str

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")

    response = await bridge.chat(
        model="google/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Review the movie Inception"}],
        response_model=MovieReview,
    )

    review: MovieReview = response.parsed
    print(f"{review.title}: {review.rating}/10")
    print(f"Pros: {', '.join(review.pros)}")
    print(f"Cons: {', '.join(review.cons)}")
    print(f"Verdict: {review.verdict}")
```

`response.content` still contains the raw JSON string. `response.parsed` holds the validated `MovieReview` instance. If parsing fails, an error is raised before `response` is returned.

---

### ChatParameters Reference

`ChatParameters` is a Pydantic model — all fields are optional and default to `None` (provider defaults apply when `None`).

```python
from llm_bridge import ChatParameters

params = ChatParameters(
    max_tokens=2048,
    temperature=0.5,
    top_p=0.9,
    stop_sequences=["END"],
    system_prompt="You are a helpful assistant.",
)
```

| Field | Type | Default | OpenAI | Anthropic | Google |
|-------|------|---------|--------|-----------|--------|
| `max_tokens` | `int` | `None` | `max_completion_tokens` | `max_tokens` (defaults to 4096 if `None`) | `max_output_tokens` |
| `temperature` | `float` | `None` | Supported (0.0–2.0) | Supported (0.0–1.0) | Supported (0.0–2.0) |
| `top_p` | `float` | `None` | Supported | Supported | Supported |
| `top_k` | `int` | `None` | Not supported (ignored) | Supported | Supported |
| `frequency_penalty` | `float` | `None` | Supported (−2.0–2.0) | Not supported (ignored) | Supported |
| `presence_penalty` | `float` | `None` | Supported (−2.0–2.0) | Not supported (ignored) | Supported |
| `stop_sequences` | `list[str]` | `None` | `stop` (max 4) | `stop_sequences` | `stop_sequences` |
| `seed` | `int` | `None` | Supported | Not supported (ignored) | Supported |
| `system_prompt` | `str` | `None` | Injected as `role="system"` message | Top-level `system` param | `system_instruction` |

**Notes:**
- Parameters unsupported by a provider are silently ignored — your code is portable across providers without changes.
- `temperature` and `top_p` should generally not be set at the same time. Use one or the other.
- Config-level `max_tokens` and `temperature` are used as defaults when the corresponding `ChatParameters` fields are `None`. Call-level params always win.

---

### Config File Reference

`llm_bridge_config.yaml` is loaded via `LLMBridge.from_config(path)`. Environment variables are resolved before validation — use `${VAR}` or `${VAR:-default}` syntax anywhere in the file.

**Full annotated example:**

```yaml
# llm_bridge_config.yaml

# ------------------------------------------------------------------
# Retry settings — applied to every model call.
# ------------------------------------------------------------------
retry:
  max_retries: 3          # Number of attempts before giving up.
  base_delay: 1.0         # Seconds before the first retry.
  max_delay: 60.0         # Cap on delay growth (seconds).
  exponential_base: 2.0   # Delay multiplier per retry (1s, 2s, 4s, …).

# ------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------
log_usage: true           # Log input/output/total tokens after each call.
default_model: my-gpt4o  # Used when no model is passed to bridge.chat().
skills_dir: ./skills      # Path to skills directory (optional).
                          # Relative paths resolve from this config file's location.
                          # If omitted, looks for 'skills/' next to this file.

# ------------------------------------------------------------------
# Model definitions — each key is an alias used in bridge.chat().
# ------------------------------------------------------------------
models:
  my-gpt4o:
    provider: openai                # "openai" | "anthropic" | "google" | "vertexai"
    model: gpt-4o                   # Model name passed to the SDK.
    api_keys:
      - ${OPENAI_API_KEY}           # Multiple keys → round-robin rotation.
      - ${OPENAI_API_KEY_2}
    max_tokens: 4096                # Default max output tokens for this model.
    temperature: 0.7                # Default sampling temperature.
    fallback_models:
      - my-claude                   # Try these aliases if this model fails.
    extra: {}                       # Provider-specific kwargs passed to the SDK.

  my-claude:
    provider: anthropic
    model: claude-sonnet-4
    api_keys:
      - ${ANTHROPIC_API_KEY}
    max_tokens: 4096
    temperature: 0.7

  my-gemini:
    provider: google
    model: gemini-2.0-flash
    api_keys:
      - ${GOOGLE_API_KEY}
    max_tokens: 4096
    temperature: 0.7

  # Vertex AI — uses Application Default Credentials, no api_keys needed.
  my-vertex:
    provider: vertexai
    model: gemini-2.0-flash
    api_keys: []
    max_tokens: 4096
    extra:
      vertexai: true
      project: my-gcp-project
      location: us-central1
```

**`models` entry fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | Required | One of: `openai`, `anthropic`, `google`, `vertexai`. |
| `model` | `str` | Required | Model name passed to the provider SDK. |
| `api_keys` | `list[str]` | `[]` | One or more API keys. Multiple keys are rotated round-robin. |
| `max_tokens` | `int` | `4096` | Default max output tokens. Overridden by `ChatParameters.max_tokens`. |
| `temperature` | `float` | `0.7` | Default temperature. Overridden by `ChatParameters.temperature`. |
| `fallback_models` | `list[str]` | `[]` | Ordered list of model aliases to try if this model fails. |
| `extra` | `dict` | `{}` | Additional kwargs forwarded to the provider adapter. |

**Top-level fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models` | `dict` | `{}` | Map of alias to model config. |
| `retry` | `RetryConfig` | See above | Retry and backoff settings. |
| `default_model` | `str` | `None` | Alias used when `bridge.chat()` cannot resolve the model string. |
| `log_usage` | `bool` | `true` | Whether to log token usage after each call. |
| `skills_dir` | `str` | `None` | Path to skills directory. Defaults to `skills/` next to the config file. |

---

### Inline Model Syntax

Instead of defining a model alias in YAML, you can use the `"provider/model"` shorthand directly in `bridge.chat()`:

```python
# OpenAI
response = await bridge.chat(
    model="openai/gpt-4o",
    messages=[...],
)

# Anthropic
response = await bridge.chat(
    model="anthropic/claude-sonnet-4",
    messages=[...],
)

# Google Gemini (AI Studio)
response = await bridge.chat(
    model="google/gemini-2.0-flash",
    messages=[...],
)

# Vertex AI
response = await bridge.chat(
    model="vertexai/gemini-2.0-flash",
    messages=[...],
)
```

The part before `/` must match a registered provider name. The part after `/` is passed directly to the provider SDK as the model identifier.

Inline syntax uses no API keys and no config-level defaults (max_tokens, temperature, fallback_models). For production use, prefer named aliases in the YAML config so you get key rotation, fallback, and per-model defaults.

---

## Part B — Skills

### What Is a Skill

A skill is a reusable prompt template that gets injected as the system prompt for a `bridge.chat()` call. Skills are stored as `SKILL.md` files with optional YAML frontmatter and a Jinja2 template body.

Instead of duplicating long system prompts across your codebase, you define them once in a skill file and reference them by name.

---

### Directory Layout and SKILL.md Format

Skills live in a `skills/` directory. Each skill is a subdirectory named after the skill, containing a required `SKILL.md` file:

```
skills/
├── code-reviewer/
│   └── SKILL.md
├── summarizer/
│   ├── SKILL.md
│   └── examples.txt      # Optional supporting files (not loaded automatically)
└── translator/
    └── SKILL.md
```

**SKILL.md format:**

```markdown
---
name: code-reviewer
description: Reviews code for quality, bugs, and best practices.
---

You are an expert {{ language }} code reviewer. Your task is to review code submitted by the user.

Focus on:
- Correctness and logic errors
- {{ focus_area | default("general best practices") }}
- Security vulnerabilities
- Performance issues

Provide your review as:
1. A brief summary
2. A list of specific issues (if any)
3. Suggested improvements
```

The YAML frontmatter block (between `---` delimiters) is optional metadata stripped before rendering. The body below the frontmatter is the Jinja2 template that becomes the system prompt.

---

### Jinja2 Templating

The SKILL.md body is rendered as a Jinja2 template before being injected as a system prompt.

**Variable substitution:**

```
{{ variable_name }}
```

**Default values (if variable may not be provided):**

```
{{ variable_name | default("fallback value") }}
```

**Conditional blocks:**

```
{% if strict_mode %}
Be strict and flag all style issues, even minor ones.
{% else %}
Focus only on significant problems.
{% endif %}
```

**StrictUndefined behavior:** LLM Bridge uses Jinja2's `StrictUndefined` mode. If your template references a variable that is not passed in `skill_vars`, rendering raises a `SkillRenderError`. Always use `| default(...)` for optional variables.

**Examples:**

```
# Always required — raises SkillRenderError if missing:
You are reviewing {{ language }} code.

# Optional with fallback:
Tone: {{ tone | default("professional") }}

# Optional block:
{% if persona %}Act as {{ persona }}.{% endif %}
```

---

### Loading a Skill in bridge.chat()

Pass skill names in `skills` and template variables in `skill_vars`:

```python
response = await bridge.chat(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Review this: def add(a, b): return a + b"}],
    skills=["code-reviewer"],
    skill_vars={"language": "Python"},
)

print(response.content)
```

The rendered skill body is injected as the system prompt. If `params.system_prompt` is also set, the skill content comes first, followed by `\n\n---\n\n`, then the system prompt.

---

### Loading Multiple Skills

Pass a list of skill names to `skills`. Skills are rendered and concatenated in order, separated by `\n\n---\n\n`:

```python
response = await bridge.chat(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Review and summarize this code: ..."}],
    skills=["code-reviewer", "summarizer"],
    skill_vars={
        "language": "Python",
        "format": "bullet points",
    },
)
```

The final system prompt will be:

```
<rendered code-reviewer body>

---

<rendered summarizer body>
```

The same `skill_vars` dict is passed to every skill. Variables not referenced by a skill are ignored.

---

### Configuring skills_dir in YAML

Add `skills_dir` to your config file:

```yaml
# llm_bridge_config.yaml
skills_dir: ./skills
```

Relative paths are resolved relative to the config file's location. If your config is at `/project/llm_bridge_config.yaml` and `skills_dir` is `./skills`, the loader looks for skills at `/project/skills/`.

**Auto-discovery:** If `skills_dir` is not set, LLM Bridge automatically checks for a `skills/` directory next to the config file. If that directory exists, it is used as the skills directory. If it does not exist, skills are disabled and passing `skills=` to `bridge.chat()` raises `LLMBridgeError`.

```yaml
# Explicitly set:
skills_dir: ./skills

# Or use a path outside the project:
skills_dir: /shared/prompt-library/skills

# Or omit and rely on auto-discovery (looks for skills/ next to config):
# skills_dir: (not set)
```

---

### Full End-to-End Example

This example creates a code-reviewer skill and calls it from Python.

**Step 1: Create the skill**

```
mkdir -p skills/code-reviewer
```

Create `skills/code-reviewer/SKILL.md`:

```markdown
---
name: code-reviewer
description: Reviews code for quality, correctness, and security issues.
---

You are a senior {{ language | default("software") }} engineer performing a thorough code review.

Review criteria:
- **Correctness**: Does the code do what it claims?
- **Security**: Are there any vulnerabilities (injection, auth issues, etc.)?
- **Performance**: Are there obvious inefficiencies?
- **Readability**: Is the code easy to understand and maintain?
{% if style_guide %}
- **Style**: Follow the {{ style_guide }} style guide.
{% endif %}

Format your response as:
1. Overall assessment (1-2 sentences)
2. Issues found (numbered list, or "None found")
3. Suggested improvements (if any)
```

**Step 2: Create the config file**

Create `llm_bridge_config.yaml` in your project root:

```yaml
retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 60.0
  exponential_base: 2.0

log_usage: true
skills_dir: ./skills

models:
  gpt4o:
    provider: openai
    model: gpt-4o
    api_keys:
      - ${OPENAI_API_KEY}
    max_tokens: 2048
    temperature: 0.3
```

**Step 3: Write the Python code**

```python
import asyncio
from llm_bridge import LLMBridge

CODE_TO_REVIEW = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")

    response = await bridge.chat(
        model="gpt4o",
        messages=[
            {
                "role": "user",
                "content": f"Please review this code:\n\n```python\n{CODE_TO_REVIEW}\n```",
            }
        ],
        skills=["code-reviewer"],
        skill_vars={
            "language": "Python",
            "style_guide": "PEP 8",
        },
    )

    print(response.content)
    print(f"\nTokens used: {response.usage.total_tokens}")

asyncio.run(main())
```

**Expected project layout:**

```
project/
├── llm_bridge_config.yaml
├── .env
├── main.py
├── skills/
│   └── code-reviewer/
│       └── SKILL.md
└── llm_bridge/
    ├── __init__.py
    ├── bridge.py
    ├── config.py
    ├── models.py
    ├── router.py
    ├── skills.py
    └── providers/
        ├── __init__.py
        ├── base.py
        ├── openai_provider.py
        ├── anthropic_provider.py
        └── google_provider.py
```
