# LLM Bridge Skills Loader — Design Spec

**Date:** 2026-03-16
**Status:** Approved
**Scope:** `plugins/llm-bridge`

---

## 1. Overview

Extend the LLM Bridge Python module with a local skill-loading mechanism. Users can define reusable prompt skills (in standard SKILL.md format with Jinja2 templating) and inject them into `LLMBridge.chat()` calls via a `skills` parameter.

Additionally, write comprehensive user-facing documentation covering the LLM Gateway API and the skill loading system.

---

## 2. Goals

1. Load skills from a configurable `skills/` directory (standard format: `skills/<name>/SKILL.md`)
2. Support Jinja2 template variables in SKILL.md files
3. Inject loaded skill content into LLM calls as system prompt (prepended before any user-defined system prompt)
4. Support loading multiple skills per call (concatenated with `---` separator)
5. Minimal changes to existing `LLMBridge.chat()` signature (two new optional kwargs)
6. Write `docs/llm-bridge-guide.md` covering API usage and skill loading

---

## 3. Non-Goals

- No middleware pipeline architecture
- No skill registry or decorator pattern
- No changes to provider adapters
- No dynamic skill discovery at runtime (skills are loaded on demand, not preloaded)

---

## 4. File Changes

### 4.1 New File: `templates/llm_bridge/skills.py`

Contains the `SkillLoader` class.

```
Responsibilities:
- Discover available skills from skills_dir
- Load SKILL.md for a given skill name
- Strip YAML frontmatter (--- blocks)
- Render content with Jinja2 using provided variables
- Concatenate multiple skills with separator
```

**Public API:**

```python
class SkillLoader:
    def __init__(self, skills_dir: Path) -> None: ...

    def load(self, skill_name: str, variables: dict = {}) -> str:
        """Load and Jinja2-render a single skill. Returns rendered string."""

    def load_many(self, skill_names: list[str], variables: dict = {}) -> str:
        """Load and concatenate multiple skills separated by '\\n\\n---\\n\\n'."""

    def list_skills(self) -> list[str]:
        """Return names of all available skills in skills_dir."""
```

**Error handling:**
- `SkillNotFoundError(skill_name)` — raised when `skills/<name>/SKILL.md` does not exist
- `SkillRenderError(skill_name, cause)` — raised when Jinja2 rendering fails
- Both are subclasses of `LLMBridgeError` (new base exception class)

### 4.2 Modified: `templates/llm_bridge/models.py`

Add `skills_dir` field to `BridgeConfig`:

```python
class BridgeConfig(BaseModel):
    ...
    skills_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to skills directory. Each subdirectory must contain a SKILL.md file. "
            "Defaults to 'skills/' relative to the directory containing llm_bridge_config.yaml."
        ),
    )
```

### 4.3 Modified: `templates/llm_bridge/bridge.py`

1. Instantiate `SkillLoader` in `__init__` if `config.skills_dir` is set (or default path exists).
2. Add `skills` and `skill_vars` parameters to `chat()`:

```python
async def chat(
    self,
    model: str,
    messages: list[dict],
    *,
    response_model: Optional[Type[BaseModel]] = None,
    params: Optional[ChatParameters] = None,
    skills: Optional[list[str]] = None,
    skill_vars: Optional[dict] = None,
    **kwargs,
) -> UnifiedResponse:
```

3. Skill injection logic (before `_resolve_model`):

```python
if skills:
    skill_content = self._skill_loader.load_many(skills, skill_vars or {})
    existing = params.system_prompt if params else None
    merged_system = skill_content if not existing else f"{skill_content}\n\n---\n\n{existing}"
    params = (params or ChatParameters()).model_copy(update={"system_prompt": merged_system})
```

### 4.4 Modified: `templates/llm_bridge_config.example.yaml`

Add `skills_dir` field with comment:

```yaml
# Path to skills directory (optional).
# Each subdirectory must contain a SKILL.md file.
# Defaults to 'skills/' relative to this config file.
# skills_dir: ./skills
```

### 4.5 New File: `templates/llm_bridge/__init__.py` (export update)

Export `SkillLoader` and the new exception classes from the package's public API.

### 4.6 New File: `plugins/llm-bridge/docs/llm-bridge-guide.md`

User-facing documentation (see Section 6).

---

## 5. Skill Directory Format

```
skills/
└── <skill-name>/
    ├── SKILL.md          # Required — Jinja2 template, optional YAML frontmatter
    ├── references/       # Optional — additional context files (not auto-loaded)
    ├── scripts/          # Optional — helper scripts
    └── templates/        # Optional — output templates
```

**SKILL.md structure:**

```markdown
---
name: code-reviewer
description: Reviews code for quality and security
---

You are an expert {{ language | default("Python") }} code reviewer.

Focus on:
- Security vulnerabilities
- Performance issues
- Context: {{ project_context | default("general software project") }}
```

- Frontmatter is stripped before injection (not sent to the LLM)
- Jinja2 `{{ variable }}` syntax for dynamic values
- Jinja2 `| default(...)` filter supported for optional variables

---

## 6. Documentation Structure (`docs/llm-bridge-guide.md`)

### Part A — LLM Gateway API

1. Quick start (install deps, create config YAML, first call)
2. `bridge.chat()` parameters reference
3. Structured output with Pydantic `response_model`
4. `ChatParameters` full reference
5. Config file reference (models, retry, fallback, log_usage)
6. Multi-provider inline syntax (`"openai/gpt-4o"`, `"anthropic/claude-sonnet-4"`)

### Part B — Skills

1. What is a skill
2. Directory layout and SKILL.md format
3. Jinja2 templating in skills
4. Loading a skill in `bridge.chat()`
5. Loading multiple skills
6. Configuring `skills_dir` in YAML
7. Full end-to-end example

---

## 7. Data Flow

```
bridge.chat(model, messages, skills=["reviewer"], skill_vars={"lang": "Python"})
    │
    ├─ SkillLoader.load_many(["reviewer"], {"lang": "Python"})
    │       │
    │       └─ Read skills/reviewer/SKILL.md
    │          Strip frontmatter
    │          Jinja2 render with skill_vars
    │          Return rendered string
    │
    ├─ Merge rendered string with params.system_prompt
    │
    └─ Proceed to _resolve_model → _call_model → provider.chat()
```

---

## 8. Dependencies

- `jinja2>=3.0` — add to `requirements.txt` and `pyproject.toml` section in SKILL.md
- `pyyaml>=6.0` — already required (for frontmatter parsing, reuse existing dep)

---

## 9. Testing

- `test_skills.py` added to templates — covers:
  - Load single skill with variables
  - Load multiple skills (verify separator)
  - Missing skill raises `SkillNotFoundError`
  - Jinja2 render error raises `SkillRenderError`
  - Frontmatter is stripped correctly
  - `bridge.chat()` with `skills` param injects system prompt correctly
  - Existing `system_prompt` + skills merges correctly

---

## 10. SKILL.md Skill (for llm-bridge-python skill)

Update `plugins/llm-bridge/skills/llm-bridge-python/SKILL.md` to include:
- Step for copying `skills.py`
- Step for adding `jinja2` to dependencies
- Updated usage guide showing `skills` parameter
