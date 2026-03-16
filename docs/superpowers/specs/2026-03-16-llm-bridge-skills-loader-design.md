# LLM Bridge Skills Loader — Design Spec

**Date:** 2026-03-16
**Status:** Approved
**Scope:** `plugins/llm-bridge`

---

## 1. Overview

Extend the LLM Bridge Python module with a local skill-loading mechanism. Users define reusable prompt skills (SKILL.md with optional YAML frontmatter + Jinja2 body) and inject them into `LLMBridge.chat()` via a `skills` parameter.

Additionally, write user-facing documentation covering the LLM Gateway API and the skill loading system.

---

## 2. Goals

1. Load skills from a configurable `skills/` directory (one subdirectory per skill, each containing `SKILL.md`)
2. Jinja2 template variables with `StrictUndefined` — missing variables raise `SkillRenderError`
3. Inject skill content as system prompt, prepended before any caller-supplied system prompt
4. Support loading multiple skills per call; frontmatter stripped first, then concatenated with `"\n\n---\n\n"`
5. Two new optional kwargs on `chat()`: `skills` and `skill_vars`
6. Write `plugins/llm-bridge/docs/llm-bridge-guide.md`

---

## 3. Non-Goals

- No middleware pipeline, no skill registry, no decorator pattern
- `SkillLoader` is instantiated eagerly (validates `skills_dir` on startup); individual SKILL.md files are read lazily on each `chat()` call
- Skill loading is synchronous (local file I/O; async overhead not warranted)
- `jinja2` and `python-frontmatter` are unconditional dependencies — every consumer of the skill gets them; this is an intentional simplicity trade-off

---

## 4. File Changes

All paths relative to the skill template root:
`plugins/llm-bridge/skills/llm-bridge-python/templates/`

### 4.1 New File: `llm_bridge/skills.py`

**Exception hierarchy** (top of this file, exported via `__init__.py`):

```python
class LLMBridgeError(Exception):
    """Base exception for all LLM Bridge errors."""

class SkillNotFoundError(LLMBridgeError):
    def __init__(self, skill_name: str, skills_dir: Path) -> None:
        super().__init__(f"Skill '{skill_name}' not found in '{skills_dir}'.")
        self.skill_name = skill_name
        self.skills_dir = skills_dir

class SkillRenderError(LLMBridgeError):
    def __init__(self, skill_name: str, cause: Exception) -> None:
        super().__init__(f"Failed to render skill '{skill_name}': {cause}")
        self.skill_name = skill_name
        self.cause = cause
```

**`SkillLoader` class — full signatures:**

```python
class SkillLoader:
    def __init__(self, skills_dir: Path) -> None:
        """Store skills_dir. Raise FileNotFoundError if the directory does not exist."""

    def load(self, skill_name: str, variables: dict[str, Any] | None = None) -> str:
        """Load and render a single skill.

        1. Read skills_dir/<skill_name>/SKILL.md (UTF-8).
        2. Parse with python-frontmatter to strip the YAML frontmatter block.
        3. Render the body with Jinja2 (Environment(undefined=StrictUndefined)).
        4. Return the rendered string.

        Raises:
            SkillNotFoundError: if skills_dir/<skill_name>/SKILL.md does not exist.
            SkillRenderError:   if Jinja2 rendering fails (e.g. undefined variable).
        """

    def load_many(self, skill_names: list[str], variables: dict[str, Any] | None = None) -> str:
        """Call load() for each name in order. Join results with '\\n\\n---\\n\\n'.
        Returns a single concatenated string."""

    def list_skills(self) -> list[str]:
        """Return sorted list[str] of skill names (subdirectory names inside skills_dir)
        that contain a SKILL.md file. Returns [] if skills_dir has no valid skill dirs.
        Does not raise if a previously valid dir was deleted after __init__."""
```

Note: `variables=None` (not `{}`) avoids the mutable-default-argument Python anti-pattern. Internally, use `vars_ = variables or {}` before passing to Jinja2.

### 4.2 Modified: `llm_bridge/models.py`

Add `skills_dir` to `BridgeConfig`:

```python
class BridgeConfig(BaseModel):
    ...
    skills_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to the skills directory. Each subdirectory must contain SKILL.md. "
            "Relative paths are resolved from the config YAML file's parent directory. "
            "Defaults to 'skills/' next to the config file if that directory exists."
        ),
    )
```

### 4.3 Modified: `llm_bridge/bridge.py`

**`from_config` update** — must forward the config file path:

```python
@classmethod
def from_config(cls, path: Union[str, Path]) -> "LLMBridge":
    config_path = Path(path).resolve()
    return cls(load_config(config_path), config_path=config_path)
```

**`__init__` signature** — add `config_path` parameter:

```python
def __init__(self, config: BridgeConfig, *, config_path: Optional[Path] = None) -> None:
```

**Required new import in `bridge.py`:**

```python
from .skills import LLMBridgeError, SkillLoader
```

**`SkillLoader.__init__` contract clarification:**

`SkillLoader.__init__` raises `FileNotFoundError` if `skills_dir` does not exist. This is the contract for **direct usage** (e.g., in tests or standalone scripts).

`LLMBridge.__init__` uses a guard-then-None pattern: it checks `resolved_skills_dir.exists()` first and only constructs `SkillLoader` if the directory exists. When the directory is absent, `_skill_loader` is set to `None` silently — no error at startup, because skill-loading is optional. The error surfaces only when a caller actually passes `skills=[...]` to `chat()`.

```python
self._skill_loader: Optional[SkillLoader] = (
    SkillLoader(resolved_skills_dir) if resolved_skills_dir.exists() else None
)
```

**`skills_dir` resolution in `__init__`** (after the existing provider setup):

```python
# Resolve skills directory
if config.skills_dir:
    raw = Path(config.skills_dir)
    resolved_skills_dir = (
        raw if raw.is_absolute()
        else (config_path.parent / raw if config_path else raw)
    )
else:
    resolved_skills_dir = (
        config_path.parent / "skills" if config_path else Path("skills")
    )

self._skill_loader: Optional[SkillLoader] = (
    SkillLoader(resolved_skills_dir) if resolved_skills_dir.exists() else None
)
```

**`chat()` new parameters:**

```python
async def chat(
    self,
    model: str,
    messages: list[dict],
    *,
    response_model: Optional[Type[BaseModel]] = None,
    params: Optional[ChatParameters] = None,
    skills: Optional[list[str]] = None,
    skill_vars: Optional[dict[str, Any]] = None,
    **kwargs,
) -> UnifiedResponse:
```

**Skill injection block** (at the start of `chat()`, before `_resolve_model`):

```python
if skills:
    if self._skill_loader is None:
        # Store resolved_skills_dir on self for use in error messages
        raise LLMBridgeError(
            "Cannot load skills: no skills directory is configured or the directory "
            f"does not exist. Set 'skills_dir' in your config YAML."
        )
    skill_content = self._skill_loader.load_many(skills, skill_vars)
    existing_system = params.system_prompt if params else None
    merged_system = (
        skill_content
        if not existing_system
        else f"{skill_content}\n\n---\n\n{existing_system}"
    )
    # Create a NEW ChatParameters — never mutate the caller's object.
    # This merged object is shared across primary + all fallback partial calls.
    params = (params or ChatParameters()).model_copy(
        update={"system_prompt": merged_system}
    )
```

Note: `self._skill_loader is None` raises `LLMBridgeError` (not `SkillNotFoundError`) because there is no specific skill dir to report — the error is about configuration, not a missing skill file.

### 4.4 Modified: `llm_bridge_config.example.yaml`

```yaml
# Path to skills directory (optional).
# Each subdirectory must contain a SKILL.md file.
# Relative paths are resolved from this config file's location.
# Defaults to 'skills/' next to this config file (if that directory exists).
# skills_dir: ./skills
```

### 4.5 Modified: `llm_bridge/__init__.py`

Add import and extend `__all__` with exactly these four names:

```python
from .skills import LLMBridgeError, SkillLoader, SkillNotFoundError, SkillRenderError

__all__ = [
    # existing exports unchanged
    "LLMBridge",
    "ChatParameters",
    "UnifiedResponse",
    "UsageInfo",
    "BridgeConfig",
    "ModelConfig",
    "RetryConfig",
    # new skill-loader exports
    "SkillLoader",
    "LLMBridgeError",
    "SkillNotFoundError",
    "SkillRenderError",
]
```

`LLMBridgeError` must be exported because it is raised from the public `chat()` method and callers need to catch it by name.

---

## 5. Skill Directory Format

```
skills/
└── <skill-name>/
    ├── SKILL.md          # Required — optional YAML frontmatter + Jinja2 body
    ├── references/       # Optional — additional context files (not auto-loaded)
    ├── scripts/          # Optional — helper scripts
    └── templates/        # Optional — output templates
```

**SKILL.md format:**

```markdown
---
name: code-reviewer
description: Reviews code for quality and security
---

You are an expert {{ language }} code reviewer.

Focus on:
- Security vulnerabilities
- Performance issues
- Context: {{ project_context | default("general software project") }}
```

- Frontmatter (`---` block) is parsed and stripped by `python-frontmatter` before Jinja2 rendering
- Jinja2 uses `StrictUndefined`; every `{{ variable }}` must be in `skill_vars` or use `| default(...)`
- Files must be UTF-8 encoded

---

## 6. Data Flow

```
bridge.chat(model, messages, skills=["reviewer"], skill_vars={"language": "Python"})
    │
    ├─ SkillLoader.load_many(["reviewer"], {"language": "Python"})
    │       │
    │       └─ Read skills/reviewer/SKILL.md (UTF-8)
    │          python-frontmatter: strip YAML frontmatter
    │          Jinja2 render (StrictUndefined) with skill_vars
    │          Return rendered body string
    │
    ├─ Merge rendered string + params.system_prompt
    │  (skill content first, existing system_prompt second)
    │  Create NEW ChatParameters via model_copy() — no mutation of caller's object
    │
    ├─ Build functools.partial callables (primary + fallbacks) using merged params
    │  (skills loaded once; merged system_prompt shared across all fallback calls)
    │
    └─ fallback_chain / _call_model → provider.chat()
```

---

## 7. Documentation Structure (`plugins/llm-bridge/docs/llm-bridge-guide.md`)

### Part A — LLM Gateway API

1. Quick start (install deps, create config YAML, first call)
2. `bridge.chat()` full parameter reference
3. Structured output with Pydantic `response_model`
4. `ChatParameters` reference table (all fields, provider support matrix)
5. Config file reference (models, retry, fallback, log_usage, skills_dir)
6. Inline model syntax (`"openai/gpt-4o"`, `"anthropic/claude-sonnet-4"`)

### Part B — Skills

1. What is a skill
2. Directory layout and SKILL.md format
3. Jinja2 templating — variables, `| default()` filter, StrictUndefined behavior
4. Loading a skill in `bridge.chat()`
5. Loading multiple skills (concatenation order and separator)
6. Configuring `skills_dir` in YAML
7. Full end-to-end example

---

## 8. Dependencies

New (unconditional):
- `jinja2>=3.0`
- `python-frontmatter>=1.0`

Existing (unchanged):
- `pydantic>=2.0`, `pyyaml>=6.0`, `python-dotenv>=1.0`

Added to:
1. SKILL.md Step 5 unconditional block (alongside existing deps)
2. `requirements.txt` example in `llm-bridge-guide.md`

---

## 9. Testing (`templates/test_skills.py`)

| Test | Covers |
|------|--------|
| `test_load_single_skill_with_variables` | `{{ var }}` renders correctly |
| `test_load_skill_strips_frontmatter` | frontmatter block absent from output |
| `test_load_skill_no_frontmatter` | entire file content returned |
| `test_load_many_concatenates` | two skills joined by `\n\n---\n\n` |
| `test_load_missing_skill_raises` | `SkillNotFoundError` raised |
| `test_load_undefined_variable_raises` | `SkillRenderError` raised (StrictUndefined) |
| `test_load_default_filter_works` | `{{ var \| default("x") }}` without skill_vars |
| `test_bridge_chat_injects_skill_as_system_prompt` | skill sets system_prompt correctly |
| `test_bridge_chat_merges_with_existing_system_prompt` | skill first, existing second |
| `test_bridge_chat_does_not_mutate_caller_params` | original `ChatParameters` unchanged |

---

## 10. SKILL.md Skill Update (`plugins/llm-bridge/skills/llm-bridge-python/SKILL.md`)

- **Step 3**: Copy `skills.py` alongside other module files
- **Step 5**: Add `jinja2>=3.0` and `python-frontmatter>=1.0` to the unconditional deps block
- **Step 6**: Update usage guide to show `skills` and `skill_vars` parameters in examples
