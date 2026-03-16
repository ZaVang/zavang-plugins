# LLM Bridge Skills Loader Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local skill-loading mechanism to LLM Bridge so users can define SKILL.md prompt templates and inject them into `bridge.chat()` calls.

**Architecture:** New `skills.py` module provides `SkillLoader` (reads SKILL.md, strips frontmatter via `python-frontmatter`, renders with Jinja2 `StrictUndefined`). `LLMBridge.__init__` accepts an optional `config_path: Path` kwarg and resolves a `skills_dir` relative to that path. `bridge.chat()` gains `skills` and `skill_vars` kwargs that inject rendered skill content as system prompt before building provider callables. Skill injection creates a new `ChatParameters` via `model_copy()` — the caller's object is never mutated.

**Tech Stack:** Python 3.10+, Pydantic v2, Jinja2 ≥3.0, python-frontmatter ≥1.0, pytest + pytest-asyncio

---

## Chunk 1: `skills.py` — Exceptions + SkillLoader

**Files:**
- Create: `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/skills.py`
- Create: `plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py`

> All `pytest` commands in this chunk are run from:
> `plugins/llm-bridge/skills/llm-bridge-python/templates/`

---

### Task 1: Create `skills.py` with exception hierarchy and `SkillLoader`

- [ ] **Step 1: Write the failing tests**

Create `plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py`:

```python
"""Tests for llm_bridge.skills — SkillLoader and exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_bridge.skills import (
    LLMBridgeError,
    SkillLoader,
    SkillNotFoundError,
    SkillRenderError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    skills_dir: Path,
    name: str,
    body: str,
    frontmatter: str = "",
) -> None:
    """Write a SKILL.md file into skills_dir/<name>/SKILL.md."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        content = f"---\n{frontmatter}\n---\n\n{body}"
    else:
        content = body
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def skills_dir(tmp_path: Path) -> Path:
    """Return a tmp skills directory with two pre-populated skills."""
    _make_skill(
        tmp_path,
        "greeter",
        body="Hello, {{ name }}!",
    )
    _make_skill(
        tmp_path,
        "reviewer",
        frontmatter="name: reviewer\ndescription: code reviewer",
        body=(
            "You are an expert {{ language }} reviewer.\n"
            "Context: {{ project | default('general project') }}"
        ),
    )
    return tmp_path


# ---------------------------------------------------------------------------
# SkillLoader.__init__
# ---------------------------------------------------------------------------

def test_init_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Skills directory does not exist"):
        SkillLoader(tmp_path / "nonexistent")


def test_init_succeeds_on_existing_dir(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path)
    assert loader is not None


def test_init_succeeds_on_empty_dir(tmp_path: Path) -> None:
    # An empty skills dir is valid (no skills available, but no error)
    loader = SkillLoader(tmp_path)
    assert loader.list_skills() == []


# ---------------------------------------------------------------------------
# SkillLoader.list_skills
# ---------------------------------------------------------------------------

def test_list_skills_returns_sorted_names(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    assert loader.list_skills() == ["greeter", "reviewer"]


def test_list_skills_ignores_dirs_without_skill_md(skills_dir: Path) -> None:
    # A subdir with no SKILL.md must not appear in list_skills()
    (skills_dir / "no-skill-md").mkdir()
    loader = SkillLoader(skills_dir)
    assert "no-skill-md" not in loader.list_skills()


def test_list_skills_returns_empty_list_for_empty_dir(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path)
    assert loader.list_skills() == []


# ---------------------------------------------------------------------------
# SkillLoader.load — path construction
# ---------------------------------------------------------------------------

def test_load_reads_from_correct_path(tmp_path: Path) -> None:
    """load() must read skills_dir/<skill_name>/SKILL.md — not a flat file."""
    _make_skill(tmp_path, "my-skill", body="correct path content")
    # Also put a SKILL.md directly in skills_dir to verify it is NOT read
    (tmp_path / "SKILL.md").write_text("wrong flat content", encoding="utf-8")
    loader = SkillLoader(tmp_path)
    result = loader.load("my-skill")
    assert result == "correct path content"


# ---------------------------------------------------------------------------
# SkillLoader.load — happy path
# ---------------------------------------------------------------------------

def test_load_renders_variable(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    result = loader.load("greeter", {"name": "World"})
    assert result == "Hello, World!"


def test_load_strips_frontmatter(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    result = loader.load("reviewer", {"language": "Python"})
    # frontmatter block must not appear in output
    assert "name: reviewer" not in result
    assert "description: code reviewer" not in result
    # body content must be rendered
    assert "You are an expert Python reviewer." in result


def test_load_no_frontmatter_returns_full_content(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    result = loader.load("greeter", {"name": "Alice"})
    assert result == "Hello, Alice!"


def test_load_default_filter_works_without_providing_var(skills_dir: Path) -> None:
    """A template using | default(...) must succeed even when the var is absent."""
    loader = SkillLoader(skills_dir)
    # 'reviewer' uses {{ project | default('general project') }}
    # We don't provide 'project' — it must use the default value
    result = loader.load("reviewer", {"language": "Go"})
    assert "general project" in result


def test_load_none_variables_treated_as_empty(tmp_path: Path) -> None:
    """Passing variables=None must behave identically to variables={}."""
    _make_skill(tmp_path, "safe", body="Hello {{ who | default('World') }}!")
    loader = SkillLoader(tmp_path)
    result = loader.load("safe", None)
    assert result == "Hello World!"


# ---------------------------------------------------------------------------
# SkillLoader.load — error cases
# ---------------------------------------------------------------------------

def test_load_missing_skill_raises_not_found(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    with pytest.raises(SkillNotFoundError) as exc_info:
        loader.load("does-not-exist")
    assert exc_info.value.skill_name == "does-not-exist"
    assert exc_info.value.skills_dir == skills_dir


def test_load_undefined_variable_raises_render_error(skills_dir: Path) -> None:
    """Jinja2 UndefinedError (from StrictUndefined) must be wrapped in SkillRenderError."""
    loader = SkillLoader(skills_dir)
    # 'greeter' uses {{ name }} with no default — passing no vars triggers UndefinedError
    with pytest.raises(SkillRenderError) as exc_info:
        loader.load("greeter", {})
    assert exc_info.value.skill_name == "greeter"
    assert isinstance(exc_info.value.cause, Exception)


# ---------------------------------------------------------------------------
# SkillLoader.load_many
# ---------------------------------------------------------------------------

def test_load_many_concatenates_with_separator(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    result = loader.load_many(
        ["greeter", "reviewer"],
        {"name": "Bob", "language": "Python"},
    )
    parts = result.split("\n\n---\n\n")
    assert len(parts) == 2
    assert "Hello, Bob!" in parts[0]
    assert "You are an expert Python reviewer." in parts[1]


def test_load_many_single_skill_has_no_separator(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    result = loader.load_many(["greeter"], {"name": "X"})
    assert result == "Hello, X!"
    # single skill — no separator expected
    assert "\n\n---\n\n" not in result


def test_load_many_raises_if_any_skill_missing(skills_dir: Path) -> None:
    loader = SkillLoader(skills_dir)
    with pytest.raises(SkillNotFoundError):
        loader.load_many(["greeter", "missing"], {"name": "X"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'llm_bridge.skills'`

- [ ] **Step 3: Create `llm_bridge/skills.py`**

Create `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/skills.py`:

```python
"""
LLM Bridge — Local skill loader.

Loads SKILL.md files from a user-defined skills directory, strips YAML
frontmatter, and renders them with Jinja2 so they can be injected into
LLMBridge.chat() calls as system prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter
from jinja2 import Environment, StrictUndefined, UndefinedError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMBridgeError(Exception):
    """Base exception for all LLM Bridge errors."""


class SkillNotFoundError(LLMBridgeError):
    """Raised when a requested skill's SKILL.md file does not exist."""

    def __init__(self, skill_name: str, skills_dir: Path) -> None:
        super().__init__(
            f"Skill '{skill_name}' not found in '{skills_dir}'. "
            f"Expected: {skills_dir / skill_name / 'SKILL.md'}"
        )
        self.skill_name = skill_name
        self.skills_dir = skills_dir


class SkillRenderError(LLMBridgeError):
    """Raised when Jinja2 rendering of a skill template fails."""

    def __init__(self, skill_name: str, cause: Exception) -> None:
        super().__init__(f"Failed to render skill '{skill_name}': {cause}")
        self.skill_name = skill_name
        self.cause = cause


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------

class SkillLoader:
    """Loads and renders skill templates from a local skills directory.

    Each skill lives in a subdirectory named after the skill and must
    contain a ``SKILL.md`` file.  The file may have optional YAML
    frontmatter (stripped before rendering) and a Jinja2 template body.

    Parameters
    ----------
    skills_dir :
        Path to the directory containing skill subdirectories.

    Raises
    ------
    FileNotFoundError
        If ``skills_dir`` does not exist.
    """

    def __init__(self, skills_dir: Path) -> None:
        if not skills_dir.exists():
            raise FileNotFoundError(
                f"Skills directory does not exist: {skills_dir}"
            )
        self._skills_dir = skills_dir
        self._jinja_env = Environment(
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )

    def load(self, skill_name: str, variables: dict[str, Any] | None = None) -> str:
        """Load and Jinja2-render a single skill.

        Reads ``skills_dir/<skill_name>/SKILL.md`` (UTF-8), strips YAML
        frontmatter via ``python-frontmatter``, and renders the body
        template with the provided variables.

        Parameters
        ----------
        skill_name :
            Name of the skill subdirectory.
        variables :
            Template variables.  ``None`` is treated as ``{}``.
            Variables referenced without ``| default()`` raise
            ``SkillRenderError`` (via Jinja2 ``StrictUndefined``).

        Returns
        -------
        str
            Rendered skill body (frontmatter stripped).

        Raises
        ------
        SkillNotFoundError
            If ``skills_dir/<skill_name>/SKILL.md`` does not exist.
        SkillRenderError
            If Jinja2 rendering fails (e.g. undefined variable).
        """
        skill_path = self._skills_dir / skill_name / "SKILL.md"
        if not skill_path.exists():
            raise SkillNotFoundError(skill_name, self._skills_dir)

        raw_text = skill_path.read_text(encoding="utf-8")
        post = frontmatter.loads(raw_text)
        body = post.content

        vars_ = variables or {}
        try:
            template = self._jinja_env.from_string(body)
            return template.render(**vars_)
        except UndefinedError as exc:
            raise SkillRenderError(skill_name, exc) from exc

    def load_many(
        self,
        skill_names: list[str],
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Load and concatenate multiple skills.

        Skills are rendered in the given order and joined with
        ``'\\n\\n---\\n\\n'``.

        Parameters
        ----------
        skill_names :
            Ordered list of skill names to load.
        variables :
            Jinja2 template variables shared across all skills.

        Returns
        -------
        str
            Single concatenated string of all rendered skill bodies.
        """
        rendered = [self.load(name, variables) for name in skill_names]
        return "\n\n---\n\n".join(rendered)

    def list_skills(self) -> list[str]:
        """Return a sorted list of available skill names.

        Only subdirectories containing a ``SKILL.md`` file are included.
        Returns ``[]`` if no valid skills are found or if the directory
        was removed after ``__init__`` was called.

        Returns
        -------
        list[str]
            Sorted skill names.
        """
        if not self._skills_dir.exists():
            return []
        return sorted(
            d.name
            for d in self._skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py -v
```

Expected: All tests PASS. No errors.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/skills.py \
        plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py
git commit -m "feat(llm-bridge): add SkillLoader with Jinja2 and frontmatter support"
```

---

## Chunk 2: Wire `SkillLoader` into `LLMBridge`

**Files:**
- Modify: `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/models.py`
- Modify: `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/bridge.py`
- Modify: `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/__init__.py`
- Modify: `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge_config.example.yaml`
- Modify: `plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py` (append)

> All `pytest` commands in this chunk are run from:
> `plugins/llm-bridge/skills/llm-bridge-python/templates/`

---

### Task 2: Add `skills_dir` to `BridgeConfig`

- [ ] **Step 1: Add the field to `models.py`**

Open `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/models.py`.

In the `BridgeConfig` class, after the `log_usage` field, add:

```python
    skills_dir: Optional[str] = Field(
        default=None,
        description=(
            "Path to the skills directory. Each subdirectory must contain SKILL.md. "
            "Relative paths are resolved from the config YAML file's parent directory. "
            "Defaults to 'skills/' next to the config file if that directory exists."
        ),
    )
```

- [ ] **Step 2: Verify field is visible**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -c "from llm_bridge.models import BridgeConfig; b = BridgeConfig(); print('skills_dir default:', repr(b.skills_dir))"
```

Expected: `skills_dir default: None`

- [ ] **Step 3: Commit**

```bash
git add plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/models.py
git commit -m "feat(llm-bridge): add skills_dir field to BridgeConfig"
```

---

### Task 3: Write bridge integration tests (TDD — before touching `bridge.py`)

- [ ] **Step 0: Install async test dependency first**

```bash
pip install pytest-asyncio
```

This must be done before running any async bridge tests.

- [ ] **Step 1: Append bridge integration tests to `test_skills.py`**

Append to `plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py`:

```python
# ===========================================================================
# LLMBridge integration tests
# ===========================================================================

import pytest
from unittest.mock import patch

from llm_bridge import LLMBridge, ChatParameters
from llm_bridge.models import BridgeConfig, ModelConfig, UnifiedResponse, UsageInfo
from llm_bridge.skills import LLMBridgeError


def _make_bridge(
    skills_dir: Path | None = None,
    config_path: Path | None = None,
) -> LLMBridge:
    """Create a minimal LLMBridge for testing without real API keys."""
    config = BridgeConfig(
        models={
            "test-model": ModelConfig(
                provider="openai",
                model="gpt-4o",
                api_keys=["sk-fake"],
            )
        },
        skills_dir=str(skills_dir) if skills_dir else None,
    )
    return LLMBridge(config, config_path=config_path)


def _fake_response() -> UnifiedResponse:
    return UnifiedResponse(
        provider="openai",
        model="gpt-4o",
        content="ok",
        usage=UsageInfo(),
    )


# ---------------------------------------------------------------------------
# SkillLoader instantiation in LLMBridge.__init__
# ---------------------------------------------------------------------------

def test_bridge_skill_loader_is_none_when_no_skills_dir_configured() -> None:
    # No skills_dir in config, no config_path → no default dir to check
    bridge = _make_bridge()
    assert bridge._skill_loader is None


def test_bridge_skill_loader_created_when_skills_dir_exists(skills_dir: Path) -> None:
    bridge = _make_bridge(skills_dir=skills_dir)
    assert bridge._skill_loader is not None


def test_bridge_skill_loader_is_none_when_configured_dir_missing(tmp_path: Path) -> None:
    # skills_dir is set but the directory does not exist → silent None, no error at startup
    nonexistent = tmp_path / "no-skills"
    config = BridgeConfig(
        models={"m": ModelConfig(provider="openai", model="gpt-4o", api_keys=["k"])},
        skills_dir=str(nonexistent),
    )
    bridge = LLMBridge(config, config_path=tmp_path / "config.yaml")
    assert bridge._skill_loader is None


def test_bridge_resolves_relative_skills_dir_from_config_path(tmp_path: Path) -> None:
    """A relative skills_dir must be resolved relative to config_path, not CWD."""
    # Put the skills dir at tmp_path/my_skills/
    skills_subdir = tmp_path / "my_skills"
    skills_subdir.mkdir()
    _make_skill(skills_subdir, "ping", body="pong")

    # The config sits at tmp_path/config.yaml; skills_dir is relative "./my_skills"
    config = BridgeConfig(
        models={"m": ModelConfig(provider="openai", model="gpt-4o", api_keys=["k"])},
        skills_dir="./my_skills",
    )
    bridge = LLMBridge(config, config_path=tmp_path / "config.yaml")
    assert bridge._skill_loader is not None
    assert "ping" in bridge._skill_loader.list_skills()


# ---------------------------------------------------------------------------
# chat() — skill injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bridge_chat_raises_when_skills_but_no_loader() -> None:
    bridge = _make_bridge()
    with pytest.raises(LLMBridgeError, match="no skills directory"):
        await bridge.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            skills=["greeter"],
        )


@pytest.mark.asyncio
async def test_bridge_chat_injects_skill_as_system_prompt(skills_dir: Path) -> None:
    bridge = _make_bridge(skills_dir=skills_dir)
    captured: dict = {}

    async def fake_call_model(model_cfg, messages, response_model, params, **kw):
        captured["system_prompt"] = params.system_prompt if params else None
        return _fake_response()

    with patch.object(bridge, "_call_model", side_effect=fake_call_model):
        await bridge.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            skills=["greeter"],
            skill_vars={"name": "World"},
        )

    assert captured["system_prompt"] == "Hello, World!"


@pytest.mark.asyncio
async def test_bridge_chat_skill_is_prepended_before_existing_system_prompt(
    skills_dir: Path,
) -> None:
    """When params.system_prompt exists, skill content comes FIRST,
    separated by '\\n\\n---\\n\\n'."""
    bridge = _make_bridge(skills_dir=skills_dir)
    captured: dict = {}

    async def fake_call_model(model_cfg, messages, response_model, params, **kw):
        captured["system_prompt"] = params.system_prompt if params else None
        return _fake_response()

    with patch.object(bridge, "_call_model", side_effect=fake_call_model):
        await bridge.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            skills=["greeter"],
            skill_vars={"name": "Alice"},
            params=ChatParameters(system_prompt="Be concise."),
        )

    sp = captured["system_prompt"]
    # Exact format: <skill>\n\n---\n\n<existing>
    assert sp == "Hello, Alice!\n\n---\n\nBe concise."


@pytest.mark.asyncio
async def test_bridge_chat_does_not_mutate_caller_params(skills_dir: Path) -> None:
    """bridge.chat() must NOT modify the ChatParameters object passed by the caller."""
    bridge = _make_bridge(skills_dir=skills_dir)
    original_params = ChatParameters(system_prompt="original")

    async def fake_call_model(model_cfg, messages, response_model, params, **kw):
        return _fake_response()

    with patch.object(bridge, "_call_model", side_effect=fake_call_model):
        await bridge.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            skills=["greeter"],
            skill_vars={"name": "Bob"},
            params=original_params,
        )

    assert original_params.system_prompt == "original"


@pytest.mark.asyncio
async def test_bridge_chat_no_skills_does_not_touch_params() -> None:
    """When no skills are passed, params must reach the provider unchanged."""
    bridge = _make_bridge()
    captured: dict = {}

    async def fake_call_model(model_cfg, messages, response_model, params, **kw):
        captured["system_prompt"] = params.system_prompt if params else None
        return _fake_response()

    with patch.object(bridge, "_call_model", side_effect=fake_call_model):
        await bridge.chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            params=ChatParameters(system_prompt="custom"),
        )

    assert captured["system_prompt"] == "custom"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py -k "bridge" -v 2>&1 | tail -15
```

Expected: FAIL — `LLMBridge.__init__()` does not accept `config_path` kwarg yet.

---

### Task 4: Implement `bridge.py` changes

- [ ] **Step 1: Add import to `bridge.py`**

Open `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/bridge.py`.

Add after the existing imports block:

```python
from .skills import LLMBridgeError, SkillLoader
```

- [ ] **Step 1b: Update `__init__` signature to accept `config_path`**

Change the `__init__` signature from:

```python
    def __init__(self, config: BridgeConfig) -> None:
```

to:

```python
    def __init__(self, config: BridgeConfig, *, config_path: Optional[Path] = None) -> None:
```

- [ ] **Step 2: Add skill loader setup at the end of `__init__`**

At the **end** of the `__init__` body (after the existing provider/key-rotator loop), add:

```python
        # ------------------------------------------------------------------
        # Skill loader — resolve skills_dir and create SkillLoader if present
        # ------------------------------------------------------------------
        if config.skills_dir:
            raw = Path(config.skills_dir)
            resolved_skills_dir: Optional[Path] = (
                raw if raw.is_absolute()
                else (config_path.parent / raw if config_path else raw)
            )
        elif config_path is not None:
            # Default: look for 'skills/' next to the config file
            resolved_skills_dir = config_path.parent / "skills"
        else:
            # No config_path provided — cannot resolve a default skills dir
            resolved_skills_dir = None

        self._skill_loader: Optional[SkillLoader] = (
            SkillLoader(resolved_skills_dir)
            if resolved_skills_dir is not None and resolved_skills_dir.exists()
            else None
        )
```

Key: when both `config.skills_dir` is `None` **and** `config_path` is `None`, `_skill_loader` is set to `None` unconditionally (no CWD fallback).

- [ ] **Step 3: Update `from_config` to forward `config_path`**

Change `from_config` from:

```python
    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "LLMBridge":
        """..."""
        return cls(load_config(path))
```

to:

```python
    @classmethod
    def from_config(cls, path: Union[str, Path]) -> "LLMBridge":
        """..."""
        config_path = Path(path).resolve()
        return cls(load_config(config_path), config_path=config_path)
```

- [ ] **Step 4: Add `skills` and `skill_vars` to `chat()` AND inject (atomic change)**

Change the `chat()` signature from:

```python
    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        response_model: Optional[Type[BaseModel]] = None,
        params: Optional[ChatParameters] = None,
        **kwargs,
    ) -> UnifiedResponse:
```

to:

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

Then, at the very **start** of the `chat()` method body (before `model_cfg = self._resolve_model(model)`), add:

```python
        # ------------------------------------------------------------------
        # Skill injection — load and merge into system prompt
        # ------------------------------------------------------------------
        if skills:
            if self._skill_loader is None:
                raise LLMBridgeError(
                    "Cannot load skills: no skills directory is configured or the "
                    "directory does not exist. Set 'skills_dir' in your config YAML."
                )
            skill_content = self._skill_loader.load_many(skills, skill_vars)
            existing_system = params.system_prompt if params else None
            merged_system = (
                skill_content
                if not existing_system
                else f"{skill_content}\n\n---\n\n{existing_system}"
            )
            # Create a NEW ChatParameters — never mutate the caller's object.
            # This merged params is shared across primary + all fallback calls.
            params = (params or ChatParameters()).model_copy(
                update={"system_prompt": merged_system}
            )
```

Note: The signature change and injection block are done in **one atomic step** to prevent a window where `skills` is accepted but silently ignored.

- [ ] **Step 5: Run bridge tests to verify they pass**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py -k "bridge" -v
```

Expected: All bridge tests PASS.

- [ ] **Step 6: Run the full test suite**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/bridge.py \
        plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py
git commit -m "feat(llm-bridge): wire SkillLoader into LLMBridge.chat()"
```

---

### Task 5: Update `__init__.py` exports

- [ ] **Step 1: Write a failing import test**

Append to `test_skills.py`:

```python
# ---------------------------------------------------------------------------
# Package export tests
# ---------------------------------------------------------------------------

def test_package_exports_skill_classes() -> None:
    """All skill-related symbols must be importable from the top-level package."""
    import llm_bridge
    for name in ("SkillLoader", "LLMBridgeError", "SkillNotFoundError", "SkillRenderError"):
        assert hasattr(llm_bridge, name), f"Missing export: {name}"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py::test_package_exports_skill_classes -v
```

Expected: FAIL — `AttributeError` or `AssertionError` for the missing names.

- [ ] **Step 3: Replace `__init__.py` content**

Replace `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/__init__.py` with:

```python
"""
LLM Bridge — Lightweight multi-provider LLM gateway.

Quick start::

    from llm_bridge import LLMBridge, ChatParameters

    bridge = LLMBridge.from_config("llm_bridge_config.yaml")
    response = await bridge.chat(
        "openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        params=ChatParameters(temperature=0.5, max_tokens=1024),
    )

Skill loading::

    response = await bridge.chat(
        "openai/gpt-4o",
        messages=[{"role": "user", "content": "Review this code: ..."}],
        skills=["code-reviewer"],
        skill_vars={"language": "Python"},
    )
"""

from .bridge import LLMBridge
from .models import (
    BridgeConfig,
    ChatParameters,
    ModelConfig,
    RetryConfig,
    UnifiedResponse,
    UsageInfo,
)
from .skills import LLMBridgeError, SkillLoader, SkillNotFoundError, SkillRenderError

__all__ = [
    # Core gateway
    "LLMBridge",
    "ChatParameters",
    "UnifiedResponse",
    "UsageInfo",
    "BridgeConfig",
    "ModelConfig",
    "RetryConfig",
    # Skill loader
    "SkillLoader",
    "LLMBridgeError",
    "SkillNotFoundError",
    "SkillRenderError",
]
```

- [ ] **Step 4: Run the export test to verify it passes**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py::test_package_exports_skill_classes -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest test_skills.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge/__init__.py \
        plugins/llm-bridge/skills/llm-bridge-python/templates/test_skills.py
git commit -m "feat(llm-bridge): export SkillLoader and exceptions from package public API"
```

---

### Task 6: Update `llm_bridge_config.example.yaml`

- [ ] **Step 1: Add `skills_dir` comment block**

Open `plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge_config.example.yaml`.

After the `log_usage: true` line, add:

```yaml

# Path to skills directory (optional).
# Each subdirectory must contain a SKILL.md file.
# Relative paths are resolved from this config file's location.
# Defaults to 'skills/' next to this config file (if that directory exists).
# skills_dir: ./skills
```

- [ ] **Step 2: Commit**

```bash
git add plugins/llm-bridge/skills/llm-bridge-python/templates/llm_bridge_config.example.yaml
git commit -m "docs(llm-bridge): add skills_dir comment to example config"
```

---

## Chunk 3: Update `SKILL.md` + Write User Documentation

**Files:**
- Modify: `plugins/llm-bridge/skills/llm-bridge-python/SKILL.md`
- Create: `plugins/llm-bridge/docs/llm-bridge-guide.md`

---

### Task 7: Update the `llm-bridge-python` SKILL.md

- [ ] **Step 1: Add `skills.py` to the copy list in Step 3**

Open `plugins/llm-bridge/skills/llm-bridge-python/SKILL.md`.

In **Step 3: Copy Template Files**, add `skills.py` to the file listing:

```
$TARGET_DIR/
├── __init__.py
├── models.py
├── config.py
├── router.py
├── bridge.py
├── skills.py          # ← add this line
└── providers/
    ├── __init__.py
    ├── base.py
    ...
```

- [ ] **Step 2: Add new dependencies in Step 5**

In **Step 5: Update Dependencies**, add to the **unconditional block** (always required, regardless of provider selection):

For `requirements.txt`:
```
jinja2>=3.0
python-frontmatter>=1.0
```

For `pyproject.toml` `[project.dependencies]`, add equivalently.

- [ ] **Step 3: Add skill example to Step 6 usage guide**

In **Step 6: Display Usage Guide**, after the existing `asyncio.run(main())` example, add:

```python
    # --- Using a local skill ---

    # 1. Create: skills/assistant/SKILL.md
    #    ---
    #    name: assistant
    #    ---
    #    You are a {{ tone | default("helpful") }} assistant.

    response = await bridge.chat(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Summarize quantum computing"}],
        skills=["assistant"],
        skill_vars={"tone": "concise"},
    )
    print(response.content)
```

- [ ] **Step 4: Commit**

```bash
git add plugins/llm-bridge/skills/llm-bridge-python/SKILL.md
git commit -m "docs(llm-bridge): update SKILL.md — add skills.py to template, new deps, skill example"
```

---

### Task 8: Write `plugins/llm-bridge/docs/llm-bridge-guide.md`

- [ ] **Step 1: Create the docs directory**

```bash
mkdir -p plugins/llm-bridge/docs
```

- [ ] **Step 2: Create the guide**

Create `plugins/llm-bridge/docs/llm-bridge-guide.md`:

````markdown
# LLM Bridge — User Guide

A production-ready Python module that unifies OpenAI, Anthropic, and Google Gemini
behind a single `LLMBridge.chat()` interface, with retry, fallback, key rotation,
and local skill loading.

---

## Part A — LLM Gateway API

### Quick Start

**1. Install dependencies**

```bash
pip install pydantic>=2.0 pyyaml>=6.0 python-dotenv>=1.0 \
            jinja2>=3.0 python-frontmatter>=1.0
# Provider SDKs (install only what you need):
pip install openai>=1.0
pip install anthropic>=0.40
pip install google-genai>=1.0
```

**2. Create `llm_bridge_config.yaml`** in your project root:

```yaml
retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 60.0

log_usage: true
default_model: my-claude

models:
  my-claude:
    provider: anthropic
    model: claude-sonnet-4-6
    api_keys:
      - ${ANTHROPIC_API_KEY}
    max_tokens: 4096
    temperature: 0.7
```

**3. Add API keys to `.env`:**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

**4. Make your first call:**

```python
import asyncio
from llm_bridge import LLMBridge

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")
    response = await bridge.chat(
        model="my-claude",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.content)
    print(f"Tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

---

### `bridge.chat()` — Full Parameter Reference

```python
response = await bridge.chat(
    model="my-claude",              # config alias or "provider/model"
    messages=[...],                 # OpenAI-style message list
    response_model=MyPydanticModel, # optional: get structured output
    params=ChatParameters(...),     # optional: generation settings
    skills=["skill-name"],          # optional: inject local skills
    skill_vars={"key": "value"},    # optional: Jinja2 variables for skills
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Config alias (e.g. `"my-claude"`) or inline `"provider/model"` (e.g. `"openai/gpt-4o"`) |
| `messages` | `list[dict]` | OpenAI-style messages: `[{"role": "user", "content": "..."}]` |
| `response_model` | `Type[BaseModel]` | Pydantic model for structured JSON output. Result available in `response.parsed`. |
| `params` | `ChatParameters` | Generation settings (temperature, max_tokens, etc.) |
| `skills` | `list[str]` | Skill names to load and inject as system prompt |
| `skill_vars` | `dict` | Jinja2 template variables for rendering skill templates |

**Inline model syntax** — skip YAML config for quick tests:

```python
response = await bridge.chat(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from llm_bridge import LLMBridge, ChatParameters

class CodeReview(BaseModel):
    summary: str
    issues: list[str]
    severity: str  # "low" | "medium" | "high"

bridge = LLMBridge.from_config("llm_bridge_config.yaml")

response = await bridge.chat(
    model="my-claude",
    messages=[{"role": "user", "content": f"Review this code:\n{code}"}],
    response_model=CodeReview,
    params=ChatParameters(temperature=0.2),
)

review: CodeReview = response.parsed
print(f"Severity: {review.severity}")
print(f"Issues: {review.issues}")
```

---

### `ChatParameters` — All Settings

```python
params = ChatParameters(
    max_tokens=2048,         # Max output tokens
    temperature=0.7,         # Randomness (0.0–2.0; Anthropic: 0.0–1.0)
    top_p=0.9,               # Nucleus sampling threshold
    top_k=40,                # Top-K sampling (Anthropic + Google only)
    frequency_penalty=0.3,   # Repetition penalty (OpenAI + Google only)
    presence_penalty=0.3,    # New-topic penalty (OpenAI + Google only)
    stop_sequences=["END"],  # Custom stop strings
    seed=42,                 # Reproducibility (OpenAI + Google only)
    system_prompt="...",     # System instruction
)
```

**Provider support matrix:**

| Parameter | OpenAI | Anthropic | Google |
|-----------|--------|-----------|--------|
| `max_tokens` | ✅ | ✅ (required) | ✅ |
| `temperature` | ✅ | ✅ | ✅ |
| `top_p` | ✅ | ✅ | ✅ |
| `top_k` | ❌ | ✅ | ✅ |
| `frequency_penalty` | ✅ | ❌ | ✅ |
| `presence_penalty` | ✅ | ❌ | ✅ |
| `stop_sequences` | ✅ | ✅ | ✅ |
| `seed` | ✅ | ❌ | ✅ |
| `system_prompt` | ✅ | ✅ | ✅ |

---

### Config File Reference

```yaml
# llm_bridge_config.yaml

retry:
  max_retries: 3          # Attempts before giving up
  base_delay: 1.0         # First retry delay (seconds)
  max_delay: 60.0         # Maximum delay cap
  exponential_base: 2.0   # Backoff multiplier

log_usage: true           # Log token usage after each call

default_model: my-model   # Model alias used when none specified

# Path to local skills directory (optional).
# Relative paths resolved from this config file's location.
# skills_dir: ./skills

models:
  my-model:
    provider: openai        # openai | anthropic | google | vertexai
    model: gpt-4o
    api_keys:
      - ${OPENAI_API_KEY}   # Multiple keys = round-robin rotation
      - ${OPENAI_API_KEY_2}
    max_tokens: 4096
    temperature: 0.7
    fallback_models:
      - my-backup           # Try this alias if my-model fails
    extra: {}               # Provider-specific extra kwargs

  my-backup:
    provider: anthropic
    model: claude-sonnet-4-6
    api_keys:
      - ${ANTHROPIC_API_KEY}
    max_tokens: 4096
```

---

## Part B — Skills

### What Is a Skill?

A skill is a reusable prompt template stored as a `SKILL.md` file. When you pass
`skills=["my-skill"]` to `bridge.chat()`, the rendered skill content is injected as
the system prompt — prepended before any `system_prompt` you set in `ChatParameters`.

Skills use [Jinja2](https://jinja.palletsprojects.com/) templating, so you can pass
dynamic values via `skill_vars`.

---

### Directory Layout

```
skills/                         # Root skills directory
└── code-reviewer/
    ├── SKILL.md                # Required
    ├── references/             # Optional — extra context docs (not auto-loaded)
    └── templates/              # Optional — output templates
```

By default LLM Bridge looks for a `skills/` directory next to your
`llm_bridge_config.yaml`. Configure a custom path via `skills_dir` in the YAML.

---

### Writing a SKILL.md

```markdown
---
name: code-reviewer
description: Reviews code for quality and security
---

You are an expert {{ language | default("Python") }} code reviewer.

Review the provided code for:
- Security vulnerabilities
- Performance bottlenecks
- Code clarity and maintainability

Project context: {{ project_context | default("a general software project") }}

Respond with:
1. **Summary** — one-sentence overview
2. **Issues** — bulleted list, each with severity (low/medium/high)
3. **Recommendation** — what to fix first
```

**Rules:**
- The `---` frontmatter block is optional and is always stripped before injection
- Use `{{ variable }}` for Jinja2 variables
- Use `{{ variable | default("fallback") }}` for optional variables
- Variables without a `default` filter **must** be provided in `skill_vars`,
  otherwise a `SkillRenderError` is raised at call time

---

### Loading a Skill

```python
bridge = LLMBridge.from_config("llm_bridge_config.yaml")

response = await bridge.chat(
    model="my-claude",
    messages=[{"role": "user", "content": f"Review this:\n{code}"}],
    skills=["code-reviewer"],
    skill_vars={
        "language": "Python",
        "project_context": "FastAPI REST API",
    },
)
```

---

### Loading Multiple Skills

Skills are rendered in the order given and injected into the system prompt
separated by `---`:

```python
response = await bridge.chat(
    model="my-claude",
    messages=[{"role": "user", "content": "..."}],
    skills=["base-persona", "code-reviewer", "output-format"],
    skill_vars={"language": "TypeScript"},
)
```

The injected system prompt looks like:

```
<base-persona content>

---

<code-reviewer content>

---

<output-format content>
```

If you also set `params.system_prompt`, it is appended **after** the skills:

```
<skill content>

---

<your system_prompt>
```

---

### Configuring `skills_dir`

```yaml
# llm_bridge_config.yaml
skills_dir: ./prompts/skills    # relative to this config file
# or absolute:
# skills_dir: /home/user/project/skills
```

---

### Full End-to-End Example

**Project structure:**

```
my-project/
├── llm_bridge_config.yaml
├── .env
├── skills/
│   └── summarizer/
│       └── SKILL.md
└── main.py
```

**`skills/summarizer/SKILL.md`:**

```markdown
---
name: summarizer
description: Summarizes content concisely
---

You are a {{ tone | default("professional") }} summarizer.
Always respond with:
1. A one-sentence TL;DR
2. Three key points as bullet items
3. One recommended action
```

**`llm_bridge_config.yaml`:**

```yaml
models:
  flash:
    provider: google
    model: gemini-2.0-flash
    api_keys:
      - ${GOOGLE_API_KEY}
    max_tokens: 1024
```

**`main.py`:**

```python
import asyncio
from llm_bridge import LLMBridge

async def main():
    bridge = LLMBridge.from_config("llm_bridge_config.yaml")

    response = await bridge.chat(
        model="flash",
        messages=[{"role": "user", "content": "Summarize: " + long_article}],
        skills=["summarizer"],
        skill_vars={"tone": "friendly"},
    )
    print(response.content)

asyncio.run(main())
```

---

### Error Handling

```python
from llm_bridge import LLMBridgeError, SkillNotFoundError, SkillRenderError

try:
    response = await bridge.chat(
        model="my-model",
        messages=[...],
        skills=["my-skill"],
        skill_vars={"lang": "Python"},
    )
except SkillNotFoundError as e:
    print(f"Skill '{e.skill_name}' not found in {e.skills_dir}")
except SkillRenderError as e:
    print(f"Template error in skill '{e.skill_name}': {e.cause}")
except LLMBridgeError as e:
    print(f"LLM Bridge error: {e}")
except RuntimeError as e:
    print(f"All retries/fallbacks exhausted: {e}")
```
````

- [ ] **Step 3: Verify the file was created**

```bash
wc -l plugins/llm-bridge/docs/llm-bridge-guide.md
```

Expected: 200+ lines.

- [ ] **Step 4: Commit**

```bash
git add plugins/llm-bridge/docs/llm-bridge-guide.md
git commit -m "docs(llm-bridge): add comprehensive user guide (API reference + skills)"
```

---

### Task 9: Final verification

- [ ] **Step 1: Install dependencies**

```bash
pip install jinja2>=3.0 python-frontmatter>=1.0 pytest-asyncio
```

- [ ] **Step 2: Run the complete test suite**

```bash
cd plugins/llm-bridge/skills/llm-bridge-python/templates
python -m pytest test_skills.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Verify clean package import**

```bash
python -c "
from llm_bridge import (
    LLMBridge, ChatParameters, UnifiedResponse, UsageInfo,
    BridgeConfig, ModelConfig, RetryConfig,
    SkillLoader, LLMBridgeError, SkillNotFoundError, SkillRenderError,
)
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 4: Commit any remaining uncommitted files**

```bash
git status
# If clean, no commit needed. If there are loose files:
git add <any remaining files>
git commit -m "chore(llm-bridge): finalize skills loader feature"
```
