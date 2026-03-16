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
    # Construction must not raise — the returned loader is usable
    loader = SkillLoader(tmp_path)
    assert loader.list_skills() == []  # empty dir → no skills, not an error


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
    from jinja2 import UndefinedError
    assert isinstance(exc_info.value.cause, UndefinedError)


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


# ===========================================================================
# LLMBridge integration tests
# ===========================================================================

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
    # No skills_dir in config, no config_path → _skill_loader must be None
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
    skills_subdir = tmp_path / "my_skills"
    skills_subdir.mkdir()
    _make_skill(skills_subdir, "ping", body="pong")

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
    """Skill content comes FIRST, separated by '\\n\\n---\\n\\n'."""
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
    assert sp == "Hello, Alice!\n\n---\n\nBe concise."


@pytest.mark.asyncio
async def test_bridge_chat_does_not_mutate_caller_params(skills_dir: Path) -> None:
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
