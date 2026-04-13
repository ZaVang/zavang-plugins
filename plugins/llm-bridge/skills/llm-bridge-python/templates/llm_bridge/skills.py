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
from jinja2 import Environment, StrictUndefined, TemplateError


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
        if not skills_dir.is_dir():
            raise NotADirectoryError(
                f"Skills directory is not a directory: {skills_dir}"
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
        except TemplateError as exc:
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
        If ``skill_names`` is empty, returns an empty string ``""``.
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
