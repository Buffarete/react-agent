"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from react_agent import prompts
from pydantic import BaseModel
from pydantic.config import ConfigDict


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-5",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # Accept platform-injected context keys (ignored by our code but prevents coercion errors)
    host: str | None = field(
        default=None,
        metadata={
            "description": "Platform-injected host value; not used by the agent."
        },
    )

    # Accept common platform headers if coerced into context
    accept: str | None = field(default=None)
    user_agent: str | None = field(default=None)
    origin: str | None = field(default=None)


class AnyContext(BaseModel):
    """Permissive context model to accept platform-injected keys.

    This avoids runtime coercion errors when the platform provides extra fields
    like headers or origin. Our code reads config via Configuration.from_context().
    """

    model_config = ConfigDict(extra="allow")


@dataclass(kw_only=True)
class _ConfigAccessor:
    @classmethod
    def from_context(cls) -> Configuration:  # type: ignore[name-defined]
        """Create a Configuration instance from a RunnableConfig context.

        Pulls the current LangGraph config and maps the "configurable" keys onto
        our dataclass fields, ignoring any platform-injected extras.
        """
        try:
            from langgraph.config import get_config as _get_config  # type: ignore
        except Exception:
            _get_config = None  # type: ignore

        try:
            from langchain_core.runnables import ensure_config as _ensure_config  # type: ignore
        except Exception:
            def _ensure_config(value):  # type: ignore
                return value or {}

        try:
            raw = _get_config() if _get_config else None
        except RuntimeError:
            raw = None

        cfg = _ensure_config(raw)
        configurable = (cfg.get("configurable") if isinstance(cfg, dict) else None) or {}
        allowed = {f.name for f in fields(Configuration) if f.init}
        return Configuration(**{k: v for k, v in configurable.items() if k in allowed})


"""Attach method to Configuration to preserve public API (single implementation)."""
setattr(Configuration, "from_context", _ConfigAccessor.from_context)  # type: ignore[attr-defined]
