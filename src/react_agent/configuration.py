"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from react_agent import prompts


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

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        # Import lazily for broader runtime compatibility
        try:
            from langgraph.config import get_config as _get_config  # type: ignore
        except Exception:  # pragma: no cover - compatibility for older runtimes
            _get_config = None  # type: ignore

        try:
            from langchain_core.runnables import ensure_config as _ensure_config  # type: ignore
        except Exception:  # pragma: no cover
            def _ensure_config(value):  # type: ignore
                return value or {}

        try:
            raw_config = _get_config() if _get_config else None
        except RuntimeError:
            raw_config = None

        config = _ensure_config(raw_config)
        configurable = (config.get("configurable") if isinstance(config, dict) else None) or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
