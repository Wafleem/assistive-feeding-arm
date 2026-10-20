"""LLM backends, all behind one interface. Cloud ones are optional extras (lazy imports)."""

from __future__ import annotations

from .anthropic_backend import AnthropicBackend
from .base import BackendError, LLMBackend
from .fake import FakeBackend
from .local import LocalLLMBackend
from .openai_backend import OpenAIBackend

REGISTRY = {"local": LocalLLMBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend}
DEFAULT_ORDER = "local"  # privacy + offline by default; cloud only if configured


def backends_from_env() -> list[LLMBackend]:
    """ASSIST_LLM_BACKENDS="local,anthropic,openai" -> instances in that fallback order.
    The rule parser is always the final fallback and is not listed here."""
    raise NotImplementedError("TODO(language): backends_from_env (parse, reject unknown names with ValueError)")


__all__ = ["AnthropicBackend", "BackendError", "FakeBackend", "LLMBackend", "LocalLLMBackend", "OpenAIBackend",
           "REGISTRY", "backends_from_env"]
