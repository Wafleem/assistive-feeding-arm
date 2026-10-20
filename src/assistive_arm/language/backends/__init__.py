"""LLM backends, all behind one interface. Cloud ones are optional extras (lazy imports)."""

from __future__ import annotations

import os

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
    names = [n.strip() for n in os.environ.get("ASSIST_LLM_BACKENDS", DEFAULT_ORDER).split(",") if n.strip()]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown LLM backend(s) {unknown}; choose from {sorted(REGISTRY)}")
    return [REGISTRY[n]() for n in names]


__all__ = ["AnthropicBackend", "BackendError", "FakeBackend", "LLMBackend", "LocalLLMBackend", "OpenAIBackend",
           "REGISTRY", "backends_from_env"]
