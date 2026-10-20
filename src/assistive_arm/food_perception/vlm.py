"""Optional VLM for questions a detector can't answer.

Examples: "which of these is the soup?", "is the spoon loaded?", "did the
grape fall off?", "is this strawberry ripe or bruised?", and describing the
plate for "what's on the plate?".

TODO(food_perception):
  * Pick a small open VLM that fits next to the detector on the AGX Orin
    (Qwen2.5-VL-3B / SmolVLM / Florence-2 class; quantised). Measure latency per
    crop; it only runs at decision points, never inside the servo loop.
  * Prompt with the crop + a yes/no or short-answer question; parse strictly;
    on anything unparseable return None (the caller then asks the user).
  * Wrap as a Viam generic service (do_command {"question", "image"}).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VisualQA(Protocol):
    def ask(self, rgb: np.ndarray, question: str) -> str | None: ...


class NoVLM:
    """Default: no VLM available. Callers must handle None by asking the user."""

    def ask(self, rgb: np.ndarray, question: str) -> str | None:
        return None


def yes_no(vqa: VisualQA, rgb: np.ndarray, question: str) -> bool | None:
    a = vqa.ask(rgb, question + " Answer yes or no.")
    if a is None:
        return None
    a = a.strip().lower()
    return True if a.startswith("yes") else False if a.startswith("no") else None
