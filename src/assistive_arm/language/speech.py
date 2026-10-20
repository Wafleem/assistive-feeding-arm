"""Speech in and out.

TODO(language):
* SpeechToText: faster-whisper (small.en / distil) on the AGX, push-to-talk or VAD;
  report end-of-speech timestamps so "stop" latency can be measured.
* TextToSpeech: browser speechSynthesis in the dashboard (zero install), or Piper on a Jetson.
"""

from __future__ import annotations

from typing import Protocol


class SpeechToText(Protocol):
    def listen(self, timeout: float) -> str | None: ...


class TextToSpeech(Protocol):
    def say(self, text: str) -> None: ...
