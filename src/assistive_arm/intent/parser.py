"""Rule-based speech-to-intent parser (the offline fallback for an LLM).

Input is the transcript from the speech recogniser (Whisper / Vosk). The
parser is deliberately conservative: anything it is unsure about becomes
``Task.UNKNOWN`` with a clarifying question rather than a guess.
"""

from __future__ import annotations

import re

from .schema import Intent, IntentError, Task

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
_NAMED_BUTTONS = {
    "lobby": "L", "ground": "G", "ground floor": "G", "mezzanine": "M", "parking": "P",
    "up": "up", "down": "down", "door open": "open", "open door": "open", "door close": "close",
    "close door": "close", "alarm": "alarm",
}

_STOP_RE = re.compile(r"\b(stop|halt|freeze|cancel|abort|emergency stop|e-stop|no no)\b")
_RELEASE_RE = re.compile(r"\b(let go|release|drop it|put it down|give it to me|hand it to me)\b")
_GRAB_VERBS = r"(?:grab|grasp|pick up|get|fetch|take|hold|bring me|hand me)"
_DOOR_RE = re.compile(r"\b(door|doors)\b")
_PRESS_RE = re.compile(r"\b(press|push|hit|tap|call|select)\b")
_FILLER = re.compile(r"\b(please|could you|can you|would you|for me|thanks|thank you|the|a|an|my)\b")


def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s'-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _button_label(t: str) -> str | None:
    m = re.search(r"\bb(?:asement)?\s*([1-9])\b", t)
    if m and re.search(r"\b(basement|b[1-9])\b", t):
        return f"B{m.group(1)}"
    m = re.search(r"\bbutton ([lgmp])\b", t)  # "button L" (lobby), "button G"
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m and m.group(1) != "0":
        return str(int(m.group(1)))
    for w, n in {**_ORDINAL_WORDS, **_NUMBER_WORDS}.items():
        if re.search(rf"\b{w}\b", t):
            return str(n)
    for phrase in sorted(_NAMED_BUTTONS, key=len, reverse=True):
        if re.search(rf"\b{phrase}\b", t):
            return _NAMED_BUTTONS[phrase]
    return None


def _object_phrase(t: str) -> str | None:
    m = re.search(rf"\b{_GRAB_VERBS}\s+(.+?)(?:\s+(?:from|off|on|in|and)\b.*)?$", t)
    if not m:
        return None
    obj = _FILLER.sub(" ", m.group(1))
    obj = re.sub(r"\s+", " ", obj).strip()
    return obj or None


def parse(text: str) -> Intent:
    raw = text
    t = _norm(text)
    if not t:
        return Intent(Task.UNKNOWN, raw=raw, params={"question": "I didn't catch that."})

    # Safety words win over everything else in the utterance.
    if _STOP_RE.search(t):
        return Intent(Task.STOP, raw=raw)
    if _RELEASE_RE.search(t):
        return Intent(Task.RELEASE, raw=raw)

    is_elevator = bool(re.search(r"\b(elevator|lift|floor|button)\b", t))
    if _DOOR_RE.search(t) and not is_elevator:
        params = {"direction": "auto", "handle_type": "auto"}
        if re.search(r"\b(pull|toward me|towards me)\b", t):
            params["direction"] = "pull"
        elif re.search(r"\b(push|away)\b", t):
            params["direction"] = "push"
        if re.search(r"\b(lever|handle)\b", t):
            params["handle_type"] = "lever" if "lever" in t else "auto"
        if re.search(r"\bknob\b", t):
            params["handle_type"] = "knob"
        if re.search(r"\b(open|opening)\b", t) or re.search(r"\band open it\b", t):
            return Intent(Task.OPEN_DOOR, target="door handle", params=params, raw=raw)
        return Intent(Task.UNKNOWN, raw=raw, params={"question": "Do you want me to open the door?"})

    if is_elevator or _PRESS_RE.search(t):
        label = _button_label(t)
        if label is None:
            return Intent(Task.UNKNOWN, raw=raw, params={"question": "Which button should I press?"})
        try:
            return Intent(Task.PRESS_BUTTON, target=label, context="elevator" if is_elevator else None, raw=raw)
        except IntentError as e:
            return Intent(Task.UNKNOWN, raw=raw, params={"question": f"{e}. Which button?"})

    obj = _object_phrase(t)
    if obj:
        part = None
        if re.search(r"\b(bag|basket|bucket|tote)\b", obj):
            part = "handle"
        return Intent(Task.GRASP, target=obj, part=part, raw=raw)

    return Intent(Task.UNKNOWN, raw=raw, params={"question": "Sorry, what should I do?"})
