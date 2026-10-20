"""Spec for section 5 (language). Remove @todo as you implement."""

import pytest

from assistive_arm.language import Action, parse
from conftest import todo

EXAMPLES = [
    ("feed me the strawberries", Action.FEED, "strawberry"),
    ("a grape please", Action.FEED, "grape"),
    ("next bite", Action.FEED, None),
    ("a sip of water", Action.DRINK, None),
    ("slower", Action.ADJUST, None),
    ("smaller bite", Action.ADJUST, None),
    ("wait", Action.PAUSE, None),
    ("stop", Action.STOP, None),
    ("I'm done", Action.DONE, None),
    ("what's on the plate?", Action.QUERY, None),
    ("yes", Action.CONFIRM, None),
]


@todo
@pytest.mark.parametrize("text, action, item", EXAMPLES)
def test_rule_parser_examples(text, action, item):
    i = parse(text)
    assert i.action is action and i.item == item


@todo
def test_safety_words_win():
    assert parse("feed me a grape, no wait, stop").action is Action.STOP


# ---- provider-agnostic router (FakeBackend; no network, no keys) --------------------------

import json  # noqa: E402

from assistive_arm.language import (  # noqa: E402
    DialogueState, FakeBackend, IntentRouter, LocalLLMBackend, backends_from_env,
)

FEED_JSON = json.dumps({"action": "feed", "item": "strawberry"})


@todo
def test_default_backend_order_is_local_only(monkeypatch):
    monkeypatch.delenv("ASSIST_LLM_BACKENDS", raising=False)
    bs = backends_from_env()
    assert [b.name for b in bs] == ["local"] and isinstance(bs[0], LocalLLMBackend)


@todo
def test_backend_order_from_env_and_unknown_names_rejected(monkeypatch):
    monkeypatch.setenv("ASSIST_LLM_BACKENDS", "local,anthropic,openai")
    assert [b.name for b in backends_from_env()] == ["local", "anthropic", "openai"]
    monkeypatch.setenv("ASSIST_LLM_BACKENDS", "local,chatgpt-plus")
    import pytest as _pytest
    with _pytest.raises(ValueError):
        backends_from_env()


def test_cloud_backends_unavailable_without_keys(monkeypatch):
    from assistive_arm.language import AnthropicBackend, OpenAIBackend

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not AnthropicBackend().available() and not OpenAIBackend().available()


@todo
def test_router_falls_back_through_backends_then_rules():
    broken = FakeBackend("local", fail=True)
    cloud = FakeBackend("anthropic", default=FEED_JSON)
    r = IntentRouter([broken, cloud]).route("feed me the strawberries", DialogueState())
    assert r.source == "anthropic" and r.intent.item == "strawberry" and r.errors
    r = IntentRouter([FakeBackend("local", default="not json")]).route("feed me a grape", DialogueState())
    assert r.source == "rules"


@todo
def test_safety_words_never_reach_a_model():
    llm = FakeBackend("local", default=FEED_JSON)  # would say "feed" if asked
    for text in ["stop", "wait", "no", "I'm done"]:
        r = IntentRouter([llm]).route(text, DialogueState())
        assert r.source == "safety_rules"
    assert llm.calls == []
