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
