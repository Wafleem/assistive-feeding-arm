import pytest

from assistive_arm.intent import Intent, IntentError, Task, parse


@pytest.mark.parametrize(
    "text, task, target",
    [
        ("press button 7 on the elevator", Task.PRESS_BUTTON, "7"),
        ("Press button 7 on the elevator.", Task.PRESS_BUTTON, "7"),
        ("can you press the elevator button for floor seven please", Task.PRESS_BUTTON, "7"),
        ("take me to the 3rd floor, press the button", Task.PRESS_BUTTON, "3"),
        ("press the lobby button", Task.PRESS_BUTTON, "L"),
        ("press button L", Task.PRESS_BUTTON, "L"),
        ("hit the down button on the elevator", Task.PRESS_BUTTON, "down"),
        ("press basement 2 in the elevator", Task.PRESS_BUTTON, "B2"),
        ("grab the shopping bag", Task.GRASP, "shopping bag"),
        ("please pick up my water bottle from the table", Task.GRASP, "water bottle"),
        ("grab the door handle and open it", Task.OPEN_DOOR, "door handle"),
        ("open the door", Task.OPEN_DOOR, "door handle"),
        ("stop", Task.STOP, None),
        ("no no stop, freeze", Task.STOP, None),
        ("let go", Task.RELEASE, None),
        ("give it to me", Task.RELEASE, None),
    ],
)
def test_example_phrases(text, task, target):
    intent = parse(text)
    assert intent.task is task, intent
    assert intent.target == target


def test_bag_targets_its_handle():
    assert parse("grab the shopping bag").part == "handle"


def test_door_direction_and_handle_type():
    i = parse("push the door open")
    assert i.task is Task.OPEN_DOOR and i.params["direction"] == "push"
    i = parse("turn the knob and pull the door open")
    assert i.params["handle_type"] == "knob" and i.params["direction"] == "pull"
    i = parse("grab the door lever and open it")
    assert i.params["handle_type"] == "lever"


def test_stop_beats_everything():
    assert parse("press button 7, no wait, stop").task is Task.STOP


def test_unclear_requests_ask_instead_of_guessing():
    for text in ["press the button", "do the thing", "", "the door"]:
        i = parse(text)
        assert i.task is Task.UNKNOWN, text
        assert i.params.get("question")


def test_elevator_door_buttons_are_buttons_not_doors():
    i = parse("press the door open button in the elevator")
    assert i.task is Task.PRESS_BUTTON and i.target == "open"


def test_alarm_needs_extra_confirmation():
    assert parse("press the alarm button in the elevator").needs_extra_confirmation


def test_llm_output_is_validated():
    ok = Intent.from_dict({"task": "press_button", "target": "12", "context": "elevator"})
    assert ok.target == "12"
    with pytest.raises(IntentError):
        Intent.from_dict({"task": "press_button", "target": "launch missiles"})
    with pytest.raises(IntentError):
        Intent.from_dict({"task": "fly"})
    with pytest.raises(IntentError):
        Intent.from_dict({"task": "grasp", "target": "cup", "speed": "max"})
    with pytest.raises(IntentError):
        Intent.from_dict({"task": "open_door", "params": {"direction": "kick"}})


def test_schema_lists_every_task():
    assert set(Intent.json_schema()["properties"]["task"]["enum"]) == {t.value for t in Task}
