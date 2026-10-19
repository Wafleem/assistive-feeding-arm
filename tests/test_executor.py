"""Planner + executor with FakeRobot and a stub locator (no cameras needed)."""

import numpy as np

from assistive_arm.executor import Executor
from assistive_arm.intent import Task, parse
from assistive_arm.perception.types import Target3D
from assistive_arm.planning.task_planner import plan
from assistive_arm.robot_iface import FakeRobot, SpringSurface
from assistive_arm.skills import SkillContext


class StubLocator:
    def __init__(self, targets):
        self.targets, self.queries = targets, []

    def locate(self, robot, query, cameras=None, support_query=None):
        self.queries.append((query, support_query))
        return self.targets.get(query)


BUTTON = np.array([0.50, 0.05, 0.35])


def test_plan_reads_back_and_looks_twice():
    p = plan(parse("press button 7 on the elevator"))
    kinds = [s.kind for s in p.steps]
    assert kinds == ["clear_view", "locate", "observe", "locate", "skill"]
    assert p.steps[1].query == "elevator button 7" and p.steps[1].params["support"] == "elevator panel"
    assert "press button" in p.describe()


def test_press_button_end_to_end_on_fake():
    loc = StubLocator({"elevator button 7": Target3D("elevator button 7", BUTTON, np.array([-1.0, 0, 0]), 0.9, "scene")})
    r = FakeRobot(surfaces=[SpringSurface(BUTTON, [-1, 0, 0])])
    log = Executor(r, loc, SkillContext(confirm=lambda m: True)).run(parse("press button 7 on the elevator"))
    assert log.result.status == "success", log
    assert len(loc.queries) == 2  # coarse + close-up


def test_missing_target_fails_cleanly():
    log = Executor(FakeRobot(), StubLocator({}), SkillContext(confirm=lambda m: True)).run(parse("grab the shopping bag"))
    assert log.result.status == "failed" and "could not find" in log.result.message


def test_stop_and_clarification():
    r = FakeRobot()
    assert Executor(r, None).run(parse("stop")).result.status == "success" and r.stopped
    log = Executor(FakeRobot(), None).run(parse("press the button"))
    assert log.result.status == "refused" and "Which button" in log.result.message


def test_sensitive_button_needs_a_second_yes():
    answers = iter([False])
    log = Executor(FakeRobot(), StubLocator({}), SkillContext(confirm=lambda m: next(answers))).run(
        parse("press the alarm button in the elevator"))
    assert log.plan.intent.task is Task.PRESS_BUTTON and log.result.status == "aborted"
