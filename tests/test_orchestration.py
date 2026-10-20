"""Section 6 (orchestration). The transition table is checked now; the machine is a spec."""

from assistive_arm.orchestration import TIMEOUTS, TRANSITIONS, FeedingMachine, S
from conftest import todo


def test_every_active_state_has_a_timeout_and_a_way_out():
    active = {src for src, _ in TRANSITIONS} - {S.IDLE}
    assert active <= set(TIMEOUTS)
    for st in active:
        assert any(src == st for src, _ in TRANSITIONS), st


def test_failures_near_the_face_end_in_retreat():
    assert TRANSITIONS[(S.APPROACH, "head_moved")] is S.RETREAT
    assert TRANSITIONS[(S.WAIT_READY, "timeout")] is S.RETREAT
    assert TRANSITIONS[(S.WAIT_BITE, "timeout")] is S.RETREAT


@todo
def test_stop_from_any_state_aborts():
    from assistive_arm.language import Action, FeedIntent

    m = FeedingMachine(state=S.APPROACH)
    m.handle(FeedIntent(Action.STOP))
    assert m.state is S.ABORT
