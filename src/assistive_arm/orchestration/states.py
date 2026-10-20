"""The feeding task state machine: states, events and transitions (data, so it can be drawn and tested).

IDLE -> SELECT_ITEM -> ACQUIRE -> VERIFY_GRASP -> WAIT_READY -> APPROACH -> DELIVER
     -> (printed mouth: release into the opening | person: WAIT_BITE) -> RETREAT
     -> (next bite: SELECT_ITEM | DONE)
Any state --stop--> ABORT (halt, then retreat if near the face) ; PAUSED on "wait".
Every state has a timeout and an abort path (see TIMEOUTS).
"""

from __future__ import annotations

from enum import Enum


class S(str, Enum):
    IDLE = "idle"
    SELECT_ITEM = "select_item"
    ACQUIRE = "acquire"
    VERIFY_GRASP = "verify_grasp"
    WAIT_READY = "wait_ready"
    APPROACH = "approach"
    DELIVER = "deliver"
    WAIT_BITE = "wait_bite"
    RETREAT = "retreat"
    PAUSED = "paused"
    ABORT = "abort"
    DONE = "done"


# Seconds before a state gives up (TODO(orchestration): tune with the safety lead).
TIMEOUTS = {
    S.SELECT_ITEM: 5.0, S.ACQUIRE: 20.0, S.VERIFY_GRASP: 3.0, S.WAIT_READY: 20.0,
    S.APPROACH: 10.0, S.DELIVER: 5.0, S.WAIT_BITE: 10.0, S.RETREAT: 10.0,
}

# Normal flow (state, event) -> next state. TODO(orchestration): complete, including failures.
TRANSITIONS: dict[tuple[S, str], S] = {
    (S.IDLE, "feed_request"): S.SELECT_ITEM,
    (S.SELECT_ITEM, "item_found"): S.ACQUIRE,
    (S.SELECT_ITEM, "not_found"): S.IDLE,  # say "I can't see any strawberries"
    (S.ACQUIRE, "grasped"): S.VERIFY_GRASP,
    (S.ACQUIRE, "failed"): S.SELECT_ITEM,  # retry another instance, max N times
    (S.VERIFY_GRASP, "holding"): S.WAIT_READY,
    (S.VERIFY_GRASP, "empty"): S.SELECT_ITEM,
    (S.WAIT_READY, "ready"): S.APPROACH,
    (S.WAIT_READY, "timeout"): S.RETREAT,
    (S.APPROACH, "arrived"): S.DELIVER,
    (S.APPROACH, "head_moved"): S.RETREAT,
    (S.DELIVER, "released"): S.RETREAT,  # printed_mouth: bite dropped into the catch cup
    (S.DELIVER, "presented"): S.WAIT_BITE,  # person: hold at the standoff for the bite
    (S.WAIT_BITE, "bite_taken"): S.RETREAT,
    (S.WAIT_BITE, "timeout"): S.RETREAT,
    (S.RETREAT, "clear"): S.IDLE,
}
