# Everyday commands. Python lives in .venv (make setup). No robot needed for any of these
# except `make viam-check`.
PY     ?= .venv/bin/python
SCENE  ?= elevator
SAY    ?=
SEEDS  ?= 3

.PHONY: help setup test test-fast demo demo-all render eval eval-hard viewer viam-check clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:           ## create .venv (Python 3.12) and install the package + MuJoCo + pytest
	@if command -v uv >/dev/null; then \
		uv venv -q --python 3.12 .venv && uv pip install -q --python $(PY) -e ".[dev]"; \
	else \
		python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -e ".[dev]"; \
	fi

test:            ## every offline test (unit + MuJoCo end-to-end), ~15 s
	$(PY) -m pytest -q

test-fast:       ## unit tests only, no MuJoCo, <1 s
	$(PY) -m pytest -q --ignore=tests/test_sim.py

demo:            ## one request in sim:  make demo SCENE=bag SAY="grab the shopping bag"
	$(PY) scripts/sim_demo.py --scene $(SCENE) $(if $(SAY),--say "$(SAY)",)

demo-all:        ## the four reference scenes, one request each
	for s in elevator bag door_lever door_knob; do $(PY) scripts/sim_demo.py --scene $$s --known-hinge | tail -4; done

render:          ## save wrist + wheelchair camera RGB/depth PNGs to out/
	$(PY) scripts/sim_demo.py --scene $(SCENE) --render out/

viewer:          ## interactive MuJoCo viewer (macOS needs mjpython: .venv/bin/mjpython)
	$(if $(findstring Darwin,$(shell uname)),.venv/bin/mjpython,$(PY)) scripts/sim_demo.py --scene $(SCENE) --viewer

eval:            ## success-rate table over SEEDS seeds
	$(PY) scripts/eval_sim.py --seeds $(SEEDS)

eval-hard:       ## same, with wheelchair sway and a degraded detector
	$(PY) scripts/eval_sim.py --seeds $(SEEDS) --sway 0.004 --noisy

viam-check:      ## (hardware) connect with VIAM_ADDRESS / VIAM_API_KEY(_ID) and print the TCP pose
	$(PY) -c "from assistive_arm.robot_iface.viam import ViamRobot; r = ViamRobot(); print(r.get_tcp_pose())"

clean:
	rm -rf out .pytest_cache build *.egg-info src/*.egg-info
