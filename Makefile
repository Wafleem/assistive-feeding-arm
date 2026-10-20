# Everyday commands. Python lives in .venv (make setup). Nothing here needs the arm
# except `make viam-check`.
PY     ?= .venv/bin/python

.PHONY: help setup test test-fast render viewer todo viam-check clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:           ## create .venv (Python 3.12) and install the package + MuJoCo + pytest
	@if command -v uv >/dev/null; then \
		uv venv -q --python 3.12 .venv && uv pip install -q --python $(PY) -e ".[dev]"; \
	else \
		python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -e ".[dev]"; \
	fi

test:            ## all tests; unimplemented sections show as xfail (expected failures)
	$(PY) -m pytest -q -rx

test-fast:       ## tests without MuJoCo
	$(PY) -m pytest -q --ignore=tests/test_sim.py

render:          ## save wrist + fixed camera RGB/depth PNGs of the feeding scene to out/
	$(PY) scripts/view_feeding_scene.py --render out/ --park

viewer:          ## interactive MuJoCo viewer (macOS runs it under mjpython)
	$(if $(findstring Darwin,$(shell uname)),.venv/bin/mjpython,$(PY)) scripts/view_feeding_scene.py --viewer

todo:            ## every TODO, grouped by section
	@grep -rn "TODO(" src | sed -E 's/.*TODO\(([a-z_ /]+)\).*/\1/' | sort | uniq -c | sort -rn
	@echo; grep -rn "TODO(" src | sed 's/^src\/assistive_arm\///'

viam-check:      ## (hardware) connect with VIAM_ADDRESS / VIAM_API_KEY(_ID) and print the TCP pose
	$(PY) -c "from assistive_arm.robot_iface.viam import ViamRobot; r = ViamRobot(); print(r.get_tcp_pose())"

clean:
	rm -rf out .pytest_cache build *.egg-info src/*.egg-info
