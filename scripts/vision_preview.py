#!/usr/bin/env python3
"""Launch the observation-only camera preview; see scripts/README.md."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistive_arm.orchestration.vision_preview import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
