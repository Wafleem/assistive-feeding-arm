"""Fetch pinned public model weights; no camera or robot access."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODELS = (
    (
        "rf-detr-nano.pth",
        "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "md5", "fb6504cce7fbdc783f7a46991f07639f",
    ),
    (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "sha256", "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
)


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, url, algorithm, expected in MODELS:
        destination = directory / filename
        if destination.exists():
            if digest(destination, algorithm) != expected:
                raise RuntimeError(f"Existing {filename} differs from pinned model; retained unchanged")
            print(f"Verified existing {filename}", flush=True)
            continue
        temporary = None
        try:
            print(f"Downloading {filename} from official model storage", flush=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=filename, suffix=".partial", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(url, timeout=120) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
            if digest(temporary, algorithm) != expected:
                raise RuntimeError(f"Integrity check failed for {filename}")
            temporary.replace(destination)
            print(f"Verified {filename}: {destination.stat().st_size} bytes", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
