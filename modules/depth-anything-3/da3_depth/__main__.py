"""Module entrypoint: viam-server starts this through run.sh."""

import asyncio

from viam.module.module import Module

from .camera import DepthAnything3Camera
from .range_sensor import DepthRangeSensor

if __name__ == "__main__":
    asyncio.run(Module.run_with_models(DepthAnything3Camera, DepthRangeSensor))
