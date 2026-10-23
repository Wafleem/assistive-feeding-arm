"""Viam sensor: the depth camera's lidar-style readings (distance, nearest return, a 2D scan).

Readings are computed once per depth frame by the camera; this only forwards them, so
adding the sensor costs no extra inference. Stale depth is an error here too.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional, Sequence

from viam.components.camera import Camera
from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.utils import SensorReading, struct_to_dict

from .camera import FAMILY


class DepthRangeSensor(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "range")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        unknown = set(attrs) - {"camera"}
        if unknown:
            raise ValueError(f"unknown attributes {sorted(unknown)}; the only one is \"camera\"")
        camera = str(attrs.get("camera") or "").strip()
        if not camera:
            raise ValueError('"camera" (the depth-anything-3 camera) is required')
        return [camera], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        name = str(struct_to_dict(config.attributes)["camera"]).strip()
        self._camera = dependencies[Camera.get_resource_name(name)]

    async def get_readings(
        self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, SensorReading]:
        return await self._camera.do_command({"command": "ranges"}, timeout=timeout)  # type: ignore[union-attr]
