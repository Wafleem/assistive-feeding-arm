"""Vision service: which food is in a bowl crop, by zero-shot CLIP on the Orin GPU.

Main call (what the dashboard uses), one or more bowl crops per request:

    do_command({"command": "classify",
                "images_b64": ["<jpeg base64>", ...],
                "labels": {"cereal": ["a bowl of Froot Loops cereal", ...], ...}})   # optional
    -> {"results": [{"label": "cereal", "scores": {"cereal": 0.97, ...}, "margin": 0.95}, ...],
        "ms": 31.0, "device": "cuda", "model": "openai/clip-vit-base-patch32"}

    do_command({"command": "status"}) -> {"ready": true, "device": "cuda", ...}

GetClassifications(image) also works (default labels). The model loads in a background thread
when the service is configured, so viam-server is never blocked; calls before it is ready wait
up to ``ready_timeout_s``. Weights come from the Hugging Face cache (downloaded once).
"""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, ClassVar, Mapping, Optional, Sequence

from viam.media.video import ViamImage
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

FAMILY = ModelFamily("assistive-arm", "bowl-contents")

# Several prompts per food, averaged (the cereal is Froot Loops: multi-coloured rings, so
# colour alone does not separate it from M&Ms; the prompts name the shapes).
DEFAULT_LABELS: dict[str, list[str]] = {
    "cereal": ["a photo of a bowl of Froot Loops cereal", "a bowl of colorful ring-shaped cereal",
               "cereal loops with holes in a paper bowl", "breakfast cereal rings seen from above"],
    "mms": ["a photo of a bowl of M&M's chocolate candies", "small round colorful candy-coated chocolates in a bowl",
            "a pile of M&Ms seen from above", "a bowl of shiny round candies"],
    "empty": ["an empty paper bowl", "a white paper bowl with nothing in it", "the bottom of an empty bowl"],
}


class BowlContents(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(FAMILY, "clip")

    @classmethod
    def new(cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        svc = cls(config.name)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        unknown = set(attrs) - {"model_id", "device", "labels", "ready_timeout_s"}
        if unknown:
            raise ValueError(f"unknown attributes {sorted(unknown)}")
        return [], []

    def reconfigure(self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs = struct_to_dict(config.attributes)
        self.model_id = str(attrs.get("model_id") or "openai/clip-vit-base-patch32")
        self.want_device = str(attrs.get("device") or "auto")
        self.labels = {str(k): [str(p) for p in v] for k, v in (attrs.get("labels") or DEFAULT_LABELS).items()}
        self.ready_timeout_s = float(attrs.get("ready_timeout_s") or 120.0)
        self._ready = threading.Event()
        self._error: str | None = None
        self._text_cache: dict[tuple, Any] = {}
        threading.Thread(target=self._load, daemon=True).start()

    # ------------------------------------------------------------------ model
    def _load(self) -> None:
        try:
            # The shared env's scipy needs numpy 2 (it has 1.26) and breaks on import; CLIP never uses
            # scipy (only transformers' detection losses do), so hide it from transformers, this process only.
            import transformers.utils as _tu
            import transformers.utils.import_utils as _iu

            _iu.is_scipy_available = _tu.is_scipy_available = lambda: False
            import torch
            from transformers import CLIPModel, CLIPProcessor

            if self.want_device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.want_device
            self.torch = torch
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
            if self.device == "cuda":
                self.model = self.model.half()
            self.proc = CLIPProcessor.from_pretrained(self.model_id)
            self._text_features(self.labels)  # warm up + cache the default prompts
            self.logger.info(f"bowl-contents ready: {self.model_id} on {self.device}")
        except Exception as e:  # reported by status/classify instead of crashing the module
            self._error = f"{type(e).__name__}: {e}"
            self.logger.error(f"bowl-contents failed to load: {self._error}")
        finally:
            self._ready.set()

    def _wait(self) -> None:
        if not self._ready.wait(self.ready_timeout_s):
            raise RuntimeError("bowl-contents model still loading")
        if self._error:
            raise RuntimeError(f"bowl-contents model failed to load: {self._error}")

    def _text_features(self, labels: dict[str, list[str]]):
        key = tuple((k, tuple(v)) for k, v in labels.items())
        if key not in self._text_cache:
            torch = self.torch
            names, feats = list(labels), []
            with torch.no_grad():
                for n in names:
                    t = self.proc(text=labels[n], return_tensors="pt", padding=True).to(self.device)
                    out = self.model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
                    f = self.model.text_projection(out.pooler_output)  # explicit: any transformers version
                    f = f / f.norm(dim=-1, keepdim=True)
                    f = f.mean(dim=0)
                    feats.append(f / f.norm())
            self._text_cache[key] = (names, torch.stack(feats))
        return self._text_cache[key]

    def classify_pil(self, images, labels: dict[str, list[str]] | None = None) -> list[dict]:
        self._wait()
        torch = self.torch
        names, text = self._text_features(labels or self.labels)
        with torch.no_grad():
            px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device, dtype=text.dtype)
            img = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            img = img / img.norm(dim=-1, keepdim=True)
            probs = (self.model.logit_scale.exp() * img @ text.T).softmax(dim=-1).float().cpu().numpy()
        out = []
        for p in probs:
            order = p.argsort()[::-1]
            out.append({"label": names[order[0]], "scores": {n: round(float(s), 4) for n, s in zip(names, p)},
                        "margin": round(float(p[order[0]] - (p[order[1]] if len(p) > 1 else 0.0)), 4)})
        return out

    # ------------------------------------------------------------------ Viam API
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        cmd = str(command.get("command", ""))
        if cmd == "status":
            return {"ready": self._ready.is_set() and not self._error, "loading": not self._ready.is_set(),
                    "error": self._error or "", "device": getattr(self, "device", ""), "model": self.model_id,
                    "labels": list(self.labels)}
        if cmd == "classify":
            from PIL import Image

            images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in command.get("images_b64", [])]
            if not images:
                raise ValueError("classify needs images_b64 (a list of base64 JPEG/PNG crops)")
            labels = command.get("labels")
            labels = {str(k): [str(p) for p in v] for k, v in labels.items()} if labels else None
            t0 = time.perf_counter()
            res = self.classify_pil(images, labels)
            return {"results": res, "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "device": self.device, "model": self.model_id}
        raise ValueError(f"unknown command {cmd!r}: use classify or status")

    async def get_classifications(self, image: ViamImage, count: int, *, extra: Optional[Mapping[str, Any]] = None,
                                  timeout: Optional[float] = None, **kwargs) -> list[Classification]:
        from PIL import Image

        r = self.classify_pil([Image.open(io.BytesIO(image.data)).convert("RGB")])[0]
        ranked = sorted(r["scores"].items(), key=lambda kv: -kv[1])[: max(1, count)]
        return [Classification(class_name=n, confidence=s) for n, s in ranked]

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra=None, timeout=None,
                                              **kwargs) -> list[Classification]:
        raise NotImplementedError("send crops with do_command classify (the bowls are cropped by the caller)")

    async def get_detections_from_camera(self, camera_name: str, *, extra=None, timeout=None, **kwargs) -> list[Detection]:
        raise NotImplementedError("bowl-contents only classifies")

    async def get_detections(self, image: ViamImage, *, extra=None, timeout=None, **kwargs) -> list[Detection]:
        raise NotImplementedError("bowl-contents only classifies")

    async def get_detections_3d(self, camera_name: str, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError("bowl-contents only classifies")

    async def get_object_point_clouds(self, camera_name: str, *, extra=None, timeout=None,
                                      **kwargs) -> list[PointCloudObject]:
        raise NotImplementedError("bowl-contents only classifies")

    async def capture_all_from_camera(self, camera_name: str, return_image: bool = False,
                                      return_classifications: bool = False, return_detections: bool = False,
                                      return_object_point_clouds: bool = False, *, extra=None, timeout=None,
                                      **kwargs) -> CaptureAllResult:
        raise NotImplementedError("send crops with do_command classify")

    async def get_properties(self, *, extra=None, timeout=None, **kwargs) -> Vision.Properties:
        return Vision.Properties(classifications_supported=True, detections_supported=False,
                                 object_point_clouds_supported=False)
