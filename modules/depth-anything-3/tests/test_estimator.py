"""Device choice without real torch: a fake torch module stands in (offline)."""

import sys
import types

import pytest

from da3_depth.estimator import pick_device


def fake_torch(monkeypatch, cuda_build, gpu_works=True, mps=False):
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(cuda="13.2" if cuda_build else None)
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))

    def randn(*shape, device="cpu"):
        if device == "cuda" and not gpu_works:
            raise RuntimeError("No CUDA GPUs are available")
        return types.SimpleNamespace(sum=lambda: types.SimpleNamespace(item=lambda: 0.0))

    t.randn = randn
    t.nn = types.SimpleNamespace(functional=types.SimpleNamespace(conv2d=lambda x, w: x))
    monkeypatch.setitem(sys.modules, "torch", t)


def test_cuda_build_uses_the_gpu(monkeypatch):
    fake_torch(monkeypatch, cuda_build=True)
    assert pick_device("auto") == "cuda"


def test_cuda_build_with_a_dead_gpu_refuses_instead_of_crawling_on_the_cpu(monkeypatch):
    fake_torch(monkeypatch, cuda_build=True, gpu_works=False)
    with pytest.raises(RuntimeError, match="device: cpu"):
        pick_device("auto")
    assert pick_device("cpu") == "cpu"  # explicit CPU still allowed


def test_without_cuda_mps_then_cpu(monkeypatch):
    fake_torch(monkeypatch, cuda_build=False, mps=True)
    assert pick_device("auto") == "mps"
    fake_torch(monkeypatch, cuda_build=False)
    assert pick_device("auto") == "cpu"


def fake_depth_pro(monkeypatch, focal_estimate=1000.0):
    """A stand-in depth_pro package that records the focal it was given."""
    import dataclasses
    import numpy as np

    calls = []

    @dataclasses.dataclass(frozen=True)
    class Cfg:
        checkpoint_uri: str = "./checkpoints/depth_pro.pt"

    class Model:
        def eval(self):
            return self

        def infer(self, x, f_px=None):
            calls.append(f_px)
            f = focal_estimate if f_px is None else float(f_px)
            d = np.full(x.shape[:2], 0.25 * f / 1000.0, np.float32)  # depth scales with the focal
            d[0, 0] = np.nan
            return {"depth": FakeTensor(d), "focallength_px": f}

    class FakeTensor:
        def __init__(self, a):
            self.a = a

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    pkg = types.ModuleType("depth_pro")
    pkg.create_model_and_transforms = lambda cfg, device=None, precision=None: (Model(), lambda rgb: rgb)
    sub = types.ModuleType("depth_pro.depth_pro")
    sub.DEFAULT_MONODEPTH_CONFIG_DICT = Cfg()
    pkg.depth_pro = sub
    monkeypatch.setitem(sys.modules, "depth_pro", pkg)
    monkeypatch.setitem(sys.modules, "depth_pro.depth_pro", sub)
    return calls


def test_depth_pro_uses_the_given_focal_else_its_own(monkeypatch, tmp_path):
    import numpy as np

    from da3_depth.estimator import DepthProEstimator
    from da3_depth.geometry import Intrinsics

    fake_torch(monkeypatch, cuda_build=False)
    sys.modules["torch"].half = sys.modules["torch"].float32 = "dtype"
    sys.modules["torch"].device = lambda d: d
    sys.modules["torch"].tensor = lambda v, device=None: v

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    sys.modules["torch"].no_grad = NoGrad
    calls = fake_depth_pro(monkeypatch)
    weights = tmp_path / "depth_pro.pt"
    weights.write_bytes(b"x")
    est = DepthProEstimator(str(weights), "cpu")
    est.load()
    rgb = np.zeros((48, 64, 3), np.uint8)
    d = est.predict(rgb, Intrinsics(800.0, 790.0, 32, 24, 64, 48))
    assert calls[-1] == 800.0 and est.focal_px == 800.0  # fx, which Depth Pro pairs with the width
    assert d.shape == (48, 64) and d[0, 0] == 0.0 and d[1, 1] == pytest.approx(0.2)  # NaN -> 0 (invalid)
    est.predict(rgb, None)
    assert calls[-1] is None and est.focal_px == 1000.0
    assert est.estimates_focal
