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
