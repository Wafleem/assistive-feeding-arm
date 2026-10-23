"""Exercise RGB inference boundaries without importing RF-DETR, torch or CUDA."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from assistive_arm.vision.rfdetr import RFDETRConfig, RFDETRSpoonDetector


def prediction(boxes=(), scores=(), ids=(), names=None):
    return SimpleNamespace(
        xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
        confidence=np.asarray(scores, dtype=float),
        class_id=np.asarray(ids),
        data={} if names is None else {"class_name": np.asarray(names)},
    )


@pytest.fixture
def backend(tmp_path, monkeypatch):
    checkpoint = tmp_path / "nano.pth"
    checkpoint.write_bytes(b"offline fake checkpoint")
    state = SimpleNamespace(calls=[], inputs=[], result=prediction(), mapping={44: "bottle", 50: "spoon"})

    class FakeNano:
        def __init__(self, **kwargs):
            state.calls.append(kwargs)

        @property
        def class_names(self):
            return state.mapping

        def predict(self, image, threshold):
            state.inputs.append((image.copy(), threshold))
            return state.result

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRNano=FakeNano))
    detector = RFDETRSpoonDetector(RFDETRConfig(str(checkpoint), device="cpu"))
    return detector, state, checkpoint


def test_rgb_is_preserved_and_local_model_reused(backend):
    detector, state, checkpoint = backend
    assert not state.calls
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[:, :, 0] = 220
    rgb[:, :, 2] = 17
    rgb = rgb[:, ::-1]  # A valid camera/crop view need not be contiguous.
    detector.detect_rgb(rgb)
    detector.detect_rgb(rgb)
    assert state.calls == [{"pretrain_weights": str(checkpoint), "device": "cpu"}]
    np.testing.assert_array_equal(state.inputs[0][0], rgb)
    assert state.inputs[0][1] == 0.5


def test_only_spoons_from_models_sparse_coco_mapping_are_returned(backend):
    detector, state, _ = backend
    state.result = prediction(
        [[1, 2, 10, 8], [3, 4, 13, 9], [0, 0, 6, 4], [0, 0, 3, 4]],
        [0.99, 0.8, 0.3, 0.99], [44, 50, 50, 999],
    )
    result = detector.detect_rgb(np.zeros((12, 20, 3), dtype=np.uint8))
    assert len(result) == 1
    assert result[0].label == "spoon"
    assert result[0].bbox == (3.0, 4.0, 13.0, 9.0)
    assert result[0].score == 0.8
    assert result[0].mask is None


def test_resolved_prediction_names_override_compact_model_class_list(backend):
    detector, state, _ = backend
    state.mapping = ["spoon", "cup"]
    state.result = prediction([[0, 0, 3, 4], [2, 3, 8, 9]], [0.9, 0.8], [1, 0], ["spoon", "cup"])
    result = detector.detect_rgb(np.zeros((12, 20, 3), dtype=np.uint8))
    assert [item.extras["class_id"] for item in result] == [1]


def test_no_detection_is_empty(backend):
    detector, state, _ = backend
    state.result.confidence = None
    state.result.class_id = None
    assert detector.detect_rgb(np.zeros((12, 20, 3), dtype=np.uint8)) == []


def test_boxes_are_clipped_and_invalid_observations_are_dropped(backend):
    detector, state, _ = backend
    state.result = prediction(
        [[-3, -4, 25, 15], [22, 0, 25, 5], [8, 2, 4, 7], [0, 0, np.nan, 4],
         [0, 0, 2, 3], [0, 0, 2, 3]],
        [0.7, 0.9, 0.9, 0.9, np.nan, 1.5], [50] * 6,
    )
    result = detector.detect_rgb(np.zeros((12, 20, 3), dtype=np.uint8))
    assert len(result) == 1
    assert result[0].bbox == (0.0, 0.0, 20.0, 12.0)


@pytest.mark.parametrize("rgb", [
    [], np.zeros((3, 4, 3), dtype=float), np.zeros((3, 4), dtype=np.uint8),
    np.zeros((3, 4, 4), dtype=np.uint8), np.zeros((0, 4, 3), dtype=np.uint8),
])
def test_invalid_camera_image_rejected_before_model_load(backend, rgb):
    detector, state, _ = backend
    with pytest.raises(ValueError, match="rgb"):
        detector.detect_rgb(rgb)
    assert not state.calls


def test_missing_checkpoint_does_not_trigger_model_download(backend):
    detector, state, checkpoint = backend
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        detector.detect_rgb(np.zeros((3, 4, 3), dtype=np.uint8))
    assert not state.calls


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        RFDETRConfig("unused.pth", threshold=threshold)


def test_ambiguous_class_numbers_raise_instead_of_wrong_spoon_label(backend):
    detector, state, _ = backend
    state.mapping = ["spoon", "cup"]
    state.result = prediction([[0, 0, 3, 4]], [0.9], [0])
    with pytest.raises(ValueError, match="ID-to-name mapping"):
        detector.detect_rgb(np.zeros((8, 12, 3), dtype=np.uint8))


def test_malformed_prediction_is_error_not_empty_detection(backend):
    detector, state, _ = backend
    state.result = prediction([[0, 0, 3, 4]], [], [50])
    with pytest.raises(ValueError, match="mismatched"):
        detector.detect_rgb(np.zeros((8, 12, 3), dtype=np.uint8))


def test_invalid_class_id_cannot_become_spoon_observation(backend):
    detector, state, _ = backend
    state.result = prediction([[0, 0, 3, 4]], [0.9], [np.nan], ["spoon"])
    assert detector.detect_rgb(np.zeros((8, 12, 3), dtype=np.uint8)) == []
