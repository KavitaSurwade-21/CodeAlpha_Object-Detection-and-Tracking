"""
test_detector.py
-----------------
A couple of fast sanity checks. Not a substitute for real evaluation, but
having ANY tests in your repo is a strong signal in a code review --
showing CodeAlpha reviewers (or anyone on LinkedIn/GitHub) that you think
about correctness, not just "does it run once on my machine".

Run with:  pytest tests/
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import YOLODetector  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    return YOLODetector(model_path="yolov8n.pt", conf_threshold=0.4)


def test_detect_returns_a_list(detector):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)


def test_detection_tuple_shape(detector):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    for bbox, conf, cls_id in detections:
        assert len(bbox) == 4
        assert 0.0 <= conf <= 1.0
        assert isinstance(cls_id, int)


def test_class_name_lookup_has_known_coco_classes(detector):
    names = list(detector.class_names.values())
    assert "person" in names
    assert "car" in names
