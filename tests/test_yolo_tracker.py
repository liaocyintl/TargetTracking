from unittest.mock import MagicMock

import numpy as np

from target_tracking.trackers.yolo import _results_to_detections


def test_results_to_detections_handles_tracked_results():
    boxes = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [[10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]]
    )
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.5])
    boxes.cls.cpu.return_value.numpy.return_value = np.array([0.0, 2.0])
    boxes.id = MagicMock()
    boxes.id.cpu.return_value.numpy.return_value.astype.return_value = np.array([1, 7])

    result = MagicMock()
    result.boxes = boxes
    result.names = {0: "person", 2: "car"}

    dets = _results_to_detections([result])
    assert len(dets) == 2
    assert dets[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert dets[0].class_id == 0
    assert dets[0].class_name == "person"
    assert dets[0].confidence == 0.9
    assert dets[0].track_id == 1
    assert dets[1].track_id == 7


def test_results_to_detections_no_boxes_returns_empty():
    result = MagicMock()
    result.boxes = None
    assert _results_to_detections([result]) == []


def test_results_to_detections_no_id_returns_none_track_id():
    boxes = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0.0, 0.0, 1.0, 1.0]])
    boxes.conf.cpu.return_value.numpy.return_value = np.array([0.7])
    boxes.cls.cpu.return_value.numpy.return_value = np.array([0.0])
    boxes.id = None
    result = MagicMock()
    result.boxes = boxes
    result.names = {0: "person"}

    dets = _results_to_detections([result])
    assert len(dets) == 1
    assert dets[0].track_id is None
