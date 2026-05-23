import numpy as np
from target_tracking.core.types import Detection, FrameResult


def test_detection_defaults_track_id_none():
    d = Detection(
        bbox=(0.0, 0.0, 10.0, 20.0),
        class_id=0,
        class_name="person",
        confidence=0.9,
    )
    assert d.track_id is None


def test_detection_with_track_id():
    d = Detection(
        bbox=(1.0, 2.0, 3.0, 4.0),
        class_id=2,
        class_name="car",
        confidence=0.5,
        track_id=42,
    )
    assert d.track_id == 42


def test_frame_result_holds_detections_and_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    det = Detection((0, 0, 1, 1), 0, "person", 0.8, track_id=1)
    fr = FrameResult(frame_index=0, timestamp=0.0, frame=frame, detections=[det])
    assert fr.frame_index == 0
    assert fr.timestamp == 0.0
    assert fr.frame.shape == (10, 10, 3)
    assert len(fr.detections) == 1
    assert fr.detections[0].track_id == 1
