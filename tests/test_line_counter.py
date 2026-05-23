import numpy as np

from target_tracking.core.types import Detection, FrameResult
from target_tracking.counters.line import LineCounter


def _det(track_id: int, cx: float, cy: float, cls: int = 0, name: str = "person") -> Detection:
    half = 5.0
    return Detection(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        class_id=cls,
        class_name=name,
        confidence=0.9,
        track_id=track_id,
    )


def _frame(idx: int, dets: list[Detection]) -> FrameResult:
    return FrameResult(
        frame_index=idx,
        timestamp=float(idx) / 30.0,
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        detections=dets,
    )


def test_horizontal_line_counts_one_crossing_in_each_direction():
    counter = LineCounter(
        name="gate",
        start=(0, 50),
        end=(100, 50),
        classes_of_interest=["person"],
    )
    counter.update(_frame(0, [_det(1, 50, 40)]))
    counter.update(_frame(1, [_det(1, 50, 60)]))
    counter.update(_frame(2, [_det(2, 30, 60)]))
    counter.update(_frame(3, [_det(2, 30, 40)]))
    snap = counter.snapshot()
    assert snap["per_class"]["person"]["in"] == 1
    assert snap["per_class"]["person"]["out"] == 1
    assert snap["total_in"] == 1
    assert snap["total_out"] == 1


def test_same_track_id_only_counted_once_per_crossing():
    counter = LineCounter(name="g", start=(0, 50), end=(100, 50), classes_of_interest=["person"])
    counter.update(_frame(0, [_det(1, 50, 40)]))
    counter.update(_frame(1, [_det(1, 50, 60)]))
    counter.update(_frame(2, [_det(1, 50, 60)]))
    counter.update(_frame(3, [_det(1, 50, 65)]))
    snap = counter.snapshot()
    assert snap["total_in"] + snap["total_out"] == 1


def test_filter_by_class_of_interest():
    counter = LineCounter(name="g", start=(0, 50), end=(100, 50), classes_of_interest=["car"])
    counter.update(_frame(0, [_det(1, 50, 40, cls=0, name="person")]))
    counter.update(_frame(1, [_det(1, 50, 60, cls=0, name="person")]))
    snap = counter.snapshot()
    assert snap["total_in"] == 0
    assert snap["total_out"] == 0


def test_detection_without_track_id_ignored():
    counter = LineCounter(name="g", start=(0, 50), end=(100, 50), classes_of_interest=["person"])
    d_no_id = Detection((45, 35, 55, 45), 0, "person", 0.9, track_id=None)
    counter.update(_frame(0, [d_no_id]))
    d_no_id2 = Detection((45, 55, 55, 65), 0, "person", 0.9, track_id=None)
    counter.update(_frame(1, [d_no_id2]))
    snap = counter.snapshot()
    assert snap["total_in"] == 0


def test_classes_of_interest_none_means_all():
    counter = LineCounter(name="g", start=(0, 50), end=(100, 50), classes_of_interest=None)
    counter.update(_frame(0, [_det(1, 50, 40, cls=2, name="car")]))
    counter.update(_frame(1, [_det(1, 50, 60, cls=2, name="car")]))
    snap = counter.snapshot()
    assert snap["per_class"]["car"]["in"] == 1
