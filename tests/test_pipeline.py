from __future__ import annotations

import numpy as np

from target_tracking.core.pipeline import TrackingPipeline
from target_tracking.core.types import Detection
from target_tracking.counters.line import LineCounter
from target_tracking.sinks.base import Sink
from target_tracking.sources.base import VideoSource
from target_tracking.trackers.base import TrackerModel


class _Source(VideoSource):
    def __init__(self, n: int = 4):
        self._n = n
        self._i = 0

    def read(self):
        if self._i >= self._n:
            return None
        self._i += 1
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return (100, 100)


class _ScriptedTracker(TrackerModel):
    """Returns scripted detection sequences per frame."""
    def __init__(self, script: list[list[Detection]]):
        self._script = script
        self._i = 0

    def update(self, frame):
        out = self._script[self._i]
        self._i += 1
        return out


class _MemorySink(Sink):
    def __init__(self):
        self.calls: list[tuple[int, dict]] = []
        self.closed = False

    def write(self, result, counts):
        self.calls.append((result.frame_index, {k: dict(v) for k, v in counts.items()}))

    def close(self) -> None:
        self.closed = True


def _det(tid: int, cx: float, cy: float) -> Detection:
    h = 5.0
    return Detection((cx - h, cy - h, cx + h, cy + h), 0, "person", 0.9, track_id=tid)


def test_pipeline_drives_components_and_closes_sinks():
    script = [
        [_det(1, 50, 40)],
        [_det(1, 50, 60)],
        [_det(1, 50, 65)],
        [],
    ]
    source = _Source(n=4)
    tracker = _ScriptedTracker(script)
    counter = LineCounter("gate", (0, 50), (100, 50), classes_of_interest=["person"])
    sink = _MemorySink()
    pipe = TrackingPipeline(source=source, tracker=tracker,
                            counters=[counter], sinks=[sink])
    pipe.run()
    assert len(sink.calls) == 4
    final_counts = sink.calls[-1][1]
    assert final_counts["gate"]["total_in"] + final_counts["gate"]["total_out"] == 1
    assert sink.closed is True


def test_pipeline_invokes_on_frame_callback():
    source = _Source(n=2)
    tracker = _ScriptedTracker([[], []])
    sink = _MemorySink()
    pipe = TrackingPipeline(source=source, tracker=tracker, counters=[], sinks=[sink])
    seen: list[int] = []
    pipe.run(on_frame=lambda res, counts: seen.append(res.frame_index))
    assert seen == [0, 1]


def test_pipeline_stops_when_stop_called():
    source = _Source(n=10)
    tracker = _ScriptedTracker([[]] * 10)
    sink = _MemorySink()
    pipe = TrackingPipeline(source=source, tracker=tracker, counters=[], sinks=[sink])

    def cb(res, counts):
        if res.frame_index == 2:
            pipe.stop()

    pipe.run(on_frame=cb)
    assert len(sink.calls) == 3
