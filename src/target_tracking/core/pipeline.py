from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..counters.base import Counter
from ..sinks.base import Sink
from ..sources.base import VideoSource
from ..trackers.base import TrackerModel
from .types import FrameResult


class TrackingPipeline:
    def __init__(
        self,
        source: VideoSource,
        tracker: TrackerModel,
        counters: Sequence[Counter] = (),
        sinks: Sequence[Sink] = (),
    ) -> None:
        self._source = source
        self._tracker = tracker
        self._counters = list(counters)
        self._sinks = list(sinks)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(
        self,
        on_frame: Optional[Callable[[FrameResult, dict], None]] = None,
    ) -> None:
        idx = 0
        fps = self._source.fps or 30.0
        try:
            while not self._stop:
                frame = self._source.read()
                if frame is None:
                    break
                detections = self._tracker.update(frame)
                result = FrameResult(
                    frame_index=idx,
                    timestamp=idx / fps,
                    frame=frame,
                    detections=detections,
                )
                for c in self._counters:
                    c.update(result)
                counts = {c.name: c.snapshot() for c in self._counters}
                for s in self._sinks:
                    s.write(result, counts)
                if on_frame is not None:
                    on_frame(result, counts)
                idx += 1
        finally:
            for s in self._sinks:
                s.close()
            self._source.release()
