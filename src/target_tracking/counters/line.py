from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..core.types import Detection, FrameResult
from .base import Counter
from .registry import register


def _side(line_start: tuple[float, float], line_end: tuple[float, float],
          point: tuple[float, float]) -> int:
    """Return +1, -1, or 0 indicating which side of the directed line the point is on."""
    x1, y1 = line_start
    x2, y2 = line_end
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def _centroid(det: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@register("line")
class LineCounter(Counter):
    def __init__(
        self,
        name: str,
        start: tuple[float, float] | list[float],
        end: tuple[float, float] | list[float],
        classes_of_interest: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self._start = (float(start[0]), float(start[1]))
        self._end = (float(end[0]), float(end[1]))
        self._classes = set(classes_of_interest) if classes_of_interest is not None else None
        self._last_side: dict[int, int] = {}
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0}
        )

    def update(self, result: FrameResult) -> None:
        for det in result.detections:
            if det.track_id is None:
                continue
            if self._classes is not None and det.class_name not in self._classes:
                continue
            side = _side(self._start, self._end, _centroid(det))
            prev = self._last_side.get(det.track_id)
            self._last_side[det.track_id] = side
            if prev is None or prev == 0 or side == 0:
                continue
            if prev != side:
                direction = "in" if side > 0 else "out"
                self._counts[det.class_name][direction] += 1

    def snapshot(self) -> dict:
        per_class = {k: dict(v) for k, v in self._counts.items()}
        total_in = sum(v["in"] for v in per_class.values())
        total_out = sum(v["out"] for v in per_class.values())
        return {
            "name": self.name,
            "per_class": per_class,
            "total_in": total_in,
            "total_out": total_out,
        }
