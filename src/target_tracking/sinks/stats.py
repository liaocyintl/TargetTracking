from __future__ import annotations

import csv
import json
from pathlib import Path

from ..core.types import FrameResult
from .base import Sink
from .registry import register


@register("stats")
class StatsSink(Sink):
    """Logs per-event rows when counter values change.

    CSV columns: frame_index, timestamp, counter, class, direction, total_in, total_out.
    JSON: list of dicts with the same fields.

    Note: track_id is not currently recorded — it would require the counter to expose
    events directly. The diff-based approach detects that a count changed but not
    which specific track caused it.
    """

    _FIELDS = ["frame_index", "timestamp", "counter", "class",
               "direction", "total_in", "total_out"]

    def __init__(self, path: str, format: str = "csv") -> None:
        if format not in {"csv", "json"}:
            raise ValueError(f"Unsupported format: {format}. Use csv or json.")
        self._path = Path(path)
        self._format = format
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict] = []
        # Maps counter_name -> {class_name -> {"in": n, "out": n}}
        self._prev: dict[str, dict[str, dict[str, int]]] = {}

    def write(self, result: FrameResult, counts: dict) -> None:
        for counter_name, snap in counts.items():
            prev_per_class = self._prev.get(counter_name, {})
            for cls_name, c in snap.get("per_class", {}).items():
                prev_c = prev_per_class.get(cls_name, {"in": 0, "out": 0})
                for direction in ("in", "out"):
                    delta = c[direction] - prev_c[direction]
                    for _ in range(delta):
                        self._events.append({
                            "frame_index": result.frame_index,
                            "timestamp": round(result.timestamp, 3),
                            "counter": counter_name,
                            "class": cls_name,
                            "direction": direction,
                            "total_in": c["in"],
                            "total_out": c["out"],
                        })
            self._prev[counter_name] = {
                k: dict(v) for k, v in snap.get("per_class", {}).items()
            }

    def close(self) -> None:
        if self._format == "csv":
            with self._path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._FIELDS)
                w.writeheader()
                for ev in self._events:
                    w.writerow(ev)
        else:
            with self._path.open("w") as f:
                json.dump(self._events, f, indent=2)
