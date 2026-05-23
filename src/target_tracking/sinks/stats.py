from __future__ import annotations

import csv
import json
from pathlib import Path

from ..core.types import FrameResult
from .base import Sink
from .registry import register


@register("stats")
class StatsSink(Sink):
    def __init__(self, path: str, format: str = "csv") -> None:
        if format not in {"csv", "json"}:
            raise ValueError(f"Unsupported format: {format}. Use csv or json.")
        self._path = Path(path)
        self._format = format
        self._latest_counts: dict = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, result: FrameResult, counts: dict) -> None:
        self._latest_counts = counts

    def close(self) -> None:
        if self._format == "csv":
            with self._path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["counter", "class", "in", "out"])
                w.writeheader()
                for counter_name, snap in self._latest_counts.items():
                    for cls_name, c in snap.get("per_class", {}).items():
                        w.writerow({
                            "counter": counter_name,
                            "class": cls_name,
                            "in": c["in"],
                            "out": c["out"],
                        })
        else:
            with self._path.open("w") as f:
                json.dump(self._latest_counts, f, indent=2)
