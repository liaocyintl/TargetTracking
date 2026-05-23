import csv
import json
from pathlib import Path

import numpy as np
import pytest

from target_tracking.core.types import Detection, FrameResult
from target_tracking.sinks.stats import StatsSink


def _frame(idx: int) -> FrameResult:
    return FrameResult(
        frame_index=idx,
        timestamp=idx / 30.0,
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        detections=[],
    )


def _counts(in_n: int, out_n: int, cls: str = "person",
            counter: str = "gate") -> dict:
    return {
        counter: {
            "name": counter,
            "per_class": {cls: {"in": in_n, "out": out_n}},
            "total_in": in_n,
            "total_out": out_n,
        }
    }


def test_stats_sink_csv_writes_one_row_per_event(tmp_path: Path):
    path = tmp_path / "out.csv"
    sink = StatsSink(path=str(path), format="csv")
    sink.write(_frame(0), _counts(0, 0))         # no change → no event
    sink.write(_frame(1), _counts(1, 0))         # one "in" event
    sink.write(_frame(2), _counts(2, 1))         # one "in", one "out"
    sink.write(_frame(3), _counts(2, 1))         # no change
    sink.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 3
    assert rows[0]["frame_index"] == "1"
    assert rows[0]["direction"] == "in"
    assert rows[0]["counter"] == "gate"
    assert rows[0]["class"] == "person"
    directions_at_frame_2 = sorted(r["direction"] for r in rows if r["frame_index"] == "2")
    assert directions_at_frame_2 == ["in", "out"]


def test_stats_sink_json_writes_event_list(tmp_path: Path):
    path = tmp_path / "out.json"
    sink = StatsSink(path=str(path), format="json")
    sink.write(_frame(0), _counts(1, 0))   # one in event
    sink.write(_frame(1), _counts(1, 1))   # one out event
    sink.close()
    data = json.loads(open(path).read())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["direction"] == "in"
    assert data[1]["direction"] == "out"
    assert data[0]["total_in"] == 1
    assert data[1]["total_out"] == 1


def test_stats_sink_multi_event_in_single_frame(tmp_path: Path):
    """If a counter's value increments by >1 in one frame, emit that many rows."""
    path = tmp_path / "out.csv"
    sink = StatsSink(path=str(path), format="csv")
    sink.write(_frame(0), _counts(0, 0))
    sink.write(_frame(1), _counts(3, 0))   # 3 in events in one frame
    sink.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 3
    assert all(r["frame_index"] == "1" and r["direction"] == "in" for r in rows)


def test_stats_sink_invalid_format_raises():
    with pytest.raises(ValueError, match="format"):
        StatsSink(path="x", format="xml")


def test_stats_sink_empty_close_writes_header_only_csv(tmp_path: Path):
    path = tmp_path / "out.csv"
    sink = StatsSink(path=str(path), format="csv")
    sink.close()
    text = open(path).read()
    assert "frame_index" in text.splitlines()[0]
    assert len(text.strip().splitlines()) == 1   # header only
