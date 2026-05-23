import csv
import json
from pathlib import Path

import numpy as np
import pytest

from target_tracking.core.types import Detection, FrameResult
from target_tracking.sinks.stats import StatsSink


def _result_with_counts():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    det = Detection((0, 0, 1, 1), 0, "person", 0.9, track_id=1)
    return FrameResult(
        frame_index=5,
        timestamp=0.5,
        frame=frame,
        detections=[det],
    ), {"gate": {"name": "gate", "per_class": {"person": {"in": 2, "out": 1}},
                "total_in": 2, "total_out": 1}}


def test_stats_sink_csv_writes_summary_row_per_close(tmp_path: Path):
    path = tmp_path / "out.csv"
    sink = StatsSink(path=str(path), format="csv")
    result, counts = _result_with_counts()
    sink.write(result, counts)
    sink.close()
    rows = list(csv.DictReader(open(path)))
    assert rows[-1]["counter"] == "gate"
    assert rows[-1]["class"] == "person"
    assert int(rows[-1]["in"]) == 2
    assert int(rows[-1]["out"]) == 1


def test_stats_sink_json_writes_summary(tmp_path: Path):
    path = tmp_path / "out.json"
    sink = StatsSink(path=str(path), format="json")
    result, counts = _result_with_counts()
    sink.write(result, counts)
    sink.close()
    data = json.loads(open(path).read())
    assert data["gate"]["total_in"] == 2


def test_stats_sink_invalid_format_raises():
    with pytest.raises(ValueError, match="format"):
        StatsSink(path="x", format="xml")
