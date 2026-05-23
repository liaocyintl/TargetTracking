# Target Tracking System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-based object tracking system that counts cars/people in video or live camera, with pluggable models/counters/sinks, exposed via CLI and Gradio Web UI.

**Architecture:** Per-frame pipeline `Source → TrackerModel → Counters → Sinks`. Components communicate via `FrameResult` dataclass. Trackers/counters/sinks registered in registries so YAML config strings resolve to concrete classes — adding a new model is a single decorated class.

**Tech Stack:** Python 3.11, Ultralytics YOLO26 (with built-in BoT-SORT/ByteTrack), OpenCV, Pydantic v2, PyYAML, Gradio 5, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-23-target-tracking-design.md`

---

## File Structure (decisions locked in)

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Dependencies, packaging, tool config |
| `.gitignore` | Exclude `data/`, `outputs/`, `__pycache__`, models |
| `.python-version` | Python 3.11 |
| `src/target_tracking/core/types.py` | `Detection`, `FrameResult` dataclasses |
| `src/target_tracking/core/device.py` | `resolve_device()` auto-detect CPU/CUDA/MPS |
| `src/target_tracking/core/pipeline.py` | `TrackingPipeline.run()` orchestration loop |
| `src/target_tracking/sources/base.py` | `VideoSource` ABC |
| `src/target_tracking/sources/file.py` | `FileSource` (cv2.VideoCapture on file) |
| `src/target_tracking/sources/camera.py` | `CameraSource` (cv2.VideoCapture on device id) |
| `src/target_tracking/trackers/base.py` | `TrackerModel` ABC |
| `src/target_tracking/trackers/registry.py` | `register`/`build` for trackers |
| `src/target_tracking/trackers/yolo.py` | `YoloTracker` wrapping `model.track()` |
| `src/target_tracking/counters/base.py` | `Counter` ABC |
| `src/target_tracking/counters/registry.py` | `register`/`build` for counters |
| `src/target_tracking/counters/line.py` | `LineCounter` line-crossing logic |
| `src/target_tracking/sinks/base.py` | `Sink` ABC |
| `src/target_tracking/sinks/registry.py` | `register`/`build` for sinks |
| `src/target_tracking/sinks/display.py` | `DisplaySink` (cv2 window or callback) |
| `src/target_tracking/sinks/video.py` | `VideoSink` (cv2.VideoWriter) |
| `src/target_tracking/sinks/stats.py` | `StatsSink` (CSV/JSON Lines) |
| `src/target_tracking/visualization/draw.py` | Box/ID/line overlay rendering |
| `src/target_tracking/config.py` | Pydantic models + `load_config()` + `build_pipeline()` |
| `src/target_tracking/cli.py` | argparse entry point |
| `src/target_tracking/webui.py` | Gradio app |
| `configs/default.yaml` | Default runnable config |
| `configs/examples/car_counting.yaml` | Example: car counting |
| `configs/examples/people_counting.yaml` | Example: people counting |
| `scripts/download_models.py` | Pre-download YOLO weights |
| `tests/conftest.py` | pytest fixtures (mock tracker, tmp paths) |
| `tests/test_types.py` | Dataclass tests |
| `tests/test_device.py` | Device resolution tests |
| `tests/test_registry.py` | Registry tests |
| `tests/test_line_counter.py` | LineCounter behavior tests |
| `tests/test_pipeline.py` | Pipeline integration test with MockTracker |
| `tests/test_config.py` | Config loading and build_pipeline tests |
| `tests/test_sinks.py` | StatsSink + VideoSink tests |
| `tests/fixtures/short.mp4` | Tiny test video |
| `README.md` | Usage docs |

---

## Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `README.md`
- Create empty dirs (`__init__.py` only): `src/target_tracking/`, `src/target_tracking/core/`, `src/target_tracking/sources/`, `src/target_tracking/trackers/`, `src/target_tracking/counters/`, `src/target_tracking/sinks/`, `src/target_tracking/visualization/`, `tests/`, `tests/fixtures/`, `configs/`, `configs/examples/`, `scripts/`, `data/videos/`, `data/models/`, `outputs/`

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.venv/
venv/

# Project
data/
outputs/
*.pt
*.onnx
*.engine

# IDE
.vscode/
.idea/
.DS_Store
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "target-tracking"
version = "0.1.0"
description = "Pluggable visual object tracking and counting"
requires-python = ">=3.11"
dependencies = [
    "ultralytics>=8.3",
    "opencv-python>=4.10",
    "numpy>=1.26",
    "pyyaml>=6",
    "pydantic>=2",
    "gradio>=5",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.6",
]

[project.scripts]
target-tracking = "target_tracking.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/target_tracking"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

Note: `ultralytics>=8.3` is the pip package name; YOLO26 weights (`yolo26n.pt`) are auto-downloaded by Ultralytics when referenced. Pin tighter once the latest released ultralytics package containing YOLO26 is confirmed at install time.

- [ ] **Step 4: Create `README.md`**

```markdown
# Target Tracking

Pluggable visual object tracking and counting (cars, people, etc.) for video files and live cameras.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
# CLI with config file
python -m target_tracking.cli --config configs/default.yaml

# CLI shortcuts
python -m target_tracking.cli --source video.mp4
python -m target_tracking.cli --source camera:0

# Web UI
python -m target_tracking.webui
```

## Tests

```bash
pytest
```
```

- [ ] **Step 5: Create directory tree with `__init__.py`**

Run:
```bash
mkdir -p src/target_tracking/{core,sources,trackers,counters,sinks,visualization} \
         tests/fixtures configs/examples scripts data/videos data/models outputs
touch src/target_tracking/__init__.py \
      src/target_tracking/core/__init__.py \
      src/target_tracking/sources/__init__.py \
      src/target_tracking/trackers/__init__.py \
      src/target_tracking/counters/__init__.py \
      src/target_tracking/sinks/__init__.py \
      src/target_tracking/visualization/__init__.py \
      tests/__init__.py
```

- [ ] **Step 6: Verify install works**

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```
Expected: install completes without errors. (This may take several minutes — large deps.)

- [ ] **Step 7: Commit**

```bash
git add .gitignore .python-version pyproject.toml README.md src/ tests/ configs/ scripts/
git commit -m "chore: scaffold project structure"
```

---

## Task 2: Core data types

**Files:**
- Create: `src/target_tracking/core/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write failing test**

`tests/test_types.py`:
```python
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
```

- [ ] **Step 2: Run test, expect failure**

Run: `pytest tests/test_types.py -v`
Expected: ImportError — module doesn't exist.

- [ ] **Step 3: Implement `src/target_tracking/core/types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
```

- [ ] **Step 4: Run test, expect pass**

Run: `pytest tests/test_types.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/target_tracking/core/types.py tests/test_types.py
git commit -m "feat(core): add Detection and FrameResult dataclasses"
```

---

## Task 3: Device auto-detection

**Files:**
- Create: `src/target_tracking/core/device.py`
- Test: `tests/test_device.py`

- [ ] **Step 1: Write failing test**

`tests/test_device.py`:
```python
from unittest.mock import patch

from target_tracking.core.device import resolve_device


def test_explicit_device_returned_as_is():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_auto_picks_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"


def test_auto_picks_mps_when_no_cuda():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/test_device.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/target_tracking/core/device.py`:
```python
def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

- [ ] **Step 4: Run, expect pass**

Run: `pytest tests/test_device.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/target_tracking/core/device.py tests/test_device.py
git commit -m "feat(core): add device auto-detection"
```

---

## Task 4: Registry pattern (generic helper)

**Files:**
- Create: `src/target_tracking/_registry.py` (shared helper)
- Test: `tests/test_registry.py`

Rationale: trackers/counters/sinks all need identical register/build helpers. DRY by sharing one.

- [ ] **Step 1: Write failing test**

`tests/test_registry.py`:
```python
import pytest

from target_tracking._registry import Registry


def test_register_and_build():
    reg: Registry[object] = Registry("widget")

    @reg.register("foo")
    class Foo:
        def __init__(self, x: int = 0):
            self.x = x

    obj = reg.build("foo", x=5)
    assert isinstance(obj, Foo)
    assert obj.x == 5


def test_build_unknown_raises():
    reg: Registry[object] = Registry("widget")
    with pytest.raises(ValueError, match="Unknown widget: 'bar'"):
        reg.build("bar")


def test_names_listed_in_error():
    reg: Registry[object] = Registry("widget")

    @reg.register("a")
    class A: ...

    @reg.register("b")
    class B: ...

    with pytest.raises(ValueError, match="Available: \\['a', 'b'\\]"):
        reg.build("zzz")
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/test_registry.py -v`

- [ ] **Step 3: Implement**

`src/target_tracking/_registry.py`:
```python
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            self._items[name] = cls
            return cls
        return deco

    def build(self, name: str, **kwargs) -> T:
        if name not in self._items:
            raise ValueError(
                f"Unknown {self._kind}: {name!r}. Available: {sorted(self._items)}"
            )
        return self._items[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)
```

- [ ] **Step 4: Run, expect pass**

Run: `pytest tests/test_registry.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/target_tracking/_registry.py tests/test_registry.py
git commit -m "feat: generic Registry helper"
```

---

## Task 5: Source abstractions

**Files:**
- Create: `src/target_tracking/sources/base.py`
- Create: `src/target_tracking/sources/file.py`
- Create: `src/target_tracking/sources/camera.py`
- Modify: `src/target_tracking/sources/__init__.py`

This task has limited unit testing — cv2 video IO is best smoke-tested against a real file later (Task 14). Write the code; an end-to-end test arrives with the pipeline integration.

- [ ] **Step 1: Implement `base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VideoSource(ABC):
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return next frame (BGR ndarray) or None if stream ended."""

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height)"""

    def release(self) -> None:
        """Override if cleanup needed."""
```

- [ ] **Step 2: Implement `file.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import VideoSource


class FileSource(VideoSource):
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._w, self._h

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 3: Implement `camera.py`**

```python
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import VideoSource


class CameraSource(VideoSource):
    def __init__(self, device_id: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera id={device_id}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._w, self._h

    def release(self) -> None:
        self._cap.release()
```

- [ ] **Step 4: Update `__init__.py`**

`src/target_tracking/sources/__init__.py`:
```python
from .base import VideoSource
from .camera import CameraSource
from .file import FileSource

__all__ = ["VideoSource", "FileSource", "CameraSource"]
```

- [ ] **Step 5: Commit**

```bash
git add src/target_tracking/sources/
git commit -m "feat(sources): VideoSource ABC with FileSource and CameraSource"
```

---

## Task 6: Tracker abstraction + YOLO tracker

**Files:**
- Create: `src/target_tracking/trackers/base.py`
- Create: `src/target_tracking/trackers/registry.py`
- Create: `src/target_tracking/trackers/yolo.py`
- Modify: `src/target_tracking/trackers/__init__.py`
- Test: `tests/test_yolo_tracker.py` (light test — adapter logic only)

- [ ] **Step 1: Implement `base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.types import Detection


class TrackerModel(ABC):
    @abstractmethod
    def update(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + tracking on frame; return detections with track_id set when tracked."""

    def reset(self) -> None:
        """Override to clear tracker state (e.g. between video files)."""
```

- [ ] **Step 2: Implement `registry.py`**

```python
from .base import TrackerModel
from .._registry import Registry

REGISTRY: Registry[TrackerModel] = Registry("tracker")
register = REGISTRY.register
build = REGISTRY.build
```

- [ ] **Step 3: Write adapter conversion test**

`tests/test_yolo_tracker.py`:
```python
from unittest.mock import MagicMock

import numpy as np

from target_tracking.trackers.yolo import _results_to_detections


def test_results_to_detections_handles_tracked_results():
    # Mimic the structure ultralytics returns from model.track()
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
```

- [ ] **Step 4: Run, expect fail**

Run: `pytest tests/test_yolo_tracker.py -v`
Expected: ImportError.

- [ ] **Step 5: Implement `yolo.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.device import resolve_device
from ..core.types import Detection
from .base import TrackerModel
from .registry import register


def _results_to_detections(results) -> list[Detection]:
    if not results:
        return []
    r = results[0]
    if r.boxes is None:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    if r.boxes.id is None:
        ids = [None] * len(xyxy)
    else:
        ids = r.boxes.id.cpu().numpy().astype(int).tolist()
    names = r.names
    out: list[Detection] = []
    for i, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
        cid = int(cls)
        out.append(
            Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=cid,
                class_name=names.get(cid, str(cid)),
                confidence=float(conf),
                track_id=ids[i] if ids[i] is not None else None,
            )
        )
    return out


@register("yolo")
class YoloTracker(TrackerModel):
    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        classes: Optional[list[int]] = None,
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._tracker_cfg = tracker
        self._conf = conf
        self._iou = iou
        self._classes = classes
        self._device = resolve_device(device)

    def update(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._tracker_cfg,
            conf=self._conf,
            iou=self._iou,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )
        return _results_to_detections(results)

    def reset(self) -> None:
        # Calling track() with persist=False on a dummy frame would clear state;
        # easiest: drop and recreate the predictor on next call.
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.trackers = None
```

- [ ] **Step 6: Update `__init__.py`**

`src/target_tracking/trackers/__init__.py`:
```python
from .base import TrackerModel
from .registry import REGISTRY, build, register
from . import yolo  # noqa: F401  - side-effect: registers "yolo"

__all__ = ["TrackerModel", "REGISTRY", "build", "register"]
```

- [ ] **Step 7: Run, expect pass**

Run: `pytest tests/test_yolo_tracker.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add src/target_tracking/trackers/ tests/test_yolo_tracker.py
git commit -m "feat(trackers): TrackerModel ABC and YoloTracker"
```

---

## Task 7: LineCounter

**Files:**
- Create: `src/target_tracking/counters/base.py`
- Create: `src/target_tracking/counters/registry.py`
- Create: `src/target_tracking/counters/line.py`
- Modify: `src/target_tracking/counters/__init__.py`
- Test: `tests/test_line_counter.py`

- [ ] **Step 1: Implement `base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import FrameResult


class Counter(ABC):
    name: str

    @abstractmethod
    def update(self, result: FrameResult) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict: ...
```

- [ ] **Step 2: Implement `registry.py`**

```python
from .base import Counter
from .._registry import Registry

REGISTRY: Registry[Counter] = Registry("counter")
register = REGISTRY.register
build = REGISTRY.build
```

- [ ] **Step 3: Write failing test**

`tests/test_line_counter.py`:
```python
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
    # Horizontal line y=50, x in [0, 100]
    counter = LineCounter(
        name="gate",
        start=(0, 50),
        end=(100, 50),
        classes_of_interest=["person"],
    )
    # Track 1 crosses downward (above -> below)
    counter.update(_frame(0, [_det(1, 50, 40)]))
    counter.update(_frame(1, [_det(1, 50, 60)]))
    # Track 2 crosses upward (below -> above)
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
    counter.update(_frame(2, [_det(1, 50, 60)]))  # still below; no double count
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
```

- [ ] **Step 4: Run, expect fail**

Run: `pytest tests/test_line_counter.py -v`

- [ ] **Step 5: Implement `line.py`**

```python
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
        # track_id -> last known side (+1 / -1 / 0)
        self._last_side: dict[int, int] = {}
        # per-class in/out counts
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
```

Note on direction semantics: the cross-product sign defines "side". By convention here, crossings from negative side → positive side count as "in"; reverse counts as "out". The line's `start → end` direction is arbitrary; users orient it to match the desired "in" direction.

- [ ] **Step 6: Update `__init__.py`**

```python
from .base import Counter
from .registry import REGISTRY, build, register
from . import line  # noqa: F401

__all__ = ["Counter", "REGISTRY", "build", "register"]
```

- [ ] **Step 7: Run, expect pass**

Run: `pytest tests/test_line_counter.py -v`
Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
git add src/target_tracking/counters/ tests/test_line_counter.py
git commit -m "feat(counters): LineCounter with directional crossing logic"
```

---

## Task 8: Visualization helpers

**Files:**
- Create: `src/target_tracking/visualization/draw.py`

Visualization is best validated by eye later. Keep it simple, no tests.

- [ ] **Step 1: Implement**

`src/target_tracking/visualization/draw.py`:
```python
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Detection

_COLOR_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def _color_for_id(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return _COLOR_PALETTE[track_id % len(_COLOR_PALETTE)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = _color_for_id(det.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = det.class_name
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        label = f"{label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def draw_line(
    frame: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
) -> np.ndarray:
    out = frame  # mutate in place is fine — caller has already copied
    p1 = (int(start[0]), int(start[1]))
    p2 = (int(end[0]), int(end[1]))
    cv2.line(out, p1, p2, (0, 255, 255), 2)
    if label:
        cv2.putText(
            out, label, (p1[0] + 5, p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return out


def draw_counts(frame: np.ndarray, counts: dict) -> np.ndarray:
    out = frame
    y = 25
    for counter_name, snap in counts.items():
        header = f"[{counter_name}] in={snap['total_in']} out={snap['total_out']}"
        cv2.putText(out, header, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
        for cls_name, c in snap.get("per_class", {}).items():
            row = f"  {cls_name}: in={c['in']} out={c['out']}"
            cv2.putText(out, row, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y += 18
    return out
```

- [ ] **Step 2: Commit**

```bash
git add src/target_tracking/visualization/
git commit -m "feat(viz): box, line, and counter overlays"
```

---

## Task 9: Sinks (display, video, stats)

**Files:**
- Create: `src/target_tracking/sinks/base.py`
- Create: `src/target_tracking/sinks/registry.py`
- Create: `src/target_tracking/sinks/display.py`
- Create: `src/target_tracking/sinks/video.py`
- Create: `src/target_tracking/sinks/stats.py`
- Modify: `src/target_tracking/sinks/__init__.py`
- Test: `tests/test_sinks.py`

- [ ] **Step 1: Implement `base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import FrameResult


class Sink(ABC):
    @abstractmethod
    def write(self, result: FrameResult, counts: dict) -> None: ...

    def close(self) -> None:
        """Override to flush/release resources."""
```

- [ ] **Step 2: Implement `registry.py`**

```python
from .base import Sink
from .._registry import Registry

REGISTRY: Registry[Sink] = Registry("sink")
register = REGISTRY.register
build = REGISTRY.build
```

- [ ] **Step 3: Write tests for StatsSink and VideoSink (CSV writing is testable; display window is not)**

`tests/test_sinks.py`:
```python
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
```

- [ ] **Step 4: Run, expect fail**

Run: `pytest tests/test_sinks.py -v`

- [ ] **Step 5: Implement `stats.py`**

```python
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
```

- [ ] **Step 6: Run, expect pass for stats tests**

Run: `pytest tests/test_sinks.py -v`
Expected: 3 passed.

- [ ] **Step 7: Implement `video.py`**

```python
from __future__ import annotations

from pathlib import Path

import cv2

from ..core.types import FrameResult
from ..visualization.draw import draw_counts, draw_detections, draw_line
from .base import Sink
from .registry import register


@register("video")
class VideoSink(Sink):
    def __init__(self, path: str, fps: float = 30.0,
                 overlay_lines: list[dict] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._writer: cv2.VideoWriter | None = None
        self._overlay_lines = overlay_lines or []

    def _ensure_writer(self, shape: tuple[int, int, int]) -> None:
        if self._writer is not None:
            return
        h, w = shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (w, h))

    def write(self, result: FrameResult, counts: dict) -> None:
        self._ensure_writer(result.frame.shape)
        out = draw_detections(result.frame, result.detections)
        for ln in self._overlay_lines:
            out = draw_line(out, ln["start"], ln["end"], ln.get("name"))
        out = draw_counts(out, counts)
        assert self._writer is not None
        self._writer.write(out)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
```

- [ ] **Step 8: Implement `display.py`**

```python
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

from ..core.types import FrameResult
from ..visualization.draw import draw_counts, draw_detections, draw_line
from .base import Sink
from .registry import register


@register("display")
class DisplaySink(Sink):
    def __init__(
        self,
        window_name: str = "TargetTracking",
        show_window: bool = True,
        overlay_lines: list[dict] | None = None,
        on_frame: Optional[Callable[[np.ndarray, dict], None]] = None,
    ) -> None:
        self._window = window_name
        self._show_window = show_window
        self._overlay_lines = overlay_lines or []
        self._on_frame = on_frame

    def write(self, result: FrameResult, counts: dict) -> None:
        out = draw_detections(result.frame, result.detections)
        for ln in self._overlay_lines:
            out = draw_line(out, ln["start"], ln["end"], ln.get("name"))
        out = draw_counts(out, counts)
        if self._show_window:
            cv2.imshow(self._window, out)
            cv2.waitKey(1)
        if self._on_frame is not None:
            self._on_frame(out, counts)

    def close(self) -> None:
        if self._show_window:
            cv2.destroyWindow(self._window)
```

- [ ] **Step 9: Update `__init__.py`**

```python
from .base import Sink
from .registry import REGISTRY, build, register
from . import display, stats, video  # noqa: F401

__all__ = ["Sink", "REGISTRY", "build", "register"]
```

- [ ] **Step 10: Run all sink tests**

Run: `pytest tests/test_sinks.py -v`
Expected: 3 passed.

- [ ] **Step 11: Commit**

```bash
git add src/target_tracking/sinks/ tests/test_sinks.py
git commit -m "feat(sinks): display, video, and stats sinks"
```

---

## Task 10: Pipeline orchestration + integration test

**Files:**
- Create: `src/target_tracking/core/pipeline.py`
- Modify: `src/target_tracking/core/__init__.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write integration test using a MockTracker**

`tests/test_pipeline.py`:
```python
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
        [_det(1, 50, 60)],   # crossing
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
    assert len(sink.calls) == 3   # frames 0, 1, 2
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `pipeline.py`**

```python
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
```

- [ ] **Step 4: Update `core/__init__.py`**

```python
from .pipeline import TrackingPipeline
from .types import Detection, FrameResult

__all__ = ["TrackingPipeline", "Detection", "FrameResult"]
```

- [ ] **Step 5: Run, expect pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/target_tracking/core/pipeline.py src/target_tracking/core/__init__.py tests/test_pipeline.py
git commit -m "feat(core): TrackingPipeline orchestration loop"
```

---

## Task 11: Configuration loading + pipeline builder

**Files:**
- Create: `src/target_tracking/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

`tests/test_config.py`:
```python
from pathlib import Path

import pytest

from target_tracking.config import AppConfig, load_config, build_pipeline


YAML = """
source:
  type: file
  path: tests/fixtures/short.mp4

tracker:
  name: mock_tracker_for_test
  params: {}

counters:
  - name: gate
    type: line
    params:
      start: [0, 50]
      end: [100, 50]
      classes_of_interest: [person]

sinks:
  stats:
    enabled: true
    format: csv
    path: outputs/x.csv
  video:
    enabled: false
  display:
    enabled: false
"""


def test_load_config_parses_yaml(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(YAML)
    cfg = load_config(cfg_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.source.type == "file"
    assert cfg.tracker.name == "mock_tracker_for_test"
    assert cfg.counters[0].type == "line"
    assert cfg.sinks.stats.enabled is True


def test_load_config_rejects_unknown_source_type(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("source:\n  type: nonsense\n")
    with pytest.raises(Exception):
        load_config(cfg_path)


def test_build_pipeline_resolves_registry_names(tmp_path: Path):
    # Register a fake tracker so we don't load YOLO weights in this test
    from target_tracking.trackers import register as tracker_register
    from target_tracking.trackers.base import TrackerModel

    @tracker_register("mock_tracker_for_test")
    class _Mock(TrackerModel):
        def update(self, frame):
            return []

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(YAML)
    cfg = load_config(cfg_path)

    # We need a real video file for FileSource; cheat by switching to a mock source
    cfg.source.type = "mock_source_for_test"

    from target_tracking.sources.base import VideoSource
    from target_tracking._registry import Registry
    import target_tracking.config as config_mod

    class _SrcMock(VideoSource):
        def read(self):
            return None
        @property
        def fps(self): return 30.0
        @property
        def frame_size(self): return (10, 10)

    config_mod.SOURCE_REGISTRY.register("mock_source_for_test")(_SrcMock)

    pipe = build_pipeline(cfg)
    assert pipe is not None
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/test_config.py -v`

- [ ] **Step 3: Implement `config.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from ._registry import Registry
from .core.pipeline import TrackingPipeline
from .counters import build as build_counter
from .sinks import build as build_sink
from .sources.base import VideoSource
from .sources.camera import CameraSource
from .sources.file import FileSource
from .trackers import build as build_tracker


# A registry for sources, populated below
SOURCE_REGISTRY: Registry[VideoSource] = Registry("source")
SOURCE_REGISTRY.register("file")(FileSource)
SOURCE_REGISTRY.register("camera")(CameraSource)


class SourceConfig(BaseModel):
    type: str
    path: Optional[str] = None
    device_id: int = 0


class TrackerConfig(BaseModel):
    name: str = "yolo"
    params: dict[str, Any] = Field(default_factory=dict)


class CounterConfig(BaseModel):
    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class DisplaySinkConfig(BaseModel):
    enabled: bool = True
    window_name: str = "TargetTracking"


class VideoSinkConfig(BaseModel):
    enabled: bool = False
    path: str = "outputs/annotated.mp4"


class StatsSinkConfig(BaseModel):
    enabled: bool = False
    format: Literal["csv", "json"] = "csv"
    path: str = "outputs/counts.csv"


class SinksConfig(BaseModel):
    display: DisplaySinkConfig = Field(default_factory=DisplaySinkConfig)
    video: VideoSinkConfig = Field(default_factory=VideoSinkConfig)
    stats: StatsSinkConfig = Field(default_factory=StatsSinkConfig)


class AppConfig(BaseModel):
    source: SourceConfig
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    counters: list[CounterConfig] = Field(default_factory=list)
    sinks: SinksConfig = Field(default_factory=SinksConfig)


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text())
    return AppConfig.model_validate(data)


def _build_source(cfg: SourceConfig) -> VideoSource:
    if cfg.type == "file":
        if not cfg.path:
            raise ValueError("source.path required for file source")
        return SOURCE_REGISTRY.build("file", path=cfg.path)
    if cfg.type == "camera":
        return SOURCE_REGISTRY.build("camera", device_id=cfg.device_id)
    # Fall through to registry (extension point)
    return SOURCE_REGISTRY.build(cfg.type)


def build_pipeline(cfg: AppConfig) -> TrackingPipeline:
    source = _build_source(cfg.source)
    tracker = build_tracker(cfg.tracker.name, **cfg.tracker.params)

    counters = [
        build_counter(c.type, name=c.name, **c.params) for c in cfg.counters
    ]

    sinks = []
    if cfg.sinks.display.enabled:
        sinks.append(build_sink("display", window_name=cfg.sinks.display.window_name))
    if cfg.sinks.video.enabled:
        sinks.append(build_sink("video", path=cfg.sinks.video.path, fps=source.fps))
    if cfg.sinks.stats.enabled:
        sinks.append(
            build_sink("stats", path=cfg.sinks.stats.path, format=cfg.sinks.stats.format)
        )
    return TrackingPipeline(source=source, tracker=tracker,
                            counters=counters, sinks=sinks)
```

- [ ] **Step 4: Run, expect pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed (one will be skipped if validation behaves differently — adjust expectations).

- [ ] **Step 5: Commit**

```bash
git add src/target_tracking/config.py tests/test_config.py
git commit -m "feat(config): YAML loading + pipeline factory"
```

---

## Task 12: Default configs

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/examples/car_counting.yaml`
- Create: `configs/examples/people_counting.yaml`

- [ ] **Step 1: Create `configs/default.yaml`**

```yaml
source:
  type: file
  path: data/videos/demo.mp4

tracker:
  name: yolo
  params:
    model_name: yolo26n.pt
    tracker: bytetrack.yaml
    conf: 0.25
    iou: 0.7
    classes: [0, 2]
    device: auto

counters:
  - name: main_line
    type: line
    params:
      start: [0, 540]
      end: [1920, 540]
      classes_of_interest: [person, car]

sinks:
  display:
    enabled: true
  video:
    enabled: true
    path: outputs/annotated.mp4
  stats:
    enabled: true
    format: csv
    path: outputs/counts.csv
```

- [ ] **Step 2: Create `configs/examples/car_counting.yaml`**

```yaml
source:
  type: file
  path: data/videos/traffic.mp4

tracker:
  name: yolo
  params:
    model_name: yolo26n.pt
    tracker: bytetrack.yaml
    classes: [2, 5, 7]   # car, bus, truck
    device: auto

counters:
  - name: northbound
    type: line
    params:
      start: [200, 600]
      end: [1700, 600]
      classes_of_interest: [car, bus, truck]

sinks:
  display: { enabled: true }
  video: { enabled: true, path: outputs/cars.mp4 }
  stats: { enabled: true, format: csv, path: outputs/cars.csv }
```

- [ ] **Step 3: Create `configs/examples/people_counting.yaml`**

```yaml
source:
  type: camera
  device_id: 0

tracker:
  name: yolo
  params:
    model_name: yolo26n.pt
    tracker: bytetrack.yaml
    classes: [0]
    device: auto

counters:
  - name: doorway
    type: line
    params:
      start: [300, 400]
      end: [900, 400]
      classes_of_interest: [person]

sinks:
  display: { enabled: true }
  video: { enabled: false }
  stats: { enabled: true, format: json, path: outputs/people.json }
```

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "chore(configs): default and example configs"
```

---

## Task 13: CLI entry point

**Files:**
- Create: `src/target_tracking/cli.py`

- [ ] **Step 1: Implement `cli.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig, build_pipeline, load_config


def _parse_source(value: str):
    """Parse --source 'camera:0' or 'path/to/file.mp4'."""
    if value.startswith("camera:"):
        return {"type": "camera", "device_id": int(value.split(":", 1)[1])}
    return {"type": "file", "path": value}


def _apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source:
        cfg.source = type(cfg.source).model_validate(_parse_source(args.source))
    if args.tracker:
        cfg.tracker.name = args.tracker
    if args.model:
        cfg.tracker.params["model_name"] = args.model
    if args.device:
        cfg.tracker.params["device"] = args.device
    if args.no_display:
        cfg.sinks.display.enabled = False
    if args.no_video:
        cfg.sinks.video.enabled = False
    if args.no_stats:
        cfg.sinks.stats.enabled = False
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="target-tracking")
    sub = parser.add_subparsers(dest="command")

    # Default (no subcommand) = run pipeline
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", type=str, help="path/to/video.mp4 or camera:0")
    parser.add_argument("--tracker", type=str, help="tracker name (e.g. yolo)")
    parser.add_argument("--model", type=str, help="model weights (e.g. yolo26n.pt)")
    parser.add_argument("--device", type=str, help="auto|cpu|cuda|mps")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stats", action="store_true")

    sub.add_parser("ui", help="Launch Gradio Web UI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ui":
        from .webui import launch
        launch()
        return 0
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    pipeline = build_pipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test (manual)**

Run: `python -m target_tracking.cli --help`
Expected: usage message lists `--config`, `--source`, `ui` subcommand.

- [ ] **Step 3: Commit**

```bash
git add src/target_tracking/cli.py
git commit -m "feat(cli): argparse entry point with config override flags"
```

---

## Task 14: Gradio Web UI

**Files:**
- Create: `src/target_tracking/webui.py`

This task is hard to unit test — verify manually after implementing.

- [ ] **Step 1: Implement `webui.py`**

```python
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from .config import (
    AppConfig, CounterConfig, SinksConfig, SourceConfig, StatsSinkConfig,
    TrackerConfig, VideoSinkConfig, DisplaySinkConfig, build_pipeline,
)
from .core.pipeline import TrackingPipeline


_MODEL_CHOICES = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
_TRACKER_CHOICES = ["bytetrack.yaml", "botsort.yaml"]


class _Runner:
    def __init__(self) -> None:
        self.pipeline: Optional[TrackingPipeline] = None
        self.thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_counts: dict = {}
        self.error: Optional[str] = None

    def start(self, cfg: AppConfig) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.pipeline = build_pipeline(cfg)
        self.error = None

        def _run():
            try:
                def on_frame(_result, counts):
                    self.latest_counts = counts

                # Inject the frame capture as a display sink callback
                for s in self.pipeline._sinks:
                    if hasattr(s, "_on_frame"):
                        def cb(frame, counts, self=self):
                            # Convert BGR (OpenCV) to RGB for Gradio
                            self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        s._on_frame = cb
                self.pipeline.run(on_frame=on_frame)
            except Exception as e:  # noqa: BLE001
                self.error = str(e)

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()


def _build_config(
    video_file: str | None,
    use_camera: bool,
    camera_id: int,
    model_name: str,
    tracker_name: str,
    conf: float,
    classes_text: str,
    line_x1: int,
    line_y1: int,
    line_x2: int,
    line_y2: int,
    save_video: bool,
    save_stats: bool,
) -> AppConfig:
    classes = [int(c.strip()) for c in classes_text.split(",") if c.strip()]
    source = (
        SourceConfig(type="camera", device_id=camera_id)
        if use_camera else
        SourceConfig(type="file", path=video_file or "data/videos/demo.mp4")
    )
    return AppConfig(
        source=source,
        tracker=TrackerConfig(
            name="yolo",
            params={
                "model_name": model_name,
                "tracker": tracker_name,
                "conf": conf,
                "classes": classes,
                "device": "auto",
            },
        ),
        counters=[
            CounterConfig(
                name="gate",
                type="line",
                params={
                    "start": [line_x1, line_y1],
                    "end": [line_x2, line_y2],
                    "classes_of_interest": None,
                },
            )
        ],
        sinks=SinksConfig(
            display=DisplaySinkConfig(enabled=True, window_name="webui"),
            video=VideoSinkConfig(enabled=save_video, path="outputs/webui.mp4"),
            stats=StatsSinkConfig(enabled=save_stats, format="csv",
                                  path="outputs/webui.csv"),
        ),
    )


def _format_counts(counts: dict) -> str:
    if not counts:
        return "(no counts yet)"
    lines = []
    for name, snap in counts.items():
        lines.append(f"### {name}  —  in: {snap['total_in']}  out: {snap['total_out']}")
        for cls, c in snap.get("per_class", {}).items():
            lines.append(f"- **{cls}**: in {c['in']} / out {c['out']}")
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    runner = _Runner()

    with gr.Blocks(title="Target Tracking") as demo:
        gr.Markdown("# Target Tracking")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("File"):
                    video_file = gr.File(label="Video file", file_types=["video"])
                with gr.Tab("Camera"):
                    use_camera = gr.Checkbox(value=False, label="Use camera")
                    camera_id = gr.Number(value=0, precision=0, label="Camera id")

                model_name = gr.Dropdown(choices=_MODEL_CHOICES, value="yolo26n.pt",
                                         label="Model")
                tracker_name = gr.Dropdown(choices=_TRACKER_CHOICES,
                                           value="bytetrack.yaml", label="Tracker")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
                classes_text = gr.Textbox(value="0,2",
                                          label="COCO class ids (comma-separated)")

                gr.Markdown("**Counting line** (pixel coords)")
                with gr.Row():
                    line_x1 = gr.Number(value=0, label="x1")
                    line_y1 = gr.Number(value=540, label="y1")
                with gr.Row():
                    line_x2 = gr.Number(value=1920, label="x2")
                    line_y2 = gr.Number(value=540, label="y2")

                save_video = gr.Checkbox(value=True, label="Save annotated video")
                save_stats = gr.Checkbox(value=True, label="Save stats CSV")

                start_btn = gr.Button("Start", variant="primary")
                stop_btn = gr.Button("Stop")
                status = gr.Markdown("Idle.")

            with gr.Column(scale=2):
                live_img = gr.Image(label="Live", streaming=True, type="numpy")
                counts_md = gr.Markdown("(no counts yet)")

        def on_start(file_obj, use_cam, cam_id, model, tracker, c, cls,
                     x1, y1, x2, y2, sv, ss):
            path = file_obj.name if (file_obj is not None and not use_cam) else None
            cfg = _build_config(
                video_file=path, use_camera=use_cam, camera_id=int(cam_id),
                model_name=model, tracker_name=tracker, conf=float(c),
                classes_text=cls,
                line_x1=int(x1), line_y1=int(y1),
                line_x2=int(x2), line_y2=int(y2),
                save_video=sv, save_stats=ss,
            )
            runner.start(cfg)
            return "Running…"

        def on_stop():
            runner.stop()
            return "Stopped."

        def poll():
            if runner.error:
                return None, f"**Error:** {runner.error}"
            return runner.latest_frame, _format_counts(runner.latest_counts)

        start_btn.click(
            on_start,
            inputs=[video_file, use_camera, camera_id, model_name, tracker_name,
                    conf, classes_text, line_x1, line_y1, line_x2, line_y2,
                    save_video, save_stats],
            outputs=[status],
        )
        stop_btn.click(on_stop, outputs=[status])

        # Poll the latest frame ~10x/s
        timer = gr.Timer(0.1)
        timer.tick(poll, outputs=[live_img, counts_md])

    return demo


def launch() -> None:
    build_ui().launch()


if __name__ == "__main__":
    launch()
```

- [ ] **Step 2: Manual smoke test**

Run: `python -m target_tracking.webui`
Expected: Gradio opens a browser at `http://127.0.0.1:7860`, UI renders with left/right panels. Don't need to run inference yet — just verify layout.

- [ ] **Step 3: Commit**

```bash
git add src/target_tracking/webui.py
git commit -m "feat(webui): Gradio interface with parameter panel and live view"
```

---

## Task 15: Model download helper

**Files:**
- Create: `scripts/download_models.py`

- [ ] **Step 1: Implement**

```python
"""Pre-download YOLO weights to local cache. Usage: python scripts/download_models.py"""
from __future__ import annotations

import sys

_DEFAULT_MODELS = ["yolo26n.pt", "yolo26s.pt"]


def main(models: list[str]) -> int:
    from ultralytics import YOLO
    for name in models:
        print(f"Downloading {name} …")
        YOLO(name)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or _DEFAULT_MODELS
    raise SystemExit(main(args))
```

- [ ] **Step 2: Commit**

```bash
git add scripts/download_models.py
git commit -m "chore(scripts): YOLO weights pre-download helper"
```

---

## Task 16: End-to-end smoke test (manual)

After all tasks above, do an end-to-end manual verification.

- [ ] **Step 1: Obtain a test video**

Place any short MP4 in `data/videos/demo.mp4` (e.g., a clip of pedestrians or traffic). Several free public-domain samples exist (e.g., samplelib.com).

- [ ] **Step 2: Download model weights**

Run: `python scripts/download_models.py yolo26n.pt`
Expected: weights cached without errors.

- [ ] **Step 3: Run with default config**

Run: `python -m target_tracking.cli --config configs/default.yaml`
Expected:
- A window opens showing the video with boxes/IDs/count line/overlay text.
- `outputs/annotated.mp4` and `outputs/counts.csv` get created.
- Process exits cleanly at end of video.

- [ ] **Step 4: Test camera path**

Run: `python -m target_tracking.cli --source camera:0 --no-video --no-stats`
Expected: live webcam feed with overlays. Press Ctrl+C to stop.

- [ ] **Step 5: Test Web UI**

Run: `python -m target_tracking.webui`
Browse to printed URL, upload a video, click Start. Expected: frames stream in the right pane, counts update.

- [ ] **Step 6: Run full test suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 7: Commit any final adjustments**

```bash
git add -A
git commit -m "chore: end-to-end smoke test fixes" || true
```

---

## Spec Coverage Check

| Spec section | Covered by |
|---|---|
| 1 Goals (count, pluggable, camera/file, auto device, CLI+UI) | Tasks 3, 5, 6, 7, 10, 13, 14 |
| 3.1 Per-frame data flow | Task 10 (pipeline) |
| 3.2 Component layers | Tasks 5–10 |
| 3.3 Single-abstraction decision | Task 6 (TrackerModel ABC) |
| 4 Project layout | Task 1 + each subsequent task creates its files |
| 5.1 Data structures | Task 2 |
| 5.2 ABCs | Tasks 5, 6, 7, 9 |
| 5.3 Registry pattern | Task 4 (shared `_registry.py`) |
| 5.4 First implementations | Tasks 6, 7, 9 |
| 5.5 Pipeline | Task 10 |
| 6 Configuration | Tasks 11, 12 |
| 7 CLI | Task 13 |
| 8 Gradio Web UI | Task 14 |
| 9 Device auto-detection | Task 3 |
| 10 Testing strategy | Tasks 2, 3, 4, 7, 9, 10, 11 |
| 11 Dependencies | Task 1 (`pyproject.toml`) |
| 12 Out of scope | Documented in spec; not in plan ✓ |

All spec sections accounted for.
