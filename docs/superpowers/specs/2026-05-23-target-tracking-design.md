# Target Tracking System — Design Spec

**Date:** 2026-05-23
**Status:** Approved (pending user review)

## 1. Goals

A Python-based visual object tracking system that:

- Counts objects (e.g., cars, people) passing through a video.
- Provides a **unified, pluggable interface** for swapping detection/tracking models via configuration parameters.
- Supports **real-time camera input** as well as **video file processing**.
- Auto-detects available compute device (CPU / CUDA / Apple MPS).
- Exposes both a **CLI** and a **Gradio Web UI**.

## 2. Non-Goals

- Training new models. Inference only — we use pre-trained weights.
- Building custom detection or tracking algorithms from scratch. Initially we rely on Ultralytics' built-in pipeline (YOLO + BoT-SORT/ByteTrack).
- Multi-camera fusion, cross-camera Re-ID.
- Web-deployable production service (Gradio is for local/single-user use).

## 3. Architecture

### 3.1 Data flow (per frame)

```
VideoSource.read()  →  frame (np.ndarray, BGR)
        ↓
TrackerModel.update(frame)  →  list[Detection]  (with track_id)
        ↓
FrameResult = (frame_index, timestamp, frame, detections)
        ↓
for counter in counters:  counter.update(result)
counts = {c.name: c.snapshot() for c in counters}
        ↓
for sink in sinks:  sink.write(result, counts)
        ↓
optional: on_frame(result, counts)   # callback for Gradio live push
```

### 3.2 Component layers

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| Entry | `cli.py`, `webui.py` | Argument parsing / UI; build config and start pipeline. |
| Orchestration | `core/pipeline.py::TrackingPipeline` | Drive the per-frame loop, hold component refs. |
| Source | `sources/*` | Read frames from file / camera. |
| Tracker | `trackers/*` | Detection + tracking combined (matches YOLO native API). |
| Counter | `counters/*` | Stateful counting logic (line crossings, regions, etc.). |
| Sink | `sinks/*` | Output side-effects: display, save video, save stats. |
| Viz | `visualization/draw.py` | Draw boxes, IDs, count lines onto a frame. |

### 3.3 Key abstraction decision

**One abstraction over detection+tracking (`TrackerModel`).** YOLO's `model.track()` already integrates both; treating them as one keeps the adapter thin. If we later need decoupled detector+tracker pairs, we add a `CompositeTracker(detector, tracker)` subclass — no change to upstream/downstream code.

## 4. Project Layout

```
TargetTracking/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
│
├── configs/
│   ├── default.yaml
│   └── examples/
│       ├── car_counting.yaml
│       └── people_counting.yaml
│
├── src/target_tracking/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py          # Detection, FrameResult
│   │   ├── device.py         # auto-detect CPU/CUDA/MPS
│   │   └── pipeline.py       # TrackingPipeline
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── file.py
│   │   └── camera.py
│   ├── trackers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── yolo.py
│   ├── counters/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── line.py
│   ├── sinks/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── display.py
│   │   ├── video.py
│   │   └── stats.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── draw.py
│   ├── config.py
│   ├── cli.py
│   └── webui.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_counters.py
│   ├── test_pipeline.py
│   └── fixtures/
│
├── scripts/
│   └── download_models.py
│
├── data/          # gitignored
│   ├── videos/
│   └── models/
│
└── outputs/       # gitignored
```

## 5. Core Interfaces

### 5.1 Data structures (`core/types.py`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None

@dataclass
class FrameResult:
    frame_index: int
    timestamp: float           # seconds
    frame: np.ndarray          # original BGR frame
    detections: list[Detection]
```

### 5.2 Abstract base classes

```python
# sources/base.py
class VideoSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None: ...   # None = stream ended
    @property
    @abstractmethod
    def fps(self) -> float: ...
    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]: ...   # (w, h)
    def release(self) -> None: ...

# trackers/base.py
class TrackerModel(ABC):
    @abstractmethod
    def update(self, frame: np.ndarray) -> list[Detection]: ...
    def reset(self) -> None: ...

# counters/base.py
class Counter(ABC):
    name: str
    @abstractmethod
    def update(self, result: FrameResult) -> None: ...
    @abstractmethod
    def snapshot(self) -> dict: ...

# sinks/base.py
class Sink(ABC):
    @abstractmethod
    def write(self, result: FrameResult, counts: dict) -> None: ...
    def close(self) -> None: ...
```

### 5.3 Registry pattern (trackers / counters / sinks)

Each pluggable layer ships a `registry.py`:

```python
_REGISTRY: dict[str, type] = {}

def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def build(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
```

Adding a new model = new file with `@register("name")` + entry in config. No core changes.

### 5.4 First implementations

- **`trackers/yolo.py::YoloTracker`** — wraps Ultralytics `YOLO(model_name).track(frame, persist=True, tracker=..., conf=..., iou=..., classes=..., device=...)` and converts results to `list[Detection]`. Default model: **`yolo26n.pt`** (latest stable as of 2026-01).
- **`counters/line.py::LineCounter`** — define line via two points; track centroid trajectory per `track_id`; increment in/out counters on line crossing (sign-of-cross-product method). Per-class breakdown.
- **`sinks/display.py::DisplaySink`** — render to an OpenCV window OR yield annotated frames via a callback (used by Gradio).
- **`sinks/video.py::VideoSink`** — `cv2.VideoWriter` writing annotated frames.
- **`sinks/stats.py::StatsSink`** — append per-event rows (frame_index, timestamp, class, track_id, direction) to CSV or JSON Lines.

### 5.5 Pipeline orchestration

```python
class TrackingPipeline:
    def __init__(self, source, tracker, counters, sinks): ...

    def run(self, on_frame=None):
        for idx, frame in enumerate(self._iter()):
            dets = self.tracker.update(frame)
            result = FrameResult(idx, idx / self.source.fps, frame, dets)
            for c in self.counters:
                c.update(result)
            counts = {c.name: c.snapshot() for c in self.counters}
            for s in self.sinks:
                s.write(result, counts)
            if on_frame:
                on_frame(result, counts)
        self._close_all()
```

## 6. Configuration

`configs/default.yaml`:

```yaml
source:
  type: file               # file | camera
  path: data/videos/demo.mp4
  # device_id: 0           # for camera

tracker:
  name: yolo
  params:
    model_name: yolo26n.pt
    tracker: bytetrack.yaml    # or botsort.yaml
    conf: 0.25
    iou: 0.7
    classes: [0, 2]            # COCO: person=0, car=2
    device: auto               # auto | cpu | cuda | mps

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
    format: csv              # csv | json
    path: outputs/counts.csv
```

Config is loaded into Pydantic models (validation) and then handed to a `build_pipeline(config)` factory that resolves all `registry.build(name, **params)` calls.

## 7. CLI

```bash
python -m target_tracking.cli --config configs/default.yaml

# Shortcuts that override config fields:
python -m target_tracking.cli --source video.mp4 --tracker yolo --model yolo26n.pt
python -m target_tracking.cli --source camera:0

# Launch Web UI:
python -m target_tracking.cli ui
# or:
python -m target_tracking.webui
```

CLI flags override matching keys in the config file.

## 8. Gradio Web UI

Layout:

- **Left panel (controls):** source selector (file upload / camera id), model dropdown (yolo26 n/s/m/l/x), tracker dropdown (bytetrack / botsort), class multi-select, "draw counting line" (click two points on a preview frame), output toggles (display / record / stats), Start / Stop / Download buttons.
- **Right panel (live view):** `gr.Image(streaming=True)` showing annotated frames pushed by pipeline's `on_frame` callback; below it, a stats panel showing per-class in/out counts updated every N frames.

Implementation notes:

- Pipeline runs in a background thread; `on_frame` writes the latest annotated frame to a `gr.State` for the streaming image to pull.
- For "click to draw line", capture coordinates via `gr.Image.select` event.

## 9. Device Auto-Detection (`core/device.py`)

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

Passed to `YOLO.track(device=...)`.

## 10. Testing Strategy

- **Unit tests**
  - `LineCounter` — synthetic detection sequences crossing a line in both directions; per-class counts; no double-count for same `track_id`.
  - `Detection` / `FrameResult` dataclass behavior.
  - Registry — register/build/error on unknown name.
- **Integration test**
  - A `MockTracker` returns scripted detection sequences. Pipeline runs end-to-end with a `MemorySink`. Assert counts and sink outputs.
- **Not tested in CI**
  - YOLO inference itself (trust Ultralytics).
  - Gradio UI (manual smoke test).
- Test fixtures: a few-second `.mp4` in `tests/fixtures/`.

## 11. Dependencies (`pyproject.toml` highlights)

```
ultralytics >= 26.0       # YOLO26 + built-in trackers
opencv-python >= 4.10
numpy
pyyaml
pydantic >= 2
gradio >= 5
# dev:
pytest
pytest-cov
ruff
```

Python 3.11.

## 12. Out of Scope / Future Work

- Region (polygon) counter, dwell-time counter.
- Multi-line / multi-region in a single run (architecture already supports it; UI just needs to expose).
- Decoupled `Detector + Tracker` composite (covered by future `CompositeTracker`).
- Cross-camera Re-ID.
- Production deployment / multi-user web service.

## 13. Open Questions

None at design freeze. Any deferred decisions land in the implementation plan.
