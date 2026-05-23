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
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.trackers = None
