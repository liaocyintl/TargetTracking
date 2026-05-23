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
