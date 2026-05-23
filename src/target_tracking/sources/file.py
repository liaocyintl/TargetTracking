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
