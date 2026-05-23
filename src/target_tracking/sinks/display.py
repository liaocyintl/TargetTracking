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
