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
        writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter for {self._path} (codec mp4v unavailable?)"
            )
        self._writer = writer

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
