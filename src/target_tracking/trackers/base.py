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
