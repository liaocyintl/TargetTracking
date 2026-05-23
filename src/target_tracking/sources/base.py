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
