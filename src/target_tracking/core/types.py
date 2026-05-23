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
