from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import FrameResult


class Sink(ABC):
    @abstractmethod
    def write(self, result: FrameResult, counts: dict) -> None: ...

    def close(self) -> None:
        """Override to flush/release resources."""
