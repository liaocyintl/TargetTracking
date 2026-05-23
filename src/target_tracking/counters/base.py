from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import FrameResult


class Counter(ABC):
    name: str

    @abstractmethod
    def update(self, result: FrameResult) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict: ...
