"""Background execution with progress relayed to the GUI thread."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


class ProgressRelay(QObject):
    """A connflex-core ``Progress`` receiver that forwards stage events as Qt signals.

    It is called from the worker thread; the signals cross to the GUI thread. ``advance``
    is throttled so a stage with thousands of steps does not flood the event loop.
    """

    stageStarted = Signal(str, object)
    advanced = Signal(int)
    stageFinished = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stage = ""
        self._count = 0
        self._last = 0.0

    def start(self, stage: str, total: int | None) -> None:
        self._stage = stage
        self._count = 0
        self._last = time.monotonic()
        self.stageStarted.emit(stage, total)

    def advance(self, n: int = 1) -> None:
        self._count += n
        now = time.monotonic()
        if now - self._last >= 0.05:
            self._last = now
            self.advanced.emit(self._count)

    def finish(self) -> None:
        self.advanced.emit(self._count)
        self.stageFinished.emit(self._stage)


class Worker(QThread):
    """Run a callable off the GUI thread and report its result or failure."""

    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # noqa: D401 - QThread API
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(str(exc) or exc.__class__.__name__, traceback.format_exc())
            return
        self.succeeded.emit(result)
