"""Status-bar progress: phase title, stage trail with timings and counts, overall bar, elapsed time."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from connflex_app import theme


def _duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} min {seconds % 60:.0f} s"


class ProgressStatus(QWidget):
    """Shows what a long operation is doing: which stage, how far along, and for how long."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(12)
        self.message = QLabel("Ready")
        self.trail = QLabel()
        self.trail.setTextFormat(Qt.TextFormat.RichText)
        self.bar = QProgressBar()
        self.bar.setFixedWidth(200)
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.percent = QLabel()
        self.percent.setObjectName("muted")
        self.percent.setMinimumWidth(36)
        self.elapsed = QLabel()
        self.elapsed.setObjectName("muted")
        self.elapsed.setMinimumWidth(64)
        layout.addWidget(self.message)
        layout.addWidget(self.trail, 1)
        layout.addWidget(self.bar)
        layout.addWidget(self.percent)
        layout.addWidget(self.elapsed)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._stages: list[str] = []
        self._done: dict[str, float] = {}
        self._current: str | None = None
        self._total: int | None = None
        self._count = 0
        self._phase_started = 0.0
        self._stage_started = 0.0
        self._show_progress(False)

    # ---------------------------------------------------------------- phases
    def start_phase(self, title: str, stages: tuple[str, ...] = ()) -> None:
        self.message.setText(f"<b>{title}</b>")
        self._stages = list(stages)
        self._done = {}
        self._current = None
        self._total = None
        self._count = 0
        self._phase_started = time.perf_counter()
        self._show_progress(True)
        self._timer.start()
        self._refresh()

    def stage_started(self, name: str, total) -> None:
        if name not in self._stages:
            self._stages.append(name)
        self._current = name
        self._total = int(total) if total else None
        self._count = 0
        self._stage_started = time.perf_counter()
        self._refresh()

    def advanced(self, count: int) -> None:
        self._count = count
        self._refresh()

    def stage_finished(self, name: str) -> None:
        self._done[name] = time.perf_counter() - self._stage_started
        if self._current == name:
            self._current = None
        self._refresh()

    def finish_phase(self, message: str | None = None) -> None:
        self._timer.stop()
        self._show_progress(False)
        if message is not None:
            self.message.setText(message)

    def set_message(self, text: str) -> None:
        self.message.setText(text)

    # -------------------------------------------------------------- internals
    def _show_progress(self, on: bool) -> None:
        for widget in (self.trail, self.bar, self.percent, self.elapsed):
            widget.setVisible(on)

    def _refresh(self) -> None:
        now = time.perf_counter()
        parts = []
        timings = []
        for name in self._stages:
            if name in self._done:
                # Finished stages are folded to their name; their timings live in the tooltip.
                parts.append(f'<span style="color:{theme.MUTED}">{name}</span>')
                timings.append(f"{name}: {_duration(self._done[name])}")
            elif name == self._current:
                detail = ""
                if self._total:
                    detail = f" {self._count:,}/{self._total:,}"
                elif self._count:
                    detail = f" {self._count:,}"
                parts.append(
                    f'<span style="color:{theme.TEXT}"><b>{name}</b>{detail} · {_duration(now - self._stage_started)}</span>'
                )
            else:
                parts.append(f'<span style="color:#4b5160">{name}</span>')
        self.trail.setText(" · ".join(parts))
        if timings:
            self.trail.setToolTip("\n".join(timings) + f"\nfinished stages: {_duration(sum(self._done.values()))}")
        else:
            self.trail.setToolTip("")
        stages = max(len(self._stages), 1)
        fraction = len(self._done)
        if self._current is not None and self._total:
            fraction += min(self._count / self._total, 1.0)
        fraction = min(fraction / stages, 1.0)
        self.bar.setValue(int(fraction * 1000))
        self.percent.setText(f"{fraction * 100:.0f} %")
        self.elapsed.setText(_duration(now - self._phase_started))
