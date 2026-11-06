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


HISTORY_SHOWN = 4


class ProgressStatus(QWidget):
    """Shows what a long operation is doing: which stage, how far along, and for how long.

    When an operation ends, its total time joins a summary of recent operations that stays in
    the bar (for example "Extracting 2.3 s · Solving modes 12 ms"); hovering it lists the time
    of every stage. Opening a file or extracting again starts a new summary.
    """

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
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.message)
        layout.addWidget(self.trail, 1)
        layout.addWidget(self.summary, 1)
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
        self._title: str | None = None
        self._history: list[tuple[str, float, list[tuple[str, float]], bool]] = []
        self._show_progress(False)

    # ---------------------------------------------------------------- phases
    def start_phase(self, title: str, stages: tuple[str, ...] = (), reset: bool = False) -> None:
        """Begin ``title``; ``reset`` clears the summary of earlier operations (a new file or extraction)."""
        if reset:
            self._history.clear()
        self._title = title
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

    def finish_phase(self, message: str | None = None, ok: bool = True) -> None:
        """End the current operation and add its timings to the summary."""
        self._timer.stop()
        if self._title is not None:
            now = time.perf_counter()
            if self._current is not None and self._current not in self._done:
                self._done[self._current] = now - self._stage_started  # a stage that was never closed
            stages = [(name, self._done[name]) for name in self._stages if name in self._done]
            self._history = [entry for entry in self._history if entry[0] != self._title]
            self._history.append((self._title, now - self._phase_started, stages, ok))
            self._title = None
            self._current = None
        self._show_progress(False)
        self._render_summary()
        if message is not None:
            self.message.setText(message)

    def history(self) -> list[tuple[str, float, list[tuple[str, float]], bool]]:
        """Recent operations as (title, total seconds, [(stage, seconds)], succeeded), oldest first."""
        return list(self._history)

    def set_message(self, text: str) -> None:
        self.message.setText(text)

    # -------------------------------------------------------------- internals
    def _show_progress(self, on: bool) -> None:
        for widget in (self.trail, self.bar, self.percent, self.elapsed):
            widget.setVisible(on)
        self.summary.setVisible(not on and bool(self._history))

    def _render_summary(self) -> None:
        recent = self._history[-HISTORY_SHOWN:]
        parts, tips = [], []
        for title, total, stages, ok in recent:
            if ok:
                parts.append(f'<span style="color:{theme.MUTED}">{title}</span> {_duration(total)}')
            else:
                parts.append(f'<span style="color:{theme.DANGER}">{title} failed after {_duration(total)}</span>')
            lines = [f"{title}: {_duration(total)}" + ("" if ok else " (failed)")]
            lines += [f"    {name}  {_duration(seconds)}" for name, seconds in stages]
            tips.append("\n".join(lines))
        self.summary.setText(" · ".join(parts))
        self.summary.setToolTip("\n\n".join(tips))
        self.summary.setVisible(bool(recent) and not self.bar.isVisible())

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
