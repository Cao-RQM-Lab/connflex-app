"""GUI smoke test: drive the real window through the demo and save a screenshot."""

from __future__ import annotations

import os
import time

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from connflex_app.data import demo_config, demo_gds
from connflex_app.selftest import EXPECTED_BEAMS, EXPECTED_BODIES, EXPECTED_FIRST_MODE_HZ


class SmokeTest(QObject):
    """Opens the bundled demo, waits for extraction and the modal solve, screenshots, quits."""

    def __init__(self, window, target: str, timeout: float = 240.0) -> None:
        super().__init__(window)
        self.window = window
        self.target = target
        self.timeout = timeout
        self.exit_code = 1
        self._phase = "open"
        self._started = time.monotonic()
        self._settled_at: float | None = None
        self._gif_path = ""
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start()

    def _finish(self, code: int, message: str) -> None:
        self._timer.stop()
        self.exit_code = code
        print(("SMOKE-TEST OK: " if code == 0 else "SMOKE-TEST FAILED: ") + message)
        QApplication.instance().quit()

    def _tick(self) -> None:
        window = self.window
        if time.monotonic() - self._started > self.timeout:
            self._finish(1, f"timed out in phase {self._phase}")
            return
        if self._phase == "open":
            window.open_path(str(demo_gds()))
            self._phase = "inspect"
        elif self._phase == "inspect":
            if window._info is not None and not window._busy:
                window.setup.apply(window.setup.settings().with_config(str(demo_config())))
                window.extract()
                self._phase = "extract"
        elif self._phase == "extract":
            project = window.session.project
            if project is not None and project.modal is not None and not window._busy:
                window.session.select(1)
                self._settled_at = time.monotonic()
                self._phase = "settle"
        elif self._phase == "settle":
            if self._settled_at is not None and time.monotonic() - self._settled_at > 1.5:
                self._gif_path = os.path.splitext(self.target)[0] + ".gif"
                window.export_gif(self._gif_path, frames=8, fps=20, scale=0.5)
                self._phase = "gif"
        elif self._phase == "gif":
            if not window._busy:
                window.scene.play(False)
                image = window.grab()
                saved = image.save(self.target)
                project = window.session.project
                first = float(project.modal.frequencies_hz[0])
                summary = (
                    f"{len(project.structure.bodies)} bodies, {len(project.structure.beams)} beams, "
                    f"{len(project.modal.frequencies_hz)} modes, first {first / 1e3:.2f} kHz, "
                    f"screenshot {'saved to ' + self.target if saved else 'NOT saved'}"
                )
                problems = []
                if not saved:
                    problems.append("screenshot not saved")
                if not (os.path.isfile(self._gif_path) and os.path.getsize(self._gif_path) > 1000):
                    problems.append("GIF export failed")
                else:
                    summary += f", GIF {os.path.getsize(self._gif_path) / 1e3:.0f} kB"
                if len(project.structure.bodies) != EXPECTED_BODIES or len(project.structure.beams) != EXPECTED_BEAMS:
                    problems.append("unexpected structure")
                if abs(first - EXPECTED_FIRST_MODE_HZ) > 0.01 * EXPECTED_FIRST_MODE_HZ:
                    problems.append(f"first mode {first:.0f} Hz, expected about {EXPECTED_FIRST_MODE_HZ:.0f} Hz")
                self._finish(0 if not problems else 1, summary + ("" if not problems else " · " + "; ".join(problems)))
