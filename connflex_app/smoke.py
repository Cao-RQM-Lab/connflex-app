"""GUI smoke test: drive the real window through the demo and save a screenshot."""

from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from connflex_app.data import demo_config, demo_gds
from connflex_app.selftest import EXPECTED_BEAMS, EXPECTED_BODIES, EXPECTED_COMB_CAPACITANCE, EXPECTED_FIRST_MODE_HZ

STATOR = (1050e-6, 600e-6)  # a point on the right comb's stator, whose net the test drives
SHUTTLE = (800e-6, 600e-6)  # a point on the proof mass it pulls


class SmokeTest(QObject):
    """Opens the bundled demo and goes through every tab: extraction and modes, capacitance
    between two picked nets, and a two-step transient. Then exports, screenshots, and quits."""

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
        self._gds_path = ""
        self._notes: list[str] = []
        self._problems: list[str] = []
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

    def _gds_layers(self) -> list[tuple[int, int]]:
        try:
            import gdstk

            return sorted(gdstk.read_gds(self._gds_path).layers_and_datatypes())
        except Exception:  # noqa: BLE001 - reported as a failure
            return []

    def start(self) -> None:
        self._timer.start()

    def _finish(self, code: int, message: str) -> None:
        self._timer.stop()
        self.exit_code = code
        print(("SMOKE-TEST OK: " if code == 0 else "SMOKE-TEST FAILED: ") + message)
        QApplication.instance().quit()

    def _electrical(self) -> None:
        """Compute the capacitance model, pick the comb's two nets through the view's own click
        handling, and compare the capacitance the panel reads with the known value."""
        window = self.window
        project = window.session.project
        structure = project.structure
        groups = project.groups
        stator = structure.electrode_at(STATOR, "device")
        shuttle = structure.electrode_at(SHUTTLE, "device")
        window._group_clicked(project.render.group_at(*STATOR), 0)
        window._group_clicked(project.render.group_at(*SHUTTLE), int(window_shift()))
        panel = window.capacitance
        current = panel.grouping()
        a, b = panel.picks()
        if current is None or a < 0 or b < 0 or current.keys[a] != stator.net or current.keys[b] != shuttle.net:
            self._problems.append(f"picking nets in the view selected {a}, {b}")
            return
        from connflex_app.electrical import pair_report

        value = pair_report(project.capacitance.model, current, a, b).value
        self._notes.append(f"C(N{stator.net}, N{shuttle.net}) {value * 1e15:.1f} fF")
        if abs(value - EXPECTED_COMB_CAPACITANCE) > 0.03 * EXPECTED_COMB_CAPACITANCE:
            self._problems.append(f"comb capacitance {value * 1e15:.1f} fF, expected about {EXPECTED_COMB_CAPACITANCE * 1e15:.1f} fF")
        if groups.count != len(structure.electrodes) + structure.nets.count:
            self._problems.append("paint groups do not match the electrodes and nets")

    def _drive(self) -> None:
        """Add the stator's net to the drive by clicking it, set two short steps, and run."""
        window = self.window
        project = window.session.project
        window._group_clicked(project.render.group_at(*STATOR), 0)
        stator = project.structure.electrode_at(STATOR, "device")
        drive = window.drive
        if stator.net not in drive.nets():
            self._problems.append("clicking a net did not add it to the drive")
            return
        drive.add_step()
        for step in range(2):
            drive.set_duration(step, 30e-6)
        drive.set_voltage(stator.net, 0, 40.0)
        drive.set_voltage(stator.net, 1, 0.0)
        drive.samples.setValue(40)
        window.run_transient()

    def _transient(self) -> None:
        window = self.window
        project = window.session.project
        run = project.transient
        if run is None:
            self._problems.append("the transient gave no result")
            return
        shuttle = project.structure.electrode_at(SHUTTLE, "device")
        travel = float(np.abs(run.u[:, shuttle.body, 0]).max())
        self._notes.append(f"transient {len(run.times)} outputs over {run.plan.steps} steps, travel {travel * 1e9:.2f} nm")
        if run.ended_early or not np.isfinite(run.result.displacement).all() or not travel > 1e-10:
            self._problems.append("the transient did not move the shuttle to its last time")
        if not window.scene.has_trajectory():
            self._problems.append("the transient is not on show in the 3D view")

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
            if self._settled_at is not None and time.monotonic() - self._settled_at > 1.0:
                window.set_workspace("electrical")
                window.compute_capacitance()
                self._phase = "capacitance"
        elif self._phase == "capacitance":
            if not window._busy:
                if window.session.project.capacitance is None:
                    self._finish(1, "the capacitance model was not computed")
                    return
                self._electrical()
                window.set_workspace("electromechanical")
                self._drive()
                self._phase = "transient"
        elif self._phase == "transient":
            if not window._busy:
                self._transient()
                self._gds_path = os.path.splitext(self.target)[0] + ".gds"
                window.export_gds_file(self._gds_path, True)
                self._phase = "gds"
        elif self._phase == "gds":
            if not window._busy:
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
                    + ", ".join(self._notes)
                    + f", screenshot {'saved to ' + self.target if saved else 'NOT saved'}"
                )
                problems = list(self._problems)
                if not saved:
                    problems.append("screenshot not saved")
                if not (os.path.isfile(self._gif_path) and os.path.getsize(self._gif_path) > 1000):
                    problems.append("GIF export failed")
                else:
                    summary += f", GIF {os.path.getsize(self._gif_path) / 1e3:.0f} kB"
                layers = self._gds_layers()
                if layers != [(1, 0), (2, 0), (3, 0), (4, 0)]:
                    problems.append(f"GDS export wrote layers {layers}")
                else:
                    summary += f", GDS {os.path.getsize(self._gds_path) / 1e3:.0f} kB on 1/0 2/0 3/0 4/0"
                if len(project.structure.bodies) != EXPECTED_BODIES or len(project.structure.beams) != EXPECTED_BEAMS:
                    problems.append("unexpected structure")
                if abs(first - EXPECTED_FIRST_MODE_HZ) > 0.01 * EXPECTED_FIRST_MODE_HZ:
                    problems.append(f"first mode {first:.0f} Hz, expected about {EXPECTED_FIRST_MODE_HZ:.0f} Hz")
                self._finish(0 if not problems else 1, summary + ("" if not problems else " · " + "; ".join(problems)))


def window_shift() -> int:
    """The Shift modifier as the views report it with a click."""
    from PySide6.QtCore import Qt

    return int(Qt.KeyboardModifier.ShiftModifier.value)
