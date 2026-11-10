"""Bottom panel of the electromechanical tab: a transient over time, with playback controls."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from connflex_core import Structure

from connflex_app import paint
from connflex_app.drive import TransientRun
from connflex_app.geometry import DOF_SHORT
from connflex_app.units import fmt_si
from connflex_app.widgets.modes_panel import amplitude_from_slider, slider_from_amplitude
from connflex_app.widgets.plot import Series, TimePlot

QUANTITIES: tuple[tuple[str, str], ...] = (
    ("displacement", "Displacement of a body"),
    ("rotation", "Rotation of a body"),
    ("velocity", "Velocity of a body"),
    ("clearance", "Narrowest gap"),
    ("current", "Source currents"),
    ("charge", "Charge on driven nets"),
    ("energy", "Energy"),
)
RATES: tuple[tuple[str, float], ...] = (("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0))


def net_hex(net: int) -> str:
    """Colour of a net in tables and plots, as ``#rrggbb``."""
    r, g, b = paint.categorical(net)
    return QColor.fromRgbF(r, g, b).name()


class TimelinePanel(QWidget):
    """Plots of a transient with a cursor tied to the 3D playback."""

    scrubbed = Signal(float)
    playRequested = Signal(bool)
    rateChanged = Signal(float)
    amplitudeChanged = Signal(float)
    trueScaleChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run: TransientRun | None = None
        self._structure: Structure | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(12)

        side = QWidget()
        side.setFixedWidth(270)
        controls = QVBoxLayout(side)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        row = QHBoxLayout()
        self.play_button = QPushButton("▶  Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)
        self.play_button.setToolTip("Play or pause the transient in the 3D view (Space)")
        self.play_button.toggled.connect(self._play_toggled)
        self.rate_combo = QComboBox()
        for label, rate in RATES:
            self.rate_combo.addItem(label, rate)
        self.rate_combo.setCurrentIndex(2)
        self.rate_combo.setToolTip("Playback speed: at 1× one pass through the run takes eight seconds")
        self.rate_combo.currentIndexChanged.connect(lambda _index: self.rateChanged.emit(float(self.rate_combo.currentData())))
        row.addWidget(self.play_button, 1)
        row.addWidget(self.rate_combo)
        controls.addLayout(row)
        self.time_label = QLabel("No transient yet")
        self.time_label.setObjectName("value")
        controls.addWidget(self.time_label)

        grid = QGridLayout()
        controls.addLayout(grid)  # into its parent first: PySide mishandles widgets added to a grid layout that has no parent yet
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.quantity_combo = QComboBox()
        for key, title in QUANTITIES:
            self.quantity_combo.addItem(title, key)
        self.quantity_combo.currentIndexChanged.connect(lambda _index: self._refresh_quantity())
        self.body_combo = QComboBox()
        self.body_combo.currentIndexChanged.connect(lambda _index: self._refresh_quantity())
        self.amplitude_slider = QSlider(Qt.Orientation.Horizontal)
        self.amplitude_slider.setRange(0, 100)
        self.amplitude_slider.setValue(slider_from_amplitude(0.04))
        self.amplitude_slider.setToolTip("How far the largest motion of the run is drawn, as a fraction of the layout size")
        self.amplitude_slider.valueChanged.connect(lambda value: self.amplitudeChanged.emit(amplitude_from_slider(value)))
        self.true_scale = QCheckBox("True scale")
        self.true_scale.setToolTip("Draw the motion at its real size instead of magnified")
        self.true_scale.toggled.connect(self._true_scale_toggled)
        grid.addWidget(QLabel("Plot"), 0, 0)
        grid.addWidget(self.quantity_combo, 0, 1, 1, 2)
        grid.addWidget(QLabel("Body"), 1, 0)
        grid.addWidget(self.body_combo, 1, 1, 1, 2)
        grid.addWidget(QLabel("Motion"), 2, 0)
        grid.addWidget(self.amplitude_slider, 2, 1)
        grid.addWidget(self.true_scale, 2, 2)
        grid.setColumnStretch(1, 1)
        self.summary = QLabel("Run a transient from the Drive panel to see motion and voltages over time.")
        self.summary.setObjectName("muted")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.summary, 1)
        root.addWidget(side)

        plots = QVBoxLayout()
        plots.setSpacing(2)
        self.voltage_plot = TimePlot(show_time_axis=False)
        self.quantity_plot = TimePlot(show_time_axis=True)
        self.voltage_plot.clear("Voltages of the driven nets appear here")
        self.quantity_plot.clear("Motion over time appears here")
        self.voltage_plot.scrubbed.connect(self.scrubbed.emit)
        self.quantity_plot.scrubbed.connect(self.scrubbed.emit)
        plots.addWidget(self.voltage_plot, 2)
        plots.addWidget(self.quantity_plot, 3)
        root.addLayout(plots, 1)

    # ------------------------------------------------------------------ data
    def set_run(self, run: TransientRun | None, structure: Structure | None) -> None:
        self._run = run
        self._structure = structure
        self.body_combo.blockSignals(True)
        self.body_combo.clear()
        if run is None or structure is None:
            self.body_combo.blockSignals(False)
            self.play_button.setEnabled(False)
            self.set_playing(False)
            self.time_label.setText("No transient yet")
            self.summary.setText("Run a transient from the Drive panel to see motion and voltages over time.")
            self.voltage_plot.clear("Voltages of the driven nets appear here")
            self.quantity_plot.clear("Motion over time appears here")
            return
        for body, name in run.result.dofs:
            if name == "ux":
                self.body_combo.addItem(f"Body {body}", body)
        index = self.body_combo.findData(run.dominant_body())
        self.body_combo.setCurrentIndex(max(index, 0))
        self.body_combo.blockSignals(False)
        self.play_button.setEnabled(True)
        self.summary.setText(run.summary(structure))
        plan = run.plan
        labels = [f"step {k + 1}" for k in range(plan.steps)]
        times = run.times
        shown = list(plan.nets) if plan.nets else []
        series = [
            Series(f"N{drive.net}" + (" (floating)" if drive.mode == "float" else ""), run.source_potential(drive), net_hex(drive.net), drive.mode == "float")
            for drive in shown
        ]
        if not series:
            series = [Series("all nets at 0 V", np.zeros(len(times)), "#8d94a5")]
        self.voltage_plot.set_data(times, series, "V", "potential")
        self.voltage_plot.set_marks(plan.edges, labels)
        self.quantity_plot.set_marks(plan.edges, [])
        self._refresh_quantity()
        self.set_time(float(times[0]))

    def set_time(self, moment: float) -> None:
        """Move the cursor of both plots to simulation time ``moment``."""
        run = self._run
        if run is None:
            return
        self.voltage_plot.set_cursor_time(moment)
        self.quantity_plot.set_cursor_time(moment)
        plan = run.plan
        step = plan.step_at(moment)
        self.time_label.setText(f"t = {fmt_si(moment, 's', 4)}  ·  step {step + 1} of {plan.steps}")

    def set_playing(self, on: bool) -> None:
        self.play_button.blockSignals(True)
        self.play_button.setChecked(on)
        self.play_button.setText("❚❚  Pause" if on else "▶  Play")
        self.play_button.blockSignals(False)

    def set_amplitude(self, fraction: float) -> None:
        self.amplitude_slider.blockSignals(True)
        self.amplitude_slider.setValue(slider_from_amplitude(fraction))
        self.amplitude_slider.blockSignals(False)

    def amplitude(self) -> float:
        return amplitude_from_slider(self.amplitude_slider.value())

    def rate(self) -> float:
        return float(self.rate_combo.currentData())

    def quantity(self) -> str:
        return str(self.quantity_combo.currentData())

    # -------------------------------------------------------------- internals
    def _play_toggled(self, on: bool) -> None:
        self.play_button.setText("❚❚  Pause" if on else "▶  Play")
        self.playRequested.emit(on)

    def _true_scale_toggled(self, on: bool) -> None:
        self.amplitude_slider.setEnabled(not on)
        self.trueScaleChanged.emit(on)

    def _refresh_quantity(self) -> None:
        run, structure = self._run, self._structure
        if run is None or structure is None:
            return
        result = run.result
        key = self.quantity()
        body = self.body_combo.currentData()
        per_body = key in ("displacement", "rotation", "velocity")
        self.body_combo.setEnabled(per_body and self.body_combo.count() > 0)
        times = run.times
        series: list[Series] = []
        unit, title = "", ""
        if per_body:
            if body is None:
                self.quantity_plot.clear("This structure has no movable body")
                return
            try:
                first = result.dofs.index((int(body), "ux"))
            except ValueError:
                self.quantity_plot.clear("This body has no degrees of freedom")
                return
            source = result.velocity if key == "velocity" else result.displacement
            offset = 3 if key == "rotation" else 0
            names = ("tx", "ty", "tz") if key == "rotation" else ("ux", "uy", "uz")
            series = [Series(DOF_SHORT[name], source[:, first + offset + k]) for k, name in enumerate(names)]
            unit = {"displacement": "m", "rotation": "rad", "velocity": "m/s"}[key]
            title = f"body {body} {key}"
        elif key == "clearance":
            series = [Series("narrowest gap / at rest", result.clearance)]
            title = "1 at rest, 0 at contact"
        elif key == "current":
            driven = run.plan.driven
            series = [Series(f"N{drive.net}", result.current[:, k], net_hex(drive.net)) for k, drive in enumerate(driven)]
            unit, title = "A", "current each source drives"
            if not series:
                self.quantity_plot.clear("No net is driven by a source")
                return
        elif key == "charge":
            electrodes = structure.electrodes
            for drive in run.plan.nets:
                mine = [e.index for e in electrodes if e.net == drive.net]
                series.append(Series(f"N{drive.net}", result.charge[:, mine].sum(axis=1), net_hex(drive.net)))
            unit, title = "C", "charge on each net"
            if not series:
                self.quantity_plot.clear("No net is in the drive")
                return
        else:
            energy = result.energy
            series = [
                Series("kinetic", energy.kinetic),
                Series("strain", energy.strain),
                Series("electrostatic", energy.electrostatic - energy.electrostatic[0]),
                Series("damped", energy.damped),
                Series("supplied", energy.supplied, dashed=True),
            ]
            if np.abs(energy.resistive).max() > 0:
                series.append(Series("resistive", energy.resistive))
            unit, title = "J", "energy since the start"
        self.quantity_plot.set_data(times, series, unit, title)
        self.quantity_plot.set_marks(run.plan.edges, [])
