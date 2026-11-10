"""Right panel: list of resonant modes."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

AMPLITUDE_MIN, AMPLITUDE_MAX = 0.003, 0.3


def amplitude_from_slider(value: int) -> float:
    return AMPLITUDE_MIN * (AMPLITUDE_MAX / AMPLITUDE_MIN) ** (value / 100.0)


def slider_from_amplitude(fraction: float) -> int:
    return int(round(100.0 * math.log(fraction / AMPLITUDE_MIN) / math.log(AMPLITUDE_MAX / AMPLITUDE_MIN)))

from connflex_app.session import Project
from connflex_app.units import fmt_freq


class ModesPanel(QWidget):
    modeSelected = Signal(int)
    solveRequested = Signal()
    amplitudeChanged = Signal(float)
    speedChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hidden = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.summary = QLabel("Extract a structure to solve its modes.")
        self.summary.setObjectName("muted")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Frequency", "Dominant motion"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)
        hint = QLabel("Select a mode to animate it in the 3D view. Space toggles playback.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        controls = QGridLayout()
        layout.addLayout(controls)  # into its parent first: PySide mishandles widgets added to a grid layout that has no parent yet
        controls.setHorizontalSpacing(8)
        self.amplitude_slider = QSlider(Qt.Orientation.Horizontal)
        self.amplitude_slider.setRange(0, 100)
        self.amplitude_slider.setValue(slider_from_amplitude(0.04))
        self.amplitude_slider.setToolTip("Peak displacement of the animation as a fraction of the layout size")
        self.amplitude_value = QLabel()
        self.amplitude_value.setObjectName("muted")
        self.amplitude_value.setMinimumWidth(48)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 30)
        self.speed_slider.setValue(6)
        self.speed_slider.setToolTip("Animation rate in cycles per second")
        self.speed_value = QLabel()
        self.speed_value.setObjectName("muted")
        self.speed_value.setMinimumWidth(48)
        controls.addWidget(QLabel("Amplitude"), 0, 0)
        controls.addWidget(self.amplitude_slider, 0, 1)
        controls.addWidget(self.amplitude_value, 0, 2)
        controls.addWidget(QLabel("Speed"), 1, 0)
        controls.addWidget(self.speed_slider, 1, 1)
        controls.addWidget(self.speed_value, 1, 2)
        self.amplitude_slider.valueChanged.connect(self._amplitude_moved)
        self.speed_slider.valueChanged.connect(self._speed_moved)
        self._amplitude_moved(self.amplitude_slider.value())
        self._speed_moved(self.speed_slider.value())
        self.solve_button = QPushButton("Solve modes")
        self.solve_button.setEnabled(False)
        self.solve_button.clicked.connect(self.solveRequested.emit)
        layout.addWidget(self.solve_button)

    def amplitude(self) -> float:
        return amplitude_from_slider(self.amplitude_slider.value())

    def speed(self) -> float:
        return self.speed_slider.value() / 10.0

    def _amplitude_moved(self, value: int) -> None:
        fraction = amplitude_from_slider(value)
        self.amplitude_value.setText(f"{fraction * 100:.1f} %")
        self.amplitudeChanged.emit(fraction)

    def _speed_moved(self, value: int) -> None:
        self.speed_value.setText(f"{value / 10.0:.1f} Hz")
        self.speedChanged.emit(value / 10.0)

    def set_project(self, project: Project | None) -> None:
        self.table.setRowCount(0)
        if project is None:
            self.summary.setText("Extract a structure to solve its modes.")
            self.solve_button.setEnabled(False)
            return
        structure = project.structure
        movable = len(structure.bodies) - 1
        self.summary.setText(
            f"{movable} movable bod{'y' if movable == 1 else 'ies'}, {len(structure.beams)} beam{'s' if len(structure.beams) != 1 else ''}, "
            f"{6 * movable} degrees of freedom. Solving…"
        )
        self.solve_button.setEnabled(True)

    def set_modal(self, modal, labels: list[str], elapsed: float | None = None, hidden: int = 0, free=()) -> None:
        """List the modes of ``modal`` after the first ``hidden`` ones (rigid-body modes of ``free`` bodies)."""
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(0)
        self._hidden = hidden
        if modal is None:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)
            return
        frequencies = modal.frequencies_hz[hidden:]
        self.table.setRowCount(len(frequencies))
        for row, frequency in enumerate(frequencies):
            number = QTableWidgetItem(str(row + 1))
            number.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            freq_item = QTableWidgetItem(fmt_freq(float(frequency)))
            freq_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 0, number)
            self.table.setItem(row, 1, freq_item)
            self.table.setItem(row, 2, QTableWidgetItem(labels[row] if row < len(labels) else ""))
        self.table.setUpdatesEnabled(True)
        self.table.blockSignals(False)
        count = len(frequencies)
        if count:
            text = f"{count} mode{'s' if count != 1 else ''} from {fmt_freq(float(frequencies[0]))} to {fmt_freq(float(frequencies[-1]))}."
        else:
            text = "Every solved mode is a rigid-body mode of a free body; ask for more modes or fix the suspension."
        if elapsed is not None:
            text += f" Solved in {elapsed * 1000:.0f} ms." if elapsed < 1.0 else f" Solved in {elapsed:.2f} s."
        free = list(free)
        if free:
            names = ", ".join(f"body {b}" for b in free[:8]) + (" …" if len(free) > 8 else "")
            text += (
                f" {len(free)} free bod{'y' if len(free) == 1 else 'ies'} ({names}) ha{'s' if len(free) == 1 else 've'} no beam path "
                f"to the anchor; {hidden} rigid-body mode{'s' if hidden != 1 else ''} at 0 Hz hidden."
            )
        self.summary.setText(text)

    def select_mode(self, index: int) -> None:
        """Select the row of mode ``index`` (an index into the full modal result)."""
        self.table.blockSignals(True)
        row = index - self._hidden
        if row < 0:
            self.table.clearSelection()
        else:
            self.table.selectRow(row)
        self.table.blockSignals(False)

    def current_mode(self) -> int:
        """Index into the full modal result of the selected row, or -1."""
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() + self._hidden if rows else -1

    def _selection_changed(self) -> None:
        self.modeSelected.emit(self.current_mode())
