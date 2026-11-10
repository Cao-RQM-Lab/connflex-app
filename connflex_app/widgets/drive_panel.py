"""Right panel of the electromechanical tab: voltage steps on nets and the run options."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from connflex_core import Structure

from connflex_app.drive import RISE_FRACTION, DrivePlan, default_duration, default_source_electrode, resolve_plan
from connflex_app.electrical import describe_net
from connflex_app import theme
from connflex_app.session import Project
from connflex_app.settings import DriveNet, DriveSettings, TransientSettings
from connflex_app.units import fmt_si, parse_si
from connflex_app.widgets.chips import Collapsible, dot_icon
from connflex_app.widgets.fields import QuantityEdit
from connflex_app.widgets.timeline_panel import net_hex

FIRST_STEP_COLUMN = 2
MODES: tuple[tuple[str, str, str], ...] = (
    ("drive", "Drive", "A voltage source holds this net at the voltage of each step"),
    ("float", "Float", "No source: the net keeps zero charge and its potential follows the bodies"),
)
ROW_HEIGHT = 26


@dataclass
class _Row:
    net: int
    electrode: int
    mode: str = "drive"
    volts: list[float] = field(default_factory=list)


def _item(text: str, editable: bool, align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    if editable:
        flags |= Qt.ItemFlag.ItemIsEditable
    item.setFlags(flags)
    return item


class DrivePanel(QWidget):
    """A table of voltage steps: one column per step, one row per net in the drive."""

    changed = Signal()
    runRequested = Signal()
    equilibriumRequested = Signal(int)
    netSelected = Signal(int)
    stepSelected = Signal(int)
    modelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._structure: Structure | None = None
        self._frequencies = None
        self._durations: list[float | None] = [None]
        self._rows: list[_Row] = []
        self._saved = DriveSettings()
        self._step = 0
        self._rendering = False
        self._busy = False
        self._notes: list[str] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.hint = QLabel("Extract a structure, then click a net in either view to add it to the drive.")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.table = QTableWidget(1, FIRST_STEP_COLUMN + 1)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setToolTip(
            "Each column is one step: its duration, then the voltage every listed net is held at.\n"
            "Voltages ramp from one step to the next over the rise time. Nets that are not listed stay at 0 V.\n"
            "Type values with units and prefixes, such as 40, 2.5 mV, 180 µs, or 0.2m."
        )
        self.table.itemChanged.connect(self._item_changed)
        self.table.currentCellChanged.connect(self._cell_changed)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.add_step_button = QPushButton("+ Step")
        self.add_step_button.setToolTip("Add a step after the selected one, with the same voltages")
        self.add_step_button.clicked.connect(self.add_step)
        self.remove_step_button = QPushButton("− Step")
        self.remove_step_button.setToolTip("Remove the selected step")
        self.remove_step_button.clicked.connect(self.remove_step)
        self.add_net_button = QPushButton("+ Net")
        self.add_net_button.setToolTip("Add a net to the drive (or click it in a view)")
        self.add_net_button.clicked.connect(self._add_net_menu)
        self.remove_net_button = QPushButton("− Net")
        self.remove_net_button.setToolTip("Remove the selected net from the drive; it then stays at 0 V")
        self.remove_net_button.clicked.connect(self.remove_net)
        for button in (self.add_step_button, self.remove_step_button, self.add_net_button, self.remove_net_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.quality = QuantityEdit("", 50.0, positive=True)
        self.quality.setToolTip("Quality factor of every mode: the damping of the run. The library computes no loss mechanism.")
        self.rise = QuantityEdit("s", None, optional=True, positive=True)
        self.rise.setToolTip("Time a source takes to ramp from one step's voltage to the next. Empty: a fiftieth of the shortest step.")
        self.samples = QSpinBox()
        self.samples.setRange(2, 100000)
        self.samples.setValue(200)
        self.samples.setToolTip("Output times per step. Outputs between integration steps cost almost nothing.")
        self.method = QComboBox()
        self.method.addItem("Radau IIA (default)", "radau")
        self.method.addItem("BDF", "bdf")
        self.method.setToolTip("Radau takes fewer, more accurate steps; BDF evaluates the capacitance model about half as often per step.")
        self.settle = QCheckBox("Start settled at step 1")
        self.settle.setToolTip(
            "Find the equilibrium under the voltages of step 1 first and start there, with the sources already at those\n"
            "voltages. Without it the run starts at rest at 0 V and ramps up to step 1."
        )
        form.addRow("Quality factor", self.quality)
        form.addRow("Rise time", self.rise)
        form.addRow("Samples per step", self.samples)
        form.addRow("Integrator", self.method)
        form.addRow("", self.settle)
        layout.addLayout(form)

        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.tolerance = QuantityEdit("", 1e-6, positive=True)
        self.tolerance.setToolTip("Error the integrator allows per step, relative to each state")
        self.contact = QDoubleSpinBox()
        self.contact.setRange(0.01, 0.99)
        self.contact.setDecimals(2)
        self.contact.setSingleStep(0.05)
        self.contact.setValue(0.1)
        self.contact.setToolTip("The run ends when the narrowest gap closes to this fraction of its width at rest")
        self.resistance = QuantityEdit("Ω", 0.0, minimum=0.0)
        self.resistance.setToolTip("Series resistance of every source. Zero holds each net at its voltage.")
        self.max_step = QuantityEdit("s", None, optional=True, positive=True)
        self.max_step.set_automatic("left to error control")
        self.max_step.setToolTip("Longest integration step. Empty leaves step length to error control.")
        advanced_form.addRow("Tolerance", self.tolerance)
        advanced_form.addRow("Contact clearance", self.contact)
        advanced_form.addRow("Source resistance", self.resistance)
        advanced_form.addRow("Longest time step", self.max_step)
        layout.addWidget(Collapsible("Advanced", advanced))

        self.model_label = QLabel("")
        self.model_label.setObjectName("muted")
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(Qt.TextFormat.RichText)
        self.model_label.linkActivated.connect(lambda _link: self.modelRequested.emit())
        layout.addWidget(self.model_label)

        self.run_button = QPushButton("Run transient")
        self.run_button.setObjectName("primary")
        self.run_button.setToolTip("Follow the structure through every step (Ctrl+T)")
        self.run_button.clicked.connect(self.runRequested.emit)
        self.equilibrium_button = QPushButton("Equilibrium of step 1")
        self.equilibrium_button.setToolTip("Where the structure rests under the voltages of the selected step, and how close it is to pull-in")
        self.equilibrium_button.clicked.connect(lambda: self.equilibriumRequested.emit(self.current_step()))
        layout.addWidget(self.run_button)
        layout.addWidget(self.equilibrium_button)
        self.result = QLabel("")
        self.result.setObjectName("muted")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result)
        layout.addStretch(1)

        for edit in (self.quality, self.rise, self.tolerance, self.resistance, self.max_step):
            edit.valueChanged.connect(self._options_changed)
        self.samples.valueChanged.connect(lambda _value: self._options_changed())
        self.contact.valueChanged.connect(lambda _value: self._options_changed())
        self.method.currentIndexChanged.connect(lambda _index: self._options_changed())
        self.settle.toggled.connect(lambda _on: self._options_changed())
        self._render()
        self._update_enabled()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(400, 600)

    # ------------------------------------------------------------------ data
    def set_project(self, project: Project | None) -> None:
        """Take the structure of ``project``; the drive on show finds its nets again in it."""
        saved = self.drive_settings()
        self._structure = None if project is None else project.structure
        self._saved = saved
        self._frequencies = None
        self._resolve(saved)
        self.result.setText("")
        self._render()
        self._update_enabled()

    def set_frequencies(self, frequencies) -> None:
        """Resonant frequencies of the structure, which size the automatic step duration."""
        self._frequencies = frequencies
        self._render()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_enabled()

    def set_model_status(self, text: str) -> None:
        self.model_label.setText(text)

    def set_result(self, text: str) -> None:
        self.result.setText(text)

    def apply(self, drive: DriveSettings, transient: TransientSettings) -> None:
        """Show saved settings. Without a structure the drive is kept until one arrives."""
        self._saved = drive
        self._resolve(drive)
        self._rendering = True
        self.quality.set_value(transient.quality_factor)
        self.rise.set_value(transient.rise_time)
        self.samples.setValue(transient.samples)
        self.method.setCurrentIndex(max(self.method.findData(transient.method), 0))
        self.settle.setChecked(transient.settle_first)
        self.tolerance.set_value(transient.tolerance)
        self.contact.setValue(transient.contact_clearance)
        self.resistance.set_value(transient.source_resistance)
        self.max_step.set_value(transient.max_step)
        self._rendering = False
        self._render()
        self._update_enabled()

    def _resolve(self, drive: DriveSettings) -> None:
        self._durations = list(drive.durations) or [None]
        self._rows = []
        self._notes = []
        structure = self._structure
        if structure is None:
            return
        seen: set[int] = set()
        for saved in drive.nets:
            try:
                electrode = structure.electrode_at(tuple(saved.point), saved.layer)
            except ValueError:
                x, y = saved.point
                self._notes.append(f"a net at ({x * 1e6:.1f}, {y * 1e6:.1f}) µm is no longer there and was dropped")
                continue
            if electrode.net in seen:
                continue
            seen.add(electrode.net)
            volts = ([float(v) for v in saved.volts] + [0.0] * len(self._durations))[: len(self._durations)]
            self._rows.append(_Row(electrode.net, electrode.index, saved.mode if saved.mode in ("drive", "float") else "drive", volts))
        self._step = min(self._step, len(self._durations) - 1)

    def drive_settings(self) -> DriveSettings:
        """The drive on show, each net by a point on its source electrode."""
        structure = self._structure
        if structure is None:
            return self._saved
        nets = []
        for row in self._rows:
            electrode = structure.electrodes[row.electrode]
            point = electrode.region.geoms[0].representative_point()
            nets.append(DriveNet((float(point.x), float(point.y)), electrode.layer, row.mode, [float(v) for v in row.volts]))
        return DriveSettings(durations=list(self._durations), nets=nets)

    def transient_settings(self) -> TransientSettings:
        return TransientSettings(
            quality_factor=float(self.quality.value() or 50.0),
            rise_time=self.rise.value(),
            samples=self.samples.value(),
            method=str(self.method.currentData()),
            tolerance=float(self.tolerance.value() or 1e-6),
            max_step=self.max_step.value(),
            contact_clearance=self.contact.value(),
            source_resistance=float(self.resistance.value() or 0.0),
            settle_first=self.settle.isChecked(),
        )

    def plan(self) -> DrivePlan:
        """The drive resolved on the structure; raises ``ValueError`` with what to fix."""
        if self._structure is None:
            raise ValueError("extract a structure first")
        return resolve_plan(self._structure, self.drive_settings(), self.transient_settings(), self._frequencies)

    def steps(self) -> int:
        return len(self._durations)

    def nets(self) -> list[int]:
        return [row.net for row in self._rows]

    def current_step(self) -> int:
        return min(self._step, len(self._durations) - 1)

    def current_net(self) -> int:
        row = self.table.currentRow() - 1
        return self._rows[row].net if 0 <= row < len(self._rows) else -1

    def step_potentials(self, step: int | None = None) -> dict[int, float]:
        """Voltage by net in a step (the selected one by default), for colouring the views."""
        step = self.current_step() if step is None else step
        return {row.net: row.volts[step] for row in self._rows if row.mode == "drive" and step < len(row.volts)}

    def floating(self) -> list[int]:
        return [row.net for row in self._rows if row.mode == "float"]

    def peak_voltage(self) -> float:
        values = [abs(v) for row in self._rows if row.mode == "drive" for v in row.volts]
        return max(values, default=0.0)

    # --------------------------------------------------------------- editing
    def add_net(self, electrode: int) -> None:
        """Put the net of ``electrode`` in the drive, or select its row if it is already there."""
        structure = self._structure
        if structure is None or not 0 <= electrode < len(structure.electrodes):
            return
        picked = structure.electrodes[electrode]
        for index, row in enumerate(self._rows):
            if row.net == picked.net:
                self._select(index + 1, FIRST_STEP_COLUMN + self.current_step())
                return
        source = electrode if picked.layer == "device" and picked.body == 0 else default_source_electrode(structure, picked.net)
        self._rows.append(_Row(picked.net, electrode if source is None else source, "drive", [0.0] * len(self._durations)))
        self._render()
        self._select(len(self._rows), FIRST_STEP_COLUMN + self.current_step())
        self.table.setFocus()
        self._changed()

    def select_net(self, net: int) -> None:
        for index, row in enumerate(self._rows):
            if row.net == net:
                self._select(index + 1, FIRST_STEP_COLUMN + self.current_step())
                return

    def remove_net(self) -> None:
        row = self.table.currentRow() - 1
        if 0 <= row < len(self._rows):
            del self._rows[row]
            self._render()
            self._changed()
            self.netSelected.emit(self.current_net())

    def add_step(self) -> None:
        at = self.current_step() + 1
        self._durations.insert(at, self._durations[at - 1])
        for row in self._rows:
            row.volts.insert(at, row.volts[at - 1])
        self._step = at
        self._render()
        self._select(max(self.table.currentRow(), 0), FIRST_STEP_COLUMN + at)
        self._changed()

    def remove_step(self) -> None:
        if len(self._durations) <= 1:
            return
        at = self.current_step()
        del self._durations[at]
        for row in self._rows:
            del row.volts[at]
        self._step = min(at, len(self._durations) - 1)
        self._render()
        self._select(max(self.table.currentRow(), 0), FIRST_STEP_COLUMN + self._step)
        self._changed()

    def set_voltage(self, net: int, step: int, volts: float) -> None:
        """Set one cell from code (the smoke test and the tour drive the panel this way)."""
        for row in self._rows:
            if row.net == net and 0 <= step < len(row.volts):
                row.volts[step] = float(volts)
        self._render()
        self._changed()

    def set_duration(self, step: int, seconds: float | None) -> None:
        if 0 <= step < len(self._durations):
            self._durations[step] = seconds
            self._render()
            self._changed()

    def _add_net_menu(self) -> None:
        structure = self._structure
        if structure is None:
            return
        listed = {row.net for row in self._rows}
        menu = QMenu(self)
        for net in range(structure.nets.count):
            if net in listed:
                continue
            electrode = default_source_electrode(structure, net)
            if electrode is None:
                continue
            action = menu.addAction(dot_icon(net_hex(net)), describe_net(structure, net))
            action.triggered.connect(lambda _checked=False, e=electrode: self.add_net(e))
        if menu.isEmpty():
            menu.addAction("Every net is already in the drive").setEnabled(False)
        menu.exec(self.add_net_button.mapToGlobal(self.add_net_button.rect().bottomLeft()))

    # -------------------------------------------------------------- rendering
    def _duration_text(self, duration: float | None) -> str:
        return fmt_si(default_duration(self._frequencies) if duration is None else duration, "s", 4)

    def _render(self) -> None:
        self._rendering = True
        table = self.table
        row_now, column_now = table.currentRow(), table.currentColumn()
        steps = len(self._durations)
        table.clearContents()
        table.setRowCount(1 + len(self._rows))
        table.setColumnCount(FIRST_STEP_COLUMN + steps)
        table.setHorizontalHeaderLabels(["Net", "Mode", *(f"Step {k + 1}" for k in range(steps))])
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(1, 84)
        for column in range(FIRST_STEP_COLUMN, FIRST_STEP_COLUMN + steps):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch if steps <= 3 else QHeaderView.ResizeMode.Interactive)
            if steps > 3:
                table.setColumnWidth(column, 92)
        label = _item("Duration", False, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setToolTip("How long each step lasts")
        table.setItem(0, 0, label)
        table.setItem(0, 1, _item("", False))
        for step, duration in enumerate(self._durations):
            item = _item(self._duration_text(duration), True)
            if duration is None:  # chosen automatically: shown muted and slanted
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
                item.setForeground(QColor(theme.MUTED))
                item.setToolTip("Chosen automatically: eight periods of the lowest resonance. Type a duration to set it.")
            else:
                item.setToolTip("Duration of this step. Clear the cell, or type 'auto', for eight periods of the lowest resonance.")
            table.setItem(0, FIRST_STEP_COLUMN + step, item)
        structure = self._structure
        for index, row in enumerate(self._rows, start=1):
            name = _item(f"N{row.net}", False, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            name.setIcon(dot_icon(net_hex(row.net)))
            if structure is not None:
                electrode = structure.electrodes[row.electrode]
                point = electrode.region.geoms[0].representative_point()
                name.setToolTip(
                    f"{describe_net(structure, row.net)}\nSource bonded to electrode {row.electrode} "
                    f"({electrode.layer}) at ({point.x * 1e6:.1f}, {point.y * 1e6:.1f}) µm"
                )
            table.setItem(index, 0, name)
            combo = QComboBox()
            for key, title, tip in MODES:
                combo.addItem(title, key)
                combo.setItemData(combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
            combo.setCurrentIndex(max(combo.findData(row.mode), 0))
            combo.currentIndexChanged.connect(lambda _index, r=row, c=combo: self._mode_changed(r, str(c.currentData())))
            table.setCellWidget(index, 1, combo)
            table.setItem(index, 1, _item("", False))
            for step in range(steps):
                if row.mode == "float":
                    table.setItem(index, FIRST_STEP_COLUMN + step, _item("—", False, Qt.AlignmentFlag.AlignCenter))
                else:
                    table.setItem(index, FIRST_STEP_COLUMN + step, _item(fmt_si(row.volts[step], "V", 5), True))
        shown = min(max(table.rowCount(), 3), 8)
        table.setFixedHeight(shown * ROW_HEIGHT + header.sizeHint().height() + (16 if steps > 3 else 4))
        if 0 <= row_now < table.rowCount() and 0 <= column_now < table.columnCount():
            table.setCurrentCell(row_now, column_now)
        self._rendering = False
        durations = [default_duration(self._frequencies) if d is None else d for d in self._durations]
        self.rise.set_automatic(f"auto · {fmt_si(RISE_FRACTION * min(durations), 's')}")
        self.equilibrium_button.setText(f"Equilibrium of step {self.current_step() + 1}")
        notes = "".join(f" {note[0].upper()}{note[1:]}." for note in self._notes)
        if self._structure is None:
            self.hint.setText("Extract a structure, then click a net in either view to add it to the drive.")
        elif not self._rows:
            self.hint.setText("Click a net in either view to add it to the drive, then type its voltage in each step." + notes)
        else:
            self.hint.setText("Click a net in a view to add or select it. Nets that are not listed stay at 0 V." + notes)

    def _select(self, row: int, column: int) -> None:
        if 0 <= row < self.table.rowCount() and 0 <= column < self.table.columnCount():
            self.table.setCurrentCell(row, column)

    def _update_enabled(self) -> None:
        ready = self._structure is not None and not self._busy
        self.table.setEnabled(self._structure is not None)
        for button in (self.add_step_button, self.add_net_button):
            button.setEnabled(ready)
        self.remove_step_button.setEnabled(ready and len(self._durations) > 1)
        self.remove_net_button.setEnabled(ready and bool(self._rows))
        self.run_button.setEnabled(ready)
        self.equilibrium_button.setEnabled(ready)

    # -------------------------------------------------------------- reactions
    def _changed(self) -> None:
        self._update_enabled()
        self.changed.emit()

    def _options_changed(self) -> None:
        if not self._rendering:
            self.changed.emit()

    def _render_later(self) -> None:
        """Rebuild the table once the edit or the combo box that asked for it has finished."""
        QTimer.singleShot(0, self._render)

    def _mode_changed(self, row: _Row, mode: str) -> None:
        if self._rendering or row.mode == mode:
            return
        row.mode = mode
        self._render_later()
        self._changed()

    def _refuse(self, item: QTableWidgetItem, message: str) -> None:
        rect = self.table.visualItemRect(item)
        QToolTip.showText(self.table.viewport().mapToGlobal(rect.bottomLeft()), message, self.table)
        self._render_later()

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._rendering:
            return
        step = item.column() - FIRST_STEP_COLUMN
        if step < 0:
            return
        text = item.text().strip()
        if item.row() == 0:
            if not text or text.lower().startswith("auto"):
                value = None
            else:
                try:
                    value = parse_si(text, "s")
                except ValueError as exc:
                    self._refuse(item, str(exc))
                    return
                if value <= 0:
                    self._refuse(item, "a step needs a duration above zero")
                    return
            self._durations[step] = value
        else:
            row = self._rows[item.row() - 1]
            try:
                row.volts[step] = parse_si(text, "V")
            except ValueError as exc:
                self._refuse(item, str(exc))
                return
        self._render_later()
        self._changed()

    def _cell_changed(self, row: int, column: int, _previous_row: int, _previous_column: int) -> None:
        if self._rendering:
            return
        step = column - FIRST_STEP_COLUMN
        if 0 <= step < len(self._durations) and step != self._step:
            self._step = step
            self.equilibrium_button.setText(f"Equilibrium of step {step + 1}")
            self.stepSelected.emit(step)
        self.netSelected.emit(self.current_net())
