"""Right panel of the electrical tab: the electrostatic model and capacitance between selections."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from connflex_app import paint, theme
from connflex_app.electrical import (
    GROUPINGS,
    PARTS,
    CapacitanceResult,
    Grouping,
    Suggestion,
    couplings,
    grouping,
    member_between,
    member_total,
    mutual_matrix,
    pair_report,
    suggest_lengths,
)
from connflex_app.geometry import DOF_ORDER, DOF_SHORT
from connflex_app.session import Project
from connflex_app.settings import ElectroSettings
from connflex_app.units import fmt_capacitance, fmt_si
from connflex_app.widgets.chips import Collapsible, dot_icon
from connflex_app.widgets.fields import QuantityEdit
from connflex_app.widgets.plot import prefix_for

ROW_HEIGHT = 24
AMBER = "#fbbf24"
CYAN = "#22d3ee"
MATRIX_PARTS: tuple[tuple[str, str], ...] = (("matrix", "Total"), *PARTS)
COUPLING_ROWS = 200
ICON_LIMIT = 400


def _cell(text: str, align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item


def fmt_slope(value: float, factor: float) -> str:
    """A slope as a bare number in the unit ``factor`` stands for; tiny values round to 0."""
    scaled = value / factor
    if not np.isfinite(scaled) or abs(scaled) < 5e-4:
        return "0"
    return f"{scaled:.3g}"


class MatrixModel(QAbstractTableModel):
    """Capacitance between members: mutual off the diagonal, each member's total on it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._matrix = np.zeros((0, 0))
        self._labels: tuple[str, ...] = ()
        self._titles: tuple[str, ...] = ()
        self._peak = 1.0

    def set_matrix(self, matrix: np.ndarray, labels: tuple[str, ...], titles: tuple[str, ...]) -> None:
        self.beginResetModel()
        self._matrix = matrix
        self._labels = labels
        self._titles = titles
        positive = matrix[matrix > 0]
        self._peak = float(positive.max()) if positive.size else 1.0
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008 - Qt API
        return 0 if parent.isValid() else len(self._matrix)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802, B008 - Qt API
        return 0 if parent.isValid() else len(self._matrix)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802 - Qt API
        if role == Qt.ItemDataRole.DisplayRole and 0 <= section < len(self._labels):
            return self._labels[section]
        if role == Qt.ItemDataRole.ToolTipRole and 0 <= section < len(self._titles):
            return self._titles[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        value = float(self._matrix[row, column])
        if role == Qt.ItemDataRole.DisplayRole:
            return "·" if abs(value) < 1e-9 * self._peak else fmt_si(value, "F", 3)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            if row == column:
                return f"{self._titles[row]}\ntotal capacitance to everything else: {fmt_capacitance(value, 5)}"
            return f"{self._labels[row]} ↔ {self._labels[column]}: {fmt_capacitance(value, 5)}"
        if role == Qt.ItemDataRole.BackgroundRole and value > 0:
            strength = max(0.0, 1.0 + np.log10(value / self._peak) / 4.0)  # four decades of shade
            base = QColor(theme.MUTED if row == column else theme.ACCENT)
            base.setAlphaF(0.08 + 0.34 * strength)
            return base
        return None


class CapacitancePanel(QWidget):
    """Sets up the electrostatic model and reads capacitance between two selected members.

    Members are nets, bodies, or electrodes. The first selected member is A (amber in the
    views), the second is B (cyan). A click in a view selects A; a click with Shift, Ctrl, or
    Cmd held selects B.
    """

    computeRequested = Signal()
    picksChanged = Signal()
    settingsChanged = Signal()
    exportReportRequested = Signal()
    exportMatrixRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._result: CapacitanceResult | None = None
        self._suggestion = Suggestion(None, None, None)
        self._groupings: dict[str, Grouping] = {}
        self._a = -1
        self._b = -1
        self._hover = -1
        self._loading = False
        self._busy = False

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

        # --- model
        model_box = QGroupBox("Electrostatic model")
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(6)
        checks = QHBoxLayout()
        self.plates_check = QCheckBox("Plates")
        self.plates_check.setToolTip("Capacitance between device electrodes and the handle beneath: through surviving oxide and across the released gap")
        self.sidewalls_check = QCheckBox("Sidewalls")
        self.sidewalls_check.setToolTip("Capacitance between sidewalls of device electrodes in the layout plane, from a field solution")
        self.fringe_check = QCheckBox("Fringe")
        self.fringe_check.setToolTip("Field above and below the device layer along every wall; gives combs their vertical forces. Needs sidewalls.")
        for check in (self.plates_check, self.sidewalls_check, self.fringe_check):
            check.setChecked(True)
            check.toggled.connect(lambda _on: self._settings_edited())
            checks.addWidget(check)
        checks.addStretch(1)
        model_layout.addLayout(checks)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.element_size = QuantityEdit("m", None, optional=True, positive=True)
        self.element_size.setToolTip("Longest boundary element on conductor outlines. Empty: half the narrowest gap between device electrodes.")
        self.interaction = QuantityEdit("m", None, optional=True, positive=True)
        self.interaction.setToolTip(
            "Device electrodes within this distance interact; farther apart they share no sidewall capacitance.\n"
            "Empty: four times the narrowest gap. Raise it until the capacitance of interest settles."
        )
        self.ambient = QuantityEdit("", 1.0, minimum=1.0)
        self.ambient.setToolTip("Relative permittivity of the medium in released gaps and trenches")
        form.addRow("Element size", self.element_size)
        form.addRow("Interaction distance", self.interaction)
        form.addRow("Ambient εr", self.ambient)
        model_layout.addLayout(form)
        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.dense_cutoff = QuantityEdit("", 10.0, minimum=0.0)
        self.dense_cutoff.setToolTip("Pairs of nets whose facing walls are shorter than this many gaps are solved in one dense field system")
        self.plates_cutoff = QuantityEdit("", 100.0, minimum=0.0)
        self.plates_cutoff.setToolTip("Pairs whose facing walls are at least this many gaps long are taken as parallel plates; between the two cutoffs, the field in windows")
        self.surrogate = QuantityEdit("", 1e-4, optional=True, positive=True, maximum=0.5)
        self.surrogate.set_automatic("off: solve every displacement")
        self.surrogate.setToolTip(
            "With bodies displaced, reuse a nearby field solution expanded to second order while its error stays within\n"
            "this fraction. It makes equilibria and transients on the field method practical; the model at rest is unchanged."
        )
        self.max_elements = QSpinBox()
        self.max_elements.setRange(100, 200000)
        self.max_elements.setSingleStep(1000)
        self.max_elements.setValue(8000)
        self.max_elements.setToolTip("Most boundary elements solved as one dense system; larger problems are solved in windows")
        self.workers = QSpinBox()
        self.workers.setRange(0, 512)
        self.workers.setSpecialValueText("all processors")
        self.workers.setToolTip("Worker processes that solve the field. 1 runs everything in the application itself.")
        advanced_form.addRow("Dense cutoff", self.dense_cutoff)
        advanced_form.addRow("Plates cutoff", self.plates_cutoff)
        advanced_form.addRow("Surrogate tolerance", self.surrogate)
        advanced_form.addRow("Max elements", self.max_elements)
        advanced_form.addRow("Workers", self.workers)
        model_layout.addWidget(Collapsible("Advanced", advanced))
        self.compute_button = QPushButton("Compute capacitance")
        self.compute_button.setObjectName("primary")
        self.compute_button.setEnabled(False)
        self.compute_button.setToolTip("Solve the capacitance between all electrodes at rest (Ctrl+K)")
        self.compute_button.clicked.connect(self.computeRequested.emit)
        model_layout.addWidget(self.compute_button)
        self.status = QLabel("Extract a structure first.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        model_layout.addWidget(self.status)
        layout.addWidget(model_box)
        for edit in (self.element_size, self.interaction, self.ambient, self.dense_cutoff, self.plates_cutoff, self.surrogate):
            edit.valueChanged.connect(self._settings_edited)
        self.max_elements.valueChanged.connect(lambda _value: self._settings_edited())
        self.workers.valueChanged.connect(lambda _value: self._settings_edited())

        # --- selection
        between = QHBoxLayout()
        between.setSpacing(4)
        label = QLabel("Between")
        label.setObjectName("muted")
        between.addWidget(label)
        self.kind_group = QButtonGroup(self)
        self.kind_group.setExclusive(True)
        self.kind_buttons: dict[str, QToolButton] = {}
        for key, title in GROUPINGS:
            button = QToolButton()
            button.setObjectName("chip")
            button.setText(title)
            button.setCheckable(True)
            button.setChecked(key == "net")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.kind_group.addButton(button)
            self.kind_buttons[key] = button
            between.addWidget(button)
            button.toggled.connect(lambda on, k=key: on and self._kind_changed(k))
        between.addStretch(1)
        layout.addLayout(between)

        picks = QGridLayout()
        layout.addLayout(picks)  # into its parent first: PySide mishandles widgets added to a grid layout that has no parent yet
        picks.setHorizontalSpacing(6)
        picks.setVerticalSpacing(6)
        self.a_combo = QComboBox()
        self.b_combo = QComboBox()
        for row, (name, color, combo) in enumerate((("A", AMBER, self.a_combo), ("B", CYAN, self.b_combo))):
            tag = QLabel(name)
            tag.setStyleSheet(f"color: {color}; font-weight: 700;")
            picks.addWidget(tag, row, 0)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(16)
            picks.addWidget(combo, row, 1)
        self.swap_button = QPushButton("Swap")
        self.swap_button.clicked.connect(self._swap)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.set_picks(-1, -1))
        picks.addWidget(self.swap_button, 0, 2)
        picks.addWidget(self.clear_button, 1, 2)
        picks.setColumnStretch(1, 1)
        self.a_combo.currentIndexChanged.connect(lambda _index: self._combo_changed())
        self.b_combo.currentIndexChanged.connect(lambda _index: self._combo_changed())
        self.hint = QLabel("Click in a view to select A; hold Shift or Ctrl (⌘) and click to select B.")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        # --- pair readout
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(4)
        self.pair_title = QLabel("Capacitance")
        self.pair_title.setObjectName("muted")
        self.pair_title.setWordWrap(True)
        self.pair_value = QLabel("—")
        self.pair_value.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.pair_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card_layout.addWidget(self.pair_title)
        card_layout.addWidget(self.pair_value)
        parts = QGridLayout()
        card_layout.addLayout(parts)  # into its parent first: PySide mishandles widgets added to a grid layout that has no parent yet
        parts.setHorizontalSpacing(10)
        parts.setVerticalSpacing(2)
        self.part_values: dict[str, QLabel] = {}
        for column, (key, title) in enumerate(PARTS):
            name = QLabel(title)
            name.setObjectName("muted")
            value = QLabel("—")
            value.setObjectName("value")
            parts.addWidget(name, 0, column)
            parts.addWidget(value, 1, column)
            self.part_values[key] = value
        self.totals = QLabel("")
        self.totals.setObjectName("muted")
        self.totals.setWordWrap(True)
        card_layout.addWidget(self.totals)
        layout.addWidget(card)

        # --- couplings of A
        self.couplings_title = QLabel("Couplings of A")
        self.couplings_title.setObjectName("muted")
        layout.addWidget(self.couplings_title)
        self.couplings = QTableWidget(0, 3)
        self.couplings.setHorizontalHeaderLabels(["To", "Capacitance", "Share"])
        self.couplings.verticalHeader().setVisible(False)
        self.couplings.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.couplings.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.couplings.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.couplings.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.couplings.setShowGrid(False)
        self.couplings.setAlternatingRowColors(True)
        header = self.couplings.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.couplings.setToolTip("Capacitance from A to every other member, largest first. Click a row to make it B.")
        self.couplings.setFixedHeight(5 * ROW_HEIGHT + header.sizeHint().height() + 4)
        self.couplings.itemSelectionChanged.connect(self._coupling_clicked)
        layout.addWidget(self.couplings)

        # --- change with motion
        slopes = QWidget()
        slopes_layout = QVBoxLayout(slopes)
        slopes_layout.setContentsMargins(0, 0, 0, 0)
        slopes_layout.setSpacing(4)
        self.slopes = QTableWidget(0, 7)
        self.slopes.setHorizontalHeaderLabels(["Body", *(DOF_SHORT[name] for name in DOF_ORDER)])
        self.slopes.verticalHeader().setVisible(False)
        self.slopes.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.slopes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.slopes.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.slopes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.slopes.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.slopes.setFixedHeight(4 * ROW_HEIGHT + self.slopes.horizontalHeader().sizeHint().height() + 4)
        self.slopes.setToolTip(
            "How the capacitance on show changes when a body moves: per micrometre along X, Y, Z and per milliradian\n"
            "about X, Y, Z. Half this slope times the voltage squared is the force that voltage puts on the body."
        )
        slopes_layout.addWidget(self.slopes)
        self.slopes_note = QLabel("per µm of travel (X, Y, Z) and per mrad of turn (θ)")
        self.slopes_note.setObjectName("muted")
        slopes_layout.addWidget(self.slopes_note)
        layout.addWidget(Collapsible("Change with motion (∂C/∂q)", slopes, expanded=True))

        # --- matrix
        matrix = QWidget()
        matrix_layout = QVBoxLayout(matrix)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.setSpacing(6)
        self.part_combo = QComboBox()
        for key, title in MATRIX_PARTS:
            self.part_combo.addItem(title if key == "matrix" else f"{title} part", key)
        self.part_combo.currentIndexChanged.connect(lambda _index: self._fill_matrix())
        matrix_layout.addWidget(self.part_combo)
        self.matrix_model = MatrixModel(self)
        self.matrix = QTableView()
        self.matrix.setModel(self.matrix_model)
        self.matrix.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.matrix.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.matrix.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.matrix.horizontalHeader().setDefaultSectionSize(74)
        self.matrix.setMinimumHeight(170)
        self.matrix.setToolTip(
            "Capacitance between every two members with all others held; the diagonal is each member's total\n"
            "capacitance to the others. Click a cell to select its row as A and its column as B."
        )
        self.matrix.clicked.connect(self._matrix_clicked)
        matrix_layout.addWidget(self.matrix)
        exports = QHBoxLayout()
        self.export_matrix_button = QPushButton("Export matrix (CSV)…")
        self.export_matrix_button.clicked.connect(self.exportMatrixRequested.emit)
        self.export_report_button = QPushButton("Export report (JSON)…")
        self.export_report_button.setToolTip("connflex-core's review report: every pair with its parts, nets, clusters, and what was left out")
        self.export_report_button.clicked.connect(self.exportReportRequested.emit)
        exports.addWidget(self.export_matrix_button)
        exports.addWidget(self.export_report_button)
        matrix_layout.addLayout(exports)
        layout.addWidget(Collapsible("Capacitance matrix", matrix, expanded=True))
        layout.addStretch(1)
        self.set_project(None)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(420, 700)

    # ------------------------------------------------------------------ model settings
    def settings(self) -> ElectroSettings:
        return ElectroSettings(
            include_plates=self.plates_check.isChecked(),
            include_sidewalls=self.sidewalls_check.isChecked(),
            include_fringe=self.fringe_check.isChecked(),
            element_size=self.element_size.value(),
            interaction_distance=self.interaction.value(),
            ambient_permittivity=float(self.ambient.value() or 1.0),
            dense_cutoff=float(self.dense_cutoff.value() if self.dense_cutoff.value() is not None else 10.0),
            plates_cutoff=float(self.plates_cutoff.value() if self.plates_cutoff.value() is not None else 100.0),
            surrogate_tolerance=self.surrogate.value(),
            max_elements=self.max_elements.value(),
            workers=self.workers.value() or None,
        )

    def apply(self, settings: ElectroSettings) -> None:
        self._loading = True
        self.plates_check.setChecked(settings.include_plates)
        self.sidewalls_check.setChecked(settings.include_sidewalls)
        self.fringe_check.setChecked(settings.include_fringe)
        self.element_size.set_value(settings.element_size)
        self.interaction.set_value(settings.interaction_distance)
        self.ambient.set_value(settings.ambient_permittivity)
        self.dense_cutoff.set_value(settings.dense_cutoff)
        self.plates_cutoff.set_value(settings.plates_cutoff)
        self.surrogate.set_value(settings.surrogate_tolerance)
        self.max_elements.setValue(settings.max_elements)
        self.workers.setValue(settings.workers or 0)
        self._loading = False
        self._settings_edited(emit=False)

    def _settings_edited(self, emit: bool = True) -> None:
        if self._loading:
            return
        sidewalls = self.sidewalls_check.isChecked()
        self.fringe_check.setEnabled(sidewalls)
        self.element_size.setEnabled(sidewalls)
        self.interaction.setEnabled(sidewalls)
        self._update_status()
        if emit:
            self.settingsChanged.emit()

    # ------------------------------------------------------------------ data
    def set_project(self, project: Project | None) -> None:
        self._project = project
        self._result = None
        self._groupings = {}
        self._a = self._b = self._hover = -1
        if project is not None:
            structure = project.structure
            self._suggestion = suggest_lengths(structure)
            self._groupings = {key: grouping(structure, key) for key, _title in GROUPINGS}
        else:
            self._suggestion = Suggestion(None, None, None)
        gap = self._suggestion
        if project is None:
            self.element_size.set_automatic("auto · half the narrowest gap")
            self.interaction.set_automatic("auto · four narrowest gaps")
        elif gap.gap is None:
            note = "no gap between device electrodes"
            self.element_size.set_automatic(note)
            self.interaction.set_automatic(note)
        else:
            self.element_size.set_automatic(f"auto · {fmt_si(gap.element_size, 'm', 3)}")
            self.interaction.set_automatic(f"auto · {fmt_si(gap.interaction_distance, 'm', 3)}")
        self._fill_combos()
        self._refresh()

    def set_result(self, result: CapacitanceResult | None) -> None:
        self._result = result
        self._refresh()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.compute_button.setEnabled(not busy and self._project is not None)

    def result(self) -> CapacitanceResult | None:
        return self._result

    def is_stale(self) -> bool:
        """Whether the model on show was computed with other settings than the panel holds."""
        return self._result is not None and not self._result.matches(self.settings())

    # ------------------------------------------------------------------ selection
    def kind(self) -> str:
        for key, button in self.kind_buttons.items():
            if button.isChecked():
                return key
        return "net"

    def grouping(self) -> Grouping | None:
        return self._groupings.get(self.kind())

    def picks(self) -> tuple[int, int]:
        return self._a, self._b

    def set_kind(self, kind: str) -> None:
        if kind in self.kind_buttons:
            self.kind_buttons[kind].setChecked(True)

    def set_picks(self, a: int, b: int) -> None:
        """Select members ``a`` and ``b`` of the current grouping (-1 for none)."""
        group = self.grouping()
        count = group.count if group is not None else 0
        self._a = a if 0 <= a < count else -1
        self._b = b if 0 <= b < count and b != self._a else -1
        self._sync_combos()
        self._refresh()
        self.picksChanged.emit()

    def pick_group(self, group: int, second: bool) -> None:
        """Take a click on paint group ``group`` in a view: A, or B when ``second``."""
        current = self.grouping()
        if current is None or self._project is None:
            return
        member = current.member_of_group(group, self._project.groups)
        if second:
            if member >= 0 and member != self._a:
                self.set_picks(self._a, member)
        elif member < 0:
            self.set_picks(-1, -1)
        else:
            self.set_picks(member, self._b if self._b != member else -1)

    def hover_group(self, group: int) -> None:
        current = self.grouping()
        member = -1
        if current is not None and self._project is not None:
            member = current.member_of_group(group, self._project.groups)
        if member != self._hover:
            self._hover = member
            self.picksChanged.emit()

    def describe_group(self, group: int) -> str:
        """Status-bar text for the member under the pointer."""
        current = self.grouping()
        if current is None or self._project is None:
            return ""
        member = current.member_of_group(group, self._project.groups)
        if member < 0:
            return ""
        text = current.titles[member]
        if self._result is not None:
            model = self._result.model
            text += f" · total capacitance {fmt_capacitance(member_total(model, current, member))}"
            if self._a >= 0 and member != self._a:
                text += f" · to {current.labels[self._a]} {fmt_capacitance(member_between(model, current, self._a, member))}"
        return text

    def paint_table(self) -> np.ndarray | None:
        """Colours of every paint group for the views: by member, with A, B, and the hover marked."""
        project, current = self._project, self.grouping()
        if project is None or current is None:
            return None
        groups = project.groups
        table = paint.by_category(groups, current.kind)
        return paint.with_picks(
            table,
            current.paint_groups(self._a, groups),
            current.paint_groups(self._b, groups),
            current.paint_groups(self._hover, groups) if self._hover not in (self._a, self._b) else (),
        )

    # ---------------------------------------------------------------- internals
    def _kind_changed(self, _kind: str) -> None:
        self._a = self._b = self._hover = -1
        self._fill_combos()
        self._refresh()
        self.picksChanged.emit()

    def _fill_combos(self) -> None:
        group = self.grouping()
        icons = group is not None and group.count <= ICON_LIMIT  # thousands of colour dots take seconds to draw
        for combo in (self.a_combo, self.b_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— none —", -1)
            if group is not None:
                for member in range(group.count):
                    if icons:
                        combo.addItem(dot_icon(self._member_hex(group, member)), group.titles[member], member)
                    else:
                        combo.addItem(group.titles[member], member)
            combo.blockSignals(False)
        self._sync_combos()

    def _member_hex(self, group: Grouping, member: int) -> str:
        if group.kind == "body" and group.keys[member] == 0:
            r, g, b = paint.ANCHOR
        else:
            r, g, b = paint.categorical(group.keys[member])
        return QColor.fromRgbF(r, g, b).name()

    def _sync_combos(self) -> None:
        for combo, member in ((self.a_combo, self._a), (self.b_combo, self._b)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(member), 0))
            combo.blockSignals(False)

    def _combo_changed(self) -> None:
        a = self.a_combo.currentData()
        b = self.b_combo.currentData()
        self.set_picks(-1 if a is None else int(a), -1 if b is None else int(b))

    def _swap(self) -> None:
        if self._a >= 0 and self._b >= 0:
            self.set_picks(self._b, self._a)

    def _coupling_clicked(self) -> None:
        rows = self.couplings.selectionModel().selectedRows()
        if not rows:
            return
        item = self.couplings.item(rows[0].row(), 0)
        member = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if member is not None and int(member) != self._b:
            self.set_picks(self._a, int(member))

    def _matrix_clicked(self, index: QModelIndex) -> None:
        if index.isValid():
            self.set_picks(index.row(), index.column() if index.column() != index.row() else -1)

    def _update_status(self) -> None:
        project, result = self._project, self._result
        if project is None:
            self.status.setText("Extract a structure first.")
            return
        structure = project.structure
        base = f"{len(structure.electrodes)} electrodes on {structure.nets.count} nets."
        if result is None:
            gap = self._suggestion.gap
            if gap is not None:
                base += f" Narrowest gap between device electrodes: {fmt_si(gap, 'm', 3)}."
            self.status.setText(base + " Not computed yet.")
            return
        took = f"{result.elapsed * 1000:.0f} ms" if result.elapsed < 1.0 else f"{result.elapsed:.1f} s"
        text = f"{base} {result.summary().capitalize()}, solved in {took}."
        params = result.params
        if params.include_sidewalls and params.element_size is not None and params.interaction_distance is not None:
            text += f" Elements {fmt_si(params.element_size, 'm', 3)}, reach {fmt_si(params.interaction_distance, 'm', 3)}."
        elif self.sidewalls_check.isChecked() and not params.include_sidewalls:
            text += " No two device electrodes are near each other, so there are no sidewalls to solve."
        if self.is_stale():
            text += f' <span style="color:{theme.SELECTION}">Settings changed since: compute again to apply them.</span>'
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setText(text)

    def _refresh(self) -> None:
        """Fill every readout from the current model, grouping, and picks."""
        self._update_status()
        project, result, group = self._project, self._result, self.grouping()
        self.compute_button.setEnabled(not self._busy and project is not None)
        has_model = result is not None and group is not None
        for widget in (self.export_matrix_button, self.export_report_button):
            widget.setEnabled(has_model)
        self.swap_button.setEnabled(self._a >= 0 and self._b >= 0)
        self.clear_button.setEnabled(self._a >= 0 or self._b >= 0)
        self._fill_matrix()
        self.couplings.blockSignals(True)
        self.couplings.setRowCount(0)
        self.slopes.setRowCount(0)
        for value in self.part_values.values():
            value.setText("—")
        self.totals.setText("")
        if not has_model:
            self.pair_title.setText("Capacitance")
            self.pair_value.setText("—")
            self.couplings_title.setText("Couplings of A")
            self.couplings.blockSignals(False)
            return
        model = result.model
        a, b = self._a, self._b
        if a < 0:
            self.pair_title.setText("Select a member in a view or above to read its capacitance.")
            self.pair_value.setText("—")
            self.couplings_title.setText("Couplings of A")
            self.couplings.blockSignals(False)
            return
        report = pair_report(model, group, a, b if b >= 0 else a)
        if b >= 0:
            self.pair_title.setText(f"Between {group.labels[a]} and {group.labels[b]}, every other {group.kind} held")
            self.totals.setText(
                f"Total of {group.labels[a]}: {fmt_capacitance(report.total_a)}  ·  total of {group.labels[b]}: {fmt_capacitance(report.total_b)}"
            )
        else:
            self.pair_title.setText(f"Total of {group.labels[a]} to every other {group.kind}")
            self.totals.setText(group.titles[a])
        self.pair_value.setText(fmt_capacitance(report.value, 5))
        for key, _title in PARTS:
            self.part_values[key].setText(fmt_capacitance(report.parts[key], 3) if abs(report.parts[key]) > 0 else "0")
        rows = couplings(model, group, a)
        self.couplings_title.setText(f"Couplings of {group.labels[a]}" + (f" (largest {COUPLING_ROWS} of {len(rows)})" if len(rows) > COUPLING_ROWS else ""))
        total = report.total_a if report.total_a > 0 else 1.0
        shown = rows[:COUPLING_ROWS]
        self.couplings.setRowCount(len(shown))
        for row, (member, value) in enumerate(shown):
            name = _cell(group.titles[member], Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            name.setIcon(dot_icon(self._member_hex(group, member)))
            name.setData(Qt.ItemDataRole.UserRole, member)
            self.couplings.setItem(row, 0, name)
            self.couplings.setItem(row, 1, _cell(fmt_capacitance(value)))
            self.couplings.setItem(row, 2, _cell(f"{100.0 * value / total:.1f} %"))
            if member == b:
                self.couplings.selectRow(row)
        self.couplings.blockSignals(False)
        bodies = report.by_body()[:12]
        self.slopes.setRowCount(len(bodies))
        # One unit for the whole table: farads per micrometre of travel and per milliradian of turn.
        travel = max((float(np.abs(six[:3]).max()) for _body, six in bodies), default=0.0) * 1e-6
        turn = max((float(np.abs(six[3:]).max()) for _body, six in bodies), default=0.0) * 1e-3
        travel_factor, travel_prefix = prefix_for(travel)
        turn_factor, turn_prefix = prefix_for(turn)
        for row, (body, six) in enumerate(bodies):
            self.slopes.setItem(row, 0, _cell(f"B{body}", Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
            for column in range(6):
                rotational = column >= 3
                factor = turn_factor * 1e3 if rotational else travel_factor * 1e6
                item = _cell(fmt_slope(float(six[column]), factor))
                unit = "F/rad" if rotational else "F/m"
                item.setToolTip(f"Body {body}: ∂C/∂{DOF_SHORT[DOF_ORDER[column]]} = {six[column]:.5e} {unit}")
                self.slopes.setItem(row, column + 1, item)
        if not bodies:
            self.slopes_note.setText("No movable body changes this capacitance.")
        else:
            self.slopes_note.setText(f"X, Y, Z in {travel_prefix}F per µm of travel · θ in {turn_prefix}F per mrad of turn")

    def _fill_matrix(self) -> None:
        result, group = self._result, self.grouping()
        if result is None or group is None:
            self.matrix_model.set_matrix(np.zeros((0, 0)), (), ())
            return
        part = str(self.part_combo.currentData())
        self.matrix_model.set_matrix(mutual_matrix(result.model, group, part), group.labels, group.titles)

    def matrix_csv(self) -> str:
        """The matrix on show as CSV text, in farads."""
        result, group = self._result, self.grouping()
        if result is None or group is None:
            return ""
        matrix = mutual_matrix(result.model, group, str(self.part_combo.currentData()))
        lines = [",".join(["", *group.labels])]
        for label, row in zip(group.labels, matrix):
            lines.append(",".join([label, *(f"{value:.9e}" for value in row)]))
        return "\n".join(lines) + "\n"
