"""Right panel: stiffness of one body under a boundary condition on the others."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from connflex_app.analysis import CONDITIONS, StiffnessReport, condensed_stiffness, free_bodies
from connflex_app.geometry import DOF_LONG, DOF_ORDER, DOF_SHORT
from connflex_app.session import Project
from connflex_app.units import fmt_freq, fmt_mass, fmt_point, fmt_stiffness
from connflex_app.widgets.chips import Collapsible


ROW_HEIGHT = 26


def _compact(value: float) -> str:
    """Short scientific notation that fits a narrow cell, e.g. ``1.7e5`` or ``-2.0e-4``."""
    if value == 0 or not np.isfinite(value):
        return "0" if value == 0 else "—"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    if abs(mantissa) >= 9.95:
        mantissa /= 10
        exponent += 1
    return f"{mantissa:.1f}e{exponent}"


def _cell(text: str, align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    return item


class StiffnessPanel(QWidget):
    bodySelected = Signal(int)
    deflectionRequested = Signal(int, int, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project | None = None
        self._report: StiffnessReport | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.body_combo = QComboBox()
        self.body_combo.currentIndexChanged.connect(self._body_changed)
        self.condition_combo = QComboBox()
        for key, label in CONDITIONS:
            self.condition_combo.addItem(label, key)
        self.condition_combo.currentIndexChanged.connect(self._condition_changed)
        form.addRow("Body", self.body_combo)
        form.addRow("Condition", self.condition_combo)
        layout.addLayout(form)

        self.fixed_list = QListWidget()
        self.fixed_list.setMaximumHeight(110)
        self.fixed_list.setVisible(False)
        self.fixed_list.itemChanged.connect(lambda _item: self._recompute())
        layout.addWidget(self.fixed_list)

        self.info = QLabel("Click a body in either view, or pick one above.")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.table = QTableWidget(6, 4)
        self.table.setHorizontalHeaderLabels(["DOF", "Direct k", "Effective k", "1-DOF f"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.setToolTip(
            "Direct: force per unit motion with this body's other degrees of freedom locked (diagonal of K).\n"
            "Effective: force per unit motion with its other degrees of freedom free (1 / diagonal of K⁻¹).\n"
            "1-DOF f: resonance estimate from the effective stiffness and the body's mass or inertia."
        )
        self.table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.table.setFixedHeight(6 * ROW_HEIGHT + self.table.horizontalHeader().sizeHint().height() + 4)
        for row, name in enumerate(DOF_ORDER):
            item = _cell(DOF_SHORT[name], Qt.AlignmentFlag.AlignCenter)
            item.setToolTip(DOF_LONG[name])
            self.table.setItem(row, 0, item)
            for column in (1, 2, 3):
                self.table.setItem(row, column, _cell("—"))
        layout.addWidget(self.table)

        self.deflect_button = QPushButton("Show deflection of selected DOF in 3D")
        self.deflect_button.setEnabled(False)
        self.deflect_button.clicked.connect(self._request_deflection)
        self.table.itemSelectionChanged.connect(lambda: self.deflect_button.setEnabled(self._report is not None and self.table.currentRow() >= 0))
        self.table.itemDoubleClicked.connect(lambda _item: self._request_deflection())
        layout.addWidget(self.deflect_button)

        self.matrix_mode = QComboBox()
        self.matrix_mode.addItem("Condensed stiffness matrix K (6 × 6)", "k")
        self.matrix_mode.addItem("Compliance matrix C = K⁻¹ (6 × 6)", "c")
        self.matrix_mode.currentIndexChanged.connect(lambda _index: self._fill_matrix())
        matrices = QWidget()
        matrices_layout = QVBoxLayout(matrices)
        matrices_layout.setContentsMargins(0, 0, 0, 0)
        matrices_layout.setSpacing(6)
        matrices_layout.addWidget(self.matrix_mode)
        self.matrix = QTableWidget(6, 6)
        self.matrix.setHorizontalHeaderLabels([DOF_SHORT[n] for n in DOF_ORDER])
        self.matrix.setVerticalHeaderLabels([DOF_SHORT[n] for n in DOF_ORDER])
        self.matrix.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.matrix.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.matrix.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.matrix.verticalHeader().setDefaultSectionSize(ROW_HEIGHT - 2)
        self.matrix.setFixedHeight(6 * (ROW_HEIGHT - 2) + self.matrix.horizontalHeader().sizeHint().height() + 4)
        self.matrix.setToolTip("Rows and columns follow the body's degrees of freedom at its centre of mass: ux uy uz (N, m) then θx θy θz (N·m, rad).")
        for row in range(6):
            for column in range(6):
                self.matrix.setItem(row, column, _cell("—"))
        matrices_layout.addWidget(self.matrix)
        layout.addWidget(Collapsible("Matrices (6 × 6)", matrices, expanded=False))
        self.note = QLabel("")
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch(1)

    # ------------------------------------------------------------------ data
    def set_project(self, project: Project | None) -> None:
        self._project = project
        self._report = None
        self.body_combo.blockSignals(True)
        self.body_combo.clear()
        self.fixed_list.blockSignals(True)
        self.fixed_list.clear()
        if project is not None:
            structure = project.structure
            free = set(free_bodies(structure))
            for body in structure.bodies:
                if body.anchor:
                    continue
                beams = project.render.beam_counts.get(body.index, 0)
                tag = "  ·  free" if body.index in free else ""
                self.body_combo.addItem(f"Body {body.index}  ·  {fmt_mass(body.inertial.mass)}  ·  {beams} beam{'s' if beams != 1 else ''}{tag}", body.index)
            self.body_combo.setCurrentIndex(-1)
        self.body_combo.blockSignals(False)
        self.fixed_list.blockSignals(False)
        self._clear_results()
        self.info.setText("Click a body in either view, or pick one above." if project else "Extract a structure first.")

    def set_body(self, body: int) -> None:
        index = self.body_combo.findData(body) if body >= 0 else -1
        if index == self.body_combo.currentIndex():
            return
        self.body_combo.blockSignals(True)
        self.body_combo.setCurrentIndex(index)
        self.body_combo.blockSignals(False)
        self._rebuild_fixed_list()
        self._recompute()

    def current_body(self) -> int:
        data = self.body_combo.currentData()
        return -1 if data is None else int(data)

    def fixed_bodies(self) -> set[int]:
        project = self._project
        body = self.current_body()
        if project is None or body < 0:
            return set()
        condition = self.condition_combo.currentData()
        others = {b for b in project.movable if b != body}
        if condition == "others_fixed":
            return others
        if condition == "anchor_only":
            return set()
        fixed = set()
        for row in range(self.fixed_list.count()):
            item = self.fixed_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                fixed.add(int(item.data(Qt.ItemDataRole.UserRole)))
        return fixed

    # --------------------------------------------------------------- internals
    def _body_changed(self, _index: int) -> None:
        self._rebuild_fixed_list()
        self._recompute()
        self.bodySelected.emit(self.current_body())

    def _condition_changed(self, _index: int) -> None:
        self.fixed_list.setVisible(self.condition_combo.currentData() == "custom")
        self._recompute()

    def _rebuild_fixed_list(self) -> None:
        self.fixed_list.blockSignals(True)
        self.fixed_list.clear()
        project = self._project
        body = self.current_body()
        if project is not None and body >= 0:
            for other in project.movable:
                if other == body:
                    continue
                item = QListWidgetItem(f"Body {other} fixed")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, other)
                self.fixed_list.addItem(item)
        self.fixed_list.blockSignals(False)

    def _clear_results(self) -> None:
        for row in range(6):
            for column in (1, 2, 3):
                self.table.item(row, column).setText("—")
            for column in range(6):
                self.matrix.item(row, column).setText("—")
        self.note.setText("")
        self.deflect_button.setEnabled(False)

    def _recompute(self) -> None:
        project = self._project
        body = self.current_body()
        if project is None or body < 0:
            self._report = None
            self._clear_results()
            return
        fixed = self.fixed_bodies()
        try:
            report = condensed_stiffness(project.structure, project.system, body, fixed)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self._report = None
            self._clear_results()
            self.note.setText(f"Could not compute: {exc}")
            return
        self._report = report
        structure = project.structure
        rigid = structure.bodies[body]
        beams = [beam for beam in structure.beams if body in beam.bodies]
        neighbours = sorted({other for beam in beams for other in beam.bodies if other != body})
        self.info.setText(
            f"Body {body}: mass {fmt_mass(rigid.inertial.mass)}, centre {fmt_point(rigid.inertial.center)}, "
            f"{len(beams)} beam{'s' if len(beams) != 1 else ''} to {', '.join('anchor' if n == 0 else f'body {n}' for n in neighbours) or 'nothing'}. "
            + ("Fixed: anchor." if not report.fixed else
               "Fixed: anchor, " + ", ".join(f"body {b}" for b in report.fixed) + "." if len(report.fixed) <= 6 else
               f"Fixed: anchor and {len(report.fixed)} other bodies.")
        )
        for row, name in enumerate(DOF_ORDER):
            rotational = name.startswith("t")
            self.table.item(row, 1).setText(fmt_stiffness(float(report.direct[row]), rotational))
            self.table.item(row, 2).setText(fmt_stiffness(float(report.effective[row]), rotational))
            self.table.item(row, 3).setText(fmt_freq(float(report.frequency[row])))
        self._fill_matrix()
        notes = []
        for start, kind in ((0, "translation"), (3, "rotation")):
            block = report.effective[start : start + 3]
            if np.isfinite(block).any():
                softest = start + int(np.nanargmin(block))
                notes.append(f"Softest {kind}: {DOF_LONG[DOF_ORDER[softest]]} at {fmt_stiffness(float(report.effective[softest]), start == 3)}.")
        if report.singular:
            notes.append("Warning: the system is singular under this condition (a free body has no path to a fixed one); values use a least-squares fallback.")
        self.note.setText(" ".join(notes))
        self.deflect_button.setEnabled(self.table.currentRow() >= 0)

    def _fill_matrix(self) -> None:
        report = self._report
        if report is None:
            return
        source = report.stiffness if self.matrix_mode.currentData() == "k" else report.compliance
        for row in range(6):
            for column in range(6):
                item = self.matrix.item(row, column)
                if source is None:
                    item.setText("—")
                    item.setToolTip("")
                else:
                    item.setText(_compact(float(source[row, column])))
                    item.setToolTip(f"{source[row, column]:.6e}")

    def _request_deflection(self) -> None:
        body = self.current_body()
        row = self.table.currentRow()
        if body < 0 or row < 0 or self._report is None:
            return
        self.deflectionRequested.emit(body, row, self.fixed_bodies())
