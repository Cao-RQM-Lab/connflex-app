"""Left panel: file, layer roles, wafer stack, extraction parameters, and solve options."""

from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from connflex_app.gds_io import GdsInfo, guess_roles
from connflex_app.widgets.chips import Collapsible
from connflex_app.settings import MATERIAL_PRESETS, LayerSettings, MaterialSettings, Settings, preset_name

ROLES = (
    ("device", "Device", False),
    ("handle", "Handle", False),
    ("release", "Release", True),
    ("vias", "Vias", True),
    ("box", "Box", True),
)
ROLE_TIPS = {
    "box": "Optional: the buried oxide that survives the release, as drawn.\n"
    "When set it replaces the release calculation: the release distance is not used.\n"
    "It is clipped to where device and handle silicon overlap and to outside release openings.",
}
STACK = (("device", "Device"), ("box", "BOX"), ("handle", "Handle"))


def _key(layer: tuple[int, int] | None) -> str | None:
    return None if layer is None else f"{int(layer[0])}/{int(layer[1])}"


def _layer(key: str | None) -> tuple[int, int] | None:
    if not key:
        return None
    layer, datatype = key.split("/")
    return (int(layer), int(datatype))


class CompactSpin(QDoubleSpinBox):
    """A spin box whose width does not grow with the magnitude of its range."""

    def __init__(self, width: int = 72) -> None:
        super().__init__()
        self._width = width
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self._width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self._width, super().minimumSizeHint().height())


def _spin(minimum: float, maximum: float, decimals: int, step: float, suffix: str = "", width: int = 72) -> CompactSpin:
    box = CompactSpin(width)
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(f" {suffix}")
    return box


class SetupPanel(QWidget):
    openRequested = Signal()
    extractRequested = Signal()
    configRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info: GdsInfo | None = None
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        root = QVBoxLayout(inner)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # --- file
        file_box = QGroupBox("Layout file")
        file_layout = QVBoxLayout(file_box)
        self.path_label = QLabel("No file loaded")
        self.path_label.setObjectName("muted")
        self.path_label.setWordWrap(True)
        self.open_button = QPushButton("Open GDS…")
        self.open_button.clicked.connect(self.openRequested.emit)
        cell_row = QHBoxLayout()
        cell_row.addWidget(QLabel("Cell"))
        self.cell_combo = QComboBox()
        self.cell_combo.setEnabled(False)
        self.cell_combo.currentIndexChanged.connect(self._populate_roles)
        cell_row.addWidget(self.cell_combo, 1)
        file_layout.addWidget(self.open_button)
        file_layout.addWidget(self.path_label)
        file_layout.addLayout(cell_row)
        root.addWidget(file_box)

        # --- layer roles
        roles_box = QGroupBox("Layer roles")
        roles_form = QFormLayout(roles_box)
        roles_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.role_combos: dict[str, QComboBox] = {}
        self.invert_checks: dict[str, QCheckBox] = {}
        for role, title, optional in ROLES:
            combo = QComboBox()
            combo.setEnabled(False)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
            combo.setToolTip(ROLE_TIPS.get(role, f"GDS layer/datatype that holds the {title.lower()} polygons" + (" (optional)" if optional else "")))
            invert = QCheckBox("Invert")
            invert.setEnabled(False)
            invert.setToolTip(
                f"Use the complement of the {title.lower()} layer within the cell's bounding box,\n"
                "for layers drawn as etch openings instead of remaining material."
            )
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(combo, 1)
            row.addWidget(invert)
            self.role_combos[role] = combo
            self.invert_checks[role] = invert
            roles_form.addRow(title, row)
        root.addWidget(roles_box)

        # --- stack
        stack_box = QGroupBox("Wafer stack")
        stack_layout = QVBoxLayout(stack_box)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for column, header in enumerate(("", "Thickness", "Material")):
            label = QLabel(header)
            label.setObjectName("muted")
            grid.addWidget(label, 0, column)
        self.thickness: dict[str, CompactSpin] = {}
        self.preset: dict[str, QComboBox] = {}
        self.modulus: dict[str, CompactSpin] = {}
        self.poisson: dict[str, CompactSpin] = {}
        self.density: dict[str, CompactSpin] = {}
        properties = QWidget()
        props = QGridLayout(properties)
        props.setContentsMargins(0, 0, 0, 0)
        props.setHorizontalSpacing(8)
        for column, header in enumerate(("", "E (GPa)", "ν", "ρ (kg/m³)")):
            label = QLabel(header)
            label.setObjectName("muted")
            props.addWidget(label, 0, column)
        for row, (key, title) in enumerate(STACK, start=1):
            grid.addWidget(QLabel(title), row, 0)
            self.thickness[key] = _spin(0.001, 10000.0, 3, 1.0, "µm")
            self.preset[key] = QComboBox()
            self.preset[key].addItems([*MATERIAL_PRESETS, "Custom"])
            self.preset[key].setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.preset[key].setMinimumContentsLength(7)
            self.preset[key].currentTextChanged.connect(lambda name, k=key: self._apply_preset(k, name))
            grid.addWidget(self.thickness[key], row, 1)
            grid.addWidget(self.preset[key], row, 2)
            props.addWidget(QLabel(title), row, 0)
            self.modulus[key] = _spin(0.01, 5000.0, 1, 1.0)
            self.poisson[key] = _spin(0.0, 0.499, 3, 0.01)
            self.density[key] = _spin(1.0, 50000.0, 0, 10.0)
            for box in (self.modulus[key], self.poisson[key], self.density[key]):
                box.valueChanged.connect(lambda _value, k=key: self._material_edited(k))
            props.addWidget(self.modulus[key], row, 1)
            props.addWidget(self.poisson[key], row, 2)
            props.addWidget(self.density[key], row, 3)
        stack_layout.addLayout(grid)
        stack_layout.addWidget(Collapsible("Material properties", properties))
        root.addWidget(stack_box)

        # --- extraction
        extraction_box = QGroupBox("Extraction")
        form = QFormLayout(extraction_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.hole_area = _spin(0.001, 1e9, 3, 10.0, "µm²")
        self.hole_area.setToolTip("Holes in device polygons up to this area are release holes")
        self.release_distance = _spin(0.001, 10000.0, 3, 0.5, "µm")
        self.release_distance.setToolTip("Isotropic undercut of the buried oxide during release")
        self.body_min_width = _spin(0.001, 10000.0, 3, 1.0, "µm")
        self.body_min_width.setToolTip("Features narrower than this are beam candidates, not rigid bodies")
        self.beam_width_min = _spin(0.0, 10000.0, 3, 0.5, "µm")
        self.beam_width_max = _spin(0.0, 10000.0, 3, 1.0, "µm")
        self.beam_width_max.setSpecialValueText("auto")
        self.beam_width_max.setToolTip("0 means the body minimum width")
        form.addRow("Release hole max area", self.hole_area)
        form.addRow("Release distance", self.release_distance)
        self.box_note = QLabel("Box layer set: the surviving oxide is taken from it, so the release distance is not used. "
                               "Release holes and the release layer still remove device mass.")
        self.box_note.setObjectName("muted")
        self.box_note.setWordWrap(True)
        self.box_note.setVisible(False)
        form.addRow(self.box_note)
        form.addRow("Body min width", self.body_min_width)
        form.addRow("Beam width min", self.beam_width_min)
        form.addRow("Beam width max", self.beam_width_max)
        self.anchor_mode = QComboBox()
        self.anchor_mode.addItems(["Largest handle polygon", "Handle polygon by index", "Handle polygon at point"])
        self.anchor_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.anchor_mode.setMinimumContentsLength(14)
        self.anchor_mode.currentIndexChanged.connect(self._anchor_mode_changed)
        self.anchor_index = QSpinBox()
        self.anchor_index.setRange(0, 1_000_000)
        self.anchor_x = _spin(-1e7, 1e7, 2, 10.0, "µm")
        self.anchor_y = _spin(-1e7, 1e7, 2, 10.0, "µm")
        anchor_row = QHBoxLayout()
        anchor_row.addWidget(self.anchor_index)
        anchor_row.addWidget(self.anchor_x)
        anchor_row.addWidget(self.anchor_y)
        form.addRow("Anchor", self.anchor_mode)
        form.addRow("", anchor_row)
        self._anchor_mode_changed(0)

        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.heal_distance = _spin(0.001, 1e6, 3, 1.0, "nm")
        self.simplify_tolerance = _spin(0.0, 1e6, 3, 1.0, "nm")
        self.simplify_tolerance.setSpecialValueText("auto")
        self.overlap_margin = _spin(0.0, 1e6, 3, 1.0, "nm")
        self.overlap_margin.setSpecialValueText("auto")
        self.buffer_resolution = QSpinBox()
        self.buffer_resolution.setRange(1, 64)
        self.precision_grid = _spin(0.0, 1e6, 3, 1.0, "nm")
        self.precision_grid.setSpecialValueText("off")
        advanced_form.addRow("Heal distance", self.heal_distance)
        advanced_form.addRow("Simplify tolerance", self.simplify_tolerance)
        advanced_form.addRow("Beam overlap margin", self.overlap_margin)
        advanced_form.addRow("Buffer resolution", self.buffer_resolution)
        advanced_form.addRow("Precision grid", self.precision_grid)
        form.addRow(Collapsible("Advanced", advanced))
        root.addWidget(extraction_box)

        # --- solve
        solve_box = QGroupBox("Modal solve")
        solve_form = QFormLayout(solve_box)
        solve_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.n_modes = QSpinBox()
        self.n_modes.setRange(0, 100000)
        self.n_modes.setSpecialValueText("all")
        self.method = QComboBox()
        self.method.addItems(["dense", "sparse"])
        solve_form.addRow("Number of modes", self.n_modes)
        solve_form.addRow("Solver", self.method)
        root.addWidget(solve_box)

        # --- actions
        self.config_button = QPushButton("Load process config (TOML)…")
        self.config_button.clicked.connect(self.configRequested.emit)
        self.extract_button = QPushButton("Extract structure")
        self.extract_button.setObjectName("primary")
        self.extract_button.setEnabled(False)
        self.extract_button.clicked.connect(self.extractRequested.emit)
        root.addWidget(self.config_button)
        root.addWidget(self.extract_button)
        root.addStretch(1)

        self.role_combos["box"].currentIndexChanged.connect(self._box_role_changed)
        self.apply(Settings())

    # --------------------------------------------------------------- helpers
    def _apply_preset(self, key: str, name: str) -> None:
        preset = MATERIAL_PRESETS.get(name)
        if preset is None or self._loading:
            return
        self._loading = True
        self.modulus[key].setValue(preset.youngs_modulus / 1e9)
        self.poisson[key].setValue(preset.poisson_ratio)
        self.density[key].setValue(preset.density)
        self._loading = False

    def _material_edited(self, key: str) -> None:
        if self._loading:
            return
        material = MaterialSettings(self.density[key].value(), self.modulus[key].value() * 1e9, self.poisson[key].value())
        name = preset_name(material)
        if self.preset[key].currentText() != name:
            self._loading = True
            self.preset[key].setCurrentText(name)
            self._loading = False

    def _box_role_changed(self, *_args) -> None:
        overridden = self.role_combos["box"].currentData() is not None
        self.release_distance.setEnabled(not overridden)
        self.release_distance.setToolTip(
            "Not used: the box layer defines the surviving oxide" if overridden else "Isotropic undercut of the buried oxide during release"
        )
        self.box_note.setVisible(overridden)

    def _anchor_mode_changed(self, index: int) -> None:
        self.anchor_index.setVisible(index == 1)
        self.anchor_x.setVisible(index == 2)
        self.anchor_y.setVisible(index == 2)

    def set_anchor_point(self, x: float, y: float) -> None:
        self.anchor_mode.setCurrentIndex(2)
        self.anchor_x.setValue(x * 1e6)
        self.anchor_y.setValue(y * 1e6)

    def set_busy(self, busy: bool) -> None:
        self.extract_button.setEnabled(not busy and self._info is not None)
        self.open_button.setEnabled(not busy)

    # ---------------------------------------------------------------- GDS info
    def set_gds_info(self, info: GdsInfo | None, remembered: Settings | None = None) -> None:
        self._info = info
        self.cell_combo.blockSignals(True)
        self.cell_combo.clear()
        if info is None:
            self.path_label.setText("No file loaded")
            self.cell_combo.setEnabled(False)
            for combo in self.role_combos.values():
                combo.clear()
                combo.setEnabled(False)
            for check in self.invert_checks.values():
                check.setEnabled(False)
            self.extract_button.setEnabled(False)
            self.cell_combo.blockSignals(False)
            return
        self.path_label.setText(f"{os.path.basename(info.path)}\n{os.path.dirname(info.path)}")
        self.path_label.setToolTip(info.path)
        self.cell_combo.addItems(info.top_cells)
        self.cell_combo.setEnabled(len(info.top_cells) > 1)
        if remembered is not None and remembered.cell in info.top_cells:
            self.cell_combo.setCurrentText(remembered.cell)
        self.cell_combo.blockSignals(False)
        self._populate_roles(remembered=remembered)
        self.extract_button.setEnabled(True)

    def _populate_roles(self, *_args, remembered: Settings | None = None) -> None:
        if self._info is None:
            return
        stats = self._info.layers(self.cell_combo.currentText() or None)
        guess = guess_roles(stats)
        for role, _title, optional in ROLES:
            combo = self.role_combos[role]
            combo.blockSignals(True)
            combo.clear()
            if optional:
                combo.addItem("— none —", None)
            for entry in stats:
                area = entry.area * (self._info.unit * 1e6) ** 2
                plural = "s" if entry.count != 1 else ""
                combo.addItem(f"{entry.label}   ·   {entry.count} polygon{plural}, {area:,.0f} µm²", _key(entry.key))
            if remembered is not None:
                wanted = _key(getattr(remembered, role, None))
                if wanted is not None and combo.findData(wanted) < 0:
                    wanted = _key(guess[role])
            else:
                wanted = _key(guess[role])
            index = combo.findData(wanted) if wanted is not None else (0 if optional else -1)
            combo.setCurrentIndex(max(index, 0))
            combo.setEnabled(True)
            combo.blockSignals(False)
            check = self.invert_checks[role]
            check.blockSignals(True)
            check.setChecked(bool(getattr(remembered, f"invert_{role}", False)) if remembered is not None else False)
            check.setEnabled(True)
            check.blockSignals(False)
        self._box_role_changed()

    # ---------------------------------------------------------------- settings
    def settings(self) -> Settings:
        s = Settings()
        for role, _title, _optional in ROLES:
            setattr(s, role, _layer(self.role_combos[role].currentData()))
            setattr(s, f"invert_{role}", self.invert_checks[role].isChecked())
        s.cell = self.cell_combo.currentText() if self.cell_combo.count() > 1 else None
        for key in ("device", "box", "handle"):
            material = MaterialSettings(self.density[key].value(), self.modulus[key].value() * 1e9, self.poisson[key].value())
            setattr(s, f"stack_{key}", LayerSettings(self.thickness[key].value() * 1e-6, material))
        s.release_hole_max_area = self.hole_area.value() * 1e-12
        s.release_distance = self.release_distance.value() * 1e-6
        s.body_min_width = self.body_min_width.value() * 1e-6
        s.beam_width_min = self.beam_width_min.value() * 1e-6
        s.beam_width_max = None if self.beam_width_max.value() == 0 else self.beam_width_max.value() * 1e-6
        s.heal_distance = self.heal_distance.value() * 1e-9
        s.simplify_tolerance = None if self.simplify_tolerance.value() == 0 else self.simplify_tolerance.value() * 1e-9
        s.beam_overlap_margin = None if self.overlap_margin.value() == 0 else self.overlap_margin.value() * 1e-9
        s.buffer_resolution = self.buffer_resolution.value()
        s.precision_grid = None if self.precision_grid.value() == 0 else self.precision_grid.value() * 1e-9
        mode = self.anchor_mode.currentIndex()
        if mode == 0:
            s.anchor = None
        elif mode == 1:
            s.anchor = self.anchor_index.value()
        else:
            s.anchor = (self.anchor_x.value() * 1e-6, self.anchor_y.value() * 1e-6)
        s.n_modes = None if self.n_modes.value() == 0 else self.n_modes.value()
        s.method = self.method.currentText()
        return s

    def apply(self, s: Settings) -> None:
        """Set every widget from ``s`` (layer roles are applied when a file is loaded)."""
        self._loading = True
        for role, _title, _optional in ROLES:
            check = self.invert_checks[role]
            check.blockSignals(True)
            check.setChecked(bool(getattr(s, f"invert_{role}", False)))
            check.blockSignals(False)
        for key in ("device", "box", "handle"):
            layer: LayerSettings = getattr(s, f"stack_{key}")
            self.thickness[key].setValue(layer.thickness * 1e6)
            self.modulus[key].setValue(layer.material.youngs_modulus / 1e9)
            self.poisson[key].setValue(layer.material.poisson_ratio)
            self.density[key].setValue(layer.material.density)
            self.preset[key].setCurrentText(preset_name(layer.material))
        self.hole_area.setValue(s.release_hole_max_area * 1e12)
        self.release_distance.setValue(s.release_distance * 1e6)
        self.body_min_width.setValue(s.body_min_width * 1e6)
        self.beam_width_min.setValue(s.beam_width_min * 1e6)
        self.beam_width_max.setValue(0.0 if s.beam_width_max is None else s.beam_width_max * 1e6)
        self.heal_distance.setValue(s.heal_distance * 1e9)
        self.simplify_tolerance.setValue(0.0 if s.simplify_tolerance is None else s.simplify_tolerance * 1e9)
        self.overlap_margin.setValue(0.0 if s.beam_overlap_margin is None else s.beam_overlap_margin * 1e9)
        self.buffer_resolution.setValue(s.buffer_resolution)
        self.precision_grid.setValue(0.0 if s.precision_grid is None else s.precision_grid * 1e9)
        if s.anchor is None:
            self.anchor_mode.setCurrentIndex(0)
        elif isinstance(s.anchor, int):
            self.anchor_mode.setCurrentIndex(1)
            self.anchor_index.setValue(s.anchor)
        else:
            self.anchor_mode.setCurrentIndex(2)
            self.anchor_x.setValue(float(s.anchor[0]) * 1e6)
            self.anchor_y.setValue(float(s.anchor[1]) * 1e6)
        self.n_modes.setValue(0 if s.n_modes is None else s.n_modes)
        self.method.setCurrentText(s.method)
        self._loading = False
