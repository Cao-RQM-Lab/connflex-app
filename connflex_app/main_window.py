"""Main window: three workspaces over one pair of views.

The Mechanical, Electrical, and Electromechanical tabs share the 2D and 3D views and the
setup panel. Each tab brings its own panels, decides what a click in a view selects, and
colours the views its own way: by layer or body, by net with two selections, or by voltage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTabBar,
    QToolBar,
    QWidget,
)

import connflex_core as cf
from connflex_core.mechanics.assembly import assemble
from connflex_core.progress import stage

from connflex_app import __version__, paint, theme
from connflex_app.analysis import free_bodies, rigid_mode_count, static_response
from connflex_app.drive import EquilibriumRun, default_source_electrode, run_transient, solve_equilibrium
from connflex_app.electrical import CapacitanceResult, compute_capacitance, describe_net
from connflex_app.extraction import extract as extract_structure
from connflex_app.gds_export import LAYERS as GDS_LAYERS, export_gds
from connflex_app.gds_io import GdsInfo, inspect_gds, load_box, load_layout
from connflex_app.geometry import (
    DOF_ORDER,
    DOF_SHORT,
    KIND_BEAM,
    KIND_BOX,
    KIND_DEVICE,
    KIND_HANDLE,
    KIND_REJECTED,
    body_centers,
    build_mesh,
    describe_modes,
    motion_from_vector,
    release_holes,
    silicon_pieces,
)
from connflex_app.gif_export import encode_gif, frame_from_qimage
from connflex_app.render2d import prepare_layout, prepare_project
from connflex_app.session import Project, Session
from connflex_app.settings import DriveSettings, Settings
from connflex_app.units import fmt_capacitance, fmt_freq, fmt_si
from connflex_app.widgets.capacitance_panel import AMBER, CYAN, CapacitancePanel
from connflex_app.widgets.chips import ChipBar, OverlayHost, VoltageLegend
from connflex_app.widgets.drive_panel import DrivePanel
from connflex_app.widgets.gds_dialog import GdsExportDialog
from connflex_app.widgets.gif_dialog import GifExportDialog
from connflex_app.widgets.layout_view import LayoutView
from connflex_app.widgets.modes_panel import ModesPanel
from connflex_app.widgets.progress_status import ProgressStatus
from connflex_app.widgets.scene3d import Scene3D
from connflex_app.widgets.setup_panel import SetupPanel
from connflex_app.widgets.stiffness_panel import StiffnessPanel
from connflex_app.widgets.timeline_panel import TimelinePanel
from connflex_app.workers import ProgressRelay, Worker

SCENE_KINDS = {"device": KIND_DEVICE, "box": KIND_BOX, "handle": KIND_HANDLE, "beams": KIND_BEAM, "rejected": KIND_REJECTED}
EXTRACT_STAGES = (
    "load", "heal", "release", "prune", "nets", "opening", "bodies", "candidates", "beams", "mass", "electrodes",
    "mesh", "assemble", "prepare",
)  # fmt: skip
MODAL_STAGES = ("assemble", "solve")
CAPACITANCE_STAGES = ("plates", "clusters", "sidewalls")
EQUILIBRIUM_STAGES = ("assemble", "equilibrium")
TRANSIENT_STAGES = ("assemble", "transient")
EXPORT_GDS_STAGES = ("openings", "device", "layers", "write")
# Share of the progress bar each stage of an analysis takes: the field solve and the time
# integration dwarf the set-up stages around them.
STAGE_WEIGHTS = {
    "plates": 0.02,
    "clusters": 0.02,
    "sidewalls": 1.0,
    "assemble": 0.02,
    "assemble 2": 0.02,
    "equilibrium": 0.5,
    "transient": 4.0,
}
WORKSPACES: tuple[tuple[str, str, str], ...] = (
    ("mechanical", "Mechanical", "Bodies, beams, resonant modes, and stiffness"),
    ("electrical", "Electrical", "Nets and electrodes, and the capacitance between them"),
    ("electromechanical", "Electromechanical", "Voltages on nets, equilibria, and transients in time"),
)
STATE_VERSION = 2  # of the saved dock layout; older layouts do not know the workspace panels
SECOND_PICK = Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier


def run_extraction(path: str, settings: Settings, progress, bounds=None) -> Project:
    """Load the layout, extract the structure, and build what the views need. Runs off-thread."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    with stage("load", progress):
        layout = load_layout(path, settings, bounds)
        box = load_box(path, settings, bounds)
    timings["load"] = time.perf_counter() - started
    stack, params = settings.stack(), settings.params()
    started = time.perf_counter()
    structure = extract_structure(layout, stack, params, box=box, progress=progress)
    timings["extract"] = time.perf_counter() - started
    started = time.perf_counter()
    bodies, beams = len(structure.bodies), len(structure.beams)
    cache: dict = {}
    with stage("mesh", progress, total=bodies + beams) as current:
        holes = release_holes(layout, params.release_hole_max_area)
        pieces = silicon_pieces(structure)
        mesh = build_mesh(structure, stack, pieces, advance=current.advance, cache=cache)
    timings["mesh"] = time.perf_counter() - started
    with stage("assemble", progress):
        system = assemble(structure)
    with stage("prepare", progress, total=bodies) as current:
        render = prepare_project(layout, structure, pieces, holes, advance=current.advance, cache=cache)
    del cache
    return Project(
        path=path,
        settings=settings,
        layout=layout,
        stack=stack,
        params=params,
        structure=structure,
        holes=holes,
        pieces=pieces,
        mesh=mesh,
        system=system,
        render=render,
        timings=timings,
    )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ConnFlex")
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(int(available.width() * 0.92), int(available.height() * 0.9))
        else:
            self.resize(1540, 960)
        self.session = Session(self)
        self._qsettings = QSettings()
        self._workers: list[Worker] = []
        self._busy = False
        self._phase = ""
        self._info: GdsInfo | None = None
        self._path: str | None = None
        self._workspace = "mechanical"
        self._sized: set[str] = set()
        self._hover_group = -1
        self._paint_2d: tuple | None = None
        self._last_table: np.ndarray | None = None
        self._outlined_net = -1
        self._run_stale = False
        self._progress = ProgressRelay(self)
        # The timeline runs under the setup panel and the views; the drive beside it keeps its height.
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self._build_views()
        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._wire()
        self._restored_state = self._restore()
        self._shown = False
        self.set_workspace(self._workspace, force=True)

    # ------------------------------------------------------------------ build
    def _build_views(self) -> None:
        self.layout_view = LayoutView()
        self.scene = Scene3D()
        self.view2d = OverlayHost(self.layout_view)
        self.view3d = OverlayHost(self.scene)

        colors = theme.LAYER
        self.highlight_bar = ChipBar()
        self.highlight_bar.add_label("Highlight")
        self.highlight_bar.add_toggle("beams", "Beams", True, "Fill accepted beams in orange", colors["beam"])
        self.highlight_bar.add_toggle("bodies", "Mech nets", False, "Colour each rigid body (mechanical net) distinctly")
        self.highlight_bar.add_toggle("nets", "Elec nets", False, "Colour conductors by electrical net")
        self.highlight_bar.add_toggle("rejected", "Rejected", True, "Beam candidates that did not qualify", colors["rejected"])
        self.electrical_bar = ChipBar()
        self.electrical_legend = self.electrical_bar.add_label("")
        self.electrical_legend.setObjectName("legend")
        self.electrical_legend.setTextFormat(Qt.TextFormat.RichText)
        self.voltage_bar = ChipBar()
        self.voltage_bar.add_label("Voltage")
        self.voltage_legend = VoltageLegend(paint.NEGATIVE, paint.NEUTRAL, paint.POSITIVE)
        self.voltage_bar.add_widget(self.voltage_legend)
        self.voltage_note = self.voltage_bar.add_label("")
        self.voltage_note.setObjectName("legend")
        self.graphs_bar = ChipBar()
        self.graphs_bar.add_label("Graphs")
        self.graphs_bar.add_toggle("mech_graph", "Mechanical", False, "Bodies as nodes at their centres of mass, beams as edges", theme.ACCENT)
        self.graphs_bar.add_toggle("elec_graph", "Electrical", False, "Conductors as nodes, vias as edges", colors["vias"])
        self.layers_bar = ChipBar()
        self.layers_bar.add_label("Layers")
        for key, text, color in (("device", "Device", colors["device"]), ("handle", "Handle", "#7b8396"), ("box", "BOX", colors["box"]), ("release", "Release", colors["release"]), ("vias", "Vias", colors["vias"])):
            self.layers_bar.add_toggle(key, text, True, color=color)
        self.layers_bar.add_separator()
        self.layers_bar.add_button("Fit", self.layout_view.fit, "Fit the layout to the view (F, double-click)")
        for bar in (self.highlight_bar, self.electrical_bar, self.voltage_bar, self.graphs_bar, self.layers_bar):
            self.view2d.add_overlay(bar, "top-left")

        self.scene_bar = ChipBar()
        self.scene_bar.add_label("Show")
        for key, text, color in (("device", "Device", colors["device"]), ("box", "BOX", colors["box"]), ("handle", "Handle", "#7b8396"), ("beams", "Beams", colors["beam"]), ("rejected", "Rejected", colors["rejected"])):
            self.scene_bar.add_toggle(key, text, True, color=color)
        self.scene_options_bar = ChipBar()
        self.scene_options_bar.add_toggle("color_body", "Colour by body", False, "Tint every body with its own colour")
        self.scene_options_bar.add_button("Reset view", self.scene.reset_view, "Reset the camera (R, double-click)")
        self.view3d.add_overlay(self.scene_bar, "top-left")
        self.view3d.add_overlay(self.scene_options_bar, "top-left")

        self.playback_bar = ChipBar()
        self.play_button = self.playback_bar.add_toggle("play", "▶  Play", False, "Play or pause the animation (Space)")
        self.play_button.setEnabled(False)
        self.playback_caption = self.playback_bar.add_label("No motion selected")
        self.playback_bar.add_separator()
        self.gif_button = self.playback_bar.add_button("GIF…", self.export_gif_dialog, "Export this animation as a GIF")
        self.gif_button.setEnabled(False)
        self.playback_bar.add_button("Clear", self._clear_motion, "Show the undeformed structure")
        self.view3d.add_overlay(self.playback_bar, "bottom-center")

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.view2d)
        self.splitter.addWidget(self.view3d)
        self.splitter.setSizes([800, 800])
        self.setCentralWidget(self.splitter)

    def _build_docks(self) -> None:
        right, bottom = Qt.DockWidgetArea.RightDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea
        self.setup = SetupPanel()
        self.setup.setMinimumWidth(340)
        self.dock_setup = self._dock("Setup", self.setup, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.modes = ModesPanel()
        self.modes.setMinimumWidth(320)
        self.dock_modes = self._dock("Modes", self.modes, right)
        self.stiffness = StiffnessPanel()
        self.stiffness.setMinimumWidth(320)
        self.dock_stiffness = self._dock("Stiffness", self.stiffness, right)
        self.capacitance = CapacitancePanel()
        self.capacitance.setMinimumWidth(360)
        self.dock_capacitance = self._dock("Capacitance", self.capacitance, right)
        self.drive = DrivePanel()
        self.drive.setMinimumWidth(360)
        self.dock_drive = self._dock("Drive", self.drive, right)
        self.timeline = TimelinePanel()
        self.timeline.setMinimumHeight(210)
        self.dock_timeline = self._dock("Transient", self.timeline, bottom)
        self._workspace_docks = {
            "mechanical": [self.dock_modes, self.dock_stiffness],
            "electrical": [self.dock_capacitance],
            "electromechanical": [self.dock_drive, self.dock_timeline],
        }

    def _dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower()}")
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        return dock

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.act_open = QAction("Open GDS…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_file)
        self.act_extract = QAction("Extract", self)
        self.act_extract.setShortcut(QKeySequence("Ctrl+E"))
        self.act_extract.triggered.connect(self.extract)
        self.act_solve = QAction("Solve modes", self)
        self.act_solve.setShortcut(QKeySequence("Ctrl+M"))
        self.act_solve.triggered.connect(self.solve)
        self.act_capacitance = QAction("Compute capacitance", self)
        self.act_capacitance.setToolTip("Solve the capacitance between all electrodes at rest")
        self.act_capacitance.setShortcut(QKeySequence("Ctrl+K"))
        self.act_capacitance.triggered.connect(self.compute_capacitance)
        self.act_transient = QAction("Run transient", self)
        self.act_transient.setToolTip("Follow the structure in time through the voltage steps of the drive")
        self.act_transient.setShortcut(QKeySequence("Ctrl+T"))
        self.act_transient.triggered.connect(self.run_transient)
        self.act_equilibrium = QAction("Equilibrium", self)
        self.act_equilibrium.setToolTip("Where the structure rests under the voltages of the selected step")
        self.act_equilibrium.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_equilibrium.triggered.connect(lambda: self.solve_equilibrium())
        self.act_export_gds = QAction("Export GDS…", self)
        self.act_export_gds.setToolTip("Export the extracted, released geometry as a GDS file")
        self.act_export_gds.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_export_gds.triggered.connect(self.export_gds_dialog)
        self.act_fit = QAction("Fit 2D", self)
        self.act_fit.triggered.connect(self.layout_view.fit)
        self.act_reset = QAction("Reset 3D", self)
        self.act_reset.triggered.connect(self.scene.reset_view)
        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_extract)
        toolbar.addAction(self.act_solve)
        toolbar.addAction(self.act_capacitance)
        toolbar.addAction(self.act_transient)
        toolbar.addAction(self.act_equilibrium)
        toolbar.addAction(self.act_export_gds)
        toolbar.addSeparator()
        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_reset)
        toolbar.addSeparator()
        self.view_group = QActionGroup(self)
        self.view_group.setExclusive(True)
        self.view_actions: dict[str, QAction] = {}
        for key, text in (("split", "Split"), ("2d", "2D"), ("3d", "3D")):
            action = QAction(text, self)
            action.setCheckable(True)
            action.setChecked(key == "split")
            action.triggered.connect(lambda _checked, k=key: self._set_view_mode(k))
            self.view_group.addAction(action)
            toolbar.addAction(action)
            self.view_actions[key] = action
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        version = QLabel(f"ConnFlex {__version__} · connflex-core {cf.__version__}")
        version.setObjectName("muted")
        version.setContentsMargins(0, 0, 8, 0)
        toolbar.addWidget(version)

        self.addToolBarBreak()
        tabs_bar = QToolBar("Workspaces")
        tabs_bar.setObjectName("toolbar_workspaces")
        tabs_bar.setMovable(False)
        self.addToolBar(tabs_bar)
        self.tabs = QTabBar()
        self.tabs.setObjectName("workspaces")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.workspace_actions: dict[str, QAction] = {}
        for index, (key, title, tip) in enumerate(WORKSPACES):
            self.tabs.addTab(title)
            self.tabs.setTabToolTip(index, f"{tip} (Ctrl+{index + 1})")
            action = QAction(title, self)
            action.setShortcut(QKeySequence(f"Ctrl+{index + 1}"))
            action.setStatusTip(tip)
            action.triggered.connect(lambda _checked=False, k=key: self.set_workspace(k))
            self.workspace_actions[key] = action
        self.tabs.currentChanged.connect(lambda index: self.set_workspace(WORKSPACES[index][0]))
        tabs_bar.addWidget(self.tabs)
        self.workspace_hint = QLabel("")
        self.workspace_hint.setObjectName("muted")
        self.workspace_hint.setContentsMargins(16, 0, 0, 0)
        tabs_bar.addWidget(self.workspace_hint)

    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(self.act_open)
        self.act_config = QAction("Load process config (TOML)…", self)
        self.act_config.triggered.connect(self.load_config)
        file_menu.addAction(self.act_config)
        file_menu.addSeparator()
        file_menu.addAction(self.act_extract)
        file_menu.addAction(self.act_solve)
        file_menu.addAction(self.act_capacitance)
        file_menu.addAction(self.act_transient)
        file_menu.addAction(self.act_equilibrium)
        file_menu.addSeparator()
        self.act_export_structure = QAction("Export structure as JSON…", self)
        self.act_export_structure.triggered.connect(self.export_structure)
        self.act_export_modes = QAction("Export modes as NPZ…", self)
        self.act_export_modes.triggered.connect(self.export_modes)
        self.act_export_capacitance = QAction("Export capacitance report as JSON…", self)
        self.act_export_capacitance.triggered.connect(self.export_capacitance_report)
        self.act_export_matrix = QAction("Export capacitance matrix as CSV…", self)
        self.act_export_matrix.triggered.connect(self.export_capacitance_matrix)
        self.act_export_transient = QAction("Export transient as NPZ…", self)
        self.act_export_transient.triggered.connect(self.export_transient)
        self.act_export_gif = QAction("Export animation as GIF…", self)
        self.act_export_gif.triggered.connect(self.export_gif_dialog)
        for action in (
            self.act_export_gds,
            self.act_export_structure,
            self.act_export_modes,
            self.act_export_capacitance,
            self.act_export_matrix,
            self.act_export_transient,
            self.act_export_gif,
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = menu.addMenu("&View")
        for action in self.workspace_actions.values():
            view_menu.addAction(action)
        view_menu.addSeparator()
        for action in self.view_actions.values():
            view_menu.addAction(action)
        view_menu.addSeparator()
        self.panels_menu = view_menu.addMenu("Panels")
        view_menu.addSeparator()
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_reset)
        help_menu = menu.addMenu("&Help")
        demo = QAction("Open demo layout", self)
        demo.triggered.connect(self.open_demo)
        help_menu.addAction(demo)
        about = QAction("About ConnFlex", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _build_statusbar(self) -> None:
        self.progress = ProgressStatus()
        self.progress.set_message("Open a GDS file to begin.")
        self.status_label = self.progress.message
        self._progress.stageStarted.connect(self.progress.stage_started)
        self._progress.advanced.connect(self.progress.advanced)
        self._progress.described.connect(self.progress.set_detail)
        self._progress.stageFinished.connect(self.progress.stage_finished)
        self.progress.cancelRequested.connect(self._progress.cancel)
        self.hover_label = QLabel("")
        self.hover_label.setObjectName("muted")
        bar = self.statusBar()
        bar.addWidget(self.progress, 1)
        bar.addPermanentWidget(self.hover_label)

    def _wire(self) -> None:
        self.setup.openRequested.connect(self.open_file)
        self.setup.extractRequested.connect(self.extract)
        self.setup.configRequested.connect(self.load_config)
        for combo in self.setup.role_combos.values():
            combo.currentIndexChanged.connect(lambda _index: self._preview())
        for check in self.setup.invert_checks.values():
            check.toggled.connect(lambda _on: self._preview())
        self.setup.cell_combo.currentIndexChanged.connect(lambda _index: self._preview())
        self.modes.modeSelected.connect(self._mode_selected)
        self.modes.solveRequested.connect(self.solve)
        self.stiffness.bodySelected.connect(self.session.select)
        self.stiffness.deflectionRequested.connect(self._show_deflection)
        self.layout_view.hovered.connect(self._hover_info)
        self.layout_view.bodyClicked.connect(self.session.select)
        self.layout_view.groupClicked.connect(self._group_clicked)
        self.layout_view.groupHovered.connect(self._group_hovered)
        self.layout_view.anchorPicked.connect(self.setup.set_anchor_point)
        self.scene.bodyClicked.connect(self.session.select)
        self.scene.hovered.connect(self._hover_body)
        self.scene.groupClicked.connect(self._group_clicked)
        self.scene.groupHovered.connect(self._group_hovered)
        self.scene.playingChanged.connect(self._playing_changed)
        self.scene.timeChanged.connect(self._transient_time)
        self.session.projectChanged.connect(self._project_changed)
        self.session.selectionChanged.connect(self._selection_changed)
        self.session.capacitanceChanged.connect(self._capacitance_changed)
        self.session.transientChanged.connect(self._transient_changed)
        self.highlight_bar.toggled.connect(self._highlight_toggled)
        self.graphs_bar.toggled.connect(lambda key, on: self.layout_view.set_options(**{key: on}))
        self.layers_bar.toggled.connect(lambda key, on: self.layout_view.set_options(**{key: on}))
        self.scene_bar.toggled.connect(self._scene_toggled)
        self.scene_options_bar.toggled.connect(self._scene_toggled)
        self.play_button.toggled.connect(self._play_requested)
        self.modes.amplitudeChanged.connect(self._amplitude_changed)
        self.modes.speedChanged.connect(self.scene.set_speed)
        self.scene.set_amplitude(self.modes.amplitude())
        self.scene.set_speed(self.modes.speed())

        self.capacitance.computeRequested.connect(self.compute_capacitance)
        self.capacitance.picksChanged.connect(self._refresh_paint)
        self.capacitance.settingsChanged.connect(self._update_model_status)
        self.capacitance.exportReportRequested.connect(self.export_capacitance_report)
        self.capacitance.exportMatrixRequested.connect(self.export_capacitance_matrix)

        self.drive.runRequested.connect(self.run_transient)
        self.drive.equilibriumRequested.connect(self.solve_equilibrium)
        self.drive.changed.connect(self._drive_changed)
        self.drive.netSelected.connect(lambda _net: self._refresh_paint())
        self.drive.stepSelected.connect(self._step_selected)
        self.drive.modelRequested.connect(lambda: self.set_workspace("electrical"))
        self.timeline.scrubbed.connect(self._scrubbed)
        self.timeline.playRequested.connect(self._play_requested)
        self.timeline.rateChanged.connect(self.scene.set_rate)
        self.timeline.amplitudeChanged.connect(self._amplitude_changed)
        self.timeline.trueScaleChanged.connect(self.scene.set_true_scale)
        self._update_actions()
        self._update_model_status()

    # --------------------------------------------------------------- workspaces
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, key: str, force: bool = False) -> None:
        """Show the panels, colours, and picking of one of the three tabs."""
        keys = [entry[0] for entry in WORKSPACES]
        if key not in keys or (key == self._workspace and not force):
            return
        self._workspace = key
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(keys.index(key))
        self.tabs.blockSignals(False)
        for name, docks in self._workspace_docks.items():
            for dock in docks:
                dock.setVisible(name == key)
        self.panels_menu.clear()
        for dock in (self.dock_setup, *self._workspace_docks[key]):
            self.panels_menu.addAction(dock.toggleViewAction())
        if self._shown:
            self._size_docks(key)
        mechanical = key == "mechanical"
        self.highlight_bar.setVisible(mechanical)
        self.electrical_bar.setVisible(key == "electrical")
        self.voltage_bar.setVisible(key == "electromechanical")
        self.scene_options_bar.button("color_body").setVisible(mechanical)
        self.act_solve.setVisible(mechanical)
        self.act_capacitance.setVisible(key == "electrical")
        self.act_transient.setVisible(key == "electromechanical")
        self.act_equilibrium.setVisible(key == "electromechanical")
        pick = "body" if mechanical else "group"
        self.layout_view.set_pick_mode(pick)
        self.scene.set_pick_mode(pick)
        self._outlined_net = -1
        self.layout_view.set_selected(self.session.selected if mechanical else -1)  # the body outline belongs to this tab
        self._hover_group = -1
        self.hover_label.setText("")
        self.workspace_hint.setText(
            {
                "mechanical": "Click a body to inspect its stiffness; select a mode to animate it.",
                "electrical": "Click a net to select it (A); Shift- or Ctrl-click a second one (B) to read the capacitance between them.",
                "electromechanical": "Click a net to add it to the drive, type its voltage in each step, then run the transient.",
            }[key]
        )
        self.view2d.relayout()
        self.view3d.relayout()
        self._show_workspace_motion()
        self._refresh_paint()
        self._update_actions()

    def _size_docks(self, key: str) -> None:
        """Give a workspace's panels a sensible size the first time they are shown."""
        if key in self._sized or self._restored_state:
            return
        self._sized.add(key)
        if key == "mechanical":
            self.resizeDocks([self.dock_modes, self.dock_stiffness], [430, 470], Qt.Orientation.Vertical)
            self.resizeDocks([self.dock_setup], [360], Qt.Orientation.Horizontal)
        elif key == "electrical":
            self.resizeDocks([self.dock_capacitance], [430], Qt.Orientation.Horizontal)
        else:
            self.resizeDocks([self.dock_drive], [430], Qt.Orientation.Horizontal)
            self.resizeDocks([self.dock_timeline], [290], Qt.Orientation.Vertical)

    def _show_workspace_motion(self) -> None:
        """Put the motion that belongs to the current tab into the 3D view."""
        project = self.session.project
        if project is None:
            return
        if self._workspace == "mechanical":
            self._mode_selected(self.modes.current_mode())
        elif self._workspace == "electromechanical" and project.transient is not None:
            self._show_transient(play=False)
        elif self._workspace == "electromechanical" and project.equilibrium is not None:
            self._show_equilibrium(project.equilibrium)
        else:
            self.scene.clear_motion()
            self._set_caption("No motion selected")

    # --------------------------------------------------------------- persistence
    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            self._size_docks(self._workspace)

    def _restore(self) -> bool:
        geometry = self._qsettings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._qsettings.value("window/state")
        restored = False
        if state is not None:
            restored = bool(self.restoreState(state, STATE_VERSION))
        workspace = self._qsettings.value("window/workspace")
        if workspace in [entry[0] for entry in WORKSPACES]:
            self._workspace = str(workspace)
        defaults = self._qsettings.value("settings/defaults")
        if defaults:
            try:
                settings = Settings.from_json(defaults)
                settings.drive = DriveSettings()  # a drive names nets of one layout; it is kept per file only
                self._apply_settings(settings)
            except Exception:  # noqa: BLE001 - stale settings are not fatal
                pass
        return restored

    def _settings(self) -> Settings:
        """Everything the panels hold: setup, electrostatic model, drive, and run options."""
        settings = self.setup.settings()
        settings.electro = self.capacitance.settings()
        settings.transient = self.drive.transient_settings()
        settings.drive = self.drive.drive_settings()
        return settings

    def _apply_settings(self, settings: Settings) -> None:
        self.setup.apply(settings)
        self.capacitance.apply(settings.electro)
        self.drive.apply(settings.drive, settings.transient)

    def _remember(self, path: str | None, settings: Settings) -> None:
        if path is None:
            return
        key = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
        self._qsettings.setValue(f"settings/file_{key}", settings.to_json())
        self._qsettings.setValue("settings/defaults", settings.to_json())

    def _remembered(self, path: str) -> Settings | None:
        key = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
        text = self._qsettings.value(f"settings/file_{key}")
        if not text:
            return None
        try:
            return Settings.from_json(text)
        except Exception:  # noqa: BLE001
            return None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._progress.cancel()
        self._qsettings.setValue("window/geometry", self.saveGeometry())
        self._qsettings.setValue("window/state", self.saveState(STATE_VERSION))
        self._qsettings.setValue("window/workspace", self._workspace)
        try:
            settings = self._settings()
            if self._info is not None:
                self._remember(self._path, settings)
            else:
                self._qsettings.setValue("settings/defaults", settings.to_json())
        except Exception:  # noqa: BLE001
            pass
        self.scene.play(False)
        for worker in self._workers:
            worker.wait(3000)
        super().closeEvent(event)

    # ------------------------------------------------------------------ workers
    def _run(self, fn, on_success, failure_title: str) -> None:
        worker = Worker(fn, self)
        self._workers.append(worker)

        def done(result) -> None:
            self._workers.remove(worker)
            on_success(result)

        def failed(message: str, trace: str) -> None:
            self._workers.remove(worker)
            self._failed(failure_title, message, trace)

        def cancelled() -> None:
            self._workers.remove(worker)
            title = self._phase
            self._set_busy(False, ok=None)
            self.status_label.setText(f"{title} cancelled." if title else "Cancelled.")

        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.cancelled.connect(cancelled)
        worker.start()

    def _set_busy(
        self,
        busy: bool,
        phase: str = "",
        stages: tuple[str, ...] = (),
        reset_timings: bool = False,
        ok: bool | None = True,
        cancellable: bool = False,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._busy = busy
        self._phase = phase
        self.setup.set_busy(busy)
        self.capacitance.set_busy(busy)
        self.drive.set_busy(busy)
        self._update_actions()
        if busy:
            self._progress.reset()
            self.progress.start_phase(phase, stages, reset=reset_timings, cancellable=cancellable, weights=weights)
        else:
            self.progress.finish_phase(ok=ok)

    def _update_actions(self) -> None:
        busy = self._busy
        project = self.session.project
        ready = project is not None and not busy
        self.act_open.setEnabled(not busy)
        self.act_extract.setEnabled(not busy and self._info is not None)
        self.act_solve.setEnabled(ready)
        self.act_capacitance.setEnabled(ready)
        self.act_transient.setEnabled(ready)
        self.act_equilibrium.setEnabled(ready)
        self.act_export_gds.setEnabled(ready)
        self.modes.solve_button.setEnabled(ready)
        self.act_export_structure.setEnabled(project is not None)
        self.act_export_modes.setEnabled(project is not None and project.modal is not None)
        self.act_export_capacitance.setEnabled(project is not None and project.capacitance is not None)
        self.act_export_matrix.setEnabled(project is not None and project.capacitance is not None)
        self.act_export_transient.setEnabled(project is not None and project.transient is not None)
        self.act_export_gif.setEnabled(self.scene.has_motion())

    def _failed(self, title: str, message: str, trace: str) -> None:
        self._set_busy(False, ok=False)
        self.status_label.setText(f"{title}: {message}")
        box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
        box.setDetailedText(trace)
        box.exec()

    # --------------------------------------------------------------- open/load
    def open_file(self) -> None:
        if self._busy:
            return
        directory = self._qsettings.value("lastDir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Open GDS layout", directory, "GDSII layouts (*.gds *.gds2 *.GDS *.GDS2);;All files (*)")
        if path:
            self.open_path(path)

    def open_demo(self) -> None:
        """Open the resonator layout that ships with the application."""
        from connflex_app.data import demo_gds

        self.open_path(str(demo_gds()))

    def open_path(self, path: str) -> None:
        if self._busy:
            return
        path = os.path.abspath(path)
        self._set_busy(True, f"Reading {os.path.basename(path)}", ("inspect",), reset_timings=True)
        self.progress.stage_started("inspect", None)
        self._run(lambda: inspect_gds(path), lambda info: self._inspected(path, info), "Could not read the GDS file")

    def _inspected(self, path: str, info: GdsInfo) -> None:
        self._set_busy(False)
        if self._path is not None and self._info is not None and self.session.project is not None:
            self._remember(self._path, self._settings())  # keep the drive of the file being left
        self.session.set_project(None)
        self._path = path
        self._info = info
        remembered = self._remembered(path)
        if remembered is not None:
            self._apply_settings(remembered)
        else:
            self.drive.apply(DriveSettings(), self.drive.transient_settings())
        self.setup.set_gds_info(info, remembered)
        self._qsettings.setValue("lastDir", os.path.dirname(path))
        self.setWindowTitle(f"{os.path.basename(path)} — ConnFlex")
        self._update_actions()
        layers = ", ".join(s.label for s in info.layers(self.setup.cell_combo.currentText() or None))
        self.status_label.setText(f"Loaded {os.path.basename(path)} · layers {layers}. Check the layer roles, then extract.")
        self._preview()

    def _preview(self) -> None:
        """Show the raw layers of the chosen roles in the 2D view before extraction."""
        if self._busy or self._path is None or self._info is None:
            return
        try:
            settings = self.setup.settings()
            settings.layer_map()
        except ValueError:
            return
        path, bounds = self._path, self._info.bounds_m(settings.cell)
        if not self._busy:
            self.progress.set_message("Loading the layer preview…")

        def load():
            layout = load_layout(path, settings, bounds)
            return layout, prepare_layout(layout, box=load_box(path, settings, bounds))

        self._run(load, self._previewed, "Could not read the layers")

    def _previewed(self, result) -> None:
        layout, render = result
        if self.session.project is None:
            self.layout_view.set_layout(layout, render)
            if not self._busy:
                self.progress.set_message("Layer preview ready. Check the layer roles, then extract.")

    def load_config(self) -> None:
        directory = self._qsettings.value("lastDir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Load process configuration", directory, "TOML files (*.toml);;All files (*)")
        if not path:
            return
        try:
            self._apply_settings(self._settings().with_config(path))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Could not load the configuration", str(exc))
            return
        self.status_label.setText(f"Loaded process configuration from {os.path.basename(path)}.")

    # ------------------------------------------------------------- extract/solve
    def extract(self) -> None:
        if self._busy or self._path is None:
            return
        try:
            settings = self._settings()
            settings.layer_map()
            settings.stack()
            settings.params()
            settings.solve_options()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the setup", str(exc))
            return
        self._remember(self._path, settings)
        self.scene.clear_motion()
        self._set_busy(True, "Extracting", EXTRACT_STAGES, reset_timings=True)
        path = self._path
        bounds = self._info.bounds_m(settings.cell) if self._info is not None else None
        self._run(lambda: run_extraction(path, settings, self._progress, bounds), self._extracted, "Extraction failed")

    def _extracted(self, project: Project) -> None:
        self._set_busy(False)
        self.session.set_project(project)
        structure = project.structure
        rejected = sum(1 for c in structure.rejected if len(c.bodies) != 1)
        self.status_label.setText(
            f"Extracted {len(structure.bodies)} bodies, {len(structure.beams)} beams, {rejected} rejected candidates, "
            f"{structure.nets.count} electrical nets on {len(structure.electrodes)} electrodes in "
            f"{project.timings.get('extract', 0.0):.2f} s · 3D mesh {project.mesh.triangles:,} triangles."
        )
        self.solve()

    def solve(self) -> None:
        project = self.session.project
        if project is None or self._busy:
            return
        try:
            options = self.setup.settings().solve_options()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the solve options", str(exc))
            return
        self._set_busy(True, "Solving modes", MODAL_STAGES)
        started = time.perf_counter()
        self._run(
            lambda: (project.structure.modal(options, progress=self._progress), time.perf_counter() - started),
            self._solved,
            "Modal solve failed",
        )

    def _solved(self, result) -> None:
        modal, elapsed = result
        self._set_busy(False)
        project = self.session.project
        if project is None:
            return
        self.session.set_modal(modal)
        hidden = min(rigid_mode_count(project.structure), len(modal.frequencies_hz))
        free = free_bodies(project.structure)
        labels = describe_modes(project.structure, modal.dofs, modal.modes[:, hidden:])
        self.modes.set_modal(modal, labels, elapsed, hidden=hidden, free=free)
        self.drive.set_frequencies(modal.frequencies_hz[hidden:])
        self._update_actions()
        took = f"{elapsed * 1000:.0f} ms" if elapsed < 1.0 else f"{elapsed:.2f} s"
        shown = len(modal.frequencies_hz) - hidden
        text = f"Solved {len(modal.frequencies_hz)} modes in {took}."
        if hidden:
            text += f" {hidden} rigid-body modes of {len(free)} free bod{'y' if len(free) == 1 else 'ies'} hidden."
        if shown > 0:
            text += f" First mode at {fmt_freq(float(modal.frequencies_hz[hidden]))}."
            self.status_label.setText(text)
            self.modes.select_mode(hidden)
            if self._workspace == "mechanical":
                self._mode_selected(hidden)
        else:
            self.status_label.setText(text)
            if self._workspace == "mechanical":
                self._clear_motion()

    # ------------------------------------------------------------- capacitance
    def _current_capacitance(self) -> CapacitanceResult | None:
        """The capacitance model of the project, when the panel's settings would still give it."""
        project = self.session.project
        if project is None or project.capacitance is None:
            return None
        return project.capacitance if project.capacitance.matches(self.capacitance.settings()) else None

    def compute_capacitance(self) -> None:
        project = self.session.project
        if project is None or self._busy:
            return
        settings = self.capacitance.settings()
        self._remember(self._path, self._settings())
        self._set_busy(True, "Computing capacitance", CAPACITANCE_STAGES, cancellable=True, weights=STAGE_WEIGHTS)
        structure = project.structure
        self._run(
            lambda: compute_capacitance(structure, settings, self._progress),
            self._capacitance_done,
            "Capacitance could not be computed",
        )

    def _capacitance_done(self, result: CapacitanceResult) -> None:
        self._set_busy(False)
        project = self.session.project
        if project is None:
            return
        self.session.set_capacitance(result)
        structure = project.structure
        took = f"{result.elapsed * 1000:.0f} ms" if result.elapsed < 1.0 else f"{result.elapsed:.1f} s"
        totals = np.diag(result.model.net_matrix)
        largest = int(np.argmax(totals)) if len(totals) else -1
        text = f"Capacitance of {len(structure.electrodes)} electrodes on {structure.nets.count} nets in {took}: {result.summary()}."
        if largest >= 0:
            text += f" Net {largest} holds the most, {fmt_capacitance(float(totals[largest]))} to the others."
        self.status_label.setText(text)

    def _capacitance_changed(self) -> None:
        project = self.session.project
        self.capacitance.set_result(None if project is None else project.capacitance)
        self._update_model_status()
        self._update_actions()
        self._refresh_paint()

    def _update_model_status(self) -> None:
        """Tell the drive panel which capacitance model a run would use."""
        project = self.session.project
        link = '<a href="#model" style="color:#6ba1ff">Electrical tab</a>'
        if project is None:
            self.drive.set_model_status("")
        elif project.capacitance is None:
            self.drive.set_model_status(f"Capacitance model: not computed yet. A run solves it first, with the settings of the {link}.")
        elif self._current_capacitance() is None:
            self.drive.set_model_status(f"Capacitance model: its settings changed in the {link}, so a run solves it again first.")
        else:
            self.drive.set_model_status(f"Capacitance model: {project.capacitance.summary()} ({link}).")

    # ---------------------------------------------------------- electromechanical
    def _plan(self):
        """The drive resolved on the structure, or None after telling the user what to fix."""
        try:
            self.drive.transient_settings().options()
            return self.drive.plan()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the drive", str(exc))
            return None

    def run_transient(self) -> None:
        project = self.session.project
        if project is None or self._busy:
            return
        plan = self._plan()
        if plan is None:
            return
        if not plan.driven:
            answer = QMessageBox.question(
                self,
                "No net is driven",
                "The drive lists no net with a voltage source, so every net stays at 0 V and nothing will move.\n\n"
                "Click a net in a view to add it, then type its voltage in each step. Run anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        settings = self._settings()
        self._remember(self._path, settings)
        options = settings.transient.options()
        electro = settings.electro
        existing = self._current_capacitance()
        stages = (
            *(() if existing is not None else CAPACITANCE_STAGES),
            *(EQUILIBRIUM_STAGES if plan.settle_first else ()),
            *(("assemble 2", "transient") if plan.settle_first else TRANSIENT_STAGES),
        )
        self.scene.play(False)
        self._set_busy(True, "Running transient", stages, cancellable=True, weights=STAGE_WEIGHTS)
        structure = project.structure

        def work():
            capacitance = existing if existing is not None else compute_capacitance(structure, electro, self._progress)
            run = run_transient(structure, capacitance.model, plan, options, electro.workers, self._progress)
            return capacitance, run

        self._run(work, self._transient_done, "The transient failed")

    def _transient_done(self, result) -> None:
        capacitance, run = result
        self._set_busy(False)
        project = self.session.project
        if project is None:
            return
        if project.capacitance is not capacitance:
            self.session.set_capacitance(capacitance)
        project.equilibrium = None
        self._run_stale = False
        self.session.set_transient(run)
        summary = run.summary(project.structure)
        self.status_label.setText(f"Transient: {summary}.")
        self.drive.set_result(summary + ".")
        if run.ended_early:
            QMessageBox.information(
                self,
                "The transient ended early",
                f"The run stopped at {fmt_si(float(run.times[-1]), 's')} of {fmt_si(run.plan.duration, 's')}: a gap closed to the "
                "contact clearance, or the integrator could not step on. Nothing in the model describes bodies that touch.\n\n"
                + summary,
            )

    def _transient_changed(self) -> None:
        project = self.session.project
        run = None if project is None else project.transient
        self.timeline.set_run(run, None if project is None else project.structure)
        self._update_actions()
        if run is not None and self._workspace == "electromechanical":
            self._show_transient(play=True)
        elif run is None and self.scene.has_trajectory():
            self.scene.clear_motion()
            self._set_caption("No motion selected")
        self._refresh_paint()

    def _show_transient(self, play: bool) -> None:
        project = self.session.project
        if project is None or project.transient is None:
            return
        run = project.transient
        self.scene.set_rate(self.timeline.rate())
        self.scene.set_true_scale(self.timeline.true_scale.isChecked())
        self.scene.set_trajectory(run.times, run.u, run.theta, "Transient")
        if not play:
            self.scene.play(False)
        self._set_caption(f"Transient · {run.plan.steps} step{'s' if run.plan.steps != 1 else ''} · {fmt_si(float(run.times[-1]), 's')}")
        self._transient_time(self.scene.time())

    def _transient_time(self, moment: float) -> None:
        """The 3D view moved to simulation time ``moment``: follow with the plots and the colours."""
        project = self.session.project
        if project is None or project.transient is None or not self.scene.has_trajectory():
            return
        plan = project.transient.plan
        self.timeline.set_time(moment)
        self.scene.set_caption(f"Transient · t = {fmt_si(moment, 's', 4)} · step {plan.step_at(moment) + 1}/{plan.steps}")
        if self._workspace == "electromechanical":
            self._refresh_paint()

    def _scrubbed(self, moment: float) -> None:
        project = self.session.project
        if project is None or project.transient is None:
            return
        if not self.scene.has_trajectory():
            self._show_transient(play=False)
        self.scene.play(False)
        self.scene.set_time(moment)

    def _step_selected(self, step: int) -> None:
        """A step column was selected: show its voltages, or jump the playback to it."""
        project = self.session.project
        run = None if project is None else project.transient
        if run is not None and self.scene.has_trajectory() and not self._run_stale and step < run.plan.steps:
            self.scene.play(False)
            edges = run.plan.edges
            self.scene.set_time(min(float(edges[step]) + run.plan.rise_time, float(run.times[-1])))
        self._refresh_paint()

    def _drive_changed(self) -> None:
        project = self.session.project
        if project is not None and project.transient is not None and not self._run_stale:
            self._run_stale = True
            self.timeline.summary.setText("The drive changed since this run; run the transient again to update it.\n\n" + self.timeline.summary.text())
        self._refresh_paint()

    def solve_equilibrium(self, step: int | None = None) -> None:
        project = self.session.project
        if project is None or self._busy:
            return
        plan = self._plan()
        if plan is None:
            return
        step = self.drive.current_step() if step is None else int(step)
        settings = self._settings()
        self._remember(self._path, settings)
        electro = settings.electro
        existing = self._current_capacitance()
        stages = (*(() if existing is not None else CAPACITANCE_STAGES), *EQUILIBRIUM_STAGES)
        self.scene.play(False)
        self._set_busy(True, f"Solving equilibrium of step {step + 1}", stages, cancellable=True, weights=STAGE_WEIGHTS)
        structure = project.structure

        def work():
            capacitance = existing if existing is not None else compute_capacitance(structure, electro, self._progress)
            return capacitance, solve_equilibrium(structure, capacitance.model, plan, step, electro.workers, self._progress)

        self._run(work, self._equilibrium_done, "No equilibrium was found")

    def _equilibrium_done(self, result) -> None:
        capacitance, run = result
        self._set_busy(False)
        project = self.session.project
        if project is None:
            return
        if project.capacitance is not capacitance:
            self.session.set_capacitance(capacitance)
        project.equilibrium = run
        summary = run.summary(project.structure)
        self.status_label.setText(summary + ".")
        self.drive.set_result(summary + ".")
        if self._workspace == "electromechanical":
            self._show_equilibrium(run)
            self._refresh_paint()

    def _show_equilibrium(self, run: EquilibriumRun) -> None:
        project = self.session.project
        if project is None:
            return
        u, theta = motion_from_vector(project.structure, run.result.dofs, run.result.displacement)
        caption = f"Equilibrium · step {run.step + 1}"
        if np.abs(u).max(initial=0.0) < 1e-15 and np.abs(theta).max(initial=0.0) < 1e-15:
            self.scene.clear_motion()
            self._set_caption("Equilibrium at rest: nothing moves")
            return
        self.scene.set_motion(u, theta, animated=False, caption=caption)
        self._set_caption(caption)

    # ----------------------------------------------------------------- display
    def _project_changed(self) -> None:
        project = self.session.project
        self.layout_view.set_project(project)
        if project is not None:
            self.scene.set_mesh(project.mesh, body_centers(project.structure))
        else:
            self.scene.set_mesh(None)
        self._paint_2d = None
        self._last_table = None
        self._outlined_net = -1
        self._run_stale = False
        self._hover_group = -1
        self.modes.set_project(project)
        self.modes.set_modal(None, [])
        self.stiffness.set_project(project)
        self.capacitance.set_project(project)
        self.drive.set_project(project)
        self.timeline.set_run(None, None)
        self._enable_structure_chips(project is not None)
        self._set_caption("No motion selected")
        self._update_model_status()
        self._update_actions()
        self._refresh_paint()

    def _enable_structure_chips(self, on: bool) -> None:
        for key in ("bodies", "nets"):
            self.highlight_bar.set_enabled(key, on)
        for key in ("mech_graph", "elec_graph"):
            self.graphs_bar.set_enabled(key, on)

    def _selection_changed(self, body: int) -> None:
        self.layout_view.set_selected(body if self._workspace == "mechanical" else -1)
        self.scene.set_selected(body)
        self.stiffness.set_body(body)

    def _mode_selected(self, index: int) -> None:
        project = self.session.project
        if self._workspace != "mechanical":
            return
        if project is None or project.modal is None or index < 0:
            if index < 0:
                self.scene.clear_motion()
                self._set_caption("No motion selected")
            return
        modal = project.modal
        u, theta = motion_from_vector(project.structure, modal.dofs, modal.modes[:, index])
        caption = f"Mode {index + 1} · {fmt_freq(float(modal.frequencies_hz[index]))}"
        self.scene.set_motion(u, theta, animated=True, caption=caption)
        self._set_caption(caption)

    def _show_deflection(self, body: int, dof: int, fixed) -> None:
        project = self.session.project
        if project is None:
            return
        try:
            vector = static_response(project.system, body, dof, set(fixed))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Could not compute the deflection", str(exc))
            return
        u, theta = motion_from_vector(project.structure, project.system.dofs, vector)
        caption = f"Deflection · body {body} · unit {DOF_SHORT[DOF_ORDER[dof]]} load"
        self.modes.select_mode(-1)
        self.scene.set_motion(u, theta, animated=False, caption=caption)
        self._set_caption(caption)
        if not self.view3d.isVisible():
            self.view_actions["split"].trigger()

    def _clear_motion(self) -> None:
        self.modes.select_mode(-1)
        self.scene.clear_motion()
        self._set_caption("No motion selected")
        self._refresh_paint()

    def _set_caption(self, text: str) -> None:
        self.playback_caption.setText(text)
        self.play_button.setEnabled(self.scene.is_animated())
        self.gif_button.setEnabled(self.scene.has_motion())
        self.act_export_gif.setEnabled(self.scene.has_motion())
        self._playing_changed(self.scene.is_playing())
        self.view3d.relayout()

    def _play_requested(self, on: bool) -> None:
        """Play or pause from the playback bar or the timeline; a finished transient is loaded on demand."""
        project = self.session.project
        if on and self._workspace == "electromechanical" and not self.scene.has_trajectory():
            if project is not None and project.transient is not None:
                self._show_transient(play=True)
                return
        self.scene.play(on)
        self._playing_changed(self.scene.is_playing())

    def _playing_changed(self, playing: bool) -> None:
        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.setText("❚❚  Pause" if playing else "▶  Play")
        self.play_button.blockSignals(False)
        self.timeline.set_playing(playing and self.scene.has_trajectory())

    def _amplitude_changed(self, fraction: float) -> None:
        self.scene.set_amplitude(fraction)
        self.timeline.set_amplitude(fraction)
        self.modes.amplitude_slider.blockSignals(True)
        from connflex_app.widgets.modes_panel import slider_from_amplitude

        self.modes.amplitude_slider.setValue(slider_from_amplitude(fraction))
        self.modes.amplitude_value.setText(f"{fraction * 100:.1f} %")
        self.modes.amplitude_slider.blockSignals(False)

    def _highlight_toggled(self, key: str, on: bool) -> None:
        if key == "bodies":
            if on:
                self.highlight_bar.set_checked("nets", False)
            self._refresh_paint()
        elif key == "nets":
            if on:
                self.highlight_bar.set_checked("bodies", False)
            self._refresh_paint()
        else:
            self.layout_view.set_options(**{key: on})

    def _scene_toggled(self, key: str, on: bool) -> None:
        if key == "color_body":
            self._refresh_paint()
        else:
            self.scene.set_visible(SCENE_KINDS[key], on)

    # ------------------------------------------------------------------ colours
    def _set_paint_2d(self, mode: str, links: bool, highlights: bool = True) -> None:
        """Switch the 2D view between layer colours and the group table; rebuilds only on a change.

        ``links`` gives beams and rejected candidates their net's colour; with ``highlights``
        the Beams and Rejected chips still win over it, as in the Mechanical tab.
        """
        if self._paint_2d != (mode, links, highlights):
            self._paint_2d = (mode, links, highlights)
            self.layout_view.set_options(color_mode=mode, paint_links=links, highlight_links=highlights)

    def _voltages(self) -> tuple[np.ndarray, float, list[int]]:
        """Potential of every paint group to show now, the scale of the colour map, and the
        floating nets: from the transient at the time on show, else from the selected step."""
        project = self.session.project
        assert project is not None
        groups = project.groups
        run = project.transient
        if run is not None and self.scene.has_trajectory() and not self._run_stale:
            times = run.times
            position = float(np.interp(self.scene.time(), times, np.arange(len(times))))
            low = min(int(position), len(times) - 2)
            weight = position - low
            electrodes = (1.0 - weight) * run.result.potentials[low] + weight * run.result.potentials[low + 1]
            nets = (1.0 - weight) * run.nets[low] + weight * run.nets[low + 1]
            return paint.group_potentials(groups, electrodes, nets), run.scale, list(run.plan.floating)
        nets = np.zeros(groups.nets)
        for net, volts in self.drive.step_potentials().items():
            if 0 <= net < groups.nets:
                nets[net] = volts
        return nets[groups.net], self.drive.peak_voltage() or 1.0, self.drive.floating()

    def _outline_net(self, net: int) -> None:
        """Outline the silicon of ``net`` in the 2D view (the net selected in the drive), or nothing."""
        if net == self._outlined_net:
            return
        self._outlined_net = net
        project = self.session.project
        if project is None or net < 0:
            if self._workspace != "mechanical":
                self.layout_view.set_outline(None)
            return
        geometry = [g for group in project.groups.in_net(net) for g in project.render.group_geometry.get(int(group), [])]
        self.layout_view.set_outline(geometry)

    def _refresh_paint(self) -> None:
        """Colour both views for the current tab, selection, and time."""
        project = self.session.project
        if project is None:
            self._set_paint_2d("layer", False)
            self.scene.set_color_mode("layer")
            return
        groups = project.groups
        workspace = self._workspace
        self._outline_net(self.drive.current_net() if workspace == "electromechanical" else -1)
        if workspace == "mechanical":
            kind = "body" if self.highlight_bar.is_checked("bodies") else ("net" if self.highlight_bar.is_checked("nets") else None)
            if kind is None:
                self._set_paint_2d("layer", False)
            else:
                self.layout_view.set_group_colors(paint.by_category(groups, kind))
                self._set_paint_2d("group", kind == "net")  # by net, a beam that is not highlighted shows its net
            self._last_table = None
            self.scene.set_color_mode("body" if self.scene_options_bar.is_checked("color_body") else "layer")
            return
        if workspace == "electrical":
            table = self.capacitance.paint_table()
            current = self.capacitance.grouping()
            if current is not None:
                a, b = self.capacitance.picks()
                text = f"Colour by {current.kind}"
                if a >= 0:
                    text += f' · <span style="color:{AMBER}">■</span> A {current.labels[a]}'
                if b >= 0:
                    text += f' · <span style="color:{CYAN}">■</span> B {current.labels[b]}'
                self.electrical_legend.setText(text)
                self.view2d.relayout()
        else:
            potentials, scale, floating = self._voltages()
            table = paint.by_voltage(groups, potentials, scale, floating)
            selected = self.drive.current_net()
            marks = groups.in_net(selected) if selected >= 0 else ()
            hover = groups.in_net(int(groups.net[self._hover_group])) if 0 <= self._hover_group < groups.count else ()
            if len(marks) or len(hover):
                table = paint.with_outline(table, marks, hover)
            self.voltage_legend.set_scale(fmt_si(scale, "V", 3))
            note = "floating nets in violet" if floating else ""
            if self.voltage_note.text() != note:
                self.voltage_note.setText(note)
                self.view2d.relayout()
        if table is None:
            return
        if self._last_table is None or self._last_table.shape != table.shape or not np.array_equal(self._last_table, table):
            self._last_table = table
            self.layout_view.set_group_colors(table)
            self.scene.set_group_colors(table)
        self._set_paint_2d("group", True, highlights=False)
        self.scene.set_color_mode("group")

    # ------------------------------------------------------------------ picking
    def _group_clicked(self, group: int, modifiers: int) -> None:
        project = self.session.project
        if project is None:
            return
        if self._workspace == "electrical":
            second = bool(Qt.KeyboardModifier(modifiers) & SECOND_PICK)
            self.capacitance.pick_group(group, second)
        elif self._workspace == "electromechanical" and group >= 0:
            groups = project.groups
            electrode = group if group < groups.electrodes else default_source_electrode(project.structure, int(groups.net[group]))
            if electrode is not None:
                self.drive.add_net(int(electrode))
                self._refresh_paint()

    def _group_hovered(self, group: int) -> None:
        project = self.session.project
        if project is None:
            self.hover_label.setText("")
            return
        self._hover_group = group
        if self._workspace == "electrical":
            self.capacitance.hover_group(group)
            self.hover_label.setText(self.capacitance.describe_group(group))
        elif self._workspace == "electromechanical":
            text = ""
            if group >= 0:
                net = int(project.groups.net[group])
                text = describe_net(project.structure, net)
                potentials, _scale, floating = self._voltages()
                text += f" · {fmt_si(float(potentials[group]), 'V', 4)}" + (" · floating" if net in floating else "")
                if net not in self.drive.nets():
                    text += " · click to add it to the drive"
            self.hover_label.setText(text)
            self._refresh_paint()

    def _hover_info(self, info) -> None:
        self.hover_label.setText(info["text"] if info else "")

    def _hover_body(self, body: int) -> None:
        info = self.layout_view.describe("body", body) if body >= 0 else None
        self.hover_label.setText(info["text"] if info else "")

    def _set_view_mode(self, mode: str) -> None:
        self.view2d.setVisible(mode in ("split", "2d"))
        self.view3d.setVisible(mode in ("split", "3d"))

    # ------------------------------------------------------------------ export
    def export_gds_dialog(self) -> None:
        if self.session.project is None or self._busy:
            return
        dialog = GdsExportDialog(self, GdsExportDialog.suggest_path(self.session.project.path))
        if dialog.exec() != int(QDialog.DialogCode.Accepted) or not dialog.path:
            return
        self.export_gds_file(dialog.path, dialog.separate_release())

    def export_gds_file(self, path: str, separate_release: bool) -> None:
        """Write the extracted, released structure to ``path`` (see :mod:`connflex_app.gds_export`)."""
        project = self.session.project
        if project is None or self._busy:
            return
        source = project.settings.cell or (self._info.top_cells[0] if self._info and self._info.top_cells else "CONNFLEX")
        cell = "".join(ch if ch.isalnum() or ch in "_$" else "_" for ch in source)[:24] + "_RELEASED"
        self._set_busy(True, "Exporting GDS", EXPORT_GDS_STAGES)

        def done(result) -> None:
            self._set_busy(False)
            parts = [f"{GDS_LAYERS[name][0]}/{GDS_LAYERS[name][1]} {name} {result.polygons[name]:,}" for name in GDS_LAYERS]
            mode = "separate release holes" if separate_release else "holes in device"
            self.status_label.setText(f"Exported {os.path.basename(path)} ({mode}): " + ", ".join(parts) + " polygons.")

        self._run(
            lambda: export_gds(project.layout, project.structure, project.params, path, separate_release, cell, self._progress),
            done,
            "Could not export the GDS file",
        )

    def _save_path(self, title: str, suffix: str, pattern: str) -> str | None:
        project = self.session.project
        if project is None:
            return None
        suggested = os.path.splitext(project.path)[0] + suffix
        path, _ = QFileDialog.getSaveFileName(self, title, suggested, pattern)
        return path or None

    def export_structure(self) -> None:
        project = self.session.project
        path = self._save_path("Export structure", "_structure.json", "JSON (*.json)")
        if project is None or path is None:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(project.structure.to_dict(), handle)
        self.status_label.setText(f"Exported structure to {os.path.basename(path)}.")

    def export_modes(self) -> None:
        project = self.session.project
        if project is None or project.modal is None:
            return
        path = self._save_path("Export modes", "_modes.npz", "NumPy archive (*.npz)")
        if path is None:
            return
        project.modal.to_npz(path)
        self.status_label.setText(f"Exported modes to {os.path.basename(path)}.")

    def export_capacitance_report(self, path: str | None = None) -> None:
        """Write connflex-core's capacitance review report as JSON."""
        project = self.session.project
        if project is None or project.capacitance is None:
            return
        path = path or self._save_path("Export capacitance report", "_capacitance.json", "JSON (*.json)")
        if path is None:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(project.capacitance.model.to_dict(), handle)
        self.status_label.setText(f"Exported the capacitance report to {os.path.basename(path)}.")

    def export_capacitance_matrix(self, path: str | None = None) -> None:
        """Write the matrix the capacitance panel shows as CSV, in farads."""
        project = self.session.project
        if project is None or project.capacitance is None:
            return
        path = path or self._save_path("Export capacitance matrix", f"_capacitance_{self.capacitance.kind()}s.csv", "CSV (*.csv)")
        if path is None:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.capacitance.matrix_csv())
        self.status_label.setText(f"Exported the capacitance matrix between {self.capacitance.kind()}s to {os.path.basename(path)}.")

    def export_transient(self, path: str | None = None) -> None:
        """Write every series of the transient to a NumPy archive."""
        project = self.session.project
        if project is None or project.transient is None:
            return
        path = path or self._save_path("Export transient", "_transient.npz", "NumPy archive (*.npz)")
        if path is None:
            return
        run = project.transient
        result = run.result
        energy = result.energy
        np.savez_compressed(
            path,
            times=result.times,
            displacement=result.displacement,
            velocity=result.velocity,
            dof_body=np.array([body for body, _name in result.dofs]),
            dof_name=np.array([name for _body, name in result.dofs]),
            potentials=result.potentials,
            charge=result.charge,
            net_potentials=run.nets,
            current=result.current,
            source_net=np.array([drive.net for drive in run.plan.driven]),
            clearance=result.clearance,
            step_edges=run.plan.edges,
            energy_kinetic=energy.kinetic,
            energy_strain=energy.strain,
            energy_electrostatic=energy.electrostatic,
            energy_damped=energy.damped,
            energy_resistive=energy.resistive,
            energy_supplied=energy.supplied,
        )
        self.status_label.setText(f"Exported the transient to {os.path.basename(path)}.")

    def export_gif_dialog(self) -> None:
        if not self.scene.has_motion() or self._busy:
            return
        suggested = GifExportDialog.suggest_path(self._path, self.playback_caption.text())
        dialog = GifExportDialog(self, suggested, self.scene.frame_size(), transient=self.scene.has_trajectory())
        if dialog.exec() != int(QDialog.DialogCode.Accepted) or not dialog.path:
            return
        frames, fps, scale = dialog.values()
        self.export_gif(dialog.path, frames=frames, fps=fps, scale=scale)

    def export_gif(self, path: str, frames: int = 36, fps: int = 24, scale: float = 1.0) -> None:
        """Capture the current motion from the 3D view and write it to ``path``: one cycle of a
        mode, or a transient from its first time to its last."""
        if not self.scene.has_motion() or self._busy:
            return
        was_playing = self.scene.is_playing()
        self.scene.play(False)
        self._set_busy(True, "Exporting GIF", ("frames", "encode"))
        self.progress.stage_started("frames", frames)
        captured = []
        for index in range(frames):
            image = self.scene.capture_frame(2.0 * math.pi * index / frames)
            captured.append(frame_from_qimage(image, scale))
            self.progress.advanced(index + 1)
            QApplication.processEvents()
        self.progress.stage_finished("frames")
        width, height = captured[0][1], captured[0][2]

        def encode() -> int:
            with stage("encode", self._progress, total=len(captured)) as current:
                return encode_gif(captured, path, fps, advance=current.advance)

        def done(size: int) -> None:
            self._set_busy(False)
            self.scene.play(was_playing)
            self.status_label.setText(f"Saved {frames} frames ({width} × {height} px, {size / 1e6:.1f} MB) to {os.path.basename(path)}.")

        self._run(encode, done, "Could not write the GIF")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About ConnFlex",
            f"<b>ConnFlex {__version__}</b><br>A desktop front end for connflex-core {cf.__version__}.<br><br>"
            "Open a GDS layout, map its layers to device, handle, release, and via roles, and extract a "
            "reduced-order model of rigid bodies joined by beams.<br><br>"
            "<b>Mechanical</b>: beams, bodies, and their graphs in 2D, resonant modes animated in 3D, and the "
            "stiffness any body feels under different boundary conditions.<br>"
            "<b>Electrical</b>: nets and electrodes, and the capacitance between any two nets, bodies, or "
            "electrodes, with its parts and how it changes as bodies move.<br>"
            "<b>Electromechanical</b>: voltages on nets in steps, the equilibrium under each, and transients "
            "that follow the motion in time.",
        )
