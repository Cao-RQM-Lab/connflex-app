"""Main window: wires the setup panel, both views, the mode list, and the stiffness inspector."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QApplication,
    QDialog,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QWidget,
)

import connflex_core as cf
from connflex_core.mechanics.assembly import assemble
from connflex_core.progress import stage

from connflex_app import __version__, theme
from connflex_app.analysis import free_bodies, rigid_mode_count, static_response
from connflex_app.gds_export import LAYERS as GDS_LAYERS, export_gds
from connflex_app.extraction import extract as extract_structure
from connflex_app.gds_io import GdsInfo, inspect_gds, load_box, load_layout
from connflex_app.gif_export import encode_gif, frame_from_qimage
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
    closed_plates,
    describe_modes,
    motion_from_vector,
    release_holes,
)
from connflex_app.render2d import prepare_layout, prepare_project
from connflex_app.session import Project, Session
from connflex_app.settings import Settings
from connflex_app.units import fmt_freq
from connflex_app.widgets.chips import ChipBar, OverlayHost
from connflex_app.widgets.gds_dialog import GdsExportDialog
from connflex_app.widgets.gif_dialog import GifExportDialog
from connflex_app.widgets.layout_view import LayoutView
from connflex_app.widgets.modes_panel import ModesPanel
from connflex_app.widgets.progress_status import ProgressStatus
from connflex_app.widgets.scene3d import Scene3D
from connflex_app.widgets.setup_panel import SetupPanel
from connflex_app.widgets.stiffness_panel import StiffnessPanel
from connflex_app.workers import ProgressRelay, Worker

SCENE_KINDS = {"device": KIND_DEVICE, "box": KIND_BOX, "handle": KIND_HANDLE, "beams": KIND_BEAM, "rejected": KIND_REJECTED}
EXTRACT_STAGES = ("load", "heal", "release", "prune", "nets", "opening", "bodies", "candidates", "beams", "mass", "mesh", "assemble", "prepare")
MODAL_STAGES = ("assemble", "solve")
EXPORT_GDS_STAGES = ("openings", "device", "layers", "write")


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
        plates = closed_plates(structure)
        mesh = build_mesh(structure, stack, plates, advance=current.advance, cache=cache)
    timings["mesh"] = time.perf_counter() - started
    with stage("assemble", progress):
        system = assemble(structure)
    with stage("prepare", progress, total=bodies) as current:
        render = prepare_project(layout, structure, plates, holes, advance=current.advance, cache=cache)
    del cache
    return Project(
        path=path,
        settings=settings,
        layout=layout,
        stack=stack,
        params=params,
        structure=structure,
        holes=holes,
        plates=plates,
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
        self._progress = ProgressRelay(self)
        self._build_views()
        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._wire()
        self._restored_state = self._restore()
        self._shown = False

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
        self.view2d.add_overlay(self.highlight_bar, "top-left")
        self.view2d.add_overlay(self.graphs_bar, "top-left")
        self.view2d.add_overlay(self.layers_bar, "top-left")

        self.scene_bar = ChipBar()
        self.scene_bar.add_label("Show")
        for key, text, color in (("device", "Device", colors["device"]), ("box", "BOX", colors["box"]), ("handle", "Handle", "#7b8396"), ("beams", "Beams", colors["beam"]), ("rejected", "Rejected", colors["rejected"])):
            self.scene_bar.add_toggle(key, text, True, color=color)
        self.scene_options_bar = ChipBar()
        self.scene_options_bar.add_toggle("color_body", "Colour by body", False, "Tint every body with its own colour")
        self.scene_options_bar.add_separator()
        self.scene_options_bar.add_button("Reset view", self.scene.reset_view, "Reset the camera (R, double-click)")
        self.view3d.add_overlay(self.scene_bar, "top-left")
        self.view3d.add_overlay(self.scene_options_bar, "top-left")

        self.playback_bar = ChipBar()
        self.play_button = self.playback_bar.add_toggle("play", "▶  Play", False, "Play or pause the animation (Space)")
        self.play_button.setEnabled(False)
        self.playback_caption = self.playback_bar.add_label("No motion selected")
        self.playback_bar.add_separator()
        self.gif_button = self.playback_bar.add_button("GIF…", self.export_gif_dialog, "Export one cycle of this animation as a GIF")
        self.gif_button.setEnabled(False)
        self.playback_bar.add_button("Clear", self._clear_motion, "Show the undeformed structure")
        self.view3d.add_overlay(self.playback_bar, "bottom-center")

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.view2d)
        self.splitter.addWidget(self.view3d)
        self.splitter.setSizes([800, 800])
        self.setCentralWidget(self.splitter)

    def _build_docks(self) -> None:
        self.setup = SetupPanel()
        self.setup.setMinimumWidth(340)
        self.dock_setup = self._dock("Setup", self.setup, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.modes = ModesPanel()
        self.modes.setMinimumWidth(320)
        self.dock_modes = self._dock("Modes", self.modes, Qt.DockWidgetArea.RightDockWidgetArea)
        self.stiffness = StiffnessPanel()
        self.stiffness.setMinimumWidth(320)
        self.dock_stiffness = self._dock("Stiffness", self.stiffness, Qt.DockWidgetArea.RightDockWidgetArea)

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
        self.act_extract.setEnabled(False)
        self.act_extract.triggered.connect(self.extract)
        self.act_solve = QAction("Solve modes", self)
        self.act_solve.setShortcut(QKeySequence("Ctrl+M"))
        self.act_solve.setEnabled(False)
        self.act_solve.triggered.connect(self.solve)
        self.act_export_gds = QAction("Export GDS…", self)
        self.act_export_gds.setToolTip("Export the extracted, released geometry as a GDS file")
        self.act_export_gds.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_export_gds.setEnabled(False)
        self.act_export_gds.triggered.connect(self.export_gds_dialog)
        self.act_fit = QAction("Fit 2D", self)
        self.act_fit.triggered.connect(self.layout_view.fit)
        self.act_reset = QAction("Reset 3D", self)
        self.act_reset.triggered.connect(self.scene.reset_view)
        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_extract)
        toolbar.addAction(self.act_solve)
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
        file_menu.addSeparator()
        self.act_export_structure = QAction("Export structure as JSON…", self)
        self.act_export_structure.setEnabled(False)
        self.act_export_structure.triggered.connect(self.export_structure)
        self.act_export_modes = QAction("Export modes as NPZ…", self)
        self.act_export_modes.setEnabled(False)
        self.act_export_modes.triggered.connect(self.export_modes)
        self.act_export_gif = QAction("Export animation as GIF…", self)
        self.act_export_gif.setEnabled(False)
        self.act_export_gif.triggered.connect(self.export_gif_dialog)
        file_menu.addAction(self.act_export_gds)
        file_menu.addAction(self.act_export_structure)
        file_menu.addAction(self.act_export_modes)
        file_menu.addAction(self.act_export_gif)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        view_menu = menu.addMenu("&View")
        for action in self.view_actions.values():
            view_menu.addAction(action)
        view_menu.addSeparator()
        for dock in (self.dock_setup, self.dock_modes, self.dock_stiffness):
            view_menu.addAction(dock.toggleViewAction())
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
        self._progress.stageFinished.connect(self.progress.stage_finished)
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
        self.layout_view.anchorPicked.connect(self.setup.set_anchor_point)
        self.scene.bodyClicked.connect(self.session.select)
        self.scene.hovered.connect(self._hover_body)
        self.scene.playingChanged.connect(self._playing_changed)
        self.session.projectChanged.connect(self._project_changed)
        self.session.selectionChanged.connect(self._selection_changed)
        self.highlight_bar.toggled.connect(self._highlight_toggled)
        self.graphs_bar.toggled.connect(self._highlight_toggled)
        self.layers_bar.toggled.connect(lambda key, on: self.layout_view.set_options(**{key: on}))
        self.scene_bar.toggled.connect(self._scene_toggled)
        self.scene_options_bar.toggled.connect(self._scene_toggled)
        self.play_button.toggled.connect(self.scene.play)
        self.modes.amplitudeChanged.connect(self.scene.set_amplitude)
        self.modes.speedChanged.connect(self.scene.set_speed)
        self.scene.set_amplitude(self.modes.amplitude())
        self.scene.set_speed(self.modes.speed())
        self._enable_structure_chips(False)

    # --------------------------------------------------------------- persistence
    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._shown:
            self._shown = True
            if not self._restored_state:
                self.resizeDocks([self.dock_modes, self.dock_stiffness], [430, 470], Qt.Orientation.Vertical)
                self.resizeDocks([self.dock_setup], [360], Qt.Orientation.Horizontal)

    def _restore(self) -> bool:
        geometry = self._qsettings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._qsettings.value("window/state")
        restored = False
        if state is not None:
            restored = bool(self.restoreState(state))
        defaults = self._qsettings.value("settings/defaults")
        if defaults:
            try:
                self.setup.apply(Settings.from_json(defaults))
            except Exception:  # noqa: BLE001 - stale settings are not fatal
                pass
        return restored

    def _remember(self, path: str, settings: Settings) -> None:
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
        self._qsettings.setValue("window/geometry", self.saveGeometry())
        self._qsettings.setValue("window/state", self.saveState())
        try:
            self._qsettings.setValue("settings/defaults", self.setup.settings().to_json())
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

        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.start()

    def _set_busy(self, busy: bool, phase: str = "", stages: tuple[str, ...] = (), reset_timings: bool = False, ok: bool = True) -> None:
        self._busy = busy
        self._phase = phase
        self.setup.set_busy(busy)
        self.act_open.setEnabled(not busy)
        self.act_extract.setEnabled(not busy and self._info is not None)
        self.act_solve.setEnabled(not busy and self.session.project is not None)
        self.act_export_gds.setEnabled(not busy and self.session.project is not None)
        self.modes.solve_button.setEnabled(not busy and self.session.project is not None)
        if busy:
            self.progress.start_phase(phase, stages, reset=reset_timings)
        else:
            self.progress.finish_phase(ok=ok)

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
        self._path = path
        self._info = info
        remembered = self._remembered(path)
        if remembered is not None:
            self.setup.apply(remembered)
        self.setup.set_gds_info(info, remembered)
        self.session.set_project(None)
        self._qsettings.setValue("lastDir", os.path.dirname(path))
        self.setWindowTitle(f"{os.path.basename(path)} — ConnFlex")
        self.act_extract.setEnabled(True)
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
            self.setup.apply(self.setup.settings().with_config(path))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Could not load the configuration", str(exc))
            return
        self.status_label.setText(f"Loaded process configuration from {os.path.basename(path)}.")

    # ------------------------------------------------------------- extract/solve
    def extract(self) -> None:
        if self._busy or self._path is None:
            return
        try:
            settings = self.setup.settings()
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
            f"{structure.nets.count} electrical nets in {project.timings.get('extract', 0.0):.2f} s · "
            f"3D mesh {project.mesh.triangles:,} triangles."
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
        self.act_export_modes.setEnabled(True)
        took = f"{elapsed * 1000:.0f} ms" if elapsed < 1.0 else f"{elapsed:.2f} s"
        shown = len(modal.frequencies_hz) - hidden
        text = f"Solved {len(modal.frequencies_hz)} modes in {took}."
        if hidden:
            text += f" {hidden} rigid-body modes of {len(free)} free bod{'y' if len(free) == 1 else 'ies'} hidden."
        if shown > 0:
            text += f" First mode at {fmt_freq(float(modal.frequencies_hz[hidden]))}."
            self.status_label.setText(text)
            self.modes.select_mode(hidden)
            self._mode_selected(hidden)
        else:
            self.status_label.setText(text)
            self._clear_motion()

    # ----------------------------------------------------------------- display
    def _project_changed(self) -> None:
        project = self.session.project
        self.layout_view.set_project(project)
        if project is not None:
            self.scene.set_mesh(project.mesh, body_centers(project.structure))
        else:
            self.scene.set_mesh(None)
        self.modes.set_project(project)
        self.modes.set_modal(None, [])
        self.stiffness.set_project(project)
        has_project = project is not None
        self.act_solve.setEnabled(has_project and not self._busy)
        self.act_export_structure.setEnabled(has_project)
        self.act_export_gds.setEnabled(has_project and not self._busy)
        self.act_export_modes.setEnabled(False)
        self._enable_structure_chips(has_project)
        self._set_caption("No motion selected")

    def _enable_structure_chips(self, on: bool) -> None:
        for key in ("bodies", "nets"):
            self.highlight_bar.set_enabled(key, on)
        for key in ("mech_graph", "elec_graph"):
            self.graphs_bar.set_enabled(key, on)

    def _selection_changed(self, body: int) -> None:
        self.layout_view.set_selected(body)
        self.scene.set_selected(body)
        self.stiffness.set_body(body)

    def _mode_selected(self, index: int) -> None:
        project = self.session.project
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

    def _set_caption(self, text: str) -> None:
        self.playback_caption.setText(text)
        self.play_button.setEnabled(self.scene.is_animated())
        self.gif_button.setEnabled(self.scene.has_motion())
        self.act_export_gif.setEnabled(self.scene.has_motion())
        self._playing_changed(self.scene.is_playing())
        self.view3d._place()

    def _playing_changed(self, playing: bool) -> None:
        self.play_button.blockSignals(True)
        self.play_button.setChecked(playing)
        self.play_button.setText("❚❚  Pause" if playing else "▶  Play")
        self.play_button.blockSignals(False)

    def _highlight_toggled(self, key: str, on: bool) -> None:
        if key == "bodies":
            if on:
                self.highlight_bar.set_checked("nets", False)
            self.layout_view.set_options(color_mode="body" if on else "layer")
        elif key == "nets":
            if on:
                self.highlight_bar.set_checked("bodies", False)
            self.layout_view.set_options(color_mode="net" if on else "layer")
        else:
            self.layout_view.set_options(**{key: on})

    def _scene_toggled(self, key: str, on: bool) -> None:
        if key == "color_body":
            self.scene.set_color_mode("body" if on else "layer")
        else:
            self.scene.set_visible(SCENE_KINDS[key], on)

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

    def export_structure(self) -> None:
        project = self.session.project
        if project is None:
            return
        suggested = os.path.splitext(project.path)[0] + "_structure.json"
        path, _ = QFileDialog.getSaveFileName(self, "Export structure", suggested, "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(project.structure.to_dict(), handle)
        self.status_label.setText(f"Exported structure to {os.path.basename(path)}.")

    def export_modes(self) -> None:
        project = self.session.project
        if project is None or project.modal is None:
            return
        suggested = os.path.splitext(project.path)[0] + "_modes.npz"
        path, _ = QFileDialog.getSaveFileName(self, "Export modes", suggested, "NumPy archive (*.npz)")
        if not path:
            return
        project.modal.to_npz(path)
        self.status_label.setText(f"Exported modes to {os.path.basename(path)}.")

    def export_gif_dialog(self) -> None:
        if not self.scene.has_motion() or self._busy:
            return
        suggested = GifExportDialog.suggest_path(self._path, self.playback_caption.text())
        dialog = GifExportDialog(self, suggested, self.scene.frame_size())
        if dialog.exec() != int(QDialog.DialogCode.Accepted) or not dialog.path:
            return
        frames, fps, scale = dialog.values()
        self.export_gif(dialog.path, frames=frames, fps=fps, scale=scale)

    def export_gif(self, path: str, frames: int = 36, fps: int = 24, scale: float = 1.0) -> None:
        """Capture one cycle of the current motion from the 3D view and write it to ``path``."""
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
            "reduced-order model of rigid bodies joined by beams. Inspect beams, bodies, electrical nets and "
            "their graphs in 2D, animate resonant modes in 3D, and read the stiffness any body feels under "
            "different boundary conditions.",
        )
