"""Loaded project state shared by every view."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal
from shapely import Polygon

from connflex_core import ExtractionParams, Layout, ModalResult, Stack, Structure
from connflex_core.mechanics.assembly import SystemMatrices

from connflex_app.drive import EquilibriumRun, TransientRun
from connflex_app.electrical import CapacitanceResult
from connflex_app.geometry import Groups, Mesh, Piece
from connflex_app.render2d import RenderData
from connflex_app.settings import Settings


@dataclass
class Project:
    """A layout, the structure extracted from it, and everything derived for display."""

    path: str
    settings: Settings
    layout: Layout
    stack: Stack
    params: ExtractionParams
    structure: Structure
    holes: list[Polygon]
    pieces: list[Piece]
    mesh: Mesh
    system: SystemMatrices
    render: RenderData
    modal: ModalResult | None = None
    capacitance: CapacitanceResult | None = None
    transient: TransientRun | None = None
    equilibrium: EquilibriumRun | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def movable(self) -> list[int]:
        return [body.index for body in self.structure.bodies if not body.anchor]

    @property
    def groups(self) -> Groups:
        """Paint groups of the structure: its electrodes, then the beams of each net."""
        assert self.render.groups is not None
        return self.render.groups


class Session(QObject):
    """Holds the current project and the selected body; views subscribe to its signals."""

    projectChanged = Signal()
    modalChanged = Signal()
    capacitanceChanged = Signal()
    transientChanged = Signal()
    selectionChanged = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project: Project | None = None
        self.selected: int = -1

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.selected = -1
        self.projectChanged.emit()
        self.selectionChanged.emit(-1)

    def set_modal(self, modal: ModalResult | None) -> None:
        if self.project is None:
            return
        self.project.modal = modal
        self.modalChanged.emit()

    def set_capacitance(self, result: CapacitanceResult | None) -> None:
        if self.project is None:
            return
        self.project.capacitance = result
        self.capacitanceChanged.emit()

    def set_transient(self, run: TransientRun | None) -> None:
        if self.project is None:
            return
        self.project.transient = run
        self.transientChanged.emit()

    def select(self, body: int) -> None:
        if self.project is None:
            body = -1
        if body != self.selected:
            self.selected = body
            self.selectionChanged.emit(body)
