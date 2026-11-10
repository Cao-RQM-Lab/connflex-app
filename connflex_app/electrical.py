"""Capacitance between nets, bodies, and electrodes of an extracted structure.

connflex-core returns one capacitance matrix over the electrodes of a structure, the pieces of
silicon that share one potential and one rigid motion. This module ties electrodes together
into the things a user points at (a net, a body, or one electrode) and reads the matrix for
them: capacitance between two of them, its parts, and how it changes when a body moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from shapely import STRtree

from connflex_core import CapacitanceModel, ElectrostaticParams, Structure

from connflex_app.geometry import Groups
from connflex_app.settings import ElectroSettings

GROUPINGS: tuple[tuple[str, str], ...] = (("net", "Nets"), ("body", "Bodies"), ("electrode", "Electrodes"))
PARTS: tuple[tuple[str, str], ...] = (
    ("oxide", "Oxide"),
    ("released", "Released gap"),
    ("sidewall", "Sidewalls"),
    ("fringe", "Fringe"),
)
ELEMENT_FRACTION = 0.5  # element size as a fraction of the narrowest gap
REACH_FACTOR = 4.0  # interaction distance in narrowest gaps


@dataclass(frozen=True)
class Suggestion:
    """Lengths of the sidewall model sized from the layout.

    Attributes:
        gap: Narrowest gap between two device electrodes, in metres. None when no two device
            electrodes are near enough to interact, and then there are no sidewalls to solve.
        element_size: Half that gap, connflex-core's starting value.
        interaction_distance: Four times that gap.
    """

    gap: float | None
    element_size: float | None
    interaction_distance: float | None


def suggest_lengths(structure: Structure) -> Suggestion:
    """Element size and interaction distance from the narrowest gap between device electrodes."""
    regions = [e.region for e in structure.electrodes if e.layer == "device"]
    if len(regions) < 2:
        return Suggestion(None, None, None)
    tree = STRtree(regions)
    _pairs, distances = tree.query_nearest(regions, exclusive=True, return_distance=True, all_matches=False)
    distances = distances[distances > 0]
    if distances.size == 0:
        return Suggestion(None, None, None)
    gap = float(distances.min())
    return Suggestion(gap, ELEMENT_FRACTION * gap, REACH_FACTOR * gap)


@dataclass
class CapacitanceResult:
    """A capacitance model with what it was computed from."""

    model: CapacitanceModel
    params: ElectrostaticParams
    settings: ElectroSettings
    suggestion: Suggestion
    elapsed: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)

    def matches(self, settings: ElectroSettings) -> bool:
        """Whether ``settings`` would give this model again (the worker count does not change it)."""
        mine, theirs = dict(vars(self.settings)), dict(vars(settings))
        mine.pop("workers", None)
        theirs.pop("workers", None)
        return mine == theirs

    def summary(self) -> str:
        model = self.model
        parts = []
        if self.params.include_plates:
            parts.append(f"{len(model.plates)} plate{'s' if len(model.plates) != 1 else ''}")
        if self.params.include_sidewalls:
            windows = sum(cluster.windows for cluster in model.clusters)
            methods = sorted({interaction.method for cluster in model.clusters for interaction in cluster.interactions})
            text = f"{len(model.clusters)} sidewall cluster{'s' if len(model.clusters) != 1 else ''}"
            if methods:
                text += f" ({', '.join(methods)}"
                text += f", {windows} field solution{'s' if windows != 1 else ''})" if windows else ")"
            parts.append(text)
        else:
            parts.append("no sidewalls")
        return ", ".join(parts)


def compute_capacitance(structure: Structure, settings: ElectroSettings, progress=None) -> CapacitanceResult:
    """Capacitance model of ``structure`` at rest for ``settings``. Runs off the GUI thread."""
    suggestion = suggest_lengths(structure)
    params = settings.params(suggestion.element_size, suggestion.interaction_distance)
    started = time.perf_counter()
    model = structure.capacitance(params, workers=settings.workers, progress=progress)
    return CapacitanceResult(model, params, settings, suggestion, time.perf_counter() - started)


@dataclass(frozen=True, eq=False)
class Grouping:
    """Electrodes tied together into the members a user selects.

    Attributes:
        kind: ``"net"``, ``"body"``, or ``"electrode"``.
        keys: Net, body, or electrode index of each member, ascending.
        members: Electrode indices of each member.
        labels: Short name of each member, for table headers.
        titles: Longer description of each member.
        owner: Member of each electrode. Every electrode belongs to exactly one.
    """

    kind: str
    keys: tuple[int, ...]
    members: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    titles: tuple[str, ...]
    owner: NDArray[np.intp]

    @property
    def count(self) -> int:
        return len(self.keys)

    def rows(self, member: int) -> NDArray[np.intp]:
        """Electrode indices of ``member``, as an index array."""
        return np.asarray(self.members[member], dtype=np.intp)

    def between(self, matrix: NDArray[np.float64], a: int, b: int) -> float:
        """Entry (a, b) of ``matrix`` summed onto the members, without building the whole sum."""
        return float(matrix[np.ix_(self.rows(a), self.rows(b))].sum())

    def row(self, matrix: NDArray[np.float64], a: int) -> NDArray[np.float64]:
        """Row ``a`` of ``matrix`` summed onto the members."""
        onto: NDArray[np.float64] = np.bincount(self.owner, weights=matrix[self.rows(a)].sum(axis=0), minlength=self.count)
        return onto

    def index(self, key: int) -> int:
        """Position of the member with ``key``, or -1."""
        try:
            return self.keys.index(int(key))
        except ValueError:
            return -1

    def member_of_group(self, group: int, groups: Groups) -> int:
        """Member that paint group ``group`` belongs to (see :class:`Groups`), or -1.

        Beams and bridges are silicon of a net and of no electrode, so they count toward a
        net and toward nothing else.
        """
        if group < 0 or group >= groups.count:
            return -1
        if self.kind == "net":
            return self.index(int(groups.net[group]))
        if group >= groups.electrodes:
            return -1
        if self.kind == "body":
            return self.index(int(groups.body[group]))
        return self.index(group)

    def paint_groups(self, member: int, groups: Groups) -> list[int]:
        """Paint groups that show ``member`` in the views."""
        if member < 0 or member >= self.count:
            return []
        if self.kind == "net":
            return [*self.members[member], groups.of_net(self.keys[member])]
        return list(self.members[member])

    def reduce(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """An electrode matrix summed onto the members: entry (i, j) is the charge on member
        ``i`` per volt on member ``j`` with every other member at zero."""
        if self.kind == "electrode":
            return np.array(matrix, dtype=np.float64)
        size = len(self.owner)
        tie = sparse.csr_matrix((np.ones(size), (np.arange(size), self.owner)), shape=(size, self.count))
        reduced: NDArray[np.float64] = np.asarray(tie.T @ (tie.T @ matrix.T).T)
        return reduced


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _body_list(bodies: list[int]) -> str:
    if len(bodies) <= 4:
        return ", ".join(str(body) for body in bodies)
    return f"{bodies[0]}, {bodies[1]}, … {bodies[-1]} ({len(bodies)})"


def describe_net(structure: Structure, net: int) -> str:
    """What a net is made of, e.g. ``"Net 2 · handle · 1 anchored pad"``."""
    mine = [e for e in structure.electrodes if e.net == net]
    parts = [f"Net {net}"]
    if any(e.layer == "handle" for e in mine):
        parts.append("handle")
    movable = sorted({e.body for e in mine if e.layer == "device" and e.body != 0})
    if movable:
        parts.append(f"bod{'y' if len(movable) == 1 else 'ies'} {_body_list(movable)}")
    pads = sum(1 for e in mine if e.layer == "device" and e.body == 0)
    if pads:
        parts.append(_plural(pads, "anchored pad"))
    return " · ".join(parts)


def grouping(structure: Structure, kind: str) -> Grouping:
    """Members of ``kind`` over the electrodes of ``structure``."""
    electrodes = structure.electrodes
    if kind == "net":
        keys = list(range(structure.nets.count))
        members = [tuple(e.index for e in electrodes if e.net == net) for net in keys]
        labels = [f"N{net}" for net in keys]
        titles = [describe_net(structure, net) for net in keys]
    elif kind == "body":
        keys = sorted({e.body for e in electrodes})
        members = [tuple(e.index for e in electrodes if e.body == body) for body in keys]
        labels = ["Anchor" if body == 0 else f"B{body}" for body in keys]
        titles = []
        for body, mine in zip(keys, members):
            nets = sorted({electrodes[e].net for e in mine})
            name = "Anchor body" if body == 0 else f"Body {body}"
            titles.append(f"{name} · {_plural(len(mine), 'electrode')} · net{'s' if len(nets) != 1 else ''} {_body_list(nets)}")
    elif kind == "electrode":
        keys = [e.index for e in electrodes]
        members = [(e.index,) for e in electrodes]
        labels = [f"E{e.index}" for e in electrodes]
        titles = [
            f"Electrode {e.index} · {e.layer} · {'anchor' if e.body == 0 else f'body {e.body}'} · net {e.net}" for e in electrodes
        ]
    else:
        raise ValueError(f"unknown grouping {kind!r}")
    kept = [k for k, mine in enumerate(members) if mine]
    owner = np.zeros(len(electrodes), dtype=np.intp)
    for column, k in enumerate(kept):
        owner[list(members[k])] = column
    return Grouping(
        kind,
        tuple(keys[k] for k in kept),
        tuple(members[k] for k in kept),
        tuple(labels[k] for k in kept),
        tuple(titles[k] for k in kept),
        owner,
    )


@dataclass(frozen=True, eq=False)
class PairReport:
    """Capacitance between two members of a grouping, or of one member to everything else.

    Attributes:
        a: First member.
        b: Second member; equal to ``a`` for the capacitance of ``a`` to all the others.
        value: Farads between the two with every other member held, or, for one member,
            its total capacitance to the others.
        parts: The same split by where the field is: oxide, released, sidewall, fringe.
        total_a: Total capacitance of ``a`` to every other member.
        total_b: Total capacitance of ``b``.
        slope: Change of ``value`` with each degree of freedom of ``dofs``, farads per metre
            or per radian.
        dofs: (body, degree of freedom) of each entry of ``slope``.
    """

    a: int
    b: int
    value: float
    parts: dict[str, float]
    total_a: float
    total_b: float
    slope: NDArray[np.float64]
    dofs: tuple[tuple[int, str], ...]

    def by_body(self) -> list[tuple[int, NDArray[np.float64]]]:
        """Bodies whose motion changes ``value``, each with its six slopes, strongest first.

        Bodies are ranked by the capacitance they change per micrometre of travel plus per
        milliradian of turn, a scale on which typical travel and tilt compare.
        """
        rows = []
        for row, (body, name) in enumerate(self.dofs):
            if name == "ux":
                six = self.slope[row : row + 6]
                score = float(np.abs(six[:3]).max() * 1e-6 + np.abs(six[3:]).max() * 1e-3)
                rows.append((score, body, six))
        peak = max((score for score, _body, _six in rows), default=0.0)
        if peak <= 0:
            return []
        rows.sort(key=lambda row: -row[0])
        return [(body, six) for score, body, six in rows if score > 1e-6 * peak]


def pair_report(model: CapacitanceModel, group: Grouping, a: int, b: int) -> PairReport:
    """Capacitance between members ``a`` and ``b`` of ``group`` and what it depends on."""
    sign = 1.0 if a == b else -1.0
    slope = np.zeros(len(model.dofs))
    if model.gradient.size:
        # Only the electrodes of the two members enter, so slice before summing.
        slope = sign * model.gradient[np.ix_(group.rows(a), group.rows(b))].sum(axis=(0, 1))
    return PairReport(
        a=a,
        b=b,
        value=sign * group.between(model.matrix, a, b),
        parts={name: sign * group.between(getattr(model, name), a, b) for name, _title in PARTS},
        total_a=group.between(model.matrix, a, a),
        total_b=group.between(model.matrix, b, b),
        slope=slope,
        dofs=tuple(model.dofs),
    )


def couplings(model: CapacitanceModel, group: Grouping, a: int) -> list[tuple[int, float]]:
    """Capacitance from member ``a`` to every other member, largest first, zeros left out."""
    row = -group.row(model.matrix, a)
    order = [k for k in np.argsort(-row) if k != a and row[k] > 0]
    return [(int(k), float(row[k])) for k in order]


def member_total(model: CapacitanceModel, group: Grouping, member: int) -> float:
    """Total capacitance of one member to all the others."""
    return group.between(model.matrix, member, member)


def member_between(model: CapacitanceModel, group: Grouping, a: int, b: int) -> float:
    """Capacitance between two different members."""
    return -group.between(model.matrix, a, b)


def mutual_matrix(model: CapacitanceModel, group: Grouping, part: str = "matrix") -> NDArray[np.float64]:
    """Members by members: capacitance between each two off the diagonal, and each member's
    total capacitance to the others on it. ``part`` is ``"matrix"`` for the whole or one of
    oxide, released, sidewall, and fringe."""
    reduced = group.reduce(getattr(model, part))
    mutual = -reduced
    np.fill_diagonal(mutual, np.diag(reduced))
    return mutual


def net_potentials(structure: Structure, potentials: NDArray[np.float64]) -> NDArray[np.float64]:
    """Potential of each net from potentials per electrode (last axis): the mean over its
    electrodes, which differ only while current flows through beams with resistance."""
    values = np.asarray(potentials, dtype=np.float64)
    net = np.array([electrode.net for electrode in structure.electrodes], dtype=np.intp)
    count = structure.nets.count
    tie = sparse.csr_matrix((np.ones(len(net)), (np.arange(len(net)), net)), shape=(len(net), count))
    flat = values.reshape(-1, values.shape[-1])
    sums = np.asarray((tie.T @ flat.T).T)
    sizes = np.maximum(np.bincount(net, minlength=count), 1)
    mean: NDArray[np.float64] = (sums / sizes).reshape(*values.shape[:-1], count)
    return mean
