"""Voltage steps on nets, and the transient and equilibrium analyses they drive.

A drive is a table: one column per step with its duration, one row per driven net with its
voltage in each step. Every row becomes one voltage source bonded to an electrode of its net,
with a piecewise-linear waveform that ramps from each step's voltage to the next over the rise
time. Nets that are not listed stay at zero volts; a floating net keeps zero charge.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

import connflex_core as cf
from connflex_core import CapacitanceModel, Equilibrium, Structure, Transient, TransientOptions

from connflex_app.electrical import net_potentials
from connflex_app.geometry import DOF_ORDER, DOF_SHORT, motion_from_history
from connflex_app.settings import DriveSettings, TransientSettings
from connflex_app.units import fmt_si

RISE_FRACTION = 1.0 / 50.0  # automatic rise time as a fraction of the shortest step
NO_MOTION = 1e-15  # metres: travel below a femtometre is rounding, not motion
PERIODS_PER_STEP = 8.0  # automatic step duration in periods of the lowest mode
FALLBACK_DURATION = 1e-3


class Cancelled(Exception):
    """The user stopped a running analysis."""


@dataclass(frozen=True)
class NetDrive:
    """One driven or floating net of a plan.

    Attributes:
        net: Net index.
        electrode: Electrode of that net the source is bonded to.
        mode: ``"drive"`` for a voltage source, ``"float"`` for a net that keeps zero charge.
        volts: Voltage in each step, for a driven net.
    """

    net: int
    electrode: int
    mode: str
    volts: tuple[float, ...]


@dataclass(frozen=True)
class DrivePlan:
    """Resolved drive: durations in seconds, sources by net, and how the run is sampled."""

    durations: tuple[float, ...]
    nets: tuple[NetDrive, ...]
    rise_time: float
    samples: int = 200
    settle_first: bool = False
    source_resistance: float = 0.0

    def __post_init__(self) -> None:
        if not self.durations:
            raise ValueError("a drive needs at least one step")
        for index, duration in enumerate(self.durations, start=1):
            if not (math.isfinite(duration) and duration > 0):
                raise ValueError(f"step {index} needs a duration above zero")
        if not (math.isfinite(self.rise_time) and self.rise_time > 0):
            raise ValueError("the rise time must be above zero: a source cannot jump in no time")
        if self.rise_time >= min(self.durations):
            raise ValueError(
                f"the rise time ({fmt_si(self.rise_time, 's')}) must be shorter than the shortest step "
                f"({fmt_si(min(self.durations), 's')})"
            )
        if self.samples < 2:
            raise ValueError("each step needs at least 2 output samples")
        seen: set[int] = set()
        for drive in self.nets:
            if drive.net in seen:
                raise ValueError(f"net {drive.net} is listed twice")
            seen.add(drive.net)
            if drive.mode == "drive" and len(drive.volts) != len(self.durations):
                raise ValueError(f"net {drive.net} needs one voltage for each of the {len(self.durations)} steps")

    @property
    def steps(self) -> int:
        return len(self.durations)

    @property
    def edges(self) -> NDArray[np.float64]:
        """Time at which each step begins, and the end of the last one."""
        return np.concatenate([[0.0], np.cumsum(self.durations)])

    @property
    def duration(self) -> float:
        return float(sum(self.durations))

    @property
    def driven(self) -> tuple[NetDrive, ...]:
        return tuple(drive for drive in self.nets if drive.mode == "drive")

    @property
    def floating(self) -> tuple[int, ...]:
        return tuple(drive.net for drive in self.nets if drive.mode == "float")

    def step_at(self, moment: float) -> int:
        """Step in progress at ``moment``."""
        return int(np.clip(np.searchsorted(self.edges, moment, side="right") - 1, 0, self.steps - 1))

    def potentials(self, step: int) -> dict[int, float]:
        """Voltage by net held in ``step``, for an equilibrium at that operating point."""
        return {drive.net: float(drive.volts[step]) for drive in self.driven}

    def charges(self) -> dict[int, float]:
        return dict.fromkeys(self.floating, 0.0)

    def breakpoints(self, drive: NetDrive) -> tuple[list[float], list[float]]:
        """Points of the waveform of ``drive``: a ramp over the rise time wherever the voltage changes."""
        edges = self.edges
        volts = drive.volts
        stamps: list[float] = [0.0]
        values: list[float] = [volts[0] if self.settle_first else 0.0]
        if not self.settle_first and volts[0] != 0.0:
            stamps.append(self.rise_time)
            values.append(volts[0])
        for k in range(1, self.steps):
            if volts[k] != volts[k - 1]:
                if edges[k] > stamps[-1]:
                    stamps.append(float(edges[k]))
                    values.append(volts[k - 1])
                stamps.append(float(edges[k]) + self.rise_time)
                values.append(volts[k])
        if edges[-1] > stamps[-1]:
            stamps.append(float(edges[-1]))
            values.append(values[-1])
        return stamps, values

    def waveform(self, drive: NetDrive) -> cf.PiecewiseLinear | float:
        stamps, values = self.breakpoints(drive)
        if len(set(values)) == 1:
            return float(values[0])
        return cf.PiecewiseLinear(stamps, values)

    def value(self, drive: NetDrive, moment: float) -> float:
        """Voltage the source of ``drive`` asks for at ``moment``."""
        stamps, values = self.breakpoints(drive)
        return float(np.interp(moment, stamps, values))

    def times(self) -> NDArray[np.float64]:
        """Output times: ``samples`` per step plus every corner of a waveform, ascending."""
        edges = self.edges
        grid = [np.linspace(edges[k], edges[k + 1], self.samples, endpoint=False) for k in range(self.steps)]
        corners = [stamp for drive in self.driven for stamp in self.breakpoints(drive)[0]]
        stamps = np.unique(np.concatenate([*grid, [edges[-1]], corners]))
        stamps = stamps[(stamps >= 0.0) & (stamps <= edges[-1])]
        # A corner a rounding error away from a grid time would ask for the same output twice.
        tolerance = 1e-9 * float(edges[-1])
        kept = [float(stamps[0])]
        for stamp in stamps[1:]:
            if stamp - kept[-1] > tolerance:
                kept.append(float(stamp))
        kept[-1] = float(edges[-1])
        return np.asarray(kept)

    def describe_step(self, step: int) -> str:
        """One step in words, e.g. ``"step 2/3: N0 40 V, N2 0 V"``."""
        parts = [f"N{drive.net} {fmt_si(drive.volts[step], 'V')}" for drive in self.driven]
        text = f"step {step + 1}/{self.steps}"
        return f"{text}: {', '.join(parts)}" if parts else text


def default_duration(frequencies_hz) -> float:
    """Step duration when none is given: eight periods of the lowest resonance, two digits."""
    lowest = None
    if frequencies_hz is not None:
        positive = [float(f) for f in np.atleast_1d(frequencies_hz) if np.isfinite(f) and f > 1.0]
        lowest = min(positive) if positive else None
    if lowest is None:
        return FALLBACK_DURATION
    raw = PERIODS_PER_STEP / lowest
    magnitude = 10.0 ** math.floor(math.log10(raw))
    return round(raw / magnitude, 1) * magnitude


def default_source_electrode(structure: Structure, net: int) -> int | None:
    """Electrode of ``net`` a source is most likely bonded to: its largest anchored device
    pad, else its largest device electrode, else its handle."""
    mine = [e for e in structure.electrodes if e.net == net]
    if not mine:
        return None

    def rank(electrode) -> tuple[int, float]:
        tier = 2 if electrode.layer == "handle" else (0 if electrode.body == 0 else 1)
        return (tier, -electrode.region.area)

    return min(mine, key=rank).index


def resolve_plan(
    structure: Structure,
    drive: DriveSettings,
    options: TransientSettings,
    frequencies_hz=None,
) -> DrivePlan:
    """The plan a saved drive describes on ``structure``.

    Each row finds its electrode again from its point. A point that no longer lies on an
    electrode is an error that names it.
    """
    fallback = default_duration(frequencies_hz)
    durations = tuple(fallback if d is None else float(d) for d in drive.durations)
    nets = []
    for row in drive.nets:
        try:
            electrode = structure.electrode_at(tuple(row.point), row.layer)
        except ValueError:
            x, y = row.point
            raise ValueError(
                f"the drive lists a net at ({x * 1e6:.1f}, {y * 1e6:.1f}) µm on the {row.layer} layer, "
                "but no electrode is there in this structure; remove the row or pick the net again"
            ) from None
        volts = tuple(float(v) for v in row.volts)
        volts = (volts + (0.0,) * len(durations))[: len(durations)]
        nets.append(NetDrive(electrode.net, electrode.index, row.mode, volts))
    rise = options.rise_time if options.rise_time is not None else RISE_FRACTION * min(durations)
    return DrivePlan(
        durations=durations,
        nets=tuple(nets),
        rise_time=rise,
        samples=options.samples,
        settle_first=options.settle_first,
        source_resistance=options.source_resistance,
    )


def build_sources(structure: Structure, plan: DrivePlan) -> list[cf.Source]:
    """One voltage source per driven net, bonded to its electrode."""
    return [
        cf.Source(structure.electrodes[drive.electrode], plan.waveform(drive), resistance=plan.source_resistance)
        for drive in plan.driven
    ]


class StepProgress:
    """Progress receiver that says which step a transient is in and how far in time.

    Wraps another receiver. While the ``transient`` stage runs, every output time it reports is
    described through the wrapped receiver's ``describe``, when it has one.
    """

    def __init__(self, inner, plan: DrivePlan, stamps: NDArray[np.float64]) -> None:
        self._inner = inner
        self._plan = plan
        self._stamps = stamps
        self._stage = ""
        self._count = 0

    def start(self, stage: str, total: int | None) -> None:
        self._stage = stage
        self._count = 0
        self._inner.start(stage, total)

    def advance(self, n: int = 1) -> None:
        self._count += n
        if self._stage == "transient":
            describe = getattr(self._inner, "describe", None)
            if describe is not None:
                moment = float(self._stamps[min(self._count, len(self._stamps)) - 1])
                step = self._plan.step_at(moment)
                describe(f"{self._plan.describe_step(step)} · t = {fmt_si(moment, 's')} of {fmt_si(self._plan.duration, 's')}")
        self._inner.advance(n)

    def finish(self) -> None:
        self._inner.finish()


@dataclass(eq=False)
class TransientRun:
    """A finished transient with the plan that drove it and what the views play back.

    Attributes:
        plan: The drive.
        result: connflex-core's transient.
        start: Equilibrium the run started from, when the plan settles first.
        u, theta: Translation and rotation of every body at every output time, shape
            (times, bodies, 3).
        nets: Potential of every net at every output time, shape (times, nets): the mean
            over its electrodes.
        elapsed: Seconds the run took.
    """

    plan: DrivePlan
    result: Transient
    start: Equilibrium | None
    u: NDArray[np.float64]
    theta: NDArray[np.float64]
    nets: NDArray[np.float64]
    elapsed: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def times(self) -> NDArray[np.float64]:
        return self.result.times

    @property
    def ended_early(self) -> bool:
        return self.result.contact is not None

    @property
    def scale(self) -> float:
        """Largest potential of the run in magnitude, for the colour map; 1 when all are zero."""
        peak = float(np.abs(self.result.potentials).max()) if self.result.potentials.size else 0.0
        return peak if np.isfinite(peak) and peak > 0 else 1.0

    def source_potential(self, drive: NetDrive) -> NDArray[np.float64]:
        """Potential over time where the source of ``drive`` connects, or of a floating net."""
        if drive.mode == "drive":
            return self.result.potentials[:, drive.electrode]
        return self.nets[:, drive.net]

    def dominant_body(self) -> int:
        """Movable body that travels farthest over the run, or -1."""
        travel = np.abs(self.u).max(axis=(0, 2))
        if travel.size == 0 or not np.isfinite(travel).any() or travel.max() <= 0:
            movable = [body for body, name in self.result.dofs if name == "ux"]
            return movable[0] if movable else -1
        return int(np.argmax(travel))

    def summary(self, structure: Structure) -> str:
        result = self.result
        took = f"{self.elapsed * 1000:.0f} ms" if self.elapsed < 1.0 else f"{self.elapsed:.1f} s"
        text = (
            f"{len(result.times):,} output times to {fmt_si(float(result.times[-1]), 's')} in {took} · "
            f"{result.steps:,} integration steps, {result.evaluations:,} model evaluations"
        )
        body = self.dominant_body()
        if body >= 0:
            peak = np.abs(self.u[:, body]).max(axis=0)
            axis = int(np.argmax(peak))
            if peak[axis] < NO_MOTION:
                text += " · nothing moves"
            else:
                text += f" · body {body} peaks at {fmt_si(float(peak[axis]), 'm')} along {DOF_SHORT[DOF_ORDER[axis]]}"
        text += f" · narrowest gap {float(result.clearance.min()) * 100:.1f} % of rest"
        if result.contact is not None:
            if result.contact == (-1, -1):
                text += f" · STOPPED at {fmt_si(float(result.times[-1]), 's')}: the integrator could not step on"
            else:
                a, b = result.contact
                text += (
                    f" · CONTACT at {fmt_si(float(result.times[-1]), 's')} between electrodes {a} and {b} "
                    f"(nets {structure.electrodes[a].net} and {structure.electrodes[b].net})"
                )
        supplied = float(np.abs(result.energy.supplied).max())
        if supplied > 0:
            text += f" · energy imbalance {float(np.abs(result.energy.imbalance).max()) / supplied:.1e} of supplied"
        return text


def run_transient(
    structure: Structure,
    model: CapacitanceModel,
    plan: DrivePlan,
    options: TransientOptions,
    workers: int | None = None,
    progress=None,
) -> TransientRun:
    """Follow ``structure`` through every step of ``plan``. Runs off the GUI thread."""
    started = time.perf_counter()
    stamps = plan.times()
    charges = plan.charges() or None
    start = None
    if plan.settle_first:
        start = structure.equilibrium(model, plan.potentials(0), charges=charges, workers=workers, progress=progress)
    result = structure.transient(
        model,
        build_sources(structure, plan),
        stamps,
        options,
        charges=charges,
        start=start,
        workers=workers,
        progress=None if progress is None else StepProgress(progress, plan, stamps),
    )
    u, theta = motion_from_history(structure, result.dofs, result.displacement)
    nets = net_potentials(structure, result.potentials)
    return TransientRun(plan, result, start, u, theta, nets, time.perf_counter() - started)


@dataclass(eq=False)
class EquilibriumRun:
    """Where the structure rests under the voltages of one step."""

    plan: DrivePlan
    step: int
    result: Equilibrium
    elapsed: float = 0.0

    def summary(self, structure: Structure) -> str:
        result = self.result
        state = "stable" if result.stable else "UNSTABLE"
        text = (
            f"Equilibrium of {self.plan.describe_step(self.step)} · {state} · {result.margin * 100:.1f} % of beam "
            f"stiffness left · narrowest gap {result.model.clearance * 100:.1f} % of rest · {result.iterations} iterations"
        )
        best: tuple[float, int, int] | None = None
        for row, (body, name) in enumerate(result.dofs):
            if name == "ux":
                for axis in range(3):
                    value = abs(float(result.displacement[row + axis]))
                    if best is None or value > best[0]:
                        best = (value, body, axis)
        if best is not None and best[0] >= NO_MOTION:
            signed = float(result.displacement[result.dofs.index((best[1], DOF_ORDER[best[2]]))])
            text += f" · body {best[1]} moves {fmt_si(signed, 'm')} along {DOF_SHORT[DOF_ORDER[best[2]]]}"
        elif best is not None:
            text += " · nothing moves"
        if result.loose:
            text += f" · {len(result.loose)} loose bod{'y' if len(result.loose) == 1 else 'ies'} kept at rest"
        return text


def solve_equilibrium(
    structure: Structure,
    model: CapacitanceModel,
    plan: DrivePlan,
    step: int,
    workers: int | None = None,
    progress=None,
) -> EquilibriumRun:
    """Equilibrium under the voltages of ``step``, found from rest. Runs off the GUI thread."""
    started = time.perf_counter()
    result = structure.equilibrium(
        model, plan.potentials(step), charges=plan.charges() or None, workers=workers, progress=progress
    )
    return EquilibriumRun(plan, step, result, time.perf_counter() - started)
