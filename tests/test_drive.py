"""Voltage steps on nets: waveforms, plans, transients, equilibria, and cancelling."""

from __future__ import annotations

import numpy as np
import pytest

import connflex_core as cf

from connflex_app.drive import (
    Cancelled,
    DrivePlan,
    NetDrive,
    StepProgress,
    build_sources,
    default_duration,
    default_source_electrode,
    resolve_plan,
    run_transient,
    solve_equilibrium,
)
from connflex_app.settings import DriveNet, DriveSettings, TransientSettings

STATOR, SHUTTLE, LEFT = (1050e-6, 600e-6), (800e-6, 600e-6), (150e-6, 600e-6)


def _plan(**changes) -> DrivePlan:
    values = dict(
        durations=(100e-6, 300e-6),
        nets=(NetDrive(0, 4, "drive", (40.0, 0.0)), NetDrive(2, 5, "drive", (0.0, 0.0)), NetDrive(1, 7, "float", ())),
        rise_time=2e-6,
        samples=50,
    )
    values.update(changes)
    return DrivePlan(**values)


def test_waveforms_ramp_between_steps():
    plan = _plan()
    first, second, _floating = plan.nets
    assert plan.steps == 2 and plan.duration == pytest.approx(400e-6)
    assert np.allclose(plan.edges, [0.0, 100e-6, 400e-6])
    stamps, values = plan.breakpoints(first)
    assert stamps == pytest.approx([0.0, 2e-6, 100e-6, 102e-6, 400e-6]) and values == [0.0, 40.0, 40.0, 0.0, 0.0]
    assert isinstance(plan.waveform(first), cf.PiecewiseLinear)
    assert plan.waveform(second) == 0.0  # never leaves zero: a constant source
    assert plan.value(first, 1e-6) == pytest.approx(20.0) and plan.value(first, 50e-6) == 40.0
    assert plan.value(first, 101e-6) == pytest.approx(20.0) and plan.value(first, 300e-6) == 0.0
    assert [plan.step_at(t) for t in (0.0, 99e-6, 100e-6, 399e-6, 1.0)] == [0, 0, 1, 1, 1]
    assert plan.potentials(0) == {0: 40.0, 2: 0.0} and plan.charges() == {1: 0.0} and plan.floating == (1,)
    assert plan.describe_step(0) == "step 1/2: N0 40 V, N2 0 V"
    settled = _plan(settle_first=True)
    assert settled.breakpoints(first)[1][0] == 40.0  # starts at the first step's voltage, no ramp up


def test_output_times_hold_every_corner():
    plan = _plan()
    times = plan.times()
    assert times[0] == 0.0 and times[-1] == pytest.approx(400e-6)
    assert np.all(np.diff(times) > 0)
    for corner in (2e-6, 100e-6, 102e-6):
        assert np.isclose(times, corner, rtol=0, atol=1e-15).any()
    assert len(times) >= 2 * 50 + 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(rise_time=100e-6), "shorter than the shortest step"),
        (dict(rise_time=0.0), "rise time must be above zero"),
        (dict(durations=(100e-6, 0.0)), "step 2 needs a duration"),
        (dict(samples=1), "at least 2 output samples"),
        (dict(nets=(NetDrive(0, 4, "drive", (1.0, 2.0)), NetDrive(0, 4, "drive", (1.0, 2.0)))), "listed twice"),
        (dict(nets=(NetDrive(0, 4, "drive", (1.0,)),)), "one voltage for each"),
    ],
)
def test_a_plan_says_what_is_wrong(changes, message):
    with pytest.raises(ValueError, match=message):
        _plan(**changes)


def test_default_duration_and_source():
    assert default_duration([0.0, 45_560.0, 281_000.0]) == pytest.approx(1.8e-4)  # eight periods, two digits
    assert default_duration(None) == 1e-3 and default_duration([]) == 1e-3


def test_resolving_a_saved_drive(demo):
    structure = demo["structure"]
    stator = structure.electrode_at(STATOR, "device")
    assert default_source_electrode(structure, stator.net) == stator.index
    shuttle_net = structure.electrode_at(SHUTTLE, "device").net
    pad = structure.electrodes[default_source_electrode(structure, shuttle_net)]
    assert pad.body == 0 and pad.layer == "device"  # a source is bonded to an anchored pad, not to a moving mass
    drive = DriveSettings(durations=[None, 50e-6], nets=[DriveNet(STATOR, "device", "drive", [40.0]), DriveNet(LEFT, "device", "float", [])])
    plan = resolve_plan(structure, drive, TransientSettings(), demo["modal"].frequencies_hz)
    assert plan.durations == pytest.approx((1.8e-4, 50e-6))
    assert plan.rise_time == pytest.approx(50e-6 / 50)
    assert plan.nets[0] == NetDrive(stator.net, stator.index, "drive", (40.0, 0.0))  # missing voltages are 0 V
    assert plan.nets[1].mode == "float"
    sources = build_sources(structure, plan)
    assert len(sources) == 1 and sources[0].electrode is structure.electrodes[stator.index]
    with pytest.raises(ValueError, match="no electrode is there"):
        resolve_plan(structure, DriveSettings(nets=[DriveNet((-1.0, -1.0), "device", "drive", [1.0])]), TransientSettings())


class _Recorder:
    def __init__(self, cancel_after: int | None = None) -> None:
        self.stages: list[str] = []
        self.details: list[str] = []
        self.count = 0
        self.cancel_after = cancel_after

    def start(self, stage, total):
        self.stages.append(stage)
        self.total = total

    def advance(self, n=1):
        self.count += n
        if self.cancel_after is not None and self.count >= self.cancel_after:
            raise Cancelled

    def describe(self, text):
        self.details.append(text)

    def finish(self):
        pass


def test_transient_pulls_the_masses_toward_a_biased_handle(demo, plates_model):
    structure = demo["structure"]
    left = structure.electrode_at(LEFT, "device")  # shorted to the handle through the via
    shuttle = structure.electrode_at(SHUTTLE, "device")
    plan = DrivePlan(durations=(30e-6, 30e-6), nets=(NetDrive(left.net, left.index, "drive", (20.0, 0.0)),), rise_time=1e-6, samples=30)
    recorder = _Recorder()
    run = run_transient(structure, plates_model.model, plan, TransientSettings().options(), progress=recorder)
    assert recorder.stages == ["assemble", "transient"]
    assert recorder.details[0].startswith("step 1/2: N") and recorder.details[-1].startswith("step 2/2")
    assert "of 60 µs" in recorder.details[-1]
    assert not run.ended_early and len(run.times) == len(plan.times())
    assert run.u.shape == (len(run.times), len(structure.bodies), 3) and np.all(run.u[:, 0] == 0)
    sag = run.u[:, shuttle.body, 2]
    assert sag.min() < -1e-10 and abs(sag.min()) > 5 * np.abs(run.u[:, shuttle.body, 0]).max()  # down, not sideways
    assert run.dominant_body() in (1, 2)
    assert run.nets[:, left.net].max() == pytest.approx(20.0) and run.nets[-1, left.net] == pytest.approx(0.0, abs=1e-9)
    assert np.all(run.nets[:, shuttle.net] == 0)
    supplied = np.abs(run.result.energy.supplied).max()
    assert np.abs(run.result.energy.imbalance).max() < 1e-3 * supplied
    assert "output times" in run.summary(structure) and "narrowest gap" in run.summary(structure)


def test_a_transient_can_be_cancelled(demo, plates_model):
    structure = demo["structure"]
    left = structure.electrode_at(LEFT, "device")
    plan = DrivePlan(durations=(30e-6,), nets=(NetDrive(left.net, left.index, "drive", (20.0,)),), rise_time=1e-6, samples=30)
    recorder = _Recorder(cancel_after=5)
    with pytest.raises(Cancelled):
        run_transient(structure, plates_model.model, plan, TransientSettings().options(), progress=recorder)
    assert recorder.count == 5


def test_equilibrium_and_settled_start(demo, plates_model):
    structure = demo["structure"]
    left = structure.electrode_at(LEFT, "device")
    shuttle = structure.electrode_at(SHUTTLE, "device")
    plan = DrivePlan(durations=(20e-6,), nets=(NetDrive(left.net, left.index, "drive", (20.0,)),), rise_time=1e-6, samples=20, settle_first=True)
    rest = solve_equilibrium(structure, plates_model.model, plan, 0)
    row = rest.result.dofs.index((shuttle.body, "uz"))
    assert rest.result.stable and 0 < rest.result.margin < 1 and rest.result.displacement[row] < 0
    assert "stable" in rest.summary(structure) and "along Z" in rest.summary(structure)
    run = run_transient(structure, plates_model.model, plan, TransientSettings().options())
    assert run.start is not None
    # Started at the equilibrium with the source already at its voltage: it stays there.
    assert run.u[:, shuttle.body, 2] == pytest.approx(rest.result.displacement[row], rel=1e-3)


def test_step_progress_without_describe():
    class Plain:
        events: list = []

        def start(self, stage, total):
            self.events.append(("start", stage))

        def advance(self, n=1):
            self.events.append(("advance", n))

        def finish(self):
            self.events.append(("finish",))

    plain = Plain()
    plan = _plan()
    progress = StepProgress(plain, plan, plan.times())
    progress.start("transient", 3)
    progress.advance()
    progress.finish()
    assert plain.events == [("start", "transient"), ("advance", 1), ("finish",)]
