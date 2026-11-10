"""The panels of the electrical and electromechanical tabs, without a window or OpenGL."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from connflex_app import paint  # noqa: E402
from connflex_app.drive import DrivePlan, NetDrive, run_transient  # noqa: E402
from connflex_app.settings import DriveNet, DriveSettings, ElectroSettings, TransientSettings  # noqa: E402
from connflex_app.widgets.capacitance_panel import CapacitancePanel  # noqa: E402
from connflex_app.widgets.drive_panel import FIRST_STEP_COLUMN, DrivePanel  # noqa: E402
from connflex_app.widgets.fields import QuantityEdit  # noqa: E402
from connflex_app.widgets.plot import Series, TimePlot, nice_ticks, prefix_for  # noqa: E402
from connflex_app.widgets.progress_status import ProgressStatus  # noqa: E402
from connflex_app.widgets.timeline_panel import TimelinePanel  # noqa: E402
from connflex_app.workers import ProgressRelay  # noqa: E402

STATOR, SHUTTLE, LEFT = (1050e-6, 600e-6), (800e-6, 600e-6), (150e-6, 600e-6)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_capacitance_panel_reads_a_picked_pair(app, demo_project, field_model):
    structure, render = demo_project.structure, demo_project.render
    panel = CapacitancePanel()
    changes: list[int] = []
    panel.picksChanged.connect(lambda: changes.append(1))
    panel.set_project(demo_project)
    assert panel.compute_button.isEnabled() and "auto · 1.5 µm" in panel.element_size.placeholderText()
    assert panel.paint_table().shape == (demo_project.groups.count, 4)
    panel.set_result(field_model)
    assert not panel.is_stale() and "sidewall cluster" in panel.status.text()

    panel.pick_group(render.group_at(*STATOR), second=False)
    panel.pick_group(render.group_at(*SHUTTLE), second=True)
    nets = panel.grouping()
    a, b = panel.picks()
    assert nets.keys[a] == structure.electrode_at(STATOR, "device").net and nets.keys[b] == structure.electrode_at(SHUTTLE, "device").net
    assert changes and "Between N0 and N1" in panel.pair_title.text()
    assert panel.pair_value.text().endswith("fF") and float(panel.pair_value.text().split()[0]) == pytest.approx(98.1, rel=0.03)
    assert panel.part_values["oxide"].text() == "0" and panel.part_values["sidewall"].text().endswith("fF")
    assert panel.couplings.rowCount() == 2 and panel.slopes.rowCount() == 1 and panel.slopes.item(0, 0).text() == "B1"
    assert panel.matrix_model.rowCount() == 3 and panel.matrix_csv().splitlines()[0] == ",N0,N1,N2"
    table = panel.paint_table()
    assert tuple(table[structure.electrode_at(STATOR, "device").index, :3]) == pytest.approx(paint.PICK_A)
    assert tuple(table[structure.electrode_at(SHUTTLE, "device").index, :3]) == pytest.approx(paint.PICK_B)
    assert "total capacitance" in panel.describe_group(render.group_at(*LEFT)) and panel.describe_group(-1) == ""

    panel.pick_group(-1, second=False)  # a click on nothing clears the selection
    assert panel.picks() == (-1, -1) and panel.pair_value.text() == "—"

    panel.set_kind("body")
    panel.pick_group(render.group_at(*SHUTTLE), second=False)
    assert panel.grouping().kind == "body" and "Total of B1" in panel.pair_title.text()
    beam = structure.beams[0]
    middle = ((beam.start[0] + beam.end[0]) / 2, (beam.start[1] + beam.end[1]) / 2)
    panel.pick_group(render.group_at(*middle), second=True)  # a beam belongs to no body: B stays unset
    assert panel.picks()[1] == -1
    panel.set_kind("electrode")
    assert panel.matrix_model.rowCount() == len(structure.electrodes) and panel.picks() == (-1, -1)

    panel.fringe_check.setChecked(False)  # the model on show no longer matches the settings
    assert panel.is_stale() and "compute again" in panel.status.text()
    panel.set_project(None)
    assert not panel.compute_button.isEnabled() and panel.paint_table() is None


def test_capacitance_panel_settings_round_trip(app):
    panel = CapacitancePanel()
    settings = ElectroSettings(include_fringe=False, element_size=2e-6, ambient_permittivity=1.5, plates_cutoff=40.0, surrogate_tolerance=None, workers=3)
    panel.apply(settings)
    assert panel.settings() == settings
    panel.apply(ElectroSettings())
    assert panel.settings() == ElectroSettings()


def test_drive_panel_builds_a_plan(app, demo_project):
    structure = demo_project.structure
    stator = structure.electrode_at(STATOR, "device")
    shuttle = structure.electrode_at(SHUTTLE, "device")
    panel = DrivePanel()
    assert not panel.run_button.isEnabled()
    panel.set_project(demo_project)
    panel.set_frequencies(demo_project.modal.frequencies_hz)
    assert panel.run_button.isEnabled() and panel.nets() == []
    assert panel.table.item(0, FIRST_STEP_COLUMN).text() == "180 µs"  # automatic: eight periods of 45.56 kHz

    panel.add_net(stator.index)
    panel.add_net(shuttle.index)  # a moving mass: its source goes to an anchored pad of the net
    panel.add_net(stator.index)  # already there: selected, not added twice
    assert panel.nets() == [stator.net, shuttle.net] and panel.current_net() == stator.net
    panel.add_step()
    panel.set_voltage(stator.net, 0, 40.0)
    panel.set_duration(1, 50e-6)
    assert panel.steps() == 2 and panel.step_potentials(0) == {stator.net: 40.0, shuttle.net: 0.0}
    assert panel.peak_voltage() == 40.0

    panel.table.item(1, FIRST_STEP_COLUMN + 1).setText("-2.5 mV")  # typed with a prefix and a unit
    app.processEvents()
    assert panel.step_potentials(1)[stator.net] == pytest.approx(-2.5e-3)
    panel.table.item(1, FIRST_STEP_COLUMN + 1).setText("soon")  # refused: the value stays
    app.processEvents()
    assert panel.step_potentials(1)[stator.net] == pytest.approx(-2.5e-3)
    panel.table.item(0, FIRST_STEP_COLUMN + 1).setText("")  # cleared: automatic again
    app.processEvents()

    plan = panel.plan()
    assert plan.durations == pytest.approx((1.8e-4, 1.8e-4)) and plan.rise_time == pytest.approx(1.8e-4 / 50)
    assert plan.nets[0] == NetDrive(stator.net, stator.index, "drive", (40.0, -2.5e-3))
    assert structure.electrodes[plan.nets[1].electrode].body == 0

    saved = panel.drive_settings()
    assert saved.durations == [None, None] and [net.volts for net in saved.nets] == [[40.0, -2.5e-3], [0.0, 0.0]]
    other = DrivePanel()
    other.apply(saved, TransientSettings(quality_factor=80.0, settle_first=True, samples=64))
    assert other.drive_settings() == saved  # kept until a structure arrives
    other.set_project(demo_project)
    assert other.nets() == panel.nets() and other.transient_settings().quality_factor == 80.0
    assert other.transient_settings().settle_first and other.transient_settings().samples == 64

    panel.table.setCurrentCell(2, FIRST_STEP_COLUMN)
    panel.remove_net()
    panel.remove_step()
    assert panel.nets() == [stator.net] and panel.steps() == 1
    stray = DrivePanel()
    stray.apply(DriveSettings(nets=[DriveNet((-1.0, -1.0), "device", "drive", [5.0])]), TransientSettings())
    stray.set_project(demo_project)
    assert stray.nets() == [] and "no longer there" in stray.hint.text()


def test_timeline_and_plots_draw_a_run(app, demo_project, plates_model):
    structure = demo_project.structure
    left = structure.electrode_at(LEFT, "device")
    plan = DrivePlan(durations=(20e-6, 20e-6), nets=(NetDrive(left.net, left.index, "drive", (20.0, 0.0)),), rise_time=1e-6, samples=20)
    run = run_transient(structure, plates_model.model, plan, TransientSettings().options())
    panel = TimelinePanel()
    panel.resize(1100, 280)
    scrubs: list[float] = []
    panel.scrubbed.connect(scrubs.append)
    panel.set_run(run, structure)
    assert panel.play_button.isEnabled() and panel.body_combo.count() == 2
    assert len(panel.voltage_plot.series()) == 1 and len(panel.quantity_plot.series()) == 3
    for index in range(panel.quantity_combo.count()):
        panel.quantity_combo.setCurrentIndex(index)
        assert not panel.grab().isNull()
    panel.set_time(30e-6)
    assert "step 2 of 2" in panel.time_label.text()
    assert panel.voltage_plot.value_at(0, 10e-6) == pytest.approx(20.0)
    panel.voltage_plot.scrubbed.emit(12e-6)
    assert scrubs == [12e-6]
    panel.set_run(None, None)
    assert not panel.play_button.isEnabled() and panel.voltage_plot.series() == []


def test_plot_helpers(app):
    assert nice_ticks(0.0, 1.0) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert nice_ticks(-3e-9, 7e-9, 5) == pytest.approx([-2e-9, 0.0, 2e-9, 4e-9, 6e-9])
    assert nice_ticks(1.0, 1.0) == [1.0]
    assert prefix_for(4.2e-9) == (1e-9, "n") and prefix_for(0.0) == (1.0, "") and prefix_for(2500.0) == (1e3, "k")
    plot = TimePlot()
    plot.resize(700, 160)
    times = np.linspace(0.0, 1e-3, 20000)  # more points than pixels: drawn by envelope
    plot.set_data(times, [Series("x", np.sin(times * 2e4)), Series("flat", np.zeros_like(times), dashed=True)], "m", "test")
    plot.set_marks([0.0, 5e-4, 1e-3], ["step 1", "step 2"])
    plot.set_cursor_time(2.5e-4)
    assert not plot.grab().isNull()
    plot.clear("nothing")
    assert not plot.grab().isNull()


def test_quantity_edit(app):
    edit = QuantityEdit("m", 1.5e-6, optional=True, positive=True)
    assert edit.text() == "1.5 µm"
    changes: list[int] = []
    edit.valueChanged.connect(lambda: changes.append(1))
    edit.setText("3 um")
    edit.editingFinished.emit()
    assert edit.value() == pytest.approx(3e-6) and changes == [1]
    edit.setText("-1")  # refused: not above zero
    edit.editingFinished.emit()
    assert edit.value() == pytest.approx(3e-6) and edit.text() == "3 µm"
    edit.setText("")
    edit.editingFinished.emit()
    assert edit.value() is None and changes == [1, 1]
    required = QuantityEdit("", 50.0, positive=True)
    required.setText("")
    required.editingFinished.emit()
    assert required.value() == 50.0
    bare = QuantityEdit("", 1e-6, positive=True)  # a tolerance: shown as a number, not with a prefix
    assert bare.text() == "1e-06"
    bare.setText("2e-4")
    bare.editingFinished.emit()
    assert bare.value() == pytest.approx(2e-4) and bare.text() == "0.0002"


def test_progress_shows_detail_weights_and_repeats(app):
    status = ProgressStatus()
    status.start_phase("Running transient", ("assemble", "transient"), cancellable=True, weights={"assemble": 0.02, "transient": 4.0})
    status.stage_started("assemble", None)
    status.stage_finished("assemble")
    status.stage_started("transient", 200)
    status.set_detail("step 1/2: N0 40 V · t = 10 µs of 60 µs")
    status.advanced(50)
    assert "step 1/2: N0 40 V" in status.trail.text() and "50/200" in status.trail.text()
    assert 240 < status.bar.value() < 270  # a quarter of the long stage, the short one barely counts
    assert not status.cancel_button.isHidden()
    requests: list[int] = []
    status.cancelRequested.connect(lambda: requests.append(1))
    status.cancel_button.click()
    status.cancel_button.click()
    assert requests == [1] and not status.cancel_button.isEnabled()
    status.stage_finished("transient")
    status.stage_started("assemble", None)  # the same stage again: shown as a stage of its own
    status.stage_finished("assemble")
    status.finish_phase(ok=None)  # the user cancelled: neither a result nor a failure
    title, _total, stages, ok = status.history()[-1]
    assert title == "Running transient" and ok is None and [name for name, _seconds in stages] == ["assemble", "transient", "assemble 2"]
    assert "cancelled after" in status.summary.text() and "failed" not in status.summary.text()
    assert status.cancel_button.isHidden()
    status.start_phase("Extracting", ("load",))
    assert status.cancel_button.isHidden()  # only analyses that can be stopped offer it


def test_relay_cancels_at_the_next_stage_event(app):
    from connflex_app.drive import Cancelled

    relay = ProgressRelay()
    details: list[str] = []
    relay.described.connect(details.append)
    relay.start("transient", 10)
    relay.describe("step 1/1")
    relay.advance()
    relay.finish()
    assert details[-1] == "step 1/1"
    relay.cancel()
    with pytest.raises(Cancelled):
        relay.advance()
    with pytest.raises(Cancelled):
        relay.start("sidewalls", 3)
    relay.reset()
    relay.start("sidewalls", 3)
    relay.advance()
