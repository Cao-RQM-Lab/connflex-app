"""The status bar keeps the timings of finished operations."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from connflex_app.widgets.progress_status import ProgressStatus  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _run(status: ProgressStatus, title: str, stages, reset: bool = False, ok: bool = True, close_last: bool = True) -> None:
    status.start_phase(title, tuple(stages), reset=reset)
    for index, name in enumerate(stages):
        status.stage_started(name, 10)
        status.advanced(10)
        if close_last or index < len(stages) - 1:
            status.stage_finished(name)
    status.finish_phase(ok=ok)


def test_timings_stay_after_extraction_and_solve(app):
    status = ProgressStatus()
    status.show()
    _run(status, "Extracting", ["load", "heal", "mesh"], reset=True)
    _run(status, "Solving modes", ["assemble", "solve"])
    assert not status.bar.isVisible() and status.summary.isVisible()
    titles = [entry[0] for entry in status.history()]
    assert titles == ["Extracting", "Solving modes"]
    assert "Extracting" in status.summary.text() and "Solving modes" in status.summary.text()
    tip = status.summary.toolTip()
    assert all(name in tip for name in ("load", "heal", "mesh", "assemble", "solve"))
    assert [name for name, _ in status.history()[0][2]] == ["load", "heal", "mesh"]


def test_repeating_an_operation_replaces_it_and_reset_clears(app):
    status = ProgressStatus()
    _run(status, "Extracting", ["load"], reset=True)
    _run(status, "Solving modes", ["solve"])
    _run(status, "Solving modes", ["solve"])
    assert [entry[0] for entry in status.history()] == ["Extracting", "Solving modes"]
    _run(status, "Reading demo.gds", ["inspect"], reset=True, close_last=False)
    history = status.history()
    assert [entry[0] for entry in history] == ["Reading demo.gds"]
    assert [name for name, _ in history[0][2]] == ["inspect"]  # an unclosed stage is still timed


def test_failed_operation_is_marked(app):
    status = ProgressStatus()
    _run(status, "Extracting", ["load", "heal"], reset=True, ok=False)
    assert status.history()[-1][3] is False
    assert "failed" in status.summary.text()
