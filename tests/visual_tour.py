"""Drive the real application through a scripted session and save screenshots.

Usage: python tests/visual_tour.py OUTPUT_DIR [GDS_PATH]

Needs a display: it opens the window on the native platform because Qt's offscreen
plugin has no OpenGL on macOS.
"""

from __future__ import annotations

import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from connflex_app.app import _configure_surface_format  # noqa: E402

_configure_surface_format()

import tempfile  # noqa: E402

from PySide6.QtCore import QCoreApplication, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setOrganizationName("connflex-visual-tour")
QCoreApplication.setApplicationName("ConnFlexTour")
_settings_dir = tempfile.mkdtemp(prefix="connflex-tour-")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _settings_dir)
app = QApplication(sys.argv)

from connflex_app.theme import apply_theme  # noqa: E402

apply_theme(app)

from connflex_app.main_window import MainWindow  # noqa: E402

out = sys.argv[1]
gds = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "examples", "demo_resonator.gds")
os.makedirs(out, exist_ok=True)


def pump(ms: int) -> None:
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


STALLS: dict[str, float] = {}


def wait_until(predicate, timeout: float = 300.0, what: str = "", snapshots=()) -> None:
    """Pump events until ``predicate`` holds, recording the longest single event-loop stall.

    ``snapshots`` are (delay in seconds, callable) pairs run once the delay has passed.
    """
    end = time.time() + timeout
    started = time.time()
    worst = 0.0
    pending = list(snapshots)
    while time.time() < end:
        t0 = time.perf_counter()
        app.processEvents()
        worst = max(worst, time.perf_counter() - t0)
        while pending and time.time() - started > pending[0][0]:
            pending.pop(0)[1]()
        if predicate():
            STALLS[what] = worst
            return
        time.sleep(0.005)
    raise TimeoutError(f"timed out waiting for {what}")


def shot(name: str, widget=None) -> None:
    widget = widget or window
    image = widget.grab_image() if hasattr(widget, "grab_image") else widget.grab()
    path = os.path.join(out, name)
    assert image.save(path), f"could not save {path}"
    print("saved", path)


window = MainWindow()
window.resize(1600, 1000)
window.show()
pump(400)

window.open_path(gds)
wait_until(lambda: window._info is not None and not window._busy, what="inspection")
wait_until(lambda: window.layout_view._layout is not None and not window._workers, what="preview")
pump(400)
shot("01_loaded.png")

window.extract()
wait_until(
    lambda: window.session.project is not None and window.session.project.modal is not None and not window._busy,
    what="extract+solve",
    snapshots=(
        (0.4, lambda: shot("01b_progress_during_extraction.png", window.statusBar())),
        (1.7, lambda: shot("01c_progress_later.png", window.statusBar())),
    ),
)
pump(900)
shot("02_extracted_mode1_playing.png")
shot("02b_status_after_solve.png", window.statusBar())
print("timings kept:", window.progress.summary.isVisible(), "|", [(t, round(s, 3), len(st)) for t, s, st, ok in window.progress.history()])
print("structure:", len(window.session.project.structure.bodies), "bodies,", len(window.session.project.structure.beams), "beams, mesh", f"{window.session.project.mesh.count:,}", "verts; timings", {k: round(v, 2) for k, v in window.session.project.timings.items()})

# GPU zoom: every step is a matrix change, so each paint must be quick even on large layouts.
worst = 0.0
for step in range(24):
    t0 = time.perf_counter()
    window.layout_view.zoom(1.25 if step < 16 else 0.8, QPointF(window.layout_view.width() * 0.55, window.layout_view.height() * 0.5))
    window.layout_view.repaint()
    app.processEvents()
    worst = max(worst, time.perf_counter() - t0)
STALLS["2d zoom step"] = worst
window.layout_view.fit()
pump(100)
print("modes:", [row for row in window.modes.table.item(0, 2).text()], window.modes.summary.text())
print("hover beam:", window.layout_view.describe("beam", 0)["text"])
print("settings used:", window.session.project.settings.device, window.session.project.settings.handle, window.session.project.settings.release, window.session.project.settings.vias)
print("nets:", window.session.project.structure.nets.count, "vias:", len(window.session.project.structure.nets.vias))
print("hover body:", window.layout_view.describe("body", 1)["text"])

window.session.select(1)
pump(300)
shot("03_body1_selected.png")
print("stiffness info:", window.stiffness.info.text())
print("stiffness rows:", [[window.stiffness.table.item(r, c).text() for c in range(4)] for r in range(6)])

window.highlight_bar.button("bodies").setChecked(True)
window.graphs_bar.button("mech_graph").setChecked(True)
pump(400)
shot("04_2d_bodies_mechgraph.png", window.layout_view)

window.highlight_bar.button("nets").setChecked(True)
window.graphs_bar.button("mech_graph").setChecked(False)
window.graphs_bar.button("elec_graph").setChecked(True)
pump(400)
shot("05_2d_nets_elecgraph.png", window.layout_view)
window.highlight_bar.button("nets").setChecked(False)
window.graphs_bar.button("elec_graph").setChecked(False)

window.modes.table.selectRow(1)
pump(300)
window.scene.play(False)
window.scene._phase = math.pi / 2
pump(250)
shot("06_3d_mode2_peak.png", window.scene)

import gdstk  # noqa: E402

for separate in (True, False):
    gds_path = os.path.join(out, f"released_{'separate' if separate else 'holes'}.gds")
    window.export_gds_file(gds_path, separate)
    wait_until(lambda: not window._busy, what="gds export")
    print("gds export:", "separate" if separate else "holes", sorted(gdstk.read_gds(gds_path).layers_and_datatypes()), "|", window.status_label.text())

gif_path = os.path.join(out, "mode2.gif")
window.export_gif(gif_path, frames=12, fps=20, scale=0.5)
wait_until(lambda: not window._busy, what="gif export")
from PIL import Image  # noqa: E402

with Image.open(gif_path) as gif:
    print("gif:", gif.n_frames, "frames", gif.size, f"{os.path.getsize(gif_path) / 1e3:.0f} kB", "loop", gif.info.get("loop"))

window.scene_options_bar.button("color_body").setChecked(True)
window.scene_bar.button("handle").setChecked(False)
pump(250)
shot("07_3d_colour_by_body_no_handle.png", window.scene)
window.scene_bar.button("handle").setChecked(True)

window.stiffness.condition_combo.setCurrentIndex(1)
window.stiffness.table.selectRow(0)
window.stiffness._request_deflection()
pump(400)
shot("08_deflection_x.png")
print("caption:", window.playback_caption.text())

window.scene._hover_request = QPointF(window.scene.width() * 0.5, window.scene.height() * 0.55)
window.scene.update()
pump(300)
print("3d hover body at centre:", window.scene._hover)

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

beam = window.session.project.structure.beams[0]
mid = ((beam.start[0] + beam.end[0]) / 2, (beam.start[1] + beam.end[1]) / 2)
device = window.layout_view._transform().map(QPointF(mid[0], mid[1])).toPoint()
QTest.mouseMove(window.layout_view, device)
pump(200)
print("2d hover via mouse:", window.hover_label.text())
QTest.mouseClick(window.scene, Qt.MouseButton.LeftButton, pos=QPoint(int(window.scene.width() * 0.5), int(window.scene.height() * 0.55)))
pump(300)
print("3d click selected body:", window.session.selected, "| stiffness body combo:", window.stiffness.body_combo.currentText() or "(anchor: none)")

bounds = window.layout_view._bounds
window.layout_view._center[:] = [bounds[0] + 0.25 * (bounds[2] - bounds[0]), bounds[1] + 0.5 * (bounds[3] - bounds[1])]
window.layout_view.zoom(40.0)
pump(400)
shot("09_2d_deep_zoom.png", window.layout_view)

window.view_actions["3d"].trigger()
pump(300)
shot("10_3d_only.png")
window.view_actions["split"].trigger()
pump(200)

print("worst event-loop stalls:", {k: f"{v * 1000:.0f} ms" for k, v in STALLS.items()})
print("TOUR OK")
window.close()
app.processEvents()
