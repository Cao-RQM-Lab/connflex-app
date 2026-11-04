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


def wait_until(predicate, timeout: float = 120.0, what: str = "") -> None:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
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
wait_until(lambda: window.session.project is not None and window.session.project.modal is not None and not window._busy, what="extract+solve")
pump(900)
shot("02_extracted_mode1_playing.png")
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

window.layout_view._scale *= 40
window.layout_view._center[:] = [300e-6, 600e-6]
pump(400)
shot("09_2d_deep_zoom.png", window.layout_view)

window.view_actions["3d"].trigger()
pump(300)
shot("10_3d_only.png")
window.view_actions["split"].trigger()
pump(200)

print("TOUR OK")
window.close()
app.processEvents()
