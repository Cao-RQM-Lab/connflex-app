"""Options for exporting the current animation as a GIF."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class GifExportDialog(QDialog):
    """Asks for the number of frames, frame rate, and output scale, then for the file to write.

    ``transient`` words the dialog for a transient, captured once from its first time to its
    last, instead of one cycle of a mode.
    """

    def __init__(self, parent: QWidget | None, suggested_path: str, frame_size: tuple[int, int], transient: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export animation as GIF")
        self.setModal(True)
        self._suggested = suggested_path
        self._frame_size = frame_size
        self._transient = transient
        self.path: str | None = None

        layout = QVBoxLayout(self)
        if transient:
            intro = QLabel("The transient is captured from its first time to its last, from the 3D view as it is shown now.")
        else:
            intro = QLabel("One full cycle of the current animation is captured from the 3D view as it is shown now.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.frames = QSpinBox()
        self.frames.setRange(8, 480 if transient else 240)
        self.frames.setValue(120 if transient else 36)
        self.fps = QSpinBox()
        self.fps.setRange(5, 50)
        self.fps.setValue(24)
        self.fps.setSuffix(" fps")
        self.scale = QComboBox()
        for label, value in (("100 %", 1.0), ("75 %", 0.75), ("50 %", 0.5), ("25 %", 0.25)):
            self.scale.addItem(label, value)
        self.scale.setCurrentIndex(1 if frame_size[0] > 1600 else 0)
        form.addRow("Frames" if transient else "Frames per cycle", self.frames)
        form.addRow("Frame rate", self.fps)
        form.addRow("Output size", self.scale)
        layout.addLayout(form)
        self.summary = QLabel()
        self.summary.setObjectName("muted")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        buttons = QDialogButtonBox()
        export = buttons.addButton("Choose file and export…", QDialogButtonBox.ButtonRole.AcceptRole)
        export.setObjectName("primary")
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._choose_file)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for widget in (self.frames, self.fps):
            widget.valueChanged.connect(self._refresh)
        self.scale.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def values(self) -> tuple[int, int, float]:
        return self.frames.value(), self.fps.value(), float(self.scale.currentData())

    def _refresh(self) -> None:
        frames, fps, scale = self.values()
        width, height = int(self._frame_size[0] * scale), int(self._frame_size[1] * scale)
        what = "the run plays in" if self._transient else "one cycle every"
        self.summary.setText(f"{frames} frames at {fps} fps: {what} {frames / fps:.1f} s · {width} × {height} px per frame.")

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save animation", self._suggested, "GIF image (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"
        self.path = path
        self.accept()

    @staticmethod
    def suggest_path(layout_path: str | None, caption: str) -> str:
        stem = os.path.splitext(layout_path)[0] if layout_path else os.path.join(os.path.expanduser("~"), "connflex")
        slug = "".join(ch if ch.isalnum() else "_" for ch in caption.split("·")[0].strip().lower()) or "animation"
        return f"{stem}_{slug}.gif"
