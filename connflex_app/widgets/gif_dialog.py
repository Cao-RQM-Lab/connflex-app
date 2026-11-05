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
    """Asks for frames per cycle, frame rate, and output scale, then for the file to write."""

    def __init__(self, parent: QWidget | None, suggested_path: str, frame_size: tuple[int, int]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export animation as GIF")
        self.setModal(True)
        self._suggested = suggested_path
        self._frame_size = frame_size
        self.path: str | None = None

        layout = QVBoxLayout(self)
        intro = QLabel("One full cycle of the current animation is captured from the 3D view as it is shown now.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.frames = QSpinBox()
        self.frames.setRange(8, 240)
        self.frames.setValue(36)
        self.fps = QSpinBox()
        self.fps.setRange(5, 50)
        self.fps.setValue(24)
        self.fps.setSuffix(" fps")
        self.scale = QComboBox()
        for label, value in (("100 %", 1.0), ("75 %", 0.75), ("50 %", 0.5), ("25 %", 0.25)):
            self.scale.addItem(label, value)
        self.scale.setCurrentIndex(1 if frame_size[0] > 1600 else 0)
        form.addRow("Frames per cycle", self.frames)
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
        self.summary.setText(f"{frames} frames at {fps} fps: one cycle every {frames / fps:.1f} s · {width} × {height} px per frame.")

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
