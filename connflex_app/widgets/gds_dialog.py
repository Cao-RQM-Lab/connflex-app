"""Options for exporting the extracted, released structure as GDS."""

from __future__ import annotations

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from connflex_app.gds_export import LAYER_TITLES, LAYERS

SETTINGS_KEY = "export/gds_separate_release"


class GdsExportDialog(QDialog):
    """Asks how release holes are written, then for the file to write."""

    def __init__(self, parent: QWidget | None, suggested_path: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export released geometry as GDS")
        self.setModal(True)
        self.setMinimumWidth(500)
        self._suggested = suggested_path
        self.path: str | None = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Writes the structure as extracted: only geometry connected to the anchor, with the buried "
            "oxide that survives the release etch. Units are micrometres with 1 nm precision."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        card = QFrame()
        card.setObjectName("card")
        table = QGridLayout(card)
        table.setContentsMargins(12, 8, 12, 8)
        table.setHorizontalSpacing(16)
        table.setVerticalSpacing(4)
        for row, (name, (number, datatype)) in enumerate(LAYERS.items()):
            key = QLabel(f"{number}/{datatype}")
            key.setObjectName("value")
            table.addWidget(key, row, 0)
            table.addWidget(QLabel(LAYER_TITLES[name]), row, 1)
        table.setColumnStretch(2, 1)
        layout.addWidget(card)
        layout.addSpacing(6)
        heading = QLabel("Release holes")
        heading.setObjectName("value")
        layout.addWidget(heading)

        self.separate = QRadioButton("Separate release holes")
        separate_note = QLabel("Device (1/0) is drawn with its release holes filled; the holes go on layer 2/0.")
        self.inline = QRadioButton("Release holes in the device layer")
        inline_note = QLabel("Device (1/0) keeps its release holes; layer 2/0 is left empty.")
        for note in (separate_note, inline_note):
            note.setObjectName("muted")
            note.setWordWrap(True)
            note.setContentsMargins(26, 0, 0, 6)
        group = QButtonGroup(self)
        group.addButton(self.separate)
        group.addButton(self.inline)
        separate = QSettings().value(SETTINGS_KEY, True)
        (self.separate if str(separate).lower() in ("true", "1") else self.inline).setChecked(True)
        for widget in (self.separate, separate_note, self.inline, inline_note):
            layout.addWidget(widget)
        layout.addSpacing(4)

        buttons = QDialogButtonBox()
        export = buttons.addButton("Choose file and export…", QDialogButtonBox.ButtonRole.AcceptRole)
        export.setObjectName("primary")
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._choose_file)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def separate_release(self) -> bool:
        return self.separate.isChecked()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save released geometry", self._suggested, "GDSII layout (*.gds)")
        if not path:
            return
        if not path.lower().endswith((".gds", ".gds2")):
            path += ".gds"
        QSettings().setValue(SETTINGS_KEY, self.separate_release())
        self.path = path
        self.accept()

    @staticmethod
    def suggest_path(layout_path: str | None) -> str:
        stem = os.path.splitext(layout_path)[0] if layout_path else os.path.join(os.path.expanduser("~"), "connflex")
        return f"{stem}_released.gds"
