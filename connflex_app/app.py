"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication


def _configure_surface_format() -> None:
    """Request a core-profile OpenGL context with multisampling before any window exists."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


def main(argv: list[str] | None = None) -> int:
    _configure_surface_format()
    QCoreApplication.setOrganizationName("connflex")
    QCoreApplication.setApplicationName("ConnFlex")
    app = QApplication(argv if argv is not None else sys.argv)

    from connflex_app.theme import apply_theme

    apply_theme(app)

    from connflex_app.main_window import MainWindow

    window = MainWindow()
    window.show()
    extra = app.arguments()[1:]
    if extra:
        window.open_path(extra[0])
    return app.exec()
