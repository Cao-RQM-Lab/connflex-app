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


USAGE = """usage: connflex-app [LAYOUT.gds] [--version] [--self-test] [--smoke-test [SCREENSHOT.png]]

  --version     print the version and exit
  --self-test   run the analysis pipeline on the bundled demo without a GUI and exit:
                extraction, modes, stiffness, capacitance, a transient, and the GDS export
  --smoke-test  start the GUI, run the demo through all three tabs (modes, capacitance,
                a transient), save a screenshot (default smoke_test.png), and exit
"""


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv)
    options = args[1:]
    from connflex_app.worker import is_worker, serve

    if is_worker(args):  # a field-solver process of connflex-core (see connflex_app.worker)
        serve()
        return 0
    if "--help" in options or "-h" in options:
        print(USAGE)
        return 0
    if "--version" in options:
        from connflex_app import __version__

        print(f"ConnFlex {__version__}")
        return 0
    if "--self-test" in options:
        from connflex_app.selftest import run

        return run()

    _configure_surface_format()
    QCoreApplication.setOrganizationName("connflex")
    QCoreApplication.setApplicationName("ConnFlex")
    if "--smoke-test" in options:
        # Keep the smoke test deterministic and leave the user's preferences untouched.
        import tempfile

        from PySide6.QtCore import QSettings

        scratch = tempfile.mkdtemp(prefix="connflex-smoke-")
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, scratch)
    app = QApplication([args[0]])

    from PySide6.QtGui import QIcon

    from connflex_app.data import icon_png
    from connflex_app.theme import apply_theme

    apply_theme(app)
    if icon_png().is_file():
        app.setWindowIcon(QIcon(str(icon_png())))

    from connflex_app.main_window import MainWindow

    window = MainWindow()
    window.show()
    if "--smoke-test" in options:
        from connflex_app.smoke import SmokeTest

        index = options.index("--smoke-test")
        target = options[index + 1] if index + 1 < len(options) and not options[index + 1].startswith("-") else "smoke_test.png"
        smoke = SmokeTest(window, target)
        smoke.start()
        app.exec()
        return smoke.exit_code
    paths = [option for option in options if not option.startswith("-")]
    if paths:
        window.open_path(paths[0])
    return app.exec()
