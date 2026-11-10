"""Dark theme: palette, stylesheet, and the colours shared by the 2D and 3D views."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from connflex_app import paint

BG = "#14161b"
SURFACE = "#1b1e25"
PANEL = "#20242c"
RAISED = "#262b35"
BORDER = "#2e333d"
TEXT = "#e7e9ef"
MUTED = "#8d94a5"
ACCENT = "#4f8ef7"
ACCENT_HOVER = "#6ba1ff"
SELECTION = "#fbbf24"
HOVER = "#ffffff"
DANGER = "#ef4444"

LAYER = {
    "device": "#5b9cf6",
    "device_dim": "#2f4f80",
    "handle": "#4a5060",
    "box": "#d9a441",
    "release": "#e879f9",
    "vias": "#a78bfa",
    "beam": "#f97316",
    "rejected": "#ef4444",
}

# Linear RGB for the GL view, per mesh kind: device, box, handle, beam, rejected.
GL_KIND_COLORS = (
    (0.36, 0.61, 0.96),
    (0.85, 0.64, 0.25),
    (0.33, 0.36, 0.42),
    (0.98, 0.45, 0.09),
    (0.75, 0.25, 0.25),
)
GL_ANCHOR_COLOR = (0.45, 0.50, 0.60)
GL_BACKGROUND = (0.078, 0.086, 0.106)
ACCENT_RGB = (0.31, 0.56, 0.97)
SELECTION_RGB = (0.98, 0.75, 0.14)


def qcolor(value: str, alpha: float = 1.0) -> QColor:
    color = QColor(value)
    color.setAlphaF(alpha)
    return color


def categorical_rgb(index: int, saturation: float = 0.62, lightness: float = 0.6) -> tuple[float, float, float]:
    """Well-separated colour for category ``index`` (the palette of :mod:`connflex_app.paint`)."""
    return paint.categorical(index, saturation, lightness)


def categorical(index: int, alpha: float = 1.0) -> QColor:
    r, g, b = categorical_rgb(index)
    return QColor.fromRgbF(r, g, b, alpha)


def body_rgb(index: int) -> tuple[float, float, float]:
    """Colour of a body: neutral for the anchor, categorical for the rest."""
    return GL_ANCHOR_COLOR if index == 0 else categorical_rgb(index)


def body_color(index: int, alpha: float = 1.0) -> QColor:
    r, g, b = body_rgb(index)
    return QColor.fromRgbF(r, g, b, alpha)


QSS = f"""
QMainWindow, QDialog {{ background: {BG}; }}
QMainWindow::separator {{ background: {BG}; width: 4px; height: 4px; }}
QToolBar {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 4px 8px; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; color: {TEXT}; }}
QToolBar QToolButton:hover {{ background: {RAISED}; border-color: {BORDER}; }}
QToolBar QToolButton:checked {{ background: #2a3550; border-color: {ACCENT}; }}
QToolBar QToolButton:disabled {{ color: #5c6373; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 6px 4px; }}
QDockWidget {{ color: {MUTED}; font-weight: 600; }}
QDockWidget::title {{ background: {PANEL}; padding: 7px 12px; border-bottom: 1px solid {BORDER}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 16px; padding: 12px 8px 8px 8px; font-weight: 600; color: {TEXT}; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px; color: {MUTED}; }}
QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; color: {TEXT}; }}
QPushButton:hover {{ background: #2d3340; border-color: #3a404c; }}
QPushButton:pressed {{ background: {PANEL}; }}
QPushButton:disabled {{ color: #5c6373; background: {PANEL}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: #2f4468; border-color: #2f4468; color: {MUTED}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 3px 6px; color: {TEXT}; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QAbstractSpinBox QLineEdit {{ border: none; background: transparent; padding: 0; border-radius: 0; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: #5c6373; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: #2a3550; color: {TEXT}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0px; border: none; background: transparent; }}
QSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {{ width: 0px; height: 0px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {TEXT}; }}
QSlider::handle:horizontal:hover {{ background: white; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QTableWidget, QTableView, QListWidget, QTreeView {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: #22262e; alternate-background-color: #181b21; selection-background-color: #2a3550; selection-color: {TEXT}; outline: 0; }}
QTableView::item {{ padding: 2px 6px; }}
QTableView::item:selected, QListWidget::item:selected {{ background: #2a3550; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; padding: 4px 6px; border: none; border-bottom: 1px solid {BORDER}; border-right: 1px solid {BORDER}; font-weight: 600; }}
QTableCornerButton::section {{ background: {PANEL}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3a404c; min-height: 24px; border-radius: 4px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #4a5160; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #3a404c; min-width: 24px; border-radius: 4px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ background: {RAISED}; border: none; border-radius: 3px; max-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid #3a404c; padding: 4px 6px; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: #2a3550; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QMenuBar {{ background: {SURFACE}; }}
QMenuBar::item:selected {{ background: {RAISED}; }}
QSplitter::handle {{ background: {BG}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#title {{ font-size: 15px; font-weight: 700; }}
QLabel#value {{ font-weight: 600; }}
QFrame#chipbar {{ background: rgba(20, 22, 27, 215); border: 1px solid {BORDER}; border-radius: 8px; }}
QToolButton#chip {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px 9px; color: #c9cedb; }}
QToolButton#chip:hover {{ background: {RAISED}; }}
QToolButton#chip:checked {{ background: #2a3550; border-color: {ACCENT}; color: white; }}
QToolButton#chip:disabled {{ color: #5c6373; }}
QFrame#vsep {{ background: {BORDER}; }}
QToolButton#section {{ background: transparent; border: none; color: {MUTED}; font-weight: 600; padding: 2px; }}
QToolButton#section:hover {{ color: {TEXT}; }}
QCheckBox {{ spacing: 6px; }}
QRadioButton {{ spacing: 8px; }}
QRadioButton::indicator {{ width: 12px; height: 12px; border-radius: 8px; border: 2px solid #5c6373; background: {BG}; }}
QRadioButton::indicator:hover {{ border-color: {MUTED}; }}
QRadioButton::indicator:checked {{ border: 2px solid {ACCENT}; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 {ACCENT}, stop:0.55 {ACCENT}, stop:0.62 {BG}, stop:1 {BG}); }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QToolBar#toolbar_workspaces {{ padding: 0 8px; spacing: 0; }}
QTabBar#workspaces {{ background: transparent; }}
QTabBar#workspaces::tab {{ background: transparent; color: {MUTED}; padding: 8px 18px 7px 18px; margin-right: 2px; border: none; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar#workspaces::tab:hover {{ color: {TEXT}; }}
QTabBar#workspaces::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar#workspaces::tab:disabled {{ color: #5c6373; }}
QLabel#legend {{ color: #c9cedb; padding: 0 4px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: SURFACE,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: BG,
        QPalette.ColorRole.AlternateBase: "#181b21",
        QPalette.ColorRole.ToolTipBase: RAISED,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: RAISED,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: ACCENT,
        QPalette.ColorRole.PlaceholderText: MUTED,
        QPalette.ColorRole.Mid: BORDER,
        QPalette.ColorRole.Dark: BG,
        QPalette.ColorRole.Light: "#3a404c",
    }
    for role, value in roles.items():
        palette.setColor(role, QColor(value))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#5c6373"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#5c6373"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#5c6373"))
    app.setPalette(palette)
    app.setStyleSheet(QSS)
