"""Floating control bars laid over the views."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLayout, QToolButton, QVBoxLayout, QWidget


class FlowLayout(QLayout):
    """Lays items out left to right and wraps to a new line when the width runs out."""

    def __init__(self, parent: QWidget | None = None, h_spacing: int = 2, v_spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list = []
        self._h = h_spacing
        self._v = v_spacing

    def addItem(self, item) -> None:  # noqa: N802 - Qt API
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt API
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt API
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API
        return self._arrange(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._arrange(rect, dry=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        """Size with every item on one line."""
        left, top, right, bottom = self.getContentsMargins()
        width = sum(item.sizeHint().width() for item in self._items) + self._h * max(len(self._items) - 1, 0)
        height = max((item.sizeHint().height() for item in self._items), default=0)
        return QSize(width + left + right, height + top + bottom)

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _arrange(self, rect: QRect, dry: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        area = rect.adjusted(left, top, -right, -bottom)
        lines: list[list] = [[]]
        x = area.x()
        for item in self._items:
            width = item.sizeHint().width()
            if lines[-1] and x + width > area.right() + 1:
                lines.append([])
                x = area.x()
            lines[-1].append(item)
            x += width + self._h
        y = area.y()
        for line in lines:
            height = max((item.sizeHint().height() for item in line), default=0)
            x = area.x()
            for item in line:
                hint = item.sizeHint()
                if not dry:
                    item.setGeometry(QRect(QPoint(x, y + (height - hint.height()) // 2), hint))
                x += hint.width() + self._h
            y += height + self._v
        return y - self._v - rect.y() + bottom


class Collapsible(QWidget):
    """A section header that shows or hides its content."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self._content = content
        self._button = QToolButton()
        self._button.setObjectName("section")
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setText(title)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.toggled.connect(self._toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._button)
        layout.addWidget(content)
        self._toggle(expanded)

    def _toggle(self, on: bool) -> None:
        self._button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self._content.setVisible(on)


def dot_icon(color: str, diameter: int = 10) -> QIcon:
    """A filled circle, used as a colour legend on toggles."""
    scale = 2
    pixmap = QPixmap(diameter * scale, diameter * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(1, 1, diameter - 2, diameter - 2)
    painter.end()
    return QIcon(pixmap)


class ChipBar(QFrame):
    """A translucent rounded bar of toggle chips and buttons."""

    toggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chipbar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = FlowLayout(self)
        self._layout.setContentsMargins(6, 4, 6, 4)
        self._buttons: dict[str, QToolButton] = {}

    def add_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("muted")
        label.setContentsMargins(4, 0, 4, 0)
        self._layout.addWidget(label)
        return label

    def add_toggle(self, key: str, text: str, checked: bool = False, tip: str = "", color: str | None = None) -> QToolButton:
        button = QToolButton()
        button.setObjectName("chip")
        button.setText(text)
        if color is not None:
            button.setIcon(dot_icon(color))
            button.setIconSize(QSize(10, 10))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.toggled.connect(lambda on, k=key: self.toggled.emit(k, on))
        self._layout.addWidget(button)
        self._buttons[key] = button
        return button

    def add_button(self, text: str, slot=None, tip: str = "") -> QToolButton:
        button = QToolButton()
        button.setObjectName("chip")
        button.setText(text)
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        if slot is not None:
            button.clicked.connect(slot)
        self._layout.addWidget(button)
        return button

    def add_widget(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_separator(self) -> None:
        holder = QWidget()
        box = QHBoxLayout(holder)
        box.setContentsMargins(5, 0, 5, 0)
        line = QFrame()
        line.setObjectName("vsep")
        line.setFixedSize(1, 18)
        box.addWidget(line)
        self._layout.addWidget(holder)

    def button(self, key: str) -> QToolButton:
        return self._buttons[key]

    def is_checked(self, key: str) -> bool:
        return self._buttons[key].isChecked()

    def set_checked(self, key: str, on: bool) -> None:
        button = self._buttons[key]
        button.blockSignals(True)
        button.setChecked(on)
        button.blockSignals(False)

    def set_enabled(self, key: str, on: bool) -> None:
        self._buttons[key].setEnabled(on)


class VoltageLegend(QWidget):
    """A colour bar from minus to plus the largest voltage on show, for the views' voltage colouring."""

    def __init__(self, negative, neutral, positive, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stops = [QColor.fromRgbF(*negative), QColor.fromRgbF(*neutral), QColor.fromRgbF(*positive)]
        self._low = "−1 V"
        self._high = "+1 V"
        self._font = QFont()
        self._font.setPointSize(10)
        self.setFixedHeight(22)
        self._resize()

    def set_scale(self, text: str) -> None:
        """Label both ends with ``text``, e.g. ``"40 V"``."""
        low, high = f"−{text}", f"+{text}"
        if (low, high) != (self._low, self._high):
            self._low, self._high = low, high
            self._resize()
            self.update()

    def _resize(self) -> None:
        from PySide6.QtGui import QFontMetrics

        metrics = QFontMetrics(self._font)
        self._left = metrics.horizontalAdvance(self._low) + 6
        self._right = metrics.horizontalAdvance(self._high) + 6
        self.setFixedWidth(self._left + 90 + self._right)
        self.updateGeometry()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(self.width(), 22)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        bar = QRectF(self._left, 6, 90, 10)
        gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        for position, color in zip((0.0, 0.5, 1.0), self._stops):
            gradient.setColorAt(position, color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(bar, 3, 3)
        painter.setPen(QColor("#c9cedb"))
        painter.drawText(QRectF(0, 0, self._left - 4, 22), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._low)
        painter.drawText(QRectF(bar.right() + 4, 0, self._right, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._high)
        painter.end()


class OverlayHost(QWidget):
    """Fills itself with one view and floats other widgets over it, anchored to its edges."""

    def __init__(self, view: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = view
        view.setParent(self)
        self._overlays: list[tuple[QWidget, str, int]] = []

    @property
    def view(self) -> QWidget:
        return self._view

    def add_overlay(self, widget: QWidget, anchor: str = "top-left", margin: int = 12) -> QWidget:
        widget.setParent(self)
        widget.raise_()
        self._overlays.append((widget, anchor, margin))
        self._place()
        return widget

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._view.setGeometry(self.rect())
        self._place()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._place()

    def relayout(self) -> None:
        """Place the overlays again after one was shown, hidden, or resized."""
        self._place()

    def _place(self) -> None:
        width, height = self.width(), self.height()
        stacks: dict[str, int] = {}
        for widget, anchor, margin in self._overlays:
            if widget.isHidden():
                continue
            hint = widget.sizeHint()
            w = min(hint.width(), max(width - 2 * margin, 80))
            layout = widget.layout()
            h = widget.heightForWidth(w) if layout is not None and layout.hasHeightForWidth() else hint.height()
            offset = stacks.get(anchor, 0)
            if anchor == "top-left":
                x, y = margin, margin + offset
            elif anchor == "top-right":
                x, y = width - w - margin, margin + offset
            elif anchor == "bottom-left":
                x, y = margin, height - h - margin - offset
            elif anchor == "bottom-right":
                x, y = width - w - margin, height - h - margin - offset
            else:  # bottom-center
                x, y = (width - w) // 2, height - h - margin - offset
            widget.setGeometry(int(x), int(y), w, h)
            stacks[anchor] = offset + h + 6
