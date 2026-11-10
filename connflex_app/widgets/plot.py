"""A small time-series plot drawn with QPainter: lines, step marks, and a scrub cursor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from connflex_app import theme

PREFIXES = ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f"), (1e-18, "a"))
SERIES_COLORS = ("#4f8ef7", "#f97316", "#34d399", "#f472b6", "#fbbf24", "#22d3ee", "#a78bfa", "#f87171")
LEGEND_WIDTH = 176  # gutter right of the plot that holds the legend, when the widget is wide enough
LEGEND_FROM = 560


def prefix_for(magnitude: float) -> tuple[float, str]:
    """SI factor and prefix that show ``magnitude`` as a number between 1 and 1000."""
    if not math.isfinite(magnitude) or magnitude <= 0:
        return 1.0, ""
    for factor, prefix in PREFIXES:
        if magnitude >= factor * 0.9995:
            return factor, prefix
    return PREFIXES[-1]


def nice_ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Round tick values covering ``low`` to ``high``, about ``target`` of them."""
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return [low]
    raw = span / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next((m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * magnitude >= raw), 10.0 * magnitude)
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + step * 1e-9:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _label(value: float) -> str:
    text = f"{value:.4g}"
    return "0" if text in ("-0", "0") else text


@dataclass
class Series:
    """One line of a plot: ``values`` against the plot's times."""

    name: str
    values: np.ndarray
    color: str = ""
    dashed: bool = False


class TimePlot(QWidget):
    """Lines against time with a legend that reads out the values under the cursor.

    Click or drag to move the cursor; ``scrubbed`` reports the time. Step boundaries set with
    :meth:`set_marks` are drawn as vertical guides with their labels along the top.
    """

    scrubbed = Signal(float)

    def __init__(self, parent: QWidget | None = None, show_time_axis: bool = True) -> None:
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._times = np.zeros(0)
        self._series: list[Series] = []
        self._unit = ""
        self._title = ""
        self._marks = np.zeros(0)
        self._mark_labels: list[str] = []
        self._cursor: float | None = None
        self._show_time_axis = show_time_axis
        self._empty = "No data"
        self._font = QFont()
        self._font.setPointSize(9)
        self._dragging = False

    # ------------------------------------------------------------------ data
    def set_data(self, times, series: list[Series], unit: str = "", title: str = "") -> None:
        self._times = np.asarray(times, dtype=np.float64)
        self._series = []
        for index, entry in enumerate(series):
            values = np.asarray(entry.values, dtype=np.float64)
            color = entry.color or SERIES_COLORS[index % len(SERIES_COLORS)]
            self._series.append(Series(entry.name, values, color, entry.dashed))
        self._unit = unit
        self._title = title
        self.update()

    def clear(self, message: str = "No data") -> None:
        self._times = np.zeros(0)
        self._series = []
        self._marks = np.zeros(0)
        self._mark_labels = []
        self._cursor = None
        self._empty = message
        self.update()

    def set_marks(self, edges, labels: list[str] | None = None) -> None:
        """Vertical guides at ``edges``; ``labels[k]`` names the span that starts at ``edges[k]``."""
        self._marks = np.asarray(edges, dtype=np.float64)
        self._mark_labels = list(labels or [])
        self.update()

    def set_cursor_time(self, moment: float | None) -> None:
        if moment != self._cursor:
            self._cursor = moment
            self.update()

    def series(self) -> list[Series]:
        return list(self._series)

    def value_at(self, index: int, moment: float) -> float:
        """Value of series ``index`` at ``moment``, interpolated."""
        return float(np.interp(moment, self._times, self._series[index].values))

    # --------------------------------------------------------------- geometry
    def _gutter(self) -> int:
        return LEGEND_WIDTH if self.width() >= LEGEND_FROM else 0

    def _area(self) -> QRectF:
        bottom = 20 if self._show_time_axis else 6
        right = 10 + self._gutter()
        return QRectF(58, 16, max(self.width() - 58 - right, 10), max(self.height() - 16 - bottom, 10))

    def _ranges(self) -> tuple[float, float, float, float]:
        t0, t1 = float(self._times[0]), float(self._times[-1])
        if t1 <= t0:
            t1 = t0 + 1.0
        finite = [values[np.isfinite(values)] for values in (entry.values for entry in self._series)]
        finite = [values for values in finite if values.size]
        low = min((float(values.min()) for values in finite), default=0.0)
        high = max((float(values.max()) for values in finite), default=1.0)
        if high - low <= 1e-12 * max(abs(high), abs(low), 1e-300):
            pad = abs(high) * 0.1 if high != 0 else 1.0
            low, high = low - pad, high + pad
        else:
            pad = 0.08 * (high - low)
            low, high = low - pad, high + pad
        return t0, t1, low, high

    def _time_at(self, x: float) -> float:
        area = self._area()
        t0, t1, _low, _high = self._ranges()
        fraction = min(max((x - area.left()) / area.width(), 0.0), 1.0)
        return t0 + fraction * (t1 - t0)

    # --------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.BG))
        painter.setFont(self._font)
        area = self._area()
        if len(self._times) < 2 or not self._series:
            painter.setPen(QColor(theme.MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty)
            painter.end()
            return
        t0, t1, low, high = self._ranges()

        def x_of(moment):
            return area.left() + (moment - t0) / (t1 - t0) * area.width()

        def y_of(value):
            return area.bottom() - (value - low) / (high - low) * area.height()

        factor, prefix = prefix_for(max(abs(low), abs(high)))
        grid = QPen(QColor("#22262e"), 1.0)
        muted = QColor(theme.MUTED)
        for tick in nice_ticks(low, high, max(int(area.height() / 28), 2)):
            y = y_of(tick)
            painter.setPen(grid if tick != 0 else QPen(QColor("#3a404c"), 1.0))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            painter.setPen(muted)
            painter.drawText(QRectF(0, y - 8, area.left() - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _label(tick / factor))
        time_factor, time_prefix = prefix_for(max(abs(t0), abs(t1)))
        for tick in nice_ticks(t0, t1, max(int(area.width() / 90), 2)):
            x = x_of(tick)
            painter.setPen(grid)
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            if self._show_time_axis:
                painter.setPen(muted)
                painter.drawText(QRectF(x - 40, area.bottom() + 3, 80, 14), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, _label(tick / time_factor))
        painter.setPen(muted)
        if self._show_time_axis:
            painter.drawText(QRectF(0, area.bottom() + 3, area.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"{time_prefix}s")
        painter.drawText(QRectF(0, 0, area.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{prefix}{self._unit}")

        # Step guides and their labels.
        painter.setPen(QPen(QColor("#4b5160"), 1.0, Qt.PenStyle.DashLine))
        for edge in self._marks:
            if t0 < edge < t1:
                painter.drawLine(QPointF(x_of(edge), area.top()), QPointF(x_of(edge), area.bottom()))
        if self._mark_labels:
            painter.setPen(muted)
            for k, label in enumerate(self._mark_labels):
                if k + 1 >= len(self._marks):
                    break
                left, right = x_of(max(self._marks[k], t0)), x_of(min(self._marks[k + 1], t1))
                if right - left > painter.fontMetrics().horizontalAdvance(label) + 6:
                    painter.drawText(QRectF(left, 1, right - left, 14), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, label)

        painter.setClipRect(area.adjusted(-1, -1, 1, 1))
        for entry in self._series:
            pen = QPen(QColor(entry.color), 1.6)
            if entry.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPolyline(self._polyline(entry.values, x_of, y_of, int(area.width())))
        painter.setClipping(False)
        painter.setPen(QPen(QColor(theme.BORDER), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(area)

        cursor = self._cursor
        if cursor is not None and t0 <= cursor <= t1:
            x = x_of(cursor)
            painter.setPen(QPen(QColor(theme.SELECTION), 1.2))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.setPen(Qt.PenStyle.NoPen)
            for index, entry in enumerate(self._series):
                value = self.value_at(index, cursor)
                if math.isfinite(value):
                    painter.setBrush(QColor(entry.color))
                    painter.drawEllipse(QPointF(x, y_of(value)), 3.0, 3.0)
        self._draw_legend(painter, area, factor, prefix)
        painter.end()

    def _polyline(self, values: np.ndarray, x_of, y_of, width: int) -> QPolygonF:
        times = self._times
        count = min(len(times), len(values))
        times, values = times[:count], values[:count]
        if count > 4 * max(width, 1):  # keep the envelope: lowest and highest value of every pixel column
            buckets = max(width, 1)
            edges = np.linspace(0, count, buckets + 1).astype(int)
            keep: list[int] = []
            for start, stop in zip(edges[:-1], edges[1:]):
                if stop > start:
                    chunk = values[start:stop]
                    keep += sorted({start, start + int(np.nanargmin(chunk)), start + int(np.nanargmax(chunk)), stop - 1})
            times, values = times[keep], values[keep]
        polygon = QPolygonF()
        for moment, value in zip(times, values):
            if math.isfinite(value):
                polygon.append(QPointF(x_of(moment), y_of(value)))
        return polygon

    def _draw_legend(self, painter: QPainter, area: QRectF, factor: float, prefix: str) -> None:
        metrics = painter.fontMetrics()
        entries = []
        for index, entry in enumerate(self._series):
            text = entry.name
            if self._cursor is not None:
                value = self.value_at(index, self._cursor)
                text += f"  {value / factor:.4g} {prefix}{self._unit}" if math.isfinite(value) else "  —"
            entries.append((entry.color, text))
        if self._title:
            entries.insert(0, ("", self._title))
        gutter = self._gutter()
        if gutter:
            # Beside the plot, so the legend never covers a line. Entries that do not fit are counted.
            room = max(int((self.height() - 4) // 15), 1)
            if len(entries) > room:
                hidden = len(entries) - room + 1
                entries = [*entries[: room - 1], ("", f"and {hidden} more")]
            box = QRectF(area.right() + 6, max(area.top() - 4, 0), gutter - 4, 15 * len(entries) + 6)
            width = gutter - 30
        else:
            width = max((metrics.horizontalAdvance(text) for _color, text in entries), default=0) + 26
            box = QRectF(area.left() + 8, area.top() + 4, width, 15 * len(entries) + 6)
            if box.bottom() > area.bottom() or box.right() > area.right():
                return
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.qcolor(theme.BG, 0.78))
            painter.drawRoundedRect(box, 5, 5)
        y = box.top() + 4
        for color, text in entries:
            text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(width))
            if color:
                painter.setPen(QPen(QColor(color), 2.0))
                painter.drawLine(QPointF(box.left() + 6, y + 7), QPointF(box.left() + 18, y + 7))
                painter.setPen(QColor(theme.TEXT))
                painter.drawText(QRectF(box.left() + 22, y, width, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            else:
                painter.setPen(QColor(theme.MUTED))
                painter.drawText(QRectF(box.left() + 6, y, width, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            y += 15

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and len(self._times) >= 2:
            self._dragging = True
            self.scrubbed.emit(self._time_at(event.position().x()))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._dragging:
            self.scrubbed.emit(self._time_at(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._dragging = False
