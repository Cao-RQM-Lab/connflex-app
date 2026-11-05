"""Draw the application icon and write it as PNG, ICO, and ICNS (run once; outputs are committed)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 1024.0
    # Rounded background tile.
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(64 * s, 64 * s, 896 * s, 896 * s), 200 * s, 200 * s)
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0.0, QColor("#232a3a"))
    gradient.setColorAt(1.0, QColor("#12151c"))
    painter.fillPath(tile, QBrush(gradient))
    # Handle slab, anchor pads, beams, and a perforated proof mass seen in plan view.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#3a4152"))
    painter.drawRoundedRect(QRectF(160 * s, 160 * s, 704 * s, 704 * s), 40 * s, 40 * s)
    pad = QColor("#8ea4c8")
    painter.setBrush(pad)
    for x in (210, 630):
        for y in (210, 690):
            painter.drawRoundedRect(QRectF(x * s, y * s, 184 * s, 124 * s), 18 * s, 18 * s)
    beam = QPen(QColor("#f97316"), 26 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(beam)
    for x in (302, 722):
        painter.drawLine(QPointF(x * s, 334 * s), QPointF(x * s, 690 * s))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#5b9cf6"))
    painter.drawRoundedRect(QRectF(332 * s, 372 * s, 360 * s, 280 * s), 22 * s, 22 * s)
    painter.setBrush(QColor("#1b2130"))
    for i in range(4):
        for j in range(3):
            painter.drawRect(QRectF((372 + i * 84) * s, (412 + j * 84) * s, 46 * s, 46 * s))
    painter.end()
    return image


def main() -> int:
    QGuiApplication(sys.argv)
    base = draw(1024)
    assert base.save(os.path.join(HERE, "icon.png")), "png"
    assert base.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation).save(os.path.join(HERE, "icon.ico")), "ico"
    assert base.save(os.path.join(HERE, "icon.icns")), "icns"
    base.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation).save(
        os.path.join(HERE, "..", "connflex_app", "data", "icon.png")
    )
    print("icons written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
