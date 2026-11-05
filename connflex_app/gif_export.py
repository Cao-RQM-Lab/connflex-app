"""Encode captured frames as an animated GIF with one shared palette (no flicker between frames)."""

from __future__ import annotations

import os
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

Frame = tuple[bytes, int, int]


def frame_from_qimage(image: QImage, scale: float = 1.0) -> Frame:
    """Tightly packed RGB bytes of ``image`` scaled by ``scale``, with its width and height."""
    if scale != 1.0:
        image = image.scaled(
            max(int(image.width() * scale), 1),
            max(int(image.height() * scale), 1),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    raw = bytes(image.constBits())
    if stride != width * 3:
        raw = b"".join(raw[row * stride : row * stride + width * 3] for row in range(height))
    return raw, width, height


def encode_gif(frames: list[Frame], path: str, fps: float, advance: Callable[[], None] | None = None) -> int:
    """Write ``frames`` to ``path`` as a looping GIF at ``fps``; returns the file size in bytes."""
    from PIL import Image

    if not frames:
        raise ValueError("no frames to encode")
    images = [Image.frombuffer("RGB", (width, height), data, "raw", "RGB", 0, 1) for data, width, height in frames]
    step = max(1, len(images) // 4)
    sample = images[::step][:4]
    mosaic = Image.new("RGB", (sample[0].width, sum(image.height for image in sample)))
    y = 0
    for image in sample:
        mosaic.paste(image, (0, y))
        y += image.height
    palette = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = []
    for image in images:
        quantized.append(image.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG))
        if advance is not None:
            advance()
    duration = max(int(round(1000.0 / max(fps, 1e-3))), 20)
    quantized[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=1,
    )
    return os.path.getsize(path)
