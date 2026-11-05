"""GIF encoding without a display."""

from __future__ import annotations

import os

import numpy as np

from connflex_app.gif_export import encode_gif


def test_encode_gif_round_trip(tmp_path):
    frames = []
    width, height = 64, 40
    for k in range(10):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[..., 0] = 20
        canvas[:, (k * 6) % width : (k * 6) % width + 6, 2] = 220
        frames.append((canvas.tobytes(), width, height))
    path = str(tmp_path / "anim.gif")
    advanced = []
    size = encode_gif(frames, path, fps=25, advance=lambda: advanced.append(1))
    assert size == os.path.getsize(path) > 0
    assert len(advanced) == 10
    from PIL import Image

    with Image.open(path) as gif:
        assert gif.n_frames == 10
        assert gif.size == (width, height)
        assert gif.info.get("loop") == 0
        assert gif.info.get("duration") == 40
