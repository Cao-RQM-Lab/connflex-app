"""Files shipped inside the application: the demo layout, its process file, and the icon."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """Directory of the bundled data, inside the package or inside a frozen bundle."""
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        candidate = Path(frozen) / "connflex_app" / "data"
        if candidate.is_dir():
            return candidate
    return Path(os.path.dirname(os.path.abspath(__file__)))


def demo_gds() -> Path:
    return data_dir() / "demo_resonator.gds"


def demo_config() -> Path:
    return data_dir() / "demo_process.toml"


def icon_png() -> Path:
    return data_dir() / "icon.png"
