"""Shared test setup."""

import os

# Widget tests create a QApplication; CI runners have no display, so default to Qt's
# offscreen platform unless the caller chose one (visual_tour.py needs a real display).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
