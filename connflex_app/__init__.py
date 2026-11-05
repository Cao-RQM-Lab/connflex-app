"""ConnFlex: a PySide6 desktop GUI around connflex-core.

The version is never written in the source: it comes from the git tag through setuptools-scm.
In a checkout it is computed live from git; in an installed or frozen app it is read from the
package metadata (or the generated ``_version.py``) that was produced at build time.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _metadata_version


def _detect_version() -> str:
    if not getattr(sys, "frozen", False):
        try:
            from setuptools_scm import get_version

            return get_version(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        except Exception:  # noqa: BLE001 - not a git checkout, or setuptools-scm not installed
            pass
    try:
        return _metadata_version("connflex-app")
    except PackageNotFoundError:
        pass
    try:
        from connflex_app._version import version

        return version
    except ImportError:
        return "0.0.0+unknown"


__version__ = _detect_version()
