"""The version comes from git tags, never from the source."""

from __future__ import annotations

import pathlib
import re

from packaging.version import Version

import connflex_app


def test_version_is_derived_not_hard_coded():
    assert Version(connflex_app.__version__)  # parses as PEP 440
    assert connflex_app.__version__ != "0.0.0+unknown"
    source = pathlib.Path(connflex_app.__file__).read_text()
    assert not re.search(r"__version__\s*=\s*[\"\']", source)
    pyproject = pathlib.Path(connflex_app.__file__).parents[1] / "pyproject.toml"
    assert not re.search(r"^version\s*=", pyproject.read_text(), flags=re.MULTILINE)
