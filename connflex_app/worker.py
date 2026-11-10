"""Field-solver worker process of a frozen build.

connflex-core solves the field of each window and cross-section in child interpreters that it
starts as ``<sys.executable> -m connflex_core._serve``. In a frozen bundle ``sys.executable``
is the application itself, so the launcher answers that command line here, before anything of
the GUI is imported, and the child never opens a window.
"""

from __future__ import annotations

import io
import os
import sys

SERVE_ARGUMENTS = ["-m", "connflex_core._serve"]


def is_worker(argv: list[str]) -> bool:
    """Whether this process was started as a connflex-core worker."""
    return argv[1:3] == SERVE_ARGUMENTS


def serve() -> None:
    """Run connflex-core's worker loop over the pipes the parent opened."""
    # A windowed executable can start without a standard error stream, which the worker loop
    # redirects standard output onto; give it one that discards.
    try:
        os.fstat(2)
    except OSError:
        null = os.open(os.devnull, os.O_WRONLY)
        if null != 2:
            os.dup2(null, 2)
            os.close(null)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115 - lives as long as the process
    if sys.stdout is None:
        sys.stdout = sys.stderr
    if sys.stdin is None:
        sys.stdin = io.TextIOWrapper(io.BufferedReader(io.FileIO(0, "rb", closefd=False)))
    from connflex_core.workers import serve as core_serve

    core_serve()
