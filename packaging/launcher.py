"""Entry script for the frozen application."""

import sys

from connflex_app.worker import is_worker, serve

if __name__ == "__main__":
    if is_worker(sys.argv):
        # A field-solver process that connflex-core started: no GUI, just its worker loop.
        serve()
        sys.exit(0)
    from connflex_app.app import main

    sys.exit(main())
