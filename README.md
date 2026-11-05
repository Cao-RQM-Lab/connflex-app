# ConnFlex

A desktop GUI around [connflex-core](https://pypi.org/project/connflex-core/), built with PySide6.

Open a GDS layout of an SOI MEMS process, map its layers to the device, handle, release and
via roles, and extract the reduced-order model: rigid bodies joined by Euler-Bernoulli beams,
plus the layout's electrical nets. Then:

- **2D view**: every layer rendered with pan and zoom, with toggles to highlight beams, the
  mechanical nets (rigid bodies), the electrical nets, the mechanical graph (bodies as nodes,
  beams as edges) and the electrical graph (conductors as nodes, vias as edges). Hover for
  details, click to select a body.
- **3D view**: the whole stack extruded (device, surviving buried oxide, handle), with every
  resonant mode animated live. Orbit with the left mouse button, pan with the right or middle
  button, zoom with the wheel. Click a body to select it.
- **Modes**: resonant frequencies listed from low to high with the dominant motion of each;
  select one to animate it, tune amplitude and speed, and export one cycle as a looping GIF
  (playback bar or File → Export animation as GIF).
- **Stiffness**: the 6 × 6 condensed stiffness of any body with all other bodies fixed, with
  only the anchor fixed, or with a custom set of fixed bodies, plus direct and effective
  stiffness per degree of freedom, a one-degree-of-freedom resonance estimate, the compliance
  matrix, and the static deflection shape under a unit force shown in 3D.

## Download

Self-contained builds for Linux (x64), Windows (x64), and Apple-silicon macOS are attached
to every [GitHub release](../../releases). Unpack the archive and run `ConnFlex`; nothing
else needs to be installed. The macOS app is ad-hoc signed rather than notarized, so on
first launch right-click it and choose Open (or run `xattr -dr com.apple.quarantine
ConnFlex.app`). Pick Help → Open demo layout to try it without a file of your own.

## Install from source

Python 3.11 or newer.

```sh
python -m venv venv
source venv/bin/activate
pip install -e .
```

This pulls in `connflex-core[gds]` (gdstk reader), PySide6, PyOpenGL, numpy and shapely.

## Run

```sh
connflex-app                       # or: python -m connflex_app
connflex-app path/to/layout.gds    # open a file at start
```

A demo layout ships with the repository:

```sh
python examples/make_demo_gds.py   # writes examples/demo_resonator.gds
connflex-app examples/demo_resonator.gds
```

Its process description is in `examples/demo_process.toml` and can be loaded from the Setup
panel ("Load process config"). Layer roles are guessed from the file (1/0 device, 2/0 handle,
3/0 release, 4/0 vias by convention, otherwise by area) and can be changed before extraction.
Tick **Invert** next to a role when that layer is drawn as etch openings rather than remaining
material: the role then uses the complement of the drawn polygons within the cell's bounding
box. The 2D view previews the result immediately.

## Workflow

1. **Open GDS** and check the layer roles, wafer stack and extraction parameters in the Setup
   panel. The raw layers show in the 2D view immediately.
2. **Extract**. Bodies, beams, surviving oxide and nets appear in both views, and the modal
   solve runs automatically.
3. Pick a **mode** to animate it, or select a **body** in either view and read its stiffness
   in the Stiffness panel. Double-click a row there to show the deflection shape under a unit
   force in that direction.

Settings are remembered per file. Press `F` in the 2D view to fit, `R` in the 3D view to reset
the camera, and `Space` to pause and resume the animation.

## Performance

Loading, extraction, meshing, and the modal solve run on worker threads; the status bar shows
the current stage, step counts, per-stage timings, an overall percentage, and the elapsed
time. Both views render with OpenGL: every layer is triangulated once in the worker and kept
in GPU buffers, so panning, zooming, and orbiting only change a matrix and stay smooth on
large layouts. Buffers are rebuilt only when the data or the display options change. Mode
shapes and deflections are applied on the GPU from per-body transforms, so switching modes
is instant whatever the mesh size.

The 3D model is built from the same closed shapes connflex-core analyses: release holes
(interior rings up to the release-hole area limit) are filled and hairline gaps healed before
extraction, so plates are solid and the mesh size does not depend on the number of holes.
The 2D view still shows the layout as drawn.

## Development

A Makefile wraps the common steps (`make help` lists them): `make venv` creates the
environment and installs the dev and build extras, `make check` lints, tests, and runs the
self-test, `make run ARGS=layout.gds` launches the app, `make build` creates the executable
in `dist/`, `make verify` self-tests it, `make package` archives it the way the release
workflow does, `make dist` chains those three, and `make clean` removes every build output.

- `pytest` runs the headless pipeline tests; `ruff check .` lints.
- `python -m connflex_app --self-test` runs the analysis pipeline on the bundled demo without a
  GUI, and `--smoke-test out.png` drives the real window through the demo and saves a screenshot.
- `tests/visual_tour.py OUTPUT_DIR` drives the window through every feature and saves
  screenshots. It needs a display because Qt's offscreen plugin has no OpenGL on macOS.

## Releases

Builds are produced by `.github/workflows/release.yml` with PyInstaller (see
`packaging/README.md`). To publish a release, bump `__version__` in
`connflex_app/__init__.py`, then push a matching tag:

```sh
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds and self-tests the bundle on all three platforms, then attaches
`ConnFlex-<version>-linux-x64.tar.gz`, `-windows-x64.zip`, `-macos-arm64.dmg`, and a
`SHA256SUMS.txt` to the release. `workflow_dispatch` builds the artifacts without publishing.
