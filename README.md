# ConnFlex

A desktop GUI around [connflex-core](https://pypi.org/project/connflex-core/), built with PySide6.

> This branch, `electromechanical`, targets the unreleased connflex-core 0.2.0 and installs it
> from that project's branch of the same name.

Open a GDS layout of an SOI MEMS process, map its layers to the device, handle, release and
via roles, and extract the reduced-order model: rigid bodies joined by Euler-Bernoulli beams,
plus the layout's electrical nets and electrodes. Three tabs then work on that model, sharing
one 2D and one 3D view:

| Tab | What it is for |
| --- | --- |
| **Mechanical** | Beams, bodies, resonant modes, and the stiffness any body feels. |
| **Electrical** | Nets and electrodes, and the capacitance between any two of them. |
| **Electromechanical** | Voltages on nets in steps, the equilibrium under each, and transients in time. |

The views:

- **2D view**: every layer rendered with pan and zoom, with toggles to highlight beams, the
  mechanical nets (rigid bodies), the electrical nets, the mechanical graph (bodies as nodes,
  beams as edges) and the electrical graph (conductors as nodes, vias as edges). Hover for
  details, click to select.
- **3D view**: the whole stack extruded (device, surviving buried oxide, handle), with every
  resonant mode and every transient animated live. Orbit with the left mouse button, pan with
  the right or middle button, zoom with the wheel. Click to select.

What a click selects and how the silicon is coloured follow the tab: a body, by layer or body,
in Mechanical; a net, body, or electrode, by net, in Electrical; a net, by voltage, in
Electromechanical.

Mechanical:

- **GDS export**: File → Export GDS writes the extracted, released structure: 1/0 device,
  2/0 release holes, 3/0 surviving buried oxide, 4/0 handle. Release holes can go on their
  own layer (device drawn filled) or stay cut into the device (2/0 left empty). The release
  openings are the ones connflex-core removed from body mass, and re-extracting the exported
  file reproduces the same structure.
- **Modes**: resonant frequencies listed from low to high with the dominant motion of each;
  select one to animate it, tune amplitude and speed, and export one cycle as a looping GIF
  (playback bar or File → Export animation as GIF).
- **Stiffness**: the 6 × 6 condensed stiffness of any body with all other bodies fixed, with
  only the anchor fixed, or with a custom set of fixed bodies, plus direct and effective
  stiffness per degree of freedom, a one-degree-of-freedom resonance estimate, the compliance
  matrix, and the static deflection shape under a unit force shown in 3D.

Electrical:

- **Capacitance model**: plates between device electrodes and the handle (through surviving
  oxide and across the released gap), sidewalls between device electrodes from a field
  solution, and the fringe above and below the device layer. Element size and interaction
  distance are sized from the narrowest gap between device electrodes unless you set them.
- **Between two selections**: click a net to select it (A, amber); Shift- or Ctrl-click a second
  one (B, cyan). The panel reads the capacitance between them with every other net held, its
  oxide, released-gap, sidewall, and fringe parts, and each one's total. Switch the selection
  from nets to bodies or to single electrodes to ask the same of those.
- **Around one selection**: every other member it couples to, largest first, and how the
  capacitance changes when each body moves (∂C/∂q per micrometre and per milliradian), which
  is what a voltage turns into force.
- **Matrix**: the full capacitance matrix between nets, bodies, or electrodes, in total or by
  part; click a cell to select its pair. Export it as CSV, or connflex-core's review report as
  JSON.

Electromechanical:

- **Drive**: click a net in either view to add it to the drive, then type its voltage in each
  step and each step's duration (`40`, `2.5 mV`, `180 µs`). Sources ramp from one step's
  voltage to the next over the rise time. Nets that are not listed stay at 0 V, and a net can
  be left floating with zero charge. The views colour every net by its voltage.
- **Transient**: runs the whole sequence in one integration, then plays the motion back in 3D
  (magnified, or at true scale) with the nets coloured by their voltage at each instant. The
  timeline plots the voltages against the displacement, rotation, or velocity of a body, the
  narrowest gap, source currents, charges, or the energy ledger; click or drag in it to scrub,
  or select a step to jump to it. A run that closes a gap to the contact clearance stops
  there and says between which electrodes. Export the series as NPZ or the playback as a GIF.
- **Equilibrium**: where the structure rests under the voltages of one step, whether that is
  stable, how much beam stiffness the field has left (0 at pull-in), and the narrowest gap,
  with the deflection shown in 3D. A transient can start settled at the equilibrium of its
  first step.

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

This pulls in PySide6, PyOpenGL, Pillow, numpy, shapely, and `connflex-core[gds]` (gdstk
reader), the last from the `electromechanical` branch of its repository until 0.2.0 is on
PyPI, so `git` has to be installed. To work against a local checkout of connflex-core
instead, install it over the top: `pip install -e ../connflex-core`.

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
panel ("Load process config"). The file holds `[stack.device]`, `[stack.box]`, `[stack.handle]`,
and `[extraction]` tables, and optionally `[solve]`, `[materials.<name>]`, `[electrostatics]`,
and `[transient]`, with the field names of connflex-core's parameter objects; the application
reads it, since connflex-core itself takes parameters only as Python objects. Layer roles are
guessed from the file (1/0 device, 2/0 handle,
3/0 release, 4/0 vias by convention, otherwise by area) and can be changed before extraction.
The optional **Box** role takes the surviving buried oxide from a drawn layer instead of
computing it from the release holes, the release layer, and the release distance (which is
then unused). The drawn box is clipped to where device and handle silicon overlap and to
outside release openings; release holes and the release layer still remove device mass.
Tick **Invert** next to a role when that layer is drawn as etch openings rather than remaining
material: the role then uses the complement of the drawn polygons within the cell's bounding
box. The 2D view previews the result immediately.

## Workflow

1. **Open GDS** and check the layer roles, wafer stack and extraction parameters in the Setup
   panel. The raw layers show in the 2D view immediately.
2. **Extract**. Bodies, beams, surviving oxide and nets appear in both views, and the modal
   solve runs automatically.
3. **Mechanical** tab: pick a **mode** to animate it, or select a **body** in either view and
   read its stiffness in the Stiffness panel. Double-click a row there to show the deflection
   shape under a unit force in that direction.
4. **Electrical** tab: **Compute capacitance**, then click a net and Shift-click another to
   read the capacitance between them.
5. **Electromechanical** tab: click the nets to drive, type their voltages step by step, and
   **Run transient**. If the capacitance model is missing or its settings changed, the run
   solves it first.

Settings are remembered per file, the drive included: its nets are stored by a point on
their source's electrode, so they find their net again after a new extraction. `Ctrl+1`,
`Ctrl+2`, and `Ctrl+3` switch tabs. Press `F` in the 2D view to fit, `R` in the 3D view to
reset the camera, and `Space` to pause and resume the animation.

Under the Setup panel's material properties, the buried oxide's relative permittivity sets
the capacitance through surviving oxide, and a device-layer resistivity gives every beam a
resistance, so nets behind thin beams charge late in a transient. Both are part of the
structure: change them and extract again.

## Performance

Loading, extraction, meshing, and every analysis run on worker threads; the status bar shows
the current stage, step counts, per-stage timings, an overall percentage, and the elapsed
time. A counted stage that runs for more than a couple of seconds also shows an estimate of
the time it still needs, and a transient says which step it is in and the simulation time it
has reached. Capacitance, equilibrium, and transient runs can be cancelled from the status
bar. When an operation finishes, its total stays in the status bar next to the previous ones
(hover for the per-stage times) until the next file is opened or extracted.

The capacitance field is solved in worker processes, every processor by default (the
Electrical panel's Advanced section sets their number). A transient evaluates the capacitance
model thousands of times; the surrogate tolerance, on by default, lets it reuse nearby field
solutions, which is what makes transients on the field method take seconds instead of hours.

Both views render with OpenGL: every layer is triangulated once in the worker and kept in GPU
buffers, so panning, zooming, and orbiting only change a matrix and stay smooth on large
layouts. Buffers are rebuilt only when the data or the display options change. Mode
shapes and deflections are applied on the GPU from per-body transforms, so switching modes
is instant whatever the mesh size.

The 3D model is built from the same closed shapes connflex-core analyses: release holes
(interior rings up to the release-hole area limit) are filled and hairline gaps healed before
extraction, so plates are solid and the mesh size does not depend on the number of holes.
The 2D view still shows the layout as drawn.

Colour by net, selection, and voltage costs no geometry either: every piece of silicon carries
the index of its electrode, and both views look its colour up in a small table on the GPU.

## Development

A Makefile wraps the common steps (`make help` lists them): `make venv` creates the
environment and installs the dev and build extras, `make check` lints, tests, and runs the
self-test, `make run ARGS=layout.gds` launches the app, `make build` creates the executable
in `dist/`, `make verify` self-tests it, `make package` archives it the way the release
workflow does, `make dist` chains those three, and `make clean` removes every build output.

- `pytest` runs the headless pipeline tests; `ruff check .` lints.
- `python -m connflex_app --self-test` runs the analysis pipeline on the bundled demo without a
  GUI (extraction, modes, stiffness, capacitance with worker processes, a transient, and the
  GDS export), and `--smoke-test out.png` drives the real window through all three tabs of
  the demo and saves a screenshot.
- `tests/visual_tour.py OUTPUT_DIR` drives the window through every feature and saves
  screenshots. It needs a display because Qt's offscreen plugin has no OpenGL on macOS.

## Releases

Builds are produced by `.github/workflows/release.yml` with PyInstaller (see
`packaging/README.md`). The version is never written in the source: setuptools-scm derives
it from the git tag (`vX.Y.Z` gives `X.Y.Z`; later commits give `X.Y.Z+1.devN+g<sha>`), and
`connflex_app.__version__` reports it at runtime. To publish a release, add a section for
the version to `CHANGELOG.md` (it becomes the release notes), commit, then tag and push:

```sh
git tag v0.1.2
git push origin main v0.1.2
```

With Docker and [act](https://github.com/nektos/act) installed, `make act-ci` runs the CI
workflow locally and `make act-release` runs the Linux build job of the release workflow,
both emulating the x64 GitHub runner. The workflow builds and self-tests the bundle on all
three platforms, then attaches `ConnFlex-<version>-linux-x64.tar.gz`, `-windows-x64.zip`,
`-macos-arm64.dmg`, and a `SHA256SUMS.txt` to the release. `workflow_dispatch` builds the
artifacts without publishing.
