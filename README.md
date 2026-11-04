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
  select one to animate it, and tune amplitude and speed.
- **Stiffness**: the 6 × 6 condensed stiffness of any body with all other bodies fixed, with
  only the anchor fixed, or with a custom set of fixed bodies, plus direct and effective
  stiffness per degree of freedom, a one-degree-of-freedom resonance estimate, the compliance
  matrix, and the static deflection shape under a unit force shown in 3D.

## Install

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

## Development

`tests/visual_tour.py OUTPUT_DIR` drives the real window through a scripted session and saves
screenshots of every feature. It needs a display because Qt's offscreen plugin has no OpenGL on
macOS.
