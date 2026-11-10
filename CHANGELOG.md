# Changelog

The release workflow publishes the section matching the pushed tag as the GitHub release notes.

## Unreleased

On the `electromechanical` branch, against the unreleased connflex-core 0.2.0.

### Added

- **Tabs**: the window is three workspaces, Mechanical, Electrical, and Electromechanical
  (`Ctrl+1` to `Ctrl+3`), over one shared pair of views and one setup panel. Each tab brings
  its own panels, decides what a click selects, and colours the views its own way. What was
  the whole window is the Mechanical tab.
- **Electrical tab**: the capacitance model of connflex-core (plates to the handle, sidewalls
  from a field solution, fringe), with element size and interaction distance sized from the
  narrowest gap unless set. Click a net and Shift- or Ctrl-click another, in 2D or 3D, to read
  the capacitance between them, its oxide, released-gap, sidewall, and fringe parts, every
  other member the first couples to, and how the capacitance changes as each body moves. The
  same between bodies or between single electrodes. The full matrix is shown by part and
  exports as CSV; connflex-core's review report exports as JSON.
- **Electromechanical tab**: a drive table of voltage steps. Click a net to add it, type its
  voltage in each step and each step's duration, and run the transient: the motion plays back
  in 3D with every net coloured by its voltage, beside plots of voltages, displacement,
  rotation, velocity, narrowest gap, source currents, charges, and energy over time. Click in
  the plots to scrub, or select a step to jump to it. Nets can float, sources can have a
  series resistance, and a run can start settled at the equilibrium of its first step.
- **Equilibrium** of any step: displacement, stability, the share of beam stiffness left before
  pull-in, and the narrowest gap.
- Exports: the transient as NPZ, the transient playback as a GIF.
- The status bar says which step and what simulation time a transient has reached, estimates
  the time a counted stage still needs, and offers Cancel for capacitance, equilibrium, and
  transient runs.
- Setup: relative permittivity of the buried oxide, device-layer resistivity, and via
  resistance. Process files may hold `[electrostatics]` and `[transient]` tables.

### Changed

- connflex-core is installed from its `electromechanical` branch until 0.2.0 is released.
- Process files (TOML) are read by the application; connflex-core 0.2.0 no longer reads them.
- Both views colour silicon through a table over electrodes and nets on the GPU, so colouring
  by net, selection, or voltage costs no geometry. Colour by body or net in the Mechanical tab
  uses it too.
- A frozen build answers connflex-core's worker command line itself, so the field solver's
  worker processes start from the bundled executable without opening a window.
- `make verify` writes its smoke-test output under `dist/`.

## 0.1.2 (2026-11-06)

### Added

- **GDS export** (File → Export GDS…, `Ctrl+Shift+E`) writes the extracted, released
  structure to a new GDS file: 1/0 device, 2/0 release holes, 3/0 surviving buried oxide, 4/0
  handle. Choose whether release holes go on their own layer (device drawn filled) or stay cut
  into the device (2/0 left empty). Re-extracting the exported file gives the same bodies,
  beams, and frequencies.
- **Box layer role**: an optional GDS layer that defines the surviving buried oxide directly,
  replacing the calculation from release holes, the release layer, and the release distance.
  The drawn box is clipped to where device and handle silicon overlap and to outside release
  openings. It can be inverted like the other roles, and it shows in the 2D preview.
- The status bar keeps the total time of recent operations after they finish (for example
  `Extracting 1.2 s · Solving modes 80 ms`); hover over it for per-stage times. Failed
  operations are shown in red.

### Changed

- connflex-core is limited to 0.1.x, because the box layer and the GDS export depend on its
  internals.
- The bundle self-test, run on every platform before release, also checks the box-layer
  override and a GDS export round trip.

## 0.1.1 (2026-11-04)

- Self-contained builds for Linux (x64), Windows (x64), and Apple-silicon macOS, published by
  a GitHub release workflow.
- 2D view rendered with OpenGL; 3D motion applied on the GPU; release holes closed in the 3D
  model, so large layouts stay responsive.
- Status bar with stage names, step counts, per-stage timings, and overall progress.
- Export the selected mode animation as a looping GIF.
- Free bodies (zero-frequency rigid modes) are dropped from the mode list.
- The version comes from the git tag; a Makefile wraps the build and checks.

## 0.1.0 (2026-11-04)

- First release: open a GDS layout, map and invert layer roles, extract bodies, beams, and
  nets with connflex-core, view them in 2D and 3D, animate the resonant modes, and inspect the
  condensed stiffness of any body.
