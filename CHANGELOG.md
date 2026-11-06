# Changelog

The release workflow publishes the section matching the pushed tag as the GitHub release notes.

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
