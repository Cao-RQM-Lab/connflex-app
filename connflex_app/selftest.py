"""Headless self-test: run the whole analysis pipeline on the bundled demo without a GUI.

Used by the release workflow to prove that a frozen bundle can import every dependency
and reproduce known results, including the box-layer override and the GDS export.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

EXPECTED_BODIES = 3
EXPECTED_BEAMS = 9
EXPECTED_NETS = 3
EXPECTED_FIRST_MODE_HZ = 45_560.0


def run(verbose: bool = True) -> int:
    started = time.perf_counter()
    import connflex_core as cf
    from connflex_core.mechanics.assembly import assemble
    from shapely import MultiPolygon

    from connflex_app import __version__
    from connflex_app.analysis import condensed_stiffness, static_response
    from connflex_app.data import demo_config, demo_gds
    from connflex_app.extraction import extract as extract_with_box
    from connflex_app.gds_export import LAYERS, export_gds
    from connflex_app.gds_io import inspect_gds, load_layout
    from connflex_app.geometry import build_mesh, closed_plates, describe_modes, displacement_field, motion_from_vector, normalized, release_holes
    from connflex_app.render2d import prepare_project
    from connflex_app.settings import Settings

    def say(text: str) -> None:
        if verbose:
            print(text)

    say(f"ConnFlex {__version__} self-test · connflex-core {cf.__version__}")
    path = str(demo_gds())
    info = inspect_gds(path)
    settings = Settings(device=(1, 0), handle=(2, 0), release=(3, 0), vias=(4, 0)).with_config(str(demo_config()))
    layout = load_layout(path, settings, info.bounds_m(None))
    structure = cf.extract(layout, settings.stack(), settings.params())
    say(f"extracted {len(structure.bodies)} bodies, {len(structure.beams)} beams, {structure.nets.count} nets")
    holes = release_holes(layout, settings.params().release_hole_max_area)
    plates = closed_plates(structure)
    mesh = build_mesh(structure, settings.stack(), plates)
    render = prepare_project(layout, structure, plates, holes)
    system = assemble(structure)
    modal = structure.modal(settings.solve_options())
    labels = describe_modes(structure, modal.dofs, modal.modes)
    u, theta = motion_from_vector(structure, modal.dofs, modal.modes[:, 0])
    field = normalized(displacement_field(mesh, structure, u, theta))
    body = structure.bodies[1].index
    report = condensed_stiffness(structure, system, body, set())
    response = static_response(system, body, 0, set())
    say(f"mesh {mesh.count:,} vertices, {len(render.tris)} render geometries, first mode {modal.frequencies_hz[0] / 1e3:.2f} kHz ({labels[0]})")
    say(f"body {body} effective stiffness X {report.effective[0]:.4g} N/m")
    first = modal.frequencies_hz[0]

    # The box layer patches connflex-core's oxide step: drawing the computed oxide must change nothing.
    oxide = MultiPolygon([part for item in structure.bodies for part in item.box.geoms])
    boxed = extract_with_box(layout, settings.stack(), settings.params(), box=oxide)
    boxed_first = boxed.modal(settings.solve_options()).frequencies_hz[0]
    say(f"box override: {len(boxed.bodies)} bodies, {len(boxed.beams)} beams, first mode {boxed_first / 1e3:.2f} kHz")

    # GDS export: write the released structure, read it back, and extract it again.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        gds_path = os.path.join(folder, "released.gds")
        exported = export_gds(layout, structure, settings.params(), gds_path, separate_release=True)
        roles = cf.LayerMap(device=LAYERS["device"], handle=LAYERS["handle"], release=LAYERS["release"])
        again = cf.extract(cf.load_gds(gds_path, layers=roles), settings.stack(), settings.params())
        again_first = again.modal(settings.solve_options()).frequencies_hz[0]
    say(
        "GDS export: " + ", ".join(f"{count} on {LAYERS[name][0]}/{LAYERS[name][1]}" for name, count in exported.polygons.items())
        + f"; re-extracted first mode {again_first / 1e3:.2f} kHz"
    )

    problems = []
    if len(structure.bodies) != EXPECTED_BODIES:
        problems.append(f"expected {EXPECTED_BODIES} bodies")
    if len(structure.beams) != EXPECTED_BEAMS:
        problems.append(f"expected {EXPECTED_BEAMS} beams")
    if structure.nets.count != EXPECTED_NETS:
        problems.append(f"expected {EXPECTED_NETS} nets")
    if abs(first - EXPECTED_FIRST_MODE_HZ) > 0.01 * EXPECTED_FIRST_MODE_HZ:
        problems.append(f"first mode {first:.0f} Hz, expected about {EXPECTED_FIRST_MODE_HZ:.0f} Hz")
    if len(boxed.bodies) != len(structure.bodies) or len(boxed.beams) != len(structure.beams) or abs(boxed_first - first) > 1e-6 * first:
        problems.append("the box-layer override changed the extraction")
    if any(exported.polygons.get(name, 0) == 0 for name in LAYERS):
        problems.append(f"GDS export left a layer empty: {exported.polygons}")
    if len(again.bodies) != len(structure.bodies) or len(again.beams) != len(structure.beams) or abs(again_first - first) > 1e-4 * first:
        problems.append("re-extracting the exported GDS gave a different structure")
    if not np.isfinite(field).all() or np.abs(field).max() == 0:
        problems.append("displacement field is degenerate")
    if not np.isfinite(response).all() or not np.isfinite(report.effective).all():
        problems.append("stiffness results are not finite")
    if mesh.count == 0 or not render.tris:
        problems.append("render data is empty")
    elapsed = time.perf_counter() - started
    if problems:
        say("SELF-TEST FAILED: " + "; ".join(problems))
        return 1
    say(f"SELF-TEST OK in {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
