"""Headless self-test: run the whole analysis pipeline on the bundled demo without a GUI.

Used by the release workflow to prove that a frozen bundle can import every dependency
and reproduce known results: extraction, modes, stiffness, the box-layer override, the GDS
export, capacitance with the field solved in worker processes, and a transient.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import time

import numpy as np

EXPECTED_BODIES = 3
EXPECTED_BEAMS = 9
EXPECTED_NETS = 3
EXPECTED_FIRST_MODE_HZ = 45_560.0
EXPECTED_COMB_CAPACITANCE = 98.1e-15  # farads between the right comb's stator and the proof masses
WORKERS = 2  # enough to prove that worker processes start, also from a frozen bundle


def run(verbose: bool = True) -> int:
    started = time.perf_counter()
    import connflex_core as cf
    from connflex_core.mechanics.assembly import assemble
    from shapely import MultiPolygon

    from connflex_app import __version__
    from connflex_app.analysis import condensed_stiffness, static_response
    from connflex_app.data import demo_config, demo_gds
    from connflex_app.drive import DrivePlan, NetDrive, run_transient
    from connflex_app.electrical import compute_capacitance, grouping, pair_report
    from connflex_app.extraction import extract as extract_with_box
    from connflex_app.gds_export import LAYERS, export_gds
    from connflex_app.gds_io import inspect_gds, load_layout
    from connflex_app.geometry import (
        build_mesh,
        describe_modes,
        displacement_field,
        motion_from_vector,
        normalized,
        release_holes,
        silicon_pieces,
    )
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
    pieces = silicon_pieces(structure)
    mesh = build_mesh(structure, settings.stack(), pieces)
    render = prepare_project(layout, structure, pieces, holes)
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

    # Capacitance: plates, sidewalls, and fringe, with the field solved in worker processes.
    capacitance = compute_capacitance(structure, dataclasses.replace(settings.electro, workers=WORKERS))
    nets = grouping(structure, "net")
    stator = structure.electrode_at((1050e-6, 600e-6), "device")
    shuttle = structure.electrode_at((800e-6, 600e-6), "device")
    comb = pair_report(capacitance.model, nets, nets.index(stator.net), nets.index(shuttle.net))
    say(
        f"capacitance: {capacitance.summary()}; comb N{stator.net}-N{shuttle.net} {comb.value * 1e15:.1f} fF "
        f"({comb.parts['sidewall'] * 1e15:.1f} sidewall + {comb.parts['fringe'] * 1e15:.1f} fringe) in {capacitance.elapsed:.2f} s"
    )

    # Transient: 40 V on the comb's stator for a few periods; the shuttle is pulled toward it.
    plan = DrivePlan(durations=(40e-6,), nets=(NetDrive(stator.net, stator.index, "drive", (40.0,)),), rise_time=1e-6, samples=40)
    run = run_transient(structure, capacitance.model, plan, settings.transient.options(), workers=WORKERS)
    travel = run.u[:, shuttle.body, 0]
    supplied = float(np.abs(run.result.energy.supplied).max())
    imbalance = float(np.abs(run.result.energy.imbalance).max())
    say(
        f"transient: {len(run.times)} outputs, {run.result.steps} steps, shuttle travel up to {travel.max() * 1e9:.2f} nm, "
        f"energy imbalance {imbalance / supplied if supplied else 0.0:.1e} of supplied, in {run.elapsed:.2f} s"
    )

    problems = []
    if abs(comb.value - EXPECTED_COMB_CAPACITANCE) > 0.03 * EXPECTED_COMB_CAPACITANCE:
        problems.append(f"comb capacitance {comb.value * 1e15:.1f} fF, expected about {EXPECTED_COMB_CAPACITANCE * 1e15:.1f} fF")
    if not np.allclose(capacitance.model.matrix.sum(axis=1), 0.0, atol=1e-18):
        problems.append("rows of the capacitance matrix do not sum to zero")
    if run.result.contact is not None or not np.isfinite(run.result.displacement).all():
        problems.append("the transient did not run to its last time")
    if not travel.max() > 1e-10:
        problems.append(f"the comb did not pull the shuttle ({travel.max():.3e} m)")
    if supplied and imbalance > 1e-3 * supplied:
        problems.append(f"the transient's energy does not balance ({imbalance / supplied:.1e} of supplied)")
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
