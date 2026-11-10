"""Headless tests of the analysis pipeline (no display needed)."""

from __future__ import annotations

import numpy as np
import pytest

import connflex_core as cf
from connflex_core.mechanics.assembly import assemble

from connflex_app.analysis import condensed_stiffness, free_bodies, rigid_mode_count, static_response
from connflex_app.data import demo_config, demo_gds
from connflex_app.gds_io import guess_roles, inspect_gds, load_layout
from connflex_app.geometry import (
    KIND_BEAM,
    build_mesh,
    describe_modes,
    displacement_field,
    motion_from_vector,
    normalized,
    release_holes,
    silicon_pieces,
)
from connflex_app.render2d import prepare_layout, prepare_project
from connflex_app.selftest import run as selftest
from connflex_app.settings import Settings


@pytest.fixture(scope="module")
def project():
    path = str(demo_gds())
    info = inspect_gds(path)
    settings = Settings(device=(1, 0), handle=(2, 0), release=(3, 0), vias=(4, 0)).with_config(str(demo_config()))
    layout = load_layout(path, settings, info.bounds_m(None))
    structure = cf.extract(layout, settings.stack(), settings.params())
    holes = release_holes(layout, settings.params().release_hole_max_area)
    pieces = silicon_pieces(structure)
    mesh = build_mesh(structure, settings.stack(), pieces)
    return {
        "info": info,
        "settings": settings,
        "layout": layout,
        "structure": structure,
        "holes": holes,
        "pieces": pieces,
        "mesh": mesh,
        "system": assemble(structure),
        "modal": structure.modal(),
    }


def test_inspection_and_role_guess():
    info = inspect_gds(str(demo_gds()))
    assert info.top_cells == ["RESONATOR"]
    assert guess_roles(info.layers(None)) == {"device": (1, 0), "handle": (2, 0), "release": (3, 0), "vias": (4, 0), "box": None}
    assert info.bounds_m(None) == pytest.approx((0.0, 0.0, 1.2e-3, 1.2e-3))


def test_extraction(project):
    structure = project["structure"]
    assert len(structure.bodies) == 3
    assert len(structure.beams) == 9
    assert structure.nets.count == 3
    assert structure.bodies[0].anchor


def test_inverted_vias_merge_nets(project):
    settings = Settings(device=(1, 0), handle=(2, 0), release=(3, 0), vias=(4, 0), invert_vias=True)
    layout = load_layout(str(demo_gds()), settings, project["info"].bounds_m(None))
    structure = cf.extract(layout, settings.stack(), settings.params())
    assert structure.nets.count == 1


def test_inverting_a_full_layer_fails_clearly(project):
    settings = Settings(device=(1, 0), handle=(2, 0), invert_handle=True)
    with pytest.raises(ValueError, match="leaves nothing"):
        load_layout(str(demo_gds()), settings, project["info"].bounds_m(None))


def test_mesh_and_motion(project):
    mesh, structure, modal = project["mesh"], project["structure"], project["modal"]
    assert mesh.count > 0 and np.isfinite(mesh.positions).all()
    assert modal.frequencies_hz[0] == pytest.approx(45_560, rel=0.01)
    assert np.all(np.diff(modal.frequencies_hz) >= -1e-6)
    u, theta = motion_from_vector(structure, modal.dofs, modal.modes[:, 0])
    field = normalized(displacement_field(mesh, structure, u, theta))
    assert np.linalg.norm(field, axis=1).max() == pytest.approx(1.0)
    assert np.any(np.abs(field[mesh.kind == KIND_BEAM]) > 0)
    labels = describe_modes(structure, modal.dofs, modal.modes)
    assert len(labels) == len(modal.frequencies_hz) and labels[0].endswith("X")


def test_stiffness_conditions(project):
    structure, system = project["structure"], project["system"]
    body = structure.bodies[1].index
    others = {b.index for b in structure.bodies if not b.anchor and b.index != body}
    fixed = condensed_stiffness(structure, system, body, others)
    free = condensed_stiffness(structure, system, body, set())
    assert fixed.effective[0] > free.effective[0] > 0
    assert free.effective[0] == pytest.approx(640, rel=0.02)
    assert not fixed.singular and not free.singular
    response = static_response(system, body, 0, set())
    assert np.isfinite(response).all() and response.max() > 0


def test_render_data(project):
    layout, structure = project["layout"], project["structure"]
    preview = prepare_layout(layout)
    assert preview.pick_tree is not None and len(preview.tris) == 4
    render = prepare_project(layout, structure, project["pieces"], project["holes"])
    assert render.beam_counts[structure.bodies[1].index] == 5
    assert render.graph is not None and len(render.graph["beams"]) == 9
    assert all(np.isfinite(t).all() for t in render.tris.values())


def test_settings_round_trip():
    settings = Settings(device=(1, 0), handle=(2, 0), invert_release=True, anchor=(1e-4, 2e-4), n_modes=12)
    again = Settings.from_json(settings.to_json())
    assert again == settings
    assert again.inverted_roles() == {"release"}


def test_selftest_passes():
    assert selftest(verbose=False) == 0


def _perforated(x0: float, y0: float, size: float, hole: float = 8.0, pitch: float = 15.0):
    from shapely import Polygon

    n = int(size // pitch)
    margin = (size - n * pitch) / 2
    holes = []
    for i in range(n):
        for j in range(n):
            cx, cy = x0 + margin + pitch * (i + 0.5), y0 + margin + pitch * (j + 0.5)
            holes.append([(cx - hole / 2, cy - hole / 2), (cx + hole / 2, cy - hole / 2), (cx + hole / 2, cy + hole / 2), (cx - hole / 2, cy + hole / 2)])
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)], holes)


def test_free_body_modes_are_identified():
    """A mass tied to the frame only through a tether that touches three bodies is free."""
    from shapely import box

    device = [
        box(100, 100, 300, 250),  # pad A (anchored through surviving oxide)
        box(700, 100, 900, 250),  # pad B
        _perforated(350, 400, 300),  # released mass A, held only by the tether below
        box(498, 250, 502, 400),  # tether from mass A down to the bar
        box(300, 248, 700, 252),  # bar touching pad A, pad B, and the tether: three bodies, rejected
        _perforated(700, 400, 200),  # released mass B on a proper beam
        box(798, 250, 802, 400),  # beam from pad B to mass B
    ]
    layout = cf.Layout.from_shapely(device=device, handle=[box(0, 0, 1000, 1000)], unit=1e-6)
    settings = Settings()
    structure = cf.extract(layout, settings.stack(), settings.params())
    free = free_bodies(structure)
    movable = [body.index for body in structure.bodies if not body.anchor]
    assert len(movable) == 2 and len(free) == 1
    assert rigid_mode_count(structure) == 6
    modal = structure.modal()
    assert len(modal.frequencies_hz) == 12
    assert np.all(modal.frequencies_hz[:6] < 1.0)
    assert modal.frequencies_hz[6] > 1e3
