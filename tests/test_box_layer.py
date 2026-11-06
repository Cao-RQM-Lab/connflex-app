"""The optional box layer replaces connflex-core's surviving-oxide calculation and nothing else."""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import pytest
import shapely
from shapely import MultiPolygon

import connflex_core as cf

from connflex_app.data import demo_config, demo_gds
from connflex_app.extraction import extract
from connflex_app.gds_export import _gds_polygons
from connflex_app.gds_io import inspect_gds, load_box, load_layout
from connflex_app.settings import Settings

BOX_LAYER = (5, 0)


@pytest.fixture(scope="module")
def demo():
    path = str(demo_gds())
    info = inspect_gds(path)
    settings = Settings(device=(1, 0), handle=(2, 0), release=(3, 0), vias=(4, 0)).with_config(str(demo_config()))
    layout = load_layout(path, settings, info.bounds_m(None))
    structure = cf.extract(layout, settings.stack(), settings.params())
    oxide = MultiPolygon([part for body in structure.bodies for part in body.box.geoms])
    return {"path": path, "info": info, "settings": settings, "layout": layout, "structure": structure, "oxide": oxide}


def _movable(structure) -> int:
    return sum(1 for body in structure.bodies if not body.anchor)


def test_box_equal_to_computed_oxide_reproduces_extraction(demo):
    settings, structure = demo["settings"], demo["structure"]
    again = extract(demo["layout"], settings.stack(), settings.params(), box=demo["oxide"])
    assert len(again.bodies) == len(structure.bodies) and len(again.beams) == len(structure.beams)
    for a, b in zip(again.bodies, structure.bodies):
        assert a.inertial.mass == pytest.approx(b.inertial.mass, rel=1e-12)
        assert a.box.area == pytest.approx(b.box.area, rel=1e-9)
    assert np.allclose(again.modal().frequencies_hz, structure.modal().frequencies_hz, rtol=1e-9)


def test_box_layer_takes_precedence_over_release_distance(demo):
    settings = demo["settings"]
    tiny = dataclasses.replace(settings.params(), release_distance=1e-9)  # oxide would survive under the masses
    assert _movable(cf.extract(demo["layout"], settings.stack(), tiny)) < _movable(demo["structure"])
    with_box = extract(demo["layout"], settings.stack(), tiny, box=demo["oxide"])
    assert _movable(with_box) == _movable(demo["structure"])


def test_box_under_the_whole_die_anchors_everything(demo):
    settings, layout = demo["settings"], demo["layout"]
    die = MultiPolygon([shapely.box(*layout.handle.bounds)])
    structure = extract(layout, settings.stack(), settings.params(), box=die)
    assert _movable(structure) == 0
    # Oxide only where both silicon layers are, and never under a release opening.
    assert structure.bodies[0].box.area <= layout.device.intersection(layout.handle).area + 1e-18


def test_override_is_removed_afterwards(demo):
    module = importlib.import_module("connflex_core.extract")
    original = module.surviving_box
    settings = demo["settings"]
    extract(demo["layout"], settings.stack(), settings.params(), box=demo["oxide"])
    assert module.surviving_box is original

    class Fails:
        def start(self, stage, total):
            if stage == "bodies":
                raise RuntimeError("stop")

        def advance(self, n=1):
            pass

        def finish(self):
            pass

    with pytest.raises(RuntimeError, match="stop"):
        extract(demo["layout"], settings.stack(), settings.params(), box=demo["oxide"], progress=Fails())
    assert module.surviving_box is original


def test_box_role_read_from_gds(demo, tmp_path):
    import gdstk

    library = gdstk.read_gds(demo["path"])
    top = library.top_level()[0]
    top.add(*_gds_polygons(demo["oxide"], BOX_LAYER))
    path = str(tmp_path / "with_box.gds")
    library.write_gds(path, max_points=0)

    bounds = inspect_gds(path).bounds_m(None)
    settings = dataclasses.replace(demo["settings"], box=BOX_LAYER)
    box = load_box(path, settings, bounds)
    assert box.area == pytest.approx(demo["oxide"].area, rel=1e-9)
    inverted = load_box(path, dataclasses.replace(settings, invert_box=True), bounds)
    die = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    assert inverted.area == pytest.approx(die - box.area, rel=1e-9)

    structure = extract(load_layout(path, settings, bounds), settings.stack(), settings.params(), box=box)
    assert np.allclose(structure.modal().frequencies_hz, demo["structure"].modal().frequencies_hz, rtol=1e-9)
    assert load_box(path, dataclasses.replace(settings, box=None), bounds) is None
    with pytest.raises(ValueError, match="box layer 9/0 has no polygons"):
        load_box(path, dataclasses.replace(settings, box=(9, 0)), bounds)


def test_settings_round_trip_with_box():
    settings = Settings(device=(1, 0), handle=(2, 0), box=(5, 0), invert_box=True)
    again = Settings.from_json(settings.to_json())
    assert again == settings and again.inverted_roles() == {"box"}
