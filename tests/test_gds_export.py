"""GDS export of the extracted, released structure."""

from __future__ import annotations

import numpy as np
import pytest

import connflex_core as cf

from connflex_app.data import demo_config, demo_gds
from connflex_app.gds_export import LAYERS, export_gds
from connflex_app.gds_io import inspect_gds, load_layout
from connflex_app.settings import Settings


@pytest.fixture(scope="module")
def extracted():
    path = str(demo_gds())
    info = inspect_gds(path)
    settings = Settings(device=(1, 0), handle=(2, 0), release=(3, 0), vias=(4, 0)).with_config(str(demo_config()))
    layout = load_layout(path, settings, info.bounds_m(None))
    structure = cf.extract(layout, settings.stack(), settings.params())
    return settings, layout, structure, structure.modal().frequencies_hz


@pytest.mark.parametrize("separate", [True, False])
def test_export_round_trip(extracted, tmp_path, separate):
    import gdstk

    settings, layout, structure, frequencies = extracted
    path = str(tmp_path / "released.gds")
    result = export_gds(layout, structure, settings.params(), path, separate_release=separate)

    library = gdstk.read_gds(path)
    found = set(library.layers_and_datatypes())
    assert found <= set(LAYERS.values())
    assert {LAYERS["device"], LAYERS["box"], LAYERS["handle"]} <= found
    assert (LAYERS["release"] in found) == separate
    assert (result.polygons["release"] > 0) == separate
    assert result.areas["box"] == pytest.approx(sum(body.box.area for body in structure.bodies), rel=1e-9)

    roles = cf.LayerMap(device=LAYERS["device"], handle=LAYERS["handle"], release=LAYERS["release"] if separate else None)
    again = cf.extract(cf.load_gds(path, layers=roles), settings.stack(), settings.params())
    assert len(again.bodies) == len(structure.bodies)
    assert len(again.beams) == len(structure.beams)
    assert np.allclose(again.modal().frequencies_hz, frequencies, rtol=1e-6)


def test_modes_differ_only_in_where_holes_go(extracted, tmp_path):
    settings, layout, structure, _ = extracted
    separate = export_gds(layout, structure, settings.params(), str(tmp_path / "a.gds"), separate_release=True)
    inline = export_gds(layout, structure, settings.params(), str(tmp_path / "b.gds"), separate_release=False)
    assert separate.areas["release"] > 0 and inline.areas["release"] == 0
    assert separate.areas["device"] - separate.areas["release"] == pytest.approx(inline.areas["device"], rel=1e-9)
    assert separate.areas["box"] == inline.areas["box"] and separate.areas["handle"] == inline.areas["handle"]
