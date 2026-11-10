"""Shared test setup."""

import os

# Widget tests create a QApplication; CI runners have no display, so default to Qt's
# offscreen platform unless the caller chose one (visual_tour.py needs a real display).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def demo():
    """The bundled demo, extracted once: settings, layout, structure, and its modes."""
    import connflex_core as cf

    from connflex_app.data import demo_config, demo_gds
    from connflex_app.gds_io import inspect_gds, load_layout
    from connflex_app.settings import Settings

    path = str(demo_gds())
    info = inspect_gds(path)
    settings = Settings(device=(1, 0), handle=(2, 0), release=(3, 0), vias=(4, 0)).with_config(str(demo_config()))
    layout = load_layout(path, settings, info.bounds_m(None))
    structure = cf.extract(layout, settings.stack(), settings.params())
    return {"path": path, "info": info, "settings": settings, "layout": layout, "structure": structure, "modal": structure.modal()}


@pytest.fixture(scope="session")
def demo_project(demo):
    """A :class:`Project` of the demo as the main window builds it, without opening a window."""
    from connflex_core.mechanics.assembly import assemble

    from connflex_app.geometry import build_mesh, release_holes, silicon_pieces
    from connflex_app.render2d import prepare_project
    from connflex_app.session import Project

    settings, layout, structure = demo["settings"], demo["layout"], demo["structure"]
    holes = release_holes(layout, settings.params().release_hole_max_area)
    pieces = silicon_pieces(structure)
    return Project(
        path=demo["path"],
        settings=settings,
        layout=layout,
        stack=settings.stack(),
        params=settings.params(),
        structure=structure,
        holes=holes,
        pieces=pieces,
        mesh=build_mesh(structure, settings.stack(), pieces),
        system=assemble(structure),
        render=prepare_project(layout, structure, pieces, holes),
        modal=demo["modal"],
    )


@pytest.fixture(scope="session")
def plates_model(demo):
    """Capacitance of the demo from plates to the handle alone: closed form, so quick."""
    import dataclasses

    from connflex_app.electrical import compute_capacitance

    return compute_capacitance(demo["structure"], dataclasses.replace(demo["settings"].electro, include_sidewalls=False))


@pytest.fixture(scope="session")
def field_model(demo):
    """Capacitance of the demo with sidewalls and fringe, solved in this process."""
    import dataclasses

    from connflex_app.electrical import compute_capacitance

    return compute_capacitance(demo["structure"], dataclasses.replace(demo["settings"].electro, workers=1))
