"""connflex-core extraction, optionally with a drawn buried-oxide (box) layer.

connflex-core decides which buried oxide survives the release in its "release" stage: the oxide
protected by both silicon layers, eroded by the release distance, minus the dilated release
openings. When the user supplies a box layer, that drawn layer replaces this calculation. Every
other stage (healing, release openings and their mass, pruning, bodies, beams, nets) still runs
in connflex-core unchanged.
"""

from __future__ import annotations

import importlib
import threading

import numpy as np
import shapely
from shapely import MultiPolygon, STRtree

import connflex_core as cf
from connflex_core import ExtractionParams, Layout, Stack, Structure
from connflex_core.geometry.heal import as_multipolygon, snap

_LOCK = threading.Lock()


def clip_box(box: MultiPolygon, device: MultiPolygon, handle: MultiPolygon, openings: MultiPolygon) -> MultiPolygon:
    """The drawn ``box`` where oxide can physically remain: between device and handle silicon, not under openings."""
    region = as_multipolygon(box.intersection(device).intersection(handle))
    if region.is_empty or openings.is_empty:
        return region
    parts = np.array(list(openings.geoms), dtype=object)
    near = STRtree(parts).query(region, predicate="intersects")
    if near.size == 0:
        return region
    return as_multipolygon(region.difference(shapely.union_all(parts[near])))


def extract(
    layout: Layout,
    stack: Stack,
    params: ExtractionParams,
    box: MultiPolygon | None = None,
    progress=None,
) -> Structure:
    """``connflex_core.extract``, with the surviving oxide taken from ``box`` when it is given."""
    if box is None:
        return cf.extract(layout, stack, params, progress=progress)
    drawn = as_multipolygon(box.simplify(params.simplify_tolerance, preserve_topology=True))
    if params.precision_grid is not None:
        drawn = snap(drawn, params.precision_grid)
    module = importlib.import_module("connflex_core.extract")
    original = getattr(module, "surviving_box", None)
    if original is None:
        raise RuntimeError(
            "this connflex-core version computes the buried oxide differently; a box layer cannot be applied"
        )

    def surviving_box(device, handle, openings, distance, resolution):  # noqa: ARG001 - same signature as connflex-core
        return clip_box(drawn, device, handle, openings)

    with _LOCK:  # the override is process-wide; one extraction with a box layer at a time
        module.surviving_box = surviving_box
        try:
            return module.extract(layout, stack, params, progress=progress)
        finally:
            module.surviving_box = original
