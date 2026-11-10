"""Render-ready data for the 2D view, built off the GUI thread.

Every geometry the view may draw is triangulated once (for GPU fills) and reduced to its
outline edges once (for GPU lines). Coordinates are micrometres relative to the layout
centre so that single-precision vertex buffers keep their accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import MultiPolygon, Polygon, STRtree
from shapely.geometry.base import BaseGeometry

from connflex_core import Layout, Structure

from connflex_app.geometry import (
    SCENE_UNIT,
    Groups,
    Piece,
    TriangleCache,
    device_group_lookup,
    silicon_pieces,
    triangulate,
    visible_rejected,
)

EMPTY = np.zeros((0, 2), dtype=np.float32)


def _polygons(geometry: BaseGeometry | NDArray) -> NDArray:
    parts = shapely.get_parts(geometry)
    return parts[shapely.get_type_id(parts) == 3]


def fill_triangles(geometry: BaseGeometry, origin: NDArray, cache: TriangleCache | None = None) -> NDArray[np.float32]:
    """Triangle soup (3T, 2) covering ``geometry``, in micrometres relative to ``origin``."""
    polygons = _polygons(geometry)
    if polygons.size == 0:
        return EMPTY
    try:
        tri = triangulate(geometry, cache)
    except Exception:  # noqa: BLE001 - fall back to one polygon at a time
        pieces = []
        for polygon in polygons:
            try:
                pieces.append(triangulate(MultiPolygon([polygon])))
            except Exception:  # noqa: BLE001 - skip what cannot be triangulated
                continue
        tri = np.concatenate(pieces) if pieces else np.zeros((0, 3, 2))
    return (tri.reshape(-1, 2) * SCENE_UNIT - origin).astype(np.float32)


def outline_edges(geometry: BaseGeometry, origin: NDArray) -> NDArray[np.float32]:
    """Line segments (2E, 2) of every ring of ``geometry``, in micrometres relative to ``origin``."""
    polygons = _polygons(geometry)
    if polygons.size == 0:
        return EMPTY
    rings = shapely.get_rings(polygons)
    coords, index = shapely.get_coordinates(rings, return_index=True)
    if coords.shape[0] < 2:
        return EMPTY
    same = index[:-1] == index[1:]
    segments = np.stack([coords[:-1][same], coords[1:][same]], axis=1)
    return (segments.reshape(-1, 2) * SCENE_UNIT - origin).astype(np.float32)


@dataclass
class RenderData:
    """Triangles, edges, picking index, and graph geometry for one layout or project."""

    origin: NDArray[np.float64]
    cache: TriangleCache | None = None
    tris: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    edges: dict[int, NDArray[np.float32]] = field(default_factory=dict)
    keep: list = field(default_factory=list)
    pick_tree: STRtree | None = None
    pick_tags: list[tuple[str, int]] = field(default_factory=list)
    graph: dict | None = None
    outlines: dict[int, BaseGeometry] = field(default_factory=dict)
    rejected: list = field(default_factory=list)
    beam_counts: dict[int, int] = field(default_factory=dict)
    free_bodies: set[int] = field(default_factory=set)
    holes: MultiPolygon | None = None
    box_layer: MultiPolygon | None = None
    pieces: list[Piece] = field(default_factory=list)
    groups: Groups | None = None
    beam_groups: list[int] = field(default_factory=list)
    rejected_groups: list[int] = field(default_factory=list)
    group_tree: STRtree | None = None
    group_tags: list[int] = field(default_factory=list)
    group_geometry: dict[int, list[BaseGeometry]] = field(default_factory=dict)

    @property
    def origin_m(self) -> NDArray[np.float64]:
        return self.origin / SCENE_UNIT

    def group_at(self, x: float, y: float) -> int:
        """Paint group of the topmost silicon at a point in metres: beams and bridges first,
        then the device layer, then the handle. -1 where there is none."""
        if self.group_tree is None:
            return -1
        hits = self.group_tree.query(shapely.Point(x, y), predicate="intersects")
        return self.group_tags[int(hits.min())] if hits.size else -1

    def register(self, geometry: BaseGeometry) -> None:
        key = id(geometry)
        if key not in self.tris:
            self.tris[key] = fill_triangles(geometry, self.origin, self.cache)
            self.edges[key] = outline_edges(geometry, self.origin)
            self.keep.append(geometry)

    def tri(self, geometry: BaseGeometry) -> NDArray[np.float32]:
        self.register(geometry)
        return self.tris[id(geometry)]

    def edge(self, geometry: BaseGeometry) -> NDArray[np.float32]:
        key = id(geometry)
        if key not in self.edges:
            self.edges[key] = outline_edges(geometry, self.origin)
            self.keep.append(geometry)
        return self.edges[key]


def _origin(layout: Layout) -> NDArray[np.float64]:
    xmin, ymin, xmax, ymax = shapely.total_bounds(np.array([layout.device, layout.handle], dtype=object))
    return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2]) * SCENE_UNIT


def prepare_layout(
    layout: Layout,
    advance: Callable[[], None] | None = None,
    cache: TriangleCache | None = None,
    box: MultiPolygon | None = None,
) -> RenderData:
    """Render data for the raw layers of a layout (the preview before extraction), with a drawn box layer if given."""
    data = RenderData(origin=_origin(layout), cache=cache)
    if box is not None and not box.is_empty:
        data.box_layer = box
        data.register(box)
    for geometry in (layout.handle, layout.device, layout.release, layout.vias):
        if geometry is not None:
            data.register(geometry)
            if advance is not None:
                advance()
    geoms: list = []
    tags: list[tuple[str, int]] = []
    for part in layout.device.geoms:
        geoms.append(part)
        tags.append(("device", -1))
    for part in layout.handle.geoms:
        geoms.append(part)
        tags.append(("handle_raw", -1))
    data.pick_tree = STRtree(geoms) if geoms else None
    data.pick_tags = tags
    return data


def prepare_project(
    layout: Layout,
    structure: Structure,
    pieces: list[Piece] | None,
    holes: list[Polygon],
    advance: Callable[[], None] | None = None,
    *,
    cache: TriangleCache | None = None,
) -> RenderData:
    """Render data for an extracted structure: bodies, beams, nets, graphs, and picking.

    ``pieces`` are the solid silicon pieces (see :func:`connflex_app.geometry.silicon_pieces`);
    ``cache`` can be shared with the 3D mesh builder so nothing is triangulated twice.
    """
    if pieces is None:
        pieces = silicon_pieces(structure)
    data = prepare_layout(layout, cache=cache)
    data.rejected = visible_rejected(structure)
    data.pieces = pieces
    data.groups = Groups.of(structure)
    lookup = device_group_lookup(structure, data.groups)
    data.beam_groups = [data.groups.of_net(beam.net) for beam in structure.beams]
    data.rejected_groups = [lookup(candidate.polygon) for candidate in data.rejected]
    by_body: dict[int, list[Piece]] = {}
    for piece in pieces:
        by_body.setdefault(piece.body, []).append(piece)
    if holes:
        data.holes = MultiPolygon(holes)
        data.register(data.holes)
    counts = {body.index: 0 for body in structure.bodies}
    for beam in structure.beams:
        counts[beam.bodies[0]] += 1
        counts[beam.bodies[1]] += 1
    data.beam_counts = counts
    from connflex_app.analysis import free_bodies

    data.free_bodies = set(free_bodies(structure))

    geoms: list = []
    tags: list[tuple[str, int]] = []
    for beam in structure.beams:
        data.register(beam.polygon)
        geoms.append(beam.polygon)
        tags.append(("beam", beam.index))
    for i, candidate in enumerate(data.rejected):
        data.register(candidate.polygon)
        geoms.append(candidate.polygon)
        tags.append(("rejected", i))
    for body in structure.bodies:
        mine = by_body.get(body.index, [])
        for piece in mine:
            data.register(piece.geometry)
        for geometry in (body.box, body.device, body.stubs):
            data.register(geometry)
        data.outlines[body.index] = MultiPolygon([piece.geometry for piece in mine if piece.layer == "device"])
        for part in body.stubs.geoms:
            geoms.append(part)
            tags.append(("stub", body.index))
        for part in body.device.geoms:
            geoms.append(part)
            tags.append(("body", body.index))
        if advance is not None:
            advance()
    for via in structure.nets.vias:
        data.register(via.polygon)
        geoms.append(via.polygon)
        tags.append(("via", via.index))
    for conductor in structure.nets.conductors:
        data.register(conductor.polygon)
        if conductor.layer == "handle":
            geoms.append(conductor.polygon)
            tags.append(("handle", conductor.index))
    data.pick_tree = STRtree(geoms) if geoms else None
    data.pick_tags = tags
    data.cache = None  # drop the shared float64 triangulations; the float32 copies above are what the view needs

    # Picking by paint group: what conducts on top first, then device silicon, then the handle.
    layered: list[tuple[BaseGeometry, int]] = []
    layered += [(beam.polygon, group) for beam, group in zip(structure.beams, data.beam_groups)]
    layered += [(candidate.polygon, group) for candidate, group in zip(data.rejected, data.rejected_groups)]
    layered += [(piece.geometry, piece.group) for piece in pieces if piece.layer == "device"]
    layered += [(piece.geometry, piece.group) for piece in pieces if piece.layer == "handle"]
    layered = [(geometry, group) for geometry, group in layered if group >= 0]
    data.group_tree = STRtree([geometry for geometry, _group in layered]) if layered else None
    data.group_tags = [group for _geometry, group in layered]
    for geometry, group in layered:
        data.group_geometry.setdefault(group, []).append(geometry)

    centers = {body.index: (float(body.inertial.center[0]), float(body.inertial.center[1])) for body in structure.bodies}
    beams = [(beam.bodies[0], beam.bodies[1], beam.start[:2], beam.end[:2], beam.index) for beam in structure.beams]
    conductors = {}
    for conductor in structure.nets.conductors:
        point = conductor.polygon.representative_point()
        conductors[conductor.index] = (point.x, point.y, conductor.layer, conductor.net)
    links = []
    for via in structure.nets.vias:
        centroid = via.polygon.centroid
        for a in via.device:
            for b in via.handle:
                links.append((a, b, (centroid.x, centroid.y)))
    data.graph = {"centers": centers, "beams": beams, "conductors": conductors, "links": links}
    return data
