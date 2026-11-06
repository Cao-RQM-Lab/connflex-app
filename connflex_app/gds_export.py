"""Export the extracted, released structure as a GDS file.

Layers (fixed):

- 1/0 device: every device region connflex-core kept (bodies, stubs, beams, rejected candidates).
- 2/0 release holes: the release openings connflex-core removed from body mass.
- 3/0 box: the buried oxide that survives the release etch.
- 4/0 handle: the handle regions connflex-core kept.

Release openings are computed with connflex-core's own heal and release helpers, so they are
exactly what the analysis used: holes in the device layer up to ``release_hole_max_area``
(including ones that healing closes) plus the release layer, clipped to each body the way
connflex-core clips them before subtracting their mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import MultiPolygon, Polygon, STRtree
from shapely.geometry.base import BaseGeometry

from connflex_core import ExtractionParams, Layout, Structure
from connflex_core.geometry.heal import as_multipolygon, heal, snap
from connflex_core.geometry.release import fill_release_holes, openings_within

from connflex_app.geometry import visible_rejected

LAYERS: dict[str, tuple[int, int]] = {"device": (1, 0), "release": (2, 0), "box": (3, 0), "handle": (4, 0)}
LAYER_TITLES = {"device": "device", "release": "release holes", "box": "buried oxide (released)", "handle": "handle"}
GDS_UNIT = 1e-6
GDS_PRECISION = 1e-9
MAX_POINTS = 8190  # GDSII limit; larger polygons are fractured on write


@dataclass
class ExportedLayers:
    """Geometry of every exported layer, in metres."""

    device: MultiPolygon
    release: MultiPolygon
    box: MultiPolygon
    handle: MultiPolygon

    def items(self):
        return ((name, getattr(self, name)) for name in LAYERS)


@dataclass
class ExportResult:
    path: str
    separate_release: bool
    polygons: dict[str, int] = field(default_factory=dict)
    areas: dict[str, float] = field(default_factory=dict)


def release_openings(layout: Layout, params: ExtractionParams) -> list[Polygon]:
    """Disjoint release openings, built the way ``connflex_core.extract`` builds them."""
    tolerance = params.simplify_tolerance
    device, holes = fill_release_holes(layout.device, params.release_hole_max_area)
    device = heal(device, params.heal_distance, tolerance)
    _, closed = fill_release_holes(device, params.release_hole_max_area)
    holes = holes + closed
    release = None
    if layout.release is not None:
        release = as_multipolygon(layout.release.simplify(tolerance, preserve_topology=True))
    if params.precision_grid is not None:
        release = None if release is None else snap(release, params.precision_grid)
        if holes:
            holes = list(snap(MultiPolygon(holes), params.precision_grid).geoms)
    if release is None:
        return holes
    if not holes:
        return list(release.geoms)
    return list(as_multipolygon(shapely.union_all([*holes, *release.geoms])).geoms)


def body_openings(structure: Structure, openings: list[Polygon]) -> MultiPolygon:
    """Openings inside the bodies, clipped to them, as connflex-core subtracts them from body mass."""
    if not openings:
        return MultiPolygon()
    tree = STRtree(openings)
    inside: list[Polygon] = []
    for body in structure.bodies:
        massive = MultiPolygon([*body.device.geoms, *body.stubs.geoms])
        if not massive.is_empty:
            inside.extend(openings_within(tree, openings, massive))
    return MultiPolygon([part for part in inside if isinstance(part, Polygon) and not part.is_empty])


def extracted_device(structure: Structure, params: ExtractionParams) -> MultiPolygon:
    """Every device region kept by the extraction, merged, with release holes closed."""
    parts: list[BaseGeometry] = []
    for body in structure.bodies:
        parts.extend([body.device, body.stubs])
    parts.extend(beam.polygon for beam in structure.beams)
    parts.extend(candidate.polygon for candidate in visible_rejected(structure))
    merged = shapely.union_all(parts)
    # The body opening and the beam candidates do not share edges bit for bit; close the hairlines.
    distance = params.heal_distance
    merged = merged.buffer(distance, join_style="mitre").buffer(-distance, join_style="mitre")
    return as_multipolygon(merged)


def _union(geometries) -> MultiPolygon:
    geometries = [g for g in geometries if g is not None and not g.is_empty]
    return as_multipolygon(shapely.union_all(geometries)) if geometries else MultiPolygon()


def build_layers(
    layout: Layout,
    structure: Structure,
    params: ExtractionParams,
    separate_release: bool,
    progress=None,
) -> ExportedLayers:
    """Geometry of each exported layer.

    With ``separate_release`` the device is drawn with its release holes filled and the holes go
    on their own layer; without it the holes are cut into the device and the release layer is empty.
    """
    from connflex_core.progress import stage

    with stage("openings", progress) as current:
        openings = release_openings(layout, params)
        inside = body_openings(structure, openings)
        current.count(openings=len(inside.geoms))
    with stage("device", progress) as current:
        device = extracted_device(structure, params)
        if not separate_release and not inside.is_empty:
            device = _subtract(device, inside)
        current.count(polygons=len(device.geoms))
    with stage("layers", progress):
        box = _union(body.box for body in structure.bodies)
        handle = _union(body.handle for body in structure.bodies)
    release = inside if separate_release else MultiPolygon()
    return ExportedLayers(device=device, release=release, box=box, handle=handle)


def _subtract(device: MultiPolygon, holes: MultiPolygon) -> MultiPolygon:
    """``device`` minus ``holes``, cutting each device polygon only by the holes that touch it."""
    hole_parts = np.array(list(holes.geoms), dtype=object)
    tree = STRtree(hole_parts)
    result: list[Polygon] = []
    for polygon in device.geoms:
        hits = tree.query(polygon, predicate="intersects")
        if hits.size == 0:
            result.append(polygon)
            continue
        candidates = hole_parts[hits]
        strict = shapely.contains_properly(polygon, candidates)
        if strict.all():
            # Holes strictly inside: add them as interior rings without an overlay.
            rings = [ring.coords for ring in polygon.interiors] + [h.exterior.coords for h in candidates]
            result.append(Polygon(polygon.exterior.coords, rings))
            continue
        result.extend(as_multipolygon(polygon.difference(shapely.union_all(candidates))).geoms)
    return MultiPolygon(result)


GRID = GDS_PRECISION / GDS_UNIT  # 1 nm in micrometres
_TOL = GRID * 1e-3


def _cut_heights(hole: np.ndarray):
    """Candidate starts on ``hole``'s left side: (edge index, point), best first, y on the 1 nm grid."""
    n = len(hole)
    minx = float(hole[:, 0].min())
    candidates = []
    for i in range(n):
        p, q = hole[i], hole[(i + 1) % n]
        if abs(p[1] - q[1]) <= _TOL:
            continue  # horizontal edge
        at_p, at_q = abs(p[0] - minx) <= _TOL, abs(q[0] - minx) <= _TOL
        if at_p and at_q:
            candidates.append((0, i, (0.5, 0.37, 0.63, 0.23, 0.77, 0.11, 0.89)))  # vertical edge on the far left
        elif at_p:
            candidates.append((1, i, (0.02, 0.05, 0.1, 0.2)))
        elif at_q:
            candidates.append((1, i, (0.98, 0.95, 0.9, 0.8)))
    for _, i, fractions in sorted(candidates):
        p, q = hole[i], hole[(i + 1) % n]
        for t in fractions:
            y = round((p[1] + t * (q[1] - p[1])) / GRID) * GRID
            if min(p[1], q[1]) + _TOL < y < max(p[1], q[1]) - _TOL:
                yield i, np.array([p[0] + (y - p[1]) * (q[0] - p[0]) / (q[1] - p[1]), y])


def _keyhole_ring(polygon: Polygon) -> np.ndarray | None:
    """One ring that joins every hole to the outline with zero-width cut lines, as GDSII stores holes.

    Holes are joined left to right, each by a horizontal cut from its left side to the nearest
    edge of the ring built so far (the outline and the holes already joined). A cut height is
    rejected only if a ring vertex lies on the cut itself, so the cut crosses one edge cleanly.
    Returns None when no clean cut is found; the caller then splits the polygon.
    """
    polygon = shapely.geometry.polygon.orient(polygon, 1.0)  # outline counter-clockwise, holes clockwise
    ring = np.asarray(polygon.exterior.coords)[:-1]
    holes = sorted((np.asarray(r.coords)[:-1] for r in polygon.interiors), key=lambda h: float(h[:, 0].min()))
    for hole in holes:
        nxt = np.concatenate([ring[1:], ring[:1]])
        ax, ay, bx, by = ring[:, 0], ring[:, 1], nxt[:, 0], nxt[:, 1]
        joined = None
        for k, m in _cut_heights(hole):
            da, db = ay - m[1], by - m[1]
            crossing = da * db < 0
            if not crossing.any():
                continue
            idx = np.flatnonzero(crossing)
            xs = ax[idx] + (m[1] - ay[idx]) * (bx[idx] - ax[idx]) / (by[idx] - ay[idx])
            left = xs < m[0] - _TOL
            if not left.any():
                continue
            best = int(np.argmax(np.where(left, xs, -np.inf)))
            x_hit, j = float(xs[best]), int(idx[best])
            on_cut = (np.abs(da) <= _TOL) & (ax > x_hit - _TOL) & (ax < m[0] + _TOL)
            if on_cut.any():
                continue  # a vertex sits on the cut; try another height
            joined = (j, np.array([x_hit, m[1]]), k, m)
            break
        if joined is None:
            return None
        j, hit, k, m = joined
        n = len(hole)
        loop = hole[np.arange(k + 1, k + 1 + n) % n]
        ring = np.concatenate([ring[: j + 1], hit[None], m[None], loop, m[None], hit[None], ring[j + 1 :]])
    return ring


def _repairs_to(ring: np.ndarray, polygon: Polygon) -> bool:
    """Whether a reader that repairs cut lines (as connflex-core and KLayout do) gets ``polygon`` back."""
    repaired = shapely.make_valid(Polygon(ring), method="structure", keep_collapsed=False)
    if not isinstance(repaired, Polygon) or len(repaired.interiors) != len(polygon.interiors):
        return False
    return abs(repaired.area - polygon.area) <= max(1e-6 * polygon.area, GRID * GRID)


def _split(polygon: Polygon) -> list[Polygon]:
    """Halve ``polygon`` across its longer side along a 1 nm grid line."""
    minx, miny, maxx, maxy = polygon.bounds
    if maxx - minx >= maxy - miny:
        cut = round((minx + maxx) / 2 / GRID) * GRID
        boxes = ((minx - 1, miny - 1, cut, maxy + 1), (cut, miny - 1, maxx + 1, maxy + 1))
    else:
        cut = round((miny + maxy) / 2 / GRID) * GRID
        boxes = ((minx - 1, miny - 1, maxx + 1, cut), (minx - 1, cut, maxx + 1, maxy + 1))
    return [part for box in boxes for part in as_multipolygon(shapely.clip_by_rect(polygon, *box)).geoms]


def _gds_rings(polygon: Polygon, depth: int = 0) -> list[np.ndarray]:
    """Hole-free rings of at most ``MAX_POINTS`` vertices covering ``polygon`` (micrometres, on the 1 nm grid)."""
    polygon = shapely.set_precision(polygon, GRID)
    if polygon.is_empty:
        return []
    if not isinstance(polygon, Polygon):
        return [ring for part in as_multipolygon(polygon).geoms for ring in _gds_rings(part, depth)]
    size = len(polygon.exterior.coords) + sum(len(r.coords) + 4 for r in polygon.interiors)
    if size <= MAX_POINTS:
        if not polygon.interiors:
            return [np.asarray(polygon.exterior.coords)[:-1]]
        ring = _keyhole_ring(polygon)
        if ring is not None and len(ring) <= MAX_POINTS:
            ring = np.round(ring / GRID) * GRID
            if _repairs_to(ring, polygon):
                return [ring]
    if depth > 48:
        raise ValueError(f"could not write a polygon of area {polygon.area:.6g} µm² as GDSII")
    return [ring for part in _split(polygon) for ring in _gds_rings(part, depth + 1)]


def _gds_polygons(geometry: MultiPolygon, layer: tuple[int, int]) -> list:
    """gdstk polygons for ``geometry`` (metres).

    Only gdstk's plain polygon constructor is used: its boolean and fracture operations call a
    Clipper library that, inside a Qt application on macOS, binds to an incompatible copy
    exported by a system framework and crashes. Holes are joined to the outline here instead.
    """
    import gdstk

    scaled = shapely.transform(geometry, lambda coords: coords / GDS_UNIT)
    return [
        gdstk.Polygon(ring, layer=layer[0], datatype=layer[1])
        for polygon in as_multipolygon(scaled).geoms
        for ring in _gds_rings(polygon)
    ]


def write_gds(layers: ExportedLayers, path: str, cell: str = "CONNFLEX_EXTRACTED", progress=None) -> ExportResult:
    """Write ``layers`` to ``path`` in micrometre units with 1 nm precision."""
    import gdstk

    from connflex_core.progress import stage

    result = ExportResult(path=path, separate_release=not layers.release.is_empty)
    library = gdstk.Library(unit=GDS_UNIT, precision=GDS_PRECISION)
    top = library.new_cell(cell)
    with stage("write", progress, total=len(LAYERS)) as current:
        for name, geometry in layers.items():
            polygons = _gds_polygons(geometry, LAYERS[name])
            top.add(*polygons)
            result.polygons[name] = len(geometry.geoms)
            result.areas[name] = float(geometry.area)
            current.advance()
        library.write_gds(path, max_points=0)  # polygons are already within the GDSII limit; 0 skips gdstk's fracture
    return result


def export_gds(
    layout: Layout,
    structure: Structure,
    params: ExtractionParams,
    path: str,
    separate_release: bool,
    cell: str = "CONNFLEX_EXTRACTED",
    progress=None,
) -> ExportResult:
    """Build the exported layers and write them to ``path``."""
    layers = build_layers(layout, structure, params, separate_release, progress)
    result = write_gds(layers, path, cell, progress)
    result.separate_release = separate_release
    return result
