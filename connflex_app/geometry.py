"""Render geometry: perforated plates, extruded meshes, and displacement fields.

Scene coordinates are micrometres so that single-precision GPU buffers keep their accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from connflex_core import Layout, Stack, Structure
from connflex_core.model import RejectedCandidate

SCENE_UNIT = 1e6
KIND_DEVICE, KIND_BOX, KIND_HANDLE, KIND_BEAM, KIND_REJECTED = range(5)
KIND_NAMES = ("device", "box", "handle", "beam", "rejected")

DOF_SHORT = {"ux": "X", "uy": "Y", "uz": "Z", "tx": "θx", "ty": "θy", "tz": "θz"}
DOF_LONG = {
    "ux": "translation along X",
    "uy": "translation along Y",
    "uz": "translation along Z (out of plane)",
    "tx": "rotation about X",
    "ty": "rotation about Y",
    "tz": "rotation about Z (in-plane twist)",
}
DOF_ORDER = ("ux", "uy", "uz", "tx", "ty", "tz")


def as_multipolygon(geometry: BaseGeometry | NDArray) -> MultiPolygon:
    parts = [p for p in shapely.get_parts(geometry) if isinstance(p, Polygon) and not p.is_empty]
    return MultiPolygon(parts)


def release_holes(layout: Layout, max_area: float) -> list[Polygon]:
    """Holes of the raw device layer that the extraction treats as release holes."""
    holes: list[Polygon] = []
    for polygon in layout.device.geoms:
        for ring in polygon.interiors:
            hole = Polygon(ring)
            if hole.area <= max_area:
                holes.append(hole)
    return holes


def closed_plates(structure: Structure) -> dict[int, BaseGeometry]:
    """Solid device plate of each body: its device opening plus stubs.

    connflex-core fills every interior ring up to ``release_hole_max_area`` and heals hairline
    gaps before it extracts bodies, so these shapes already have the release holes closed.
    """
    return {body.index: shapely.union_all([body.device, body.stubs]) for body in structure.bodies}


def visible_rejected(structure: Structure) -> list[RejectedCandidate]:
    """Rejected candidates that are not stubs (stubs already belong to a body)."""
    return [candidate for candidate in structure.rejected if len(candidate.bodies) != 1]


TriangleCache = dict[int, tuple[BaseGeometry, NDArray[np.float64]]]

def triangulate(
    geometry: BaseGeometry, cache: TriangleCache | None = None, key: BaseGeometry | None = None
) -> NDArray[np.float64]:
    """Triangles covering ``geometry`` as an array of shape (T, 3, 2).

    With a ``cache``, results are stored under the identity of ``key`` (default: the geometry
    itself) and the keyed object is kept alive so its identity cannot be recycled.
    """
    owner = geometry if key is None else key
    ident = id(owner)
    if cache is not None and ident in cache:
        return cache[ident][1]
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(as_multipolygon(geometry)))
    if triangles.size == 0:
        result = np.zeros((0, 3, 2))
    else:
        coords, index = shapely.get_coordinates(triangles, return_index=True)
        counts = np.bincount(index, minlength=triangles.size)
        if np.all(counts == 4):
            result = coords.reshape(-1, 4, 2)[:, :3, :]
        else:
            out = []
            for triangle in triangles:
                ring = np.asarray(triangle.exterior.coords)[:3, :2]
                if len(ring) == 3:
                    out.append(ring)
            result = np.array(out).reshape(-1, 3, 2)
    if cache is not None:
        cache[ident] = (owner, result)
    return result


def _unique_points(coords: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Unique (x, y) rows and the inverse index, through a complex view for speed."""
    view = np.ascontiguousarray(coords, dtype=np.float64).view(np.complex128).ravel()
    unique, inverse = np.unique(view, return_inverse=True)
    return np.column_stack([unique.real, unique.imag]), inverse.ravel()


def _wall_quads(geometry: MultiPolygon) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Edges (p0, p1) of every ring with the outward in-plane normal of each."""
    parts = shapely.get_parts(geometry)
    rings, owner = shapely.get_rings(parts, return_index=True)
    first = np.ones(rings.size, dtype=bool)
    first[1:] = owner[1:] != owner[:-1]
    ccw = shapely.is_ccw(rings)
    sign = np.where(ccw == first, 1.0, -1.0)
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    edge = ring_of[:-1] == ring_of[1:]
    p0, p1 = coords[:-1][edge], coords[1:][edge]
    s = sign[ring_of[:-1][edge]]
    d = p1 - p0
    length = np.hypot(d[:, 0], d[:, 1])
    keep = length > 0
    p0, p1, d, length, s = p0[keep], p1[keep], d[keep], length[keep], s[keep]
    normal = np.stack([d[:, 1], -d[:, 0]], axis=1) / length[:, None] * s[:, None]
    return p0, p1, normal


def extrude(
    geometry: BaseGeometry,
    z_top: float,
    z_bottom: float,
    cache: TriangleCache | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Indexed solid between two heights: vertices (V, 6) of position and normal, and triangle indices."""
    source = geometry
    geometry = as_multipolygon(geometry)
    if geometry.is_empty:
        return np.zeros((0, 6)), np.zeros(0, dtype=np.int64)
    vertices: list[NDArray[np.float64]] = []
    indices: list[NDArray[np.int64]] = []
    offset = 0

    def cap(shape: BaseGeometry, key: BaseGeometry, z: float, up: bool) -> None:
        nonlocal offset
        tri = triangulate(shape, cache, key=key)
        if tri.shape[0] == 0:
            return
        points, inverse = _unique_points(tri.reshape(-1, 2))
        faces = inverse.reshape(-1, 3)
        count = len(points)
        normal = np.tile([0.0, 0.0, 1.0 if up else -1.0], (count, 1))
        vertices.append(np.column_stack([points, np.full(count, z), normal]))
        indices.append((faces if up else faces[:, ::-1]).ravel() + offset)
        offset += count

    cap(geometry, source, z_top, True)
    cap(geometry, source, z_bottom, False)
    p0, p1, normal = _wall_quads(geometry)
    e = len(p0)
    if e:
        n3 = np.column_stack([normal, np.zeros(e)])
        a = np.column_stack([p0, np.full(e, z_top), n3])
        b = np.column_stack([p1, np.full(e, z_top), n3])
        c = np.column_stack([p1, np.full(e, z_bottom), n3])
        d = np.column_stack([p0, np.full(e, z_bottom), n3])
        vertices.append(np.stack([a, b, c, d], axis=1).reshape(-1, 6))
        base = np.arange(e, dtype=np.int64)[:, None] * 4 + offset
        indices.append((base + np.array([0, 1, 2, 0, 2, 3])[None, :]).ravel())
        offset += 4 * e
    if not vertices:
        return np.zeros((0, 6)), np.zeros(0, dtype=np.int64)
    return np.concatenate(vertices), np.concatenate(indices)


COL_A, COL_B, COL_S, COL_KIND = 6, 7, 8, 9


@dataclass
class Mesh:
    """Indexed triangle mesh of the whole structure in scene units, grouped by kind.

    ``vertices`` holds, per vertex: position (3), normal (3), the two bodies whose rigid
    motion the vertex follows and the weight of the second one (3), and the kind (1).
    Rigid vertices have ``a == b``, beam vertices blend from one end body to the other, and
    static vertices have ``a == b == -1``. ``ranges`` gives the index range of each kind.
    """

    vertices: NDArray[np.float32]
    indices: NDArray[np.uint32]
    ranges: dict[int, tuple[int, int]]
    body_radius: NDArray[np.float64]

    @property
    def count(self) -> int:
        return len(self.vertices)

    @property
    def positions(self) -> NDArray[np.float32]:
        return self.vertices[:, 0:3]

    @property
    def normals(self) -> NDArray[np.float32]:
        return self.vertices[:, 3:6]

    @property
    def blend_a(self) -> NDArray[np.int32]:
        return self.vertices[:, COL_A].astype(np.int32)

    @property
    def blend_b(self) -> NDArray[np.int32]:
        return self.vertices[:, COL_B].astype(np.int32)

    @property
    def blend_s(self) -> NDArray[np.float32]:
        return self.vertices[:, COL_S]

    @property
    def kind(self) -> NDArray[np.int8]:
        return self.vertices[:, COL_KIND].astype(np.int8)

    @property
    def body(self) -> NDArray[np.int32]:
        return np.where(self.kind == KIND_BEAM, -1, self.blend_a)

    @property
    def triangles(self) -> int:
        return len(self.indices) // 3

    @property
    def bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.count == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.positions.min(axis=0), self.positions.max(axis=0)

    @property
    def center(self) -> NDArray[np.float32]:
        low, high = self.bounds
        return (low + high) / 2

    @property
    def extent(self) -> float:
        low, high = self.bounds
        span = high - low
        return float(max(span[0], span[1], 1e-6))


class _MeshBuilder:
    def __init__(self) -> None:
        self.parts: dict[int, list[tuple]] = {kind: [] for kind in range(5)}

    def add(self, kind, vertices, indices, blend_a, blend_b, blend_s) -> None:
        if len(vertices) == 0:
            return
        s = np.full(len(vertices), blend_s, dtype=np.float64) if np.isscalar(blend_s) else np.asarray(blend_s, np.float64)
        self.parts[kind].append((vertices, indices, blend_a, blend_b, s))

    def build(self, centers: NDArray[np.float64]) -> Mesh:
        chunks: list[NDArray[np.float64]] = []
        index_chunks: list[NDArray[np.int64]] = []
        ranges: dict[int, tuple[int, int]] = {}
        vertex_offset = 0
        index_offset = 0
        for kind in range(5):
            kind_indices = 0
            for vertices, indices, a, b, s in self.parts[kind]:
                n = len(vertices)
                extra = np.column_stack([np.full(n, a, dtype=np.float64), np.full(n, b, dtype=np.float64), s, np.full(n, kind, dtype=np.float64)])
                chunks.append(np.column_stack([vertices[:, :3] * SCENE_UNIT, vertices[:, 3:6], extra]))
                index_chunks.append(indices + vertex_offset)
                vertex_offset += n
                kind_indices += len(indices)
            ranges[kind] = (index_offset, kind_indices)
            index_offset += kind_indices
        if vertex_offset == 0:
            return Mesh(np.zeros((0, 10), np.float32), np.zeros(0, np.uint32), ranges, np.zeros(len(centers)))
        data = np.concatenate(chunks).astype(np.float32)
        indices = np.concatenate(index_chunks).astype(np.uint32)
        radius = np.zeros(len(centers))
        for column in (COL_A, COL_B):
            ids = data[:, column].astype(np.int64)
            valid = ids >= 0
            if valid.any():
                distance = np.linalg.norm(data[valid, :3].astype(np.float64) - centers[ids[valid]], axis=1)
                np.maximum.at(radius, ids[valid], distance)
        return Mesh(data, indices, ranges, radius)


def build_mesh(
    structure: Structure,
    stack: Stack,
    plates: dict[int, BaseGeometry],
    advance: Callable[[], None] | None = None,
    *,
    cache: TriangleCache | None = None,
) -> Mesh:
    """Extrude every body layer, beam, and rejected candidate through the stack.

    ``plates`` are the closed device plates from :func:`closed_plates`; release holes are not
    modelled, which keeps the mesh small whatever the layout.
    """
    builder = _MeshBuilder()
    device_z, box_z, handle_z = stack.z_extents
    for body in structure.bodies:
        i = body.index
        builder.add(KIND_DEVICE, *extrude(plates[i], *device_z, cache), i, i, 0.0)
        builder.add(KIND_BOX, *extrude(body.box, *box_z, cache), i, i, 0.0)
        builder.add(KIND_HANDLE, *extrude(body.handle, *handle_z, cache), i, i, 0.0)
        if advance is not None:
            advance()
    for beam in structure.beams:
        if advance is not None:
            advance()
        vertices, indices = extrude(beam.polygon, *device_z, cache)
        if len(vertices) == 0:
            continue
        start = np.asarray(beam.start[:2])
        axis = np.asarray(beam.end[:2]) - start
        t = (vertices[:, :2] - start) @ axis / max(float(axis @ axis), 1e-30)
        t = np.clip(t, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        a, b = beam.bodies
        builder.add(KIND_BEAM, vertices, indices, a, b, smooth)
    for candidate in visible_rejected(structure):
        builder.add(KIND_REJECTED, *extrude(candidate.polygon, *device_z, cache), -1, -1, 0.0)
    return builder.build(body_centers(structure))


def body_centers(structure: Structure) -> NDArray[np.float64]:
    """Centres of mass of all bodies in scene units, shape (n_bodies, 3)."""
    return np.array([body.inertial.center for body in structure.bodies]) * SCENE_UNIT


def motion_from_vector(
    structure: Structure, dofs: tuple[tuple[int, str], ...], vector: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a system vector into per-body translations and rotations, anchor included as zero."""
    n = len(structure.bodies)
    u = np.zeros((n, 3))
    theta = np.zeros((n, 3))
    for row, (body, name) in enumerate(dofs):
        if name == "ux":
            u[body] = vector[row : row + 3]
            theta[body] = vector[row + 3 : row + 6]
    return u, theta


def displacement_field(
    mesh: Mesh, structure: Structure, u: NDArray, theta: NDArray
) -> NDArray[np.float32]:
    """Per-vertex displacement in scene units for rigid body motions ``u`` (m) and ``theta`` (rad)."""
    centers = body_centers(structure)
    positions = mesh.positions.astype(np.float64)
    u_scene = np.asarray(u, dtype=np.float64) * SCENE_UNIT
    theta = np.asarray(theta, dtype=np.float64)

    def rigid(index: NDArray[np.int32]) -> NDArray[np.float64]:
        valid = index >= 0
        safe = np.where(valid, index, 0)
        d = u_scene[safe] + np.cross(theta[safe], positions - centers[safe])
        d[~valid] = 0.0
        return d

    da = rigid(mesh.blend_a)
    db = rigid(mesh.blend_b)
    s = mesh.blend_s[:, None].astype(np.float64)
    return ((1.0 - s) * da + s * db).astype(np.float32)


def motion_peak(mesh: Mesh, u: NDArray, theta: NDArray) -> float:
    """Upper bound of the largest vertex displacement (scene units) for rigid motions ``u``, ``theta``."""
    translation = np.linalg.norm(np.asarray(u, dtype=np.float64), axis=1) * SCENE_UNIT
    rotation = np.linalg.norm(np.asarray(theta, dtype=np.float64), axis=1) * mesh.body_radius[: len(translation)]
    peak = float(np.max(translation + rotation)) if len(translation) else 0.0
    return peak if np.isfinite(peak) and peak > 0 else 1.0


def normalized(field: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale a displacement field so its largest vertex displacement is one scene unit."""
    if len(field) == 0:
        return field
    peak = float(np.max(np.linalg.norm(field, axis=1)))
    if peak <= 0 or not np.isfinite(peak):
        return field
    return (field / peak).astype(np.float32)


def gyration_radii(structure: Structure) -> dict[int, float]:
    """Largest radius of gyration of each movable body, to weigh rotations against translations."""
    radii = {}
    for body in structure.bodies:
        if not body.anchor:
            largest = float(np.max(np.linalg.eigvalsh(body.inertial.inertia)))
            radii[body.index] = float(np.sqrt(max(largest / body.inertial.mass, 0.0)))
    return radii


def describe_modes(structure: Structure, dofs, modes: NDArray, radii: dict[int, float] | None = None) -> list[str]:
    """Dominant body and degree of freedom of every mode column, vectorised over all modes."""
    if radii is None:
        radii = gyration_radii(structure)
    movable = [body for row, (body, name) in enumerate(dofs) if name == "ux"]
    count = modes.shape[1]
    if not movable or count == 0:
        return [""] * count
    blocks = np.abs(modes.reshape(len(movable), 6, count))
    weights = np.ones((len(movable), 6, 1))
    weights[:, 3:, 0] = np.array([radii[body] for body in movable])[:, None]
    scores = blocks * weights
    per_body = scores.max(axis=1)
    best_body = per_body.argmax(axis=0)
    best_dof = scores[best_body, :, np.arange(count)].argmax(axis=1)
    return [f"Body {movable[b]} · {DOF_SHORT[DOF_ORDER[d]]}" for b, d in zip(best_body, best_dof)]


def describe_motion(
    structure: Structure, u: NDArray, theta: NDArray, radii: dict[int, float] | None = None
) -> tuple[str, int, str]:
    """Dominant body and degree of freedom of a motion, as (label, body, dof name)."""
    if radii is None:
        radii = gyration_radii(structure)
    best: tuple[float, int, int] | None = None
    for body in structure.bodies:
        if body.anchor:
            continue
        components = np.concatenate([np.abs(u[body.index]), np.abs(theta[body.index]) * radii[body.index]])
        score = float(components.max())
        if best is None or score > best[0]:
            best = (score, body.index, int(np.argmax(components)))
    if best is None:
        return ("", -1, "")
    dof = DOF_ORDER[best[2]]
    return (f"Body {best[1]} · {DOF_SHORT[dof]}", best[1], dof)
