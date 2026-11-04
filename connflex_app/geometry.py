"""Render geometry: perforated plates, extruded meshes, and displacement fields.

Scene coordinates are micrometres so that single-precision GPU buffers keep their accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import MultiPolygon, Polygon, STRtree
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


def perforate(structure: Structure, holes: list[Polygon]) -> dict[int, MultiPolygon]:
    """Device plate of each body (its device opening plus stubs) with release holes cut back in."""
    tree = STRtree(holes) if holes else None
    hole_array = np.array(holes, dtype=object)
    plates: dict[int, MultiPolygon] = {}
    for body in structure.bodies:
        plate = as_multipolygon(shapely.union_all([body.device, body.stubs]))
        if tree is not None and not plate.is_empty:
            hits = tree.query(plate, predicate="intersects")
            if hits.size:
                plate = as_multipolygon(plate.difference(shapely.union_all(hole_array[hits])))
        plates[body.index] = plate
    return plates


def visible_rejected(structure: Structure) -> list[RejectedCandidate]:
    """Rejected candidates that are not stubs (stubs already belong to a body)."""
    return [candidate for candidate in structure.rejected if len(candidate.bodies) != 1]


def triangulate(geometry: MultiPolygon) -> NDArray[np.float64]:
    """Triangles covering ``geometry`` as an array of shape (T, 3, 2)."""
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(geometry))
    if triangles.size == 0:
        return np.zeros((0, 3, 2))
    coords, index = shapely.get_coordinates(triangles, return_index=True)
    counts = np.bincount(index, minlength=triangles.size)
    if np.all(counts == 4):
        return coords.reshape(-1, 4, 2)[:, :3, :]
    out = []
    for triangle in triangles:
        ring = np.asarray(triangle.exterior.coords)[:3, :2]
        if len(ring) == 3:
            out.append(ring)
    return np.array(out).reshape(-1, 3, 2)


def extrude(
    geometry: BaseGeometry, z_top: float, z_bottom: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solid between two heights: triangle soup positions and per-vertex outward normals."""
    geometry = as_multipolygon(geometry)
    if geometry.is_empty:
        return np.zeros((0, 3)), np.zeros((0, 3))
    tri = triangulate(geometry)
    count = tri.shape[0]
    top = np.dstack([tri, np.full((count, 3, 1), z_top)]).reshape(-1, 3)
    bottom = np.dstack([tri[:, ::-1, :], np.full((count, 3, 1), z_bottom)]).reshape(-1, 3)
    top_n = np.tile([0.0, 0.0, 1.0], (3 * count, 1))
    bottom_n = np.tile([0.0, 0.0, -1.0], (3 * count, 1))

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
    n2 = np.stack([d[:, 1], -d[:, 0]], axis=1) / length[:, None] * s[:, None]
    e = len(p0)
    a = np.column_stack([p0, np.full(e, z_top)])
    b = np.column_stack([p1, np.full(e, z_top)])
    c = np.column_stack([p1, np.full(e, z_bottom)])
    dd = np.column_stack([p0, np.full(e, z_bottom)])
    walls = np.stack([a, b, c, a, c, dd], axis=1).reshape(-1, 3)
    wall_n = np.repeat(np.column_stack([n2, np.zeros(e)]), 6, axis=0)
    positions = np.concatenate([top, bottom, walls])
    normals = np.concatenate([top_n, bottom_n, wall_n])
    return positions, normals


@dataclass
class Mesh:
    """Triangle soup of the whole structure in scene units, grouped by kind.

    ``blend_a``/``blend_b`` name the bodies whose rigid motion a vertex follows and ``blend_s``
    the weight of the second one: rigid vertices have ``a == b``, beam vertices blend from
    one end body to the other, and static vertices have ``a == b == -1``.
    """

    positions: NDArray[np.float32]
    normals: NDArray[np.float32]
    kind: NDArray[np.int8]
    body: NDArray[np.int32]
    blend_a: NDArray[np.int32]
    blend_b: NDArray[np.int32]
    blend_s: NDArray[np.float32]
    ranges: dict[int, tuple[int, int]]

    @property
    def count(self) -> int:
        return len(self.positions)

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

    def add(self, kind, positions, normals, body, blend_a, blend_b, blend_s) -> None:
        n = len(positions)
        if n == 0:
            return
        s = np.full(n, blend_s, dtype=np.float32) if np.isscalar(blend_s) else np.asarray(blend_s, np.float32)
        self.parts[kind].append((positions, normals, body, blend_a, blend_b, s))

    def build(self) -> Mesh:
        positions, normals, kinds, bodies, blend_a, blend_b, blend_s = [], [], [], [], [], [], []
        ranges: dict[int, tuple[int, int]] = {}
        start = 0
        for kind in range(5):
            count = 0
            for pos, nrm, body, a, b, s in self.parts[kind]:
                n = len(pos)
                positions.append(pos)
                normals.append(nrm)
                kinds.append(np.full(n, kind, dtype=np.int8))
                bodies.append(np.full(n, body, dtype=np.int32))
                blend_a.append(np.full(n, a, dtype=np.int32))
                blend_b.append(np.full(n, b, dtype=np.int32))
                blend_s.append(s)
                count += n
            ranges[kind] = (start, count)
            start += count
        if start == 0:
            empty3 = np.zeros((0, 3), dtype=np.float32)
            return Mesh(empty3, empty3, np.zeros(0, np.int8), np.zeros(0, np.int32),
                        np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), ranges)
        return Mesh(
            positions=(np.concatenate(positions) * SCENE_UNIT).astype(np.float32),
            normals=np.concatenate(normals).astype(np.float32),
            kind=np.concatenate(kinds),
            body=np.concatenate(bodies),
            blend_a=np.concatenate(blend_a),
            blend_b=np.concatenate(blend_b),
            blend_s=np.concatenate(blend_s),
            ranges=ranges,
        )


def build_mesh(structure: Structure, stack: Stack, plates: dict[int, MultiPolygon]) -> Mesh:
    """Extrude every body layer, beam, and rejected candidate through the stack."""
    builder = _MeshBuilder()
    device_z, box_z, handle_z = stack.z_extents
    for body in structure.bodies:
        i = body.index
        builder.add(KIND_DEVICE, *extrude(plates[i], *device_z), i, i, i, 0.0)
        builder.add(KIND_BOX, *extrude(body.box, *box_z), i, i, i, 0.0)
        builder.add(KIND_HANDLE, *extrude(body.handle, *handle_z), i, i, i, 0.0)
    for beam in structure.beams:
        positions, normals = extrude(beam.polygon, *device_z)
        if len(positions) == 0:
            continue
        start = np.asarray(beam.start[:2])
        axis = np.asarray(beam.end[:2]) - start
        t = (positions[:, :2] - start) @ axis / max(float(axis @ axis), 1e-30)
        t = np.clip(t, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        a, b = beam.bodies
        builder.add(KIND_BEAM, positions, normals, -1, a, b, smooth)
    for candidate in visible_rejected(structure):
        builder.add(KIND_REJECTED, *extrude(candidate.polygon, *device_z), -1, -1, -1, 0.0)
    return builder.build()


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
