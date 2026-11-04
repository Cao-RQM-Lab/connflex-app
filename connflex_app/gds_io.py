"""GDS file inspection: top cells and the layers each one uses."""

from __future__ import annotations

from dataclasses import dataclass, field

import shapely

from connflex_core import Layout, load_gds

from connflex_app.settings import Settings

ROLES = ("device", "handle", "release", "vias")


@dataclass(frozen=True)
class LayerStats:
    layer: int
    datatype: int
    count: int
    area: float  # in file units squared

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.datatype)

    @property
    def label(self) -> str:
        return f"{self.layer}/{self.datatype}"


@dataclass
class GdsInfo:
    path: str
    unit: float
    top_cells: list[str]
    stats: dict[str, list[LayerStats]] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    def _cell(self, cell: str | None) -> str:
        if cell is None:
            return self.top_cells[0] if self.top_cells else ""
        return cell

    def layers(self, cell: str | None) -> list[LayerStats]:
        return self.stats.get(self._cell(cell), [])

    def bounds_m(self, cell: str | None) -> tuple[float, float, float, float] | None:
        """Bounding box of every polygon in the cell, in metres."""
        box = self.bounds.get(self._cell(cell))
        if box is None:
            return None
        return tuple(value * self.unit for value in box)  # type: ignore[return-value]


def inspect_gds(path: str) -> GdsInfo:
    """Read the file once and summarize the polygons of each top-level cell by layer."""
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("reading GDS files needs gdstk: pip install 'connflex-core[gds]'") from exc
    library = gdstk.read_gds(str(path))
    tops = [cell for cell in library.top_level() if isinstance(cell, gdstk.Cell)]
    if not tops:
        raise ValueError(f"{path} has no top-level cell")
    info = GdsInfo(path=str(path), unit=library.unit, top_cells=[cell.name for cell in tops])
    for cell in tops:
        counts: dict[tuple[int, int], int] = {}
        areas: dict[tuple[int, int], float] = {}
        for polygon in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
            key = (polygon.layer, polygon.datatype)
            counts[key] = counts.get(key, 0) + 1
            areas[key] = areas.get(key, 0.0) + abs(polygon.area())
        info.stats[cell.name] = sorted(
            (LayerStats(key[0], key[1], counts[key], areas[key]) for key in counts),
            key=lambda stats: stats.key,
        )
        box = cell.bounding_box()
        if box is not None:
            (xmin, ymin), (xmax, ymax) = box
            info.bounds[cell.name] = (float(xmin), float(ymin), float(xmax), float(ymax))
    return info


def load_layout(
    path: str, settings: Settings, bounds: tuple[float, float, float, float] | None = None
) -> Layout:
    """Read the layout for ``settings`` and invert the roles it asks for.

    An inverted role becomes the complement of its drawn polygons inside ``bounds`` (the
    cell's bounding box in metres), or inside the extent of all loaded roles when no bounds
    are given. That is how layers drawn as etch openings become material.
    """
    layout = load_gds(path, layers=settings.layer_map(), cell=settings.cell)
    inverted = settings.inverted_roles()
    if not inverted:
        return layout
    geometries = {role: getattr(layout, role) for role in ROLES if getattr(layout, role) is not None}
    if bounds is None:
        bounds = tuple(shapely.total_bounds(list(geometries.values())))  # type: ignore[assignment]
    frame = shapely.box(*bounds)
    for role in inverted & set(geometries):
        complement = frame.difference(geometries[role])
        if complement.is_empty:
            raise ValueError(f"inverting the {role} layer leaves nothing inside the cell bounds")
        geometries[role] = complement
    return Layout.from_shapely(unit=1.0, **geometries)


def guess_roles(stats: list[LayerStats]) -> dict[str, tuple[int, int] | None]:
    """Guess which layer plays each role: the README convention first, then by area."""
    keys = {s.key for s in stats}
    guess: dict[str, tuple[int, int] | None] = {"device": None, "handle": None, "release": None, "vias": None}
    if (1, 0) in keys and (2, 0) in keys:
        guess["device"], guess["handle"] = (1, 0), (2, 0)
    else:
        by_area = sorted(stats, key=lambda s: s.area, reverse=True)
        if len(by_area) >= 2:
            guess["handle"], guess["device"] = by_area[0].key, by_area[1].key
        elif by_area:
            guess["device"] = guess["handle"] = by_area[0].key
    if (3, 0) in keys and (3, 0) not in (guess["device"], guess["handle"]):
        guess["release"] = (3, 0)
    if (4, 0) in keys and (4, 0) not in (guess["device"], guess["handle"]):
        guess["vias"] = (4, 0)
    return guess
