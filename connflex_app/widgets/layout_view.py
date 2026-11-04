"""2D layout view: layers, bodies, beams, nets, and graphs with pan, zoom, and picking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import QMenu, QWidget
from shapely import MultiPolygon, Point, STRtree

from connflex_app import theme
from connflex_app.geometry import visible_rejected
from connflex_app.session import Project
from connflex_app.units import fmt_length, fmt_mass, fmt_point

RASTER_LIMIT = 30000.0  # device-space extent above which geometry is clipped before painting


@dataclass
class ViewOptions:
    device: bool = True
    handle: bool = True
    box: bool = True
    release: bool = True
    vias: bool = True
    beams: bool = True
    rejected: bool = True
    color_mode: str = "layer"  # layer | body | net
    mech_graph: bool = False
    elec_graph: bool = False


def path_of(geometry) -> QPainterPath:
    """QPainterPath of every polygon in ``geometry``, holes handled by the odd-even rule."""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    polygons = shapely.get_parts(geometry)
    polygons = polygons[shapely.get_type_id(polygons) == 3]
    if polygons.size == 0:
        return path
    rings = shapely.get_rings(polygons)
    coords, index = shapely.get_coordinates(rings, return_index=True)
    if coords.shape[0] == 0:
        return path
    starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    ends = np.r_[starts[1:], len(coords)]
    for start, end in zip(starts, ends):
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in coords[start:end]]))
        path.closeSubpath()
    return path


def _cosmetic(color: QColor, width: float = 1.0, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    pen.setCosmetic(True)
    return pen


class LayoutView(QWidget):
    """Interactive 2D view of the layout and the structure extracted from it."""

    hovered = Signal(object)
    bodyClicked = Signal(int)
    anchorPicked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 200)
        self.options = ViewOptions()
        self._layout = None
        self._project: Project | None = None
        self._scale = 1.0
        self._center = np.zeros(2)
        self._fitted = False
        self._bounds: tuple[float, float, float, float] | None = None
        self._fills: list[tuple[object, QColor, QColor | None]] = []
        self._paths: dict[int, QPainterPath] = {}
        self._pick_tree: STRtree | None = None
        self._pick_tags: list[tuple[str, int]] = []
        self._pick_geoms: list = []
        self._hover: tuple[str, int] | None = None
        self._selected = -1
        self._cache: dict | None = None
        self._version = 0
        self._drag_last: QPointF | None = None
        self._drag_moved = False
        self._graph: dict | None = None
        self._outlines: dict[int, object] = {}
        self._font = QFont()
        self._font.setPointSize(10)

    # ------------------------------------------------------------------ data
    def set_layout(self, layout) -> None:
        self._layout = layout
        self._project = None
        self._fitted = False
        self._bounds = self._layout_bounds()
        self._rebuild()
        self.fit()

    def set_project(self, project: Project | None) -> None:
        self._project = project
        if project is not None:
            self._layout = project.layout
            self._bounds = self._layout_bounds()
        self._rebuild()
        if not self._fitted:
            self.fit()

    def set_options(self, **changes) -> None:
        for key, value in changes.items():
            setattr(self.options, key, value)
        self._rebuild_fills()
        self._invalidate()
        self.update()

    def set_selected(self, body: int) -> None:
        self._selected = body
        self.update()

    def _layout_bounds(self):
        if self._layout is None:
            return None
        geoms = [self._layout.device, self._layout.handle]
        return tuple(shapely.total_bounds(np.array(geoms, dtype=object)))

    def _rebuild(self) -> None:
        self._paths.clear()
        self._hover = None
        self._outlines = {}
        if self._project is not None:
            for body in self._project.structure.bodies:
                self._outlines[body.index] = shapely.union_all([body.device, body.stubs])
        self._rebuild_fills()
        self._rebuild_pick()
        self._rebuild_graph()
        self._invalidate()
        self.update()

    def _rebuild_fills(self) -> None:
        items: list[tuple[object, QColor, QColor | None]] = []
        layout, project, o = self._layout, self._project, self.options
        self._fills = items
        if layout is None:
            return
        colored = project is not None and o.color_mode in ("body", "net")
        if o.handle:
            if colored:
                items.append((layout.handle, theme.qcolor(theme.LAYER["handle"], 0.55), None))
                if o.color_mode == "body":
                    for body in project.structure.bodies:
                        items.append((body.handle, theme.body_color(body.index, 0.35), None))
                else:
                    for conductor in project.structure.nets.conductors:
                        if conductor.layer == "handle":
                            items.append((conductor.polygon, theme.categorical(conductor.net, 0.35), None))
            else:
                items.append((layout.handle, theme.qcolor(theme.LAYER["handle"], 0.9), theme.qcolor("#5c6373")))
        if o.device:
            if colored:
                items.append((layout.device, theme.qcolor(theme.LAYER["device_dim"], 0.55), None))
                if o.color_mode == "body":
                    for body in project.structure.bodies:
                        items.append((project.plates[body.index], theme.body_color(body.index, 0.92), theme.qcolor("#0f1218", 0.6)))
                else:
                    for conductor in project.structure.nets.conductors:
                        if conductor.layer == "device":
                            items.append((conductor.polygon, theme.categorical(conductor.net, 0.92), theme.qcolor("#0f1218", 0.6)))
                    if project.holes:
                        items.append((MultiPolygon(project.holes), theme.qcolor(theme.BG, 1.0), None))
            else:
                items.append((layout.device, theme.qcolor(theme.LAYER["device"], 0.92), theme.qcolor("#1d3a66")))
        if o.box and project is not None:
            for body in project.structure.bodies:
                if not body.box.is_empty:
                    items.append((body.box, theme.qcolor(theme.LAYER["box"], 0.38), theme.qcolor(theme.LAYER["box"], 0.95)))
        if o.beams and project is not None:
            for beam in project.structure.beams:
                items.append((beam.polygon, theme.qcolor(theme.LAYER["beam"], 0.95), theme.qcolor("#7c2d12")))
        if o.rejected and project is not None:
            for candidate in visible_rejected(project.structure):
                items.append((candidate.polygon, theme.qcolor(theme.LAYER["rejected"], 0.55), theme.qcolor(theme.LAYER["rejected"])))
        if o.release and layout.release is not None:
            items.append((layout.release, theme.qcolor(theme.LAYER["release"], 0.12), theme.qcolor(theme.LAYER["release"])))
        if o.vias and layout.vias is not None:
            items.append((layout.vias, theme.qcolor(theme.LAYER["vias"], 0.85), theme.qcolor("#c4b5fd")))

    def _rebuild_pick(self) -> None:
        geoms: list = []
        tags: list[tuple[str, int]] = []
        project, layout = self._project, self._layout
        if project is not None:
            structure = project.structure
            for beam in structure.beams:
                geoms.append(beam.polygon)
                tags.append(("beam", beam.index))
            for i, candidate in enumerate(visible_rejected(structure)):
                geoms.append(candidate.polygon)
                tags.append(("rejected", i))
            for body in structure.bodies:
                for part in body.stubs.geoms:
                    geoms.append(part)
                    tags.append(("stub", body.index))
                for part in body.device.geoms:
                    geoms.append(part)
                    tags.append(("body", body.index))
            for via in structure.nets.vias:
                geoms.append(via.polygon)
                tags.append(("via", via.index))
            for conductor in structure.nets.conductors:
                if conductor.layer == "handle":
                    geoms.append(conductor.polygon)
                    tags.append(("handle", conductor.index))
        elif layout is not None:
            for part in layout.device.geoms:
                geoms.append(part)
                tags.append(("device", -1))
            for part in layout.handle.geoms:
                geoms.append(part)
                tags.append(("handle_raw", -1))
        self._pick_geoms = geoms
        self._pick_tags = tags
        self._pick_tree = STRtree(geoms) if geoms else None

    def _rebuild_graph(self) -> None:
        self._graph = None
        project = self._project
        if project is None:
            return
        structure = project.structure
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
        self._graph = {"centers": centers, "beams": beams, "conductors": conductors, "links": links}

    # ------------------------------------------------------------- transform
    def _transform(self) -> QTransform:
        t = QTransform()
        t.translate(self.width() / 2, self.height() / 2)
        t.scale(self._scale, -self._scale)
        t.translate(-float(self._center[0]), -float(self._center[1]))
        return t

    def _to_world(self, pos: QPointF) -> tuple[float, float]:
        inverse, _ = self._transform().inverted()
        point = inverse.map(pos)
        return point.x(), point.y()

    def fit(self) -> None:
        if self._bounds is None:
            return
        xmin, ymin, xmax, ymax = self._bounds
        width, height = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
        self._scale = 0.92 * min(self.width() / width, self.height() / height)
        self._center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        self._fitted = True
        self._invalidate()
        self.update()

    def _invalidate(self) -> None:
        self._version += 1

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if not self._fitted:
            self.fit()

    # ---------------------------------------------------------------- input
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 ** (delta / 120.0)
        pos = event.position()
        w0 = self._to_world(pos)
        self._scale = float(np.clip(self._scale * factor, 1e2, 1e12))
        w1 = self._to_world(pos)
        self._center += np.array([w0[0] - w1[0], w0[1] - w1[1]])
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_last = event.position()
            self._drag_moved = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        pos = event.position()
        if self._drag_last is not None:
            delta = pos - self._drag_last
            if abs(delta.x()) + abs(delta.y()) > 0:
                self._drag_moved = True
            self._center[0] -= delta.x() / self._scale
            self._center[1] += delta.y() / self._scale
            self._drag_last = pos
            self.update()
            return
        hit = self._hit(pos)
        if hit != self._hover:
            self._hover = hit
            self.hovered.emit(self._info(hit))
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.unsetCursor()
        if self._drag_last is None:
            return
        self._drag_last = None
        if event.button() == Qt.MouseButton.LeftButton and not self._drag_moved:
            hit = self._hit(event.position())
            body = -1
            if hit is not None and hit[0] in ("body", "stub"):
                body = hit[1]
            self.bodyClicked.emit(body)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.fit()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._hover is not None:
            self._hover = None
            self.hovered.emit(None)
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_F:
            self.fit()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._layout is None:
            return
        x, y = self._to_world(QPointF(event.pos()))
        menu = QMenu(self)
        fit_action = QAction("Fit view", menu)
        fit_action.triggered.connect(self.fit)
        menu.addAction(fit_action)
        anchor_action = QAction(f"Use ({x * 1e6:.1f}, {y * 1e6:.1f}) µm as anchor point", menu)
        anchor_action.triggered.connect(lambda: self.anchorPicked.emit(x, y))
        menu.addAction(anchor_action)
        menu.exec(event.globalPos())

    # -------------------------------------------------------------- picking
    def _hit(self, pos: QPointF) -> tuple[str, int] | None:
        if self._pick_tree is None:
            return None
        x, y = self._to_world(pos)
        hits = self._pick_tree.query(Point(x, y), predicate="within")
        if hits.size == 0:
            return None
        return self._pick_tags[int(hits.min())]

    def describe(self, kind: str, index: int) -> dict | None:
        """Status-bar description of an item, e.g. ``describe("body", 2)``."""
        return self._info((kind, index))

    def _info(self, hit: tuple[str, int] | None) -> dict | None:
        if hit is None:
            return None
        kind, index = hit
        project = self._project
        if kind == "device":
            return {"kind": kind, "index": index, "text": "Device layer polygon"}
        if kind == "handle_raw":
            return {"kind": kind, "index": index, "text": "Handle layer polygon"}
        if project is None:
            return None
        structure = project.structure
        if kind in ("body", "stub"):
            body = structure.bodies[index]
            beams = sum(1 for beam in structure.beams if index in beam.bodies)
            nets = sorted(set(body.nets))
            role = "Anchor body" if body.anchor else f"Body {index}"
            what = " (stub region)" if kind == "stub" else ""
            text = (
                f"{role}{what} · mass {fmt_mass(body.inertial.mass)} · centre {fmt_point(body.inertial.center)}"
                f" · {beams} beam{'s' if beams != 1 else ''} · net{'s' if len(nets) != 1 else ''} {', '.join(map(str, nets)) or '—'}"
            )
            return {"kind": kind, "index": index, "text": text}
        if kind == "beam":
            beam = structure.beams[index]
            text = (
                f"Beam {index} · bodies {beam.bodies[0]}–{beam.bodies[1]} · length {fmt_length(beam.length)}"
                f" · width {fmt_length(beam.width)} · net {beam.net}"
            )
            return {"kind": kind, "index": index, "text": text}
        if kind == "rejected":
            candidate = visible_rejected(structure)[index]
            bodies = ", ".join(map(str, candidate.bodies)) or "none"
            return {"kind": kind, "index": index, "text": f"Rejected beam candidate · {candidate.reason} · bodies {bodies}"}
        if kind == "via":
            via = structure.nets.vias[index]
            return {"kind": kind, "index": index, "text": f"Via {index} · device conductors {list(via.device)} · handle conductors {list(via.handle)}"}
        if kind == "handle":
            conductor = structure.nets.conductors[index]
            return {"kind": kind, "index": index, "text": f"Handle conductor {index} · net {conductor.net}"}
        return None

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BG))
        if self._layout is None:
            painter.setPen(QColor(theme.MUTED))
            font = QFont(self._font)
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a GDS file to view its layers")
            painter.end()
            return
        pixmap, origin = self._base()
        painter.drawPixmap(origin, pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        transform = self._transform()
        if self._graph is not None:
            if self.options.mech_graph:
                self._draw_mech_graph(painter, transform)
            if self.options.elec_graph:
                self._draw_elec_graph(painter, transform)
        self._draw_highlights(painter, transform)
        self._draw_scalebar(painter)
        painter.end()

    def _base(self) -> tuple[QPixmap, QPointF]:
        dpr = self.devicePixelRatioF()
        width, height = max(self.width(), 1), max(self.height(), 1)
        cache = self._cache
        transform = self._transform()
        if (
            cache is not None
            and cache["scale"] == self._scale
            and cache["dpr"] == dpr
            and cache["version"] == self._version
            and cache["size"] == (width, height)
        ):
            at = transform.map(QPointF(*cache["center"]))
            if abs(at.x() - width / 2) <= width / 2 and abs(at.y() - height / 2) <= height / 2:
                return cache["pixmap"], QPointF(at.x() - width, at.y() - height)
        pixmap = QPixmap(int(2 * width * dpr), int(2 * height * dpr))
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(QColor(theme.BG))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = float(self._center[0]), float(self._center[1])
        base = QTransform()
        base.translate(width, height)
        base.scale(self._scale, -self._scale)
        base.translate(-cx, -cy)
        world = (cx - width / self._scale, cy - height / self._scale, cx + width / self._scale, cy + height / self._scale)
        painter.setTransform(base)
        clip = self._needs_clipping()
        for geometry, fill, outline in self._fills:
            path = self._path(geometry, world, clip)
            if path.isEmpty():
                continue
            painter.setBrush(QBrush(fill))
            painter.setPen(_cosmetic(outline, 1.0) if outline is not None else Qt.PenStyle.NoPen)
            painter.drawPath(path)
        painter.end()
        self._cache = {
            "scale": self._scale,
            "dpr": dpr,
            "version": self._version,
            "size": (width, height),
            "center": (cx, cy),
            "pixmap": pixmap,
        }
        at = transform.map(QPointF(cx, cy))
        return pixmap, QPointF(at.x() - width, at.y() - height)

    def _needs_clipping(self) -> bool:
        if self._bounds is None:
            return False
        xmin, ymin, xmax, ymax = self._bounds
        return max(xmax - xmin, ymax - ymin) * self._scale > RASTER_LIMIT

    def _path(self, geometry, world, clip: bool) -> QPainterPath:
        if clip:
            clipped = shapely.clip_by_rect(geometry, *world)
            return path_of(clipped)
        key = id(geometry)
        path = self._paths.get(key)
        if path is None:
            path = path_of(geometry)
            self._paths[key] = path
        return path

    def _draw_mech_graph(self, painter: QPainter, transform: QTransform) -> None:
        graph = self._graph
        assert graph is not None
        accent = QColor(theme.ACCENT)
        edge_pen = QPen(accent, 1.5)
        beam_pen = QPen(QColor(theme.LAYER["beam"]), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        for a, b, start, end, _ in graph["beams"]:
            pa = transform.map(QPointF(*graph["centers"][a]))
            ps = transform.map(QPointF(float(start[0]), float(start[1])))
            pe = transform.map(QPointF(float(end[0]), float(end[1])))
            pb = transform.map(QPointF(*graph["centers"][b]))
            painter.setPen(edge_pen)
            painter.drawLine(pa, ps)
            painter.drawLine(pe, pb)
            painter.setPen(beam_pen)
            painter.drawLine(ps, pe)
        painter.setFont(self._font)
        for index, center in graph["centers"].items():
            point = transform.map(QPointF(*center))
            if not self.rect().adjusted(-20, -20, 20, 20).contains(point.toPoint()):
                continue
            painter.setPen(QPen(QColor(theme.BG), 2.0))
            painter.setBrush(QBrush(QColor(theme.SELECTION) if index == self._selected else accent))
            if index == 0:
                diamond = QPolygonF([point + QPointF(0, -8), point + QPointF(8, 0), point + QPointF(0, 8), point + QPointF(-8, 0)])
                painter.drawPolygon(diamond)
            else:
                painter.drawEllipse(point, 6.5, 6.5)
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(point + QPointF(10, 4), "anchor" if index == 0 else f"B{index}")

    def _draw_elec_graph(self, painter: QPainter, transform: QTransform) -> None:
        graph = self._graph
        assert graph is not None
        via_pen = QPen(QColor(theme.LAYER["vias"]), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(via_pen)
        for a, b, centroid in graph["links"]:
            pa = transform.map(QPointF(graph["conductors"][a][0], graph["conductors"][a][1]))
            pv = transform.map(QPointF(*centroid))
            pb = transform.map(QPointF(graph["conductors"][b][0], graph["conductors"][b][1]))
            painter.drawLine(pa, pv)
            painter.drawLine(pv, pb)
        painter.setFont(self._font)
        for index, (x, y, layer, net) in graph["conductors"].items():
            point = transform.map(QPointF(x, y))
            if not self.rect().adjusted(-20, -20, 20, 20).contains(point.toPoint()):
                continue
            painter.setPen(QPen(QColor(theme.BG), 2.0))
            painter.setBrush(QBrush(theme.categorical(net)))
            if layer == "device":
                painter.drawEllipse(point, 6.0, 6.0)
            else:
                painter.drawRect(QRectF(point.x() - 6, point.y() - 6, 12, 12))
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(point + QPointF(10, 4), f"N{net}" + (" (handle)" if layer == "handle" else ""))

    def _draw_highlights(self, painter: QPainter, transform: QTransform) -> None:
        project = self._project
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if project is not None and self._selected >= 0:
            painter.setTransform(transform)
            painter.setPen(_cosmetic(QColor(theme.SELECTION), 2.5))
            outline = self._outlines.get(self._selected)
            if outline is not None and not outline.is_empty:
                painter.drawPath(path_of(shapely.clip_by_rect(outline, *self._visible_world())))
            painter.resetTransform()
        if self._hover is not None:
            geometry = self._hover_geometry()
            if geometry is not None:
                painter.setTransform(transform)
                painter.setPen(_cosmetic(QColor(theme.HOVER), 1.5))
                painter.drawPath(path_of(shapely.clip_by_rect(geometry, *self._visible_world())))
                painter.resetTransform()

    def _visible_world(self) -> tuple[float, float, float, float]:
        x0, y0 = self._to_world(QPointF(0, self.height()))
        x1, y1 = self._to_world(QPointF(self.width(), 0))
        return (x0, y0, x1, y1)

    def _hover_geometry(self):
        if self._hover is None or self._project is None:
            return None
        kind, index = self._hover
        structure = self._project.structure
        if kind == "body":
            return structure.bodies[index].device
        if kind == "stub":
            return structure.bodies[index].stubs
        if kind == "beam":
            return structure.beams[index].polygon
        if kind == "rejected":
            return visible_rejected(structure)[index].polygon
        if kind == "via":
            return structure.nets.vias[index].polygon
        if kind == "handle":
            return structure.nets.conductors[index].polygon
        return None

    def _draw_scalebar(self, painter: QPainter) -> None:
        px_per_um = self._scale * 1e-6
        if px_per_um <= 0:
            return
        target = 120.0 / px_per_um
        magnitude = 10 ** math.floor(math.log10(target))
        nice = min((m for m in (1, 2, 5, 10) if m * magnitude >= target), default=10) * magnitude
        length_px = nice * px_per_um
        x0, y = 16.0, self.height() - 18.0
        painter.setPen(QPen(QColor(theme.TEXT), 2.0))
        painter.drawLine(QPointF(x0, y), QPointF(x0 + length_px, y))
        painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        painter.drawLine(QPointF(x0 + length_px, y - 4), QPointF(x0 + length_px, y + 4))
        painter.setFont(self._font)
        label = f"{nice:g} µm" if nice < 1000 else f"{nice / 1000:g} mm"
        painter.drawText(QPointF(x0 + length_px + 8, y + 4), label)
