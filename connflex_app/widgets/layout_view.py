"""2D layout view rendered with OpenGL: pre-triangulated layers, live overlays, and picking.

Fills and outlines live in GPU buffers that only change when the data or the display
options change; panning and zooming only update one matrix. Labels, markers, and the scale
bar are drawn with QPainter on top.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QMenu
from shapely import Point

from connflex_app import theme
from connflex_app.geometry import SCENE_UNIT
from connflex_app.render2d import RenderData
from connflex_app.session import Project
from connflex_app.units import fmt_length, fmt_mass, fmt_point

LABEL_LIMIT = 400
MARKER_LIMIT = 3000

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;
uniform mat3 u_view;
uniform vec2 u_offset;
out vec4 v_color;
void main() {
    vec3 p = u_view * vec3(a_position, 1.0);
    gl_Position = vec4(p.xy + u_offset, 0.0, 1.0);
    v_color = a_color;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec4 v_color;
out vec4 frag;
void main() {
    frag = v_color;
}
"""


def _rgba(color: QColor) -> tuple[float, float, float, float]:
    return (color.redF(), color.greenF(), color.blueF(), color.alphaF())


def _compile(kind, source: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(shader).decode(errors="replace"))
    return shader


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


class _Batch:
    """One vertex array with a position and a colour buffer."""

    def __init__(self) -> None:
        self.vao = 0
        self.positions = 0
        self.colors = 0
        self.count = 0


def _colored(points: np.ndarray, color) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.float32), (len(points), 1))


class LayoutView(QOpenGLWidget):
    """Interactive 2D view of the layout and the structure extracted from it."""

    hovered = Signal(object)
    bodyClicked = Signal(int)
    anchorPicked = Signal(float, float)

    BATCHES = ("fills", "lines", "graph", "hover", "select")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 200)
        self.options = ViewOptions()
        self._layout = None
        self._project: Project | None = None
        self._render: RenderData | None = None
        self._scale = 1.0
        self._center = np.zeros(2)
        self._fitted = False
        self._bounds: tuple[float, float, float, float] | None = None
        self._items: list[tuple[object, tuple | None, tuple | None]] = []
        self._pending: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
        self._batches: dict[str, _Batch] = {}
        self._gl_ready = False
        self._program = 0
        self._uniforms: dict[str, int] = {}
        self._hover: tuple[str, int] | None = None
        self._selected = -1
        self._drag_last: QPointF | None = None
        self._drag_moved = False
        self._font = QFont()
        self._font.setPointSize(10)

    # ------------------------------------------------------------------ data
    def set_layout(self, layout, render: RenderData) -> None:
        self._layout = layout
        self._project = None
        self._render = render
        self._fitted = False
        self._bounds = self._layout_bounds()
        self._rebuild()
        self.fit()

    def set_project(self, project: Project | None) -> None:
        self._project = project
        if project is not None:
            self._layout = project.layout
            self._render = project.render
            self._bounds = self._layout_bounds()
        self._rebuild()
        if not self._fitted:
            self.fit()

    def set_options(self, **changes) -> None:
        for key, value in changes.items():
            setattr(self.options, key, value)
        self._rebuild_items()
        self._rebuild_graph()
        self.update()

    def set_selected(self, body: int) -> None:
        self._selected = body
        render = self._render
        if render is not None and body >= 0 and body in render.outlines:
            edges = render.edge(render.outlines[body])
            self._pending["select"] = (edges, _colored(edges, _rgba(QColor(theme.SELECTION))))
        else:
            self._pending["select"] = (None, None)
        self.update()

    def grab_image(self):
        """The rendered view, overlays included (``QWidget.grab`` drops QPainter overlays on GL widgets)."""
        return self.grabFramebuffer()

    def describe(self, kind: str, index: int) -> dict | None:
        """Status-bar description of an item, e.g. ``describe("body", 2)``."""
        return self._info((kind, index))

    def _layout_bounds(self):
        if self._layout is None:
            return None
        import shapely

        return tuple(shapely.total_bounds(np.array([self._layout.device, self._layout.handle], dtype=object)))

    def _rebuild(self) -> None:
        self._hover = None
        self._pending["hover"] = (None, None)
        self._pending["select"] = (None, None)
        self._rebuild_items()
        self._rebuild_graph()
        self.update()

    def _rebuild_items(self) -> None:
        """Decide what gets filled and outlined, in paint order, then pack the GPU arrays."""
        items: list[tuple[object, tuple | None, tuple | None]] = []
        layout, project, render, o = self._layout, self._project, self._render, self.options
        self._items = items
        if layout is None or render is None:
            self._pending["fills"] = (None, None)
            self._pending["lines"] = (None, None)
            return
        colored = project is not None and o.color_mode in ("body", "net")
        c = theme.LAYER
        if o.handle:
            if colored:
                items.append((layout.handle, _rgba(theme.qcolor(c["handle"], 0.55)), None))
                if o.color_mode == "body":
                    for body in project.structure.bodies:
                        items.append((body.handle, _rgba(theme.body_color(body.index, 0.35)), None))
                else:
                    for conductor in project.structure.nets.conductors:
                        if conductor.layer == "handle":
                            items.append((conductor.polygon, _rgba(theme.categorical(conductor.net, 0.35)), None))
            else:
                items.append((layout.handle, _rgba(theme.qcolor(c["handle"], 0.9)), _rgba(theme.qcolor("#6c7488"))))
        if o.device:
            if colored:
                items.append((layout.device, _rgba(theme.qcolor(c["device_dim"], 0.55)), None))
                if o.color_mode == "body":
                    for body in project.structure.bodies:
                        items.append((project.plates[body.index], _rgba(theme.body_color(body.index, 0.92)), _rgba(theme.qcolor("#0f1218", 0.7))))
                else:
                    for conductor in project.structure.nets.conductors:
                        if conductor.layer == "device":
                            items.append((conductor.polygon, _rgba(theme.categorical(conductor.net, 0.92)), _rgba(theme.qcolor("#0f1218", 0.7))))
                if render.holes is not None:  # the closed plates hide the release holes; draw them back
                    items.append((render.holes, _rgba(theme.qcolor(theme.BG, 1.0)), None))
            else:
                items.append((layout.device, _rgba(theme.qcolor(c["device"], 0.92)), _rgba(theme.qcolor("#1d3a66"))))
        if o.box and project is None and render.box_layer is not None:
            items.append((render.box_layer, _rgba(theme.qcolor(c["box"], 0.38)), _rgba(theme.qcolor(c["box"], 0.95))))
        if o.box and project is not None:
            for body in project.structure.bodies:
                if not body.box.is_empty:
                    items.append((body.box, _rgba(theme.qcolor(c["box"], 0.38)), _rgba(theme.qcolor(c["box"], 0.95))))
        if o.beams and project is not None:
            for beam in project.structure.beams:
                items.append((beam.polygon, _rgba(theme.qcolor(c["beam"], 0.95)), _rgba(theme.qcolor("#7c2d12"))))
        if o.rejected and render.rejected:
            for candidate in render.rejected:
                items.append((candidate.polygon, _rgba(theme.qcolor(c["rejected"], 0.55)), _rgba(theme.qcolor(c["rejected"]))))
        if o.release and layout.release is not None:
            items.append((layout.release, _rgba(theme.qcolor(c["release"], 0.12)), _rgba(theme.qcolor(c["release"]))))
        if o.vias and layout.vias is not None:
            items.append((layout.vias, _rgba(theme.qcolor(c["vias"], 0.85)), _rgba(theme.qcolor("#c4b5fd"))))

        fill_pos, fill_col, line_pos, line_col = [], [], [], []
        for geometry, fill, outline in items:
            if fill is not None:
                tri = render.tri(geometry)
                if len(tri):
                    fill_pos.append(tri)
                    fill_col.append(_colored(tri, fill))
            if outline is not None:
                edges = render.edge(geometry)
                if len(edges):
                    line_pos.append(edges)
                    line_col.append(_colored(edges, outline))
        self._pending["fills"] = (np.concatenate(fill_pos), np.concatenate(fill_col)) if fill_pos else (None, None)
        self._pending["lines"] = (np.concatenate(line_pos), np.concatenate(line_col)) if line_pos else (None, None)

    def _rebuild_graph(self) -> None:
        render, o = self._render, self.options
        graph = render.graph if render is not None else None
        if graph is None or not (o.mech_graph or o.elec_graph):
            self._pending["graph"] = (None, None)
            return
        origin = render.origin
        accent = _rgba(QColor(theme.ACCENT))
        beam = _rgba(QColor(theme.LAYER["beam"]))
        via = _rgba(QColor(theme.LAYER["vias"]))
        points: list = []
        colors: list = []
        if o.mech_graph:
            centers = graph["centers"]
            for a, b, start, end, _index in graph["beams"]:
                ca, cb = centers[a], centers[b]
                points += [ca, start, start, end, end, cb]
                colors += [accent, accent, beam, beam, accent, accent]
        if o.elec_graph:
            conductors = graph["conductors"]
            for a, b, centroid in graph["links"]:
                pa, pb = conductors[a][:2], conductors[b][:2]
                points += [pa, centroid, centroid, pb]
                colors += [via] * 4
        if not points:
            self._pending["graph"] = (None, None)
            return
        positions = (np.asarray(points, dtype=np.float64) * SCENE_UNIT - origin).astype(np.float32)
        self._pending["graph"] = (positions, np.asarray(colors, dtype=np.float32))

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

    def _view_matrix(self) -> np.ndarray:
        width, height = max(self.width(), 1), max(self.height(), 1)
        s = self._scale / SCENE_UNIT  # pixels per micrometre
        sx, sy = 2.0 * s / width, 2.0 * s / height
        origin = self._render.origin if self._render is not None else np.zeros(2)
        cx, cy = self._center * SCENE_UNIT - origin
        return np.array([[sx, 0.0, -cx * sx], [0.0, sy, -cy * sy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def fit(self) -> None:
        if self._bounds is None:
            return
        xmin, ymin, xmax, ymax = self._bounds
        width, height = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
        self._scale = 0.92 * min(self.width() / width, self.height() / height)
        self._center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        self._fitted = True
        self.update()

    def zoom(self, factor: float, pos: QPointF | None = None) -> None:
        """Zoom by ``factor`` about ``pos`` (widget coordinates), or about the centre."""
        if pos is None:
            pos = QPointF(self.width() / 2, self.height() / 2)
        w0 = self._to_world(pos)
        self._scale = float(np.clip(self._scale * factor, 1e2, 1e12))
        w1 = self._to_world(pos)
        self._center += np.array([w0[0] - w1[0], w0[1] - w1[1]])
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if not self._fitted:
            self.fit()

    # --------------------------------------------------------------- OpenGL
    def initializeGL(self) -> None:  # noqa: N802 - Qt API
        vertex = _compile(GL.GL_VERTEX_SHADER, VERTEX_SHADER)
        fragment = _compile(GL.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex)
        GL.glAttachShader(program, fragment)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(program).decode(errors="replace"))
        GL.glDeleteShader(vertex)
        GL.glDeleteShader(fragment)
        self._program = program
        self._uniforms = {name: GL.glGetUniformLocation(program, name) for name in ("u_view", "u_offset")}
        for name in self.BATCHES:
            batch = _Batch()
            batch.vao = GL.glGenVertexArrays(1)
            batch.positions = GL.glGenBuffers(1)
            batch.colors = GL.glGenBuffers(1)
            GL.glBindVertexArray(batch.vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, batch.positions)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, batch.colors)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(0))
            self._batches[name] = batch
        GL.glBindVertexArray(0)
        self._gl_ready = True
        if self._render is not None:
            self._rebuild_items()
            self._rebuild_graph()
            self.set_selected(self._selected)

    def _flush_uploads(self) -> None:
        for name, (positions, colors) in list(self._pending.items()):
            batch = self._batches[name]
            if positions is None or len(positions) == 0:
                batch.count = 0
            else:
                positions = np.ascontiguousarray(positions, dtype=np.float32)
                colors = np.ascontiguousarray(colors, dtype=np.float32)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, batch.positions)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, positions.nbytes, positions, GL.GL_STATIC_DRAW)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, batch.colors)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_STATIC_DRAW)
                batch.count = len(positions)
            del self._pending[name]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _draw_batch(self, name: str, mode, offsets=((0.0, 0.0),)) -> None:
        batch = self._batches[name]
        if batch.count == 0:
            return
        GL.glBindVertexArray(batch.vao)
        for dx, dy in offsets:
            GL.glUniform2f(self._uniforms["u_offset"], dx, dy)
            GL.glDrawArrays(mode, 0, batch.count)

    def paintGL(self) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.beginNativePainting()
        dpr = self.devicePixelRatioF()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        GL.glClearColor(*theme.GL_BACKGROUND, 1.0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self._gl_ready and self._render is not None:
            self._flush_uploads()
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glUseProgram(self._program)
            GL.glUniformMatrix3fv(self._uniforms["u_view"], 1, GL.GL_TRUE, self._view_matrix())
            self._draw_batch("fills", GL.GL_TRIANGLES)
            self._draw_batch("lines", GL.GL_LINES)
            self._draw_batch("graph", GL.GL_LINES)
            px, py = 2.0 / max(self.width(), 1), 2.0 / max(self.height(), 1)
            self._draw_batch("hover", GL.GL_LINES, ((0.0, 0.0), (px, 0.0), (0.0, py)))
            self._draw_batch("select", GL.GL_LINES, ((0.0, 0.0), (px, 0.0), (-px, 0.0), (0.0, py), (0.0, -py)))
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            GL.glDisable(GL.GL_BLEND)
        painter.endNativePainting()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        if self._layout is None:
            painter.setPen(QColor(theme.MUTED))
            font = QFont(self._font)
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a GDS file to view its layers")
        else:
            transform = self._transform()
            if self._render is not None and self._render.graph is not None:
                if self.options.mech_graph:
                    self._draw_mech_nodes(painter, transform)
                if self.options.elec_graph:
                    self._draw_elec_nodes(painter, transform)
            self._draw_scalebar(painter)
        painter.end()

    # -------------------------------------------------------------- overlays
    def _visible_points(self, transform: QTransform, items):
        """Yield (key, device point, payload) for items whose world point lies in the view."""
        rect = self.rect().adjusted(-20, -20, 20, 20)
        for key, x, y, payload in items:
            point = transform.map(QPointF(x, y))
            if rect.contains(point.toPoint()):
                yield key, point, payload

    def _draw_mech_nodes(self, painter: QPainter, transform: QTransform) -> None:
        graph = self._render.graph
        accent = QColor(theme.ACCENT)
        visible = list(self._visible_points(transform, ((i, c[0], c[1], None) for i, c in graph["centers"].items())))
        if len(visible) > MARKER_LIMIT:
            return
        labels = len(visible) <= LABEL_LIMIT
        for index, point, _payload in visible:
            painter.setPen(QPen(QColor(theme.BG), 2.0))
            painter.setBrush(QBrush(QColor(theme.SELECTION) if index == self._selected else accent))
            if index == 0:
                diamond = QPolygonF([point + QPointF(0, -8), point + QPointF(8, 0), point + QPointF(0, 8), point + QPointF(-8, 0)])
                painter.drawPolygon(diamond)
            else:
                painter.drawEllipse(point, 6.5, 6.5)
            if labels:
                painter.setPen(QColor(theme.TEXT))
                painter.drawText(point + QPointF(10, 4), "anchor" if index == 0 else f"B{index}")

    def _draw_elec_nodes(self, painter: QPainter, transform: QTransform) -> None:
        graph = self._render.graph
        items = ((i, x, y, (layer, net)) for i, (x, y, layer, net) in graph["conductors"].items())
        visible = list(self._visible_points(transform, items))
        if len(visible) > MARKER_LIMIT:
            return
        labels = len(visible) <= LABEL_LIMIT
        for _index, point, (layer, net) in visible:
            painter.setPen(QPen(QColor(theme.BG), 2.0))
            painter.setBrush(QBrush(theme.categorical(net)))
            if layer == "device":
                painter.drawEllipse(point, 6.0, 6.0)
            else:
                painter.drawRect(QRectF(point.x() - 6, point.y() - 6, 12, 12))
            if labels:
                painter.setPen(QColor(theme.TEXT))
                painter.drawText(point + QPointF(10, 4), f"N{net}" + (" (handle)" if layer == "handle" else ""))

    def _draw_scalebar(self, painter: QPainter) -> None:
        px_per_um = self._scale / SCENE_UNIT
        if px_per_um <= 0:
            return
        target = 120.0 / px_per_um
        magnitude = 10 ** math.floor(math.log10(target))
        nice = min((m for m in (1, 2, 5, 10) if m * magnitude >= target), default=10) * magnitude
        length_px = nice * px_per_um
        x0, y = 16.0, self.height() - 18.0
        label = f"{nice:g} µm" if nice < 1000 else f"{nice / 1000:g} mm"
        painter.setFont(self._font)
        text_width = painter.fontMetrics().horizontalAdvance(label)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(theme.qcolor(theme.BG, 0.72)))
        painter.drawRoundedRect(QRectF(x0 - 8, y - 13, length_px + text_width + 26, 26), 6, 6)
        painter.setPen(QPen(QColor(theme.TEXT), 2.0))
        painter.drawLine(QPointF(x0, y), QPointF(x0 + length_px, y))
        painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        painter.drawLine(QPointF(x0 + length_px, y - 4), QPointF(x0 + length_px, y + 4))
        painter.drawText(QPointF(x0 + length_px + 8, y + 4), label)

    # ---------------------------------------------------------------- input
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        if delta:
            self.zoom(1.15 ** (delta / 120.0), event.position())

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
            self._set_hover(hit)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.unsetCursor()
        if self._drag_last is None:
            return
        self._drag_last = None
        if event.button() == Qt.MouseButton.LeftButton and not self._drag_moved:
            hit = self._hit(event.position())
            body = hit[1] if hit is not None and hit[0] in ("body", "stub") else -1
            self.bodyClicked.emit(body)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.fit()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._hover is not None:
            self._set_hover(None)

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
        render = self._render
        if render is None or render.pick_tree is None:
            return None
        x, y = self._to_world(pos)
        hits = render.pick_tree.query(Point(x, y), predicate="within")
        if hits.size == 0:
            return None
        return render.pick_tags[int(hits.min())]

    def _set_hover(self, hit: tuple[str, int] | None) -> None:
        self._hover = hit
        geometry = self._hover_geometry()
        if geometry is not None and self._render is not None:
            edges = self._render.edge(geometry)
            self._pending["hover"] = (edges, _colored(edges, _rgba(QColor(theme.HOVER))))
        else:
            self._pending["hover"] = (None, None)
        self.hovered.emit(self._info(hit))
        self.update()

    def _hover_geometry(self):
        if self._hover is None or self._render is None:
            return None
        kind, index = self._hover
        if self._project is None:
            if kind == "device":
                return None
            return None
        structure = self._project.structure
        if kind == "body":
            return structure.bodies[index].device
        if kind == "stub":
            return structure.bodies[index].stubs
        if kind == "beam":
            return structure.beams[index].polygon
        if kind == "rejected":
            return self._render.rejected[index].polygon
        if kind == "via":
            return structure.nets.vias[index].polygon
        if kind == "handle":
            return structure.nets.conductors[index].polygon
        return None

    def _info(self, hit: tuple[str, int] | None) -> dict | None:
        if hit is None:
            return None
        kind, index = hit
        project, render = self._project, self._render
        if kind == "device":
            return {"kind": kind, "index": index, "text": "Device layer polygon"}
        if kind == "handle_raw":
            return {"kind": kind, "index": index, "text": "Handle layer polygon"}
        if project is None or render is None:
            return None
        structure = project.structure
        if kind in ("body", "stub"):
            body = structure.bodies[index]
            beams = render.beam_counts.get(index, 0)
            nets = sorted(set(body.nets))
            role = "Anchor body" if body.anchor else f"Body {index}"
            what = " (stub region)" if kind == "stub" else ""
            if not nets:
                net_text = "no net"
            elif len(nets) <= 6:
                net_text = f"net{'s' if len(nets) != 1 else ''} {', '.join(map(str, nets))}"
            else:
                net_text = f"{len(nets)} nets"
            text = (
                f"{role}{what} · mass {fmt_mass(body.inertial.mass)} · centre {fmt_point(body.inertial.center)}"
                f" · {beams} beam{'s' if beams != 1 else ''} · {net_text}"
            )
            if index in render.free_bodies:
                text += " · FREE: no beam path to the anchor"
            return {"kind": kind, "index": index, "text": text}
        if kind == "beam":
            beam = structure.beams[index]
            text = (
                f"Beam {index} · bodies {beam.bodies[0]}–{beam.bodies[1]} · length {fmt_length(beam.length)}"
                f" · width {fmt_length(beam.width)} · net {beam.net}"
            )
            return {"kind": kind, "index": index, "text": text}
        if kind == "rejected":
            candidate = render.rejected[index]
            bodies = ", ".join(map(str, candidate.bodies)) or "none"
            return {"kind": kind, "index": index, "text": f"Rejected beam candidate · {candidate.reason} · bodies {bodies}"}
        if kind == "via":
            via = structure.nets.vias[index]
            return {"kind": kind, "index": index, "text": f"Via {index} · device conductors {list(via.device)} · handle conductors {list(via.handle)}"}
        if kind == "handle":
            conductor = structure.nets.conductors[index]
            return {"kind": kind, "index": index, "text": f"Handle conductor {index} · net {conductor.net}"}
        return None
