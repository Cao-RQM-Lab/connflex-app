"""3D view of the extruded structure with live mode animation, orbit camera, and picking."""

from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QElapsedTimer, QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtOpenGL import QOpenGLFramebufferObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from connflex_app import theme
from connflex_app.geometry import KIND_BEAM, KIND_BOX, KIND_DEVICE, KIND_HANDLE, KIND_REJECTED, Mesh

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_disp;
layout(location = 3) in vec3 a_color;
layout(location = 4) in float a_body;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_scale;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
flat out float v_body;
void main() {
    vec3 p = a_position + u_scale * a_disp;
    vec4 pv = u_view * vec4(p, 1.0);
    v_position = pv.xyz;
    v_normal = mat3(u_view) * a_normal;
    v_color = a_color;
    v_body = a_body;
    gl_Position = u_proj * pv;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;
flat in float v_body;
uniform float u_selected;
uniform float u_hover;
uniform float u_pick;
uniform vec3 u_select_color;
out vec4 frag;
void main() {
    if (u_pick > 0.5) {
        int id = int(v_body + 1.5);
        frag = vec4(float(id & 255) / 255.0, float((id >> 8) & 255) / 255.0, 0.0, 1.0);
        return;
    }
    vec3 n = normalize(v_normal);
    vec3 v = normalize(-v_position);
    if (dot(n, v) < 0.0) n = -n;
    vec3 l = normalize(vec3(0.35, 0.45, 1.0));
    float diffuse = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float specular = pow(max(dot(n, h), 0.0), 40.0) * 0.22;
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.12;
    vec3 color = v_color;
    if (v_body >= 0.0 && abs(v_body - u_selected) < 0.5) color = mix(color, u_select_color, 0.65);
    else if (v_body >= 0.0 && abs(v_body - u_hover) < 0.5) color = mix(color, vec3(1.0), 0.22);
    vec3 shaded = color * (0.32 + 0.68 * diffuse) + vec3(specular) + vec3(rim);
    frag = vec4(shaded, 1.0);
}
"""

DRAW_ORDER = (KIND_HANDLE, KIND_BOX, KIND_DEVICE, KIND_BEAM, KIND_REJECTED)


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def _compile(kind, source: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(shader).decode(errors="replace"))
    return shader


class Scene3D(QOpenGLWidget):
    """Renders the mesh of a project and animates per-vertex displacement fields."""

    bodyClicked = Signal(int)
    hovered = Signal(int)
    playingChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 200)
        self._mesh: Mesh | None = None
        self._colors: np.ndarray | None = None
        self._field: np.ndarray | None = None
        self._dirty = {"mesh": False, "colors": False, "field": False}
        self._gl_ready = False
        self._program = 0
        self._vao = 0
        self._buffers: dict[str, int] = {}
        self._uniforms: dict[str, int] = {}
        self._visible = {kind: True for kind in range(5)}
        self._color_mode = "layer"
        self._selected = -1
        self._hover = -1
        self._target = np.zeros(3, dtype=np.float64)
        self._distance = 2000.0
        self._yaw = 0.55
        self._pitch = 0.55
        self._fov = 35.0
        self._extent = 1000.0
        self._animated = False
        self._playing = False
        self._phase = 0.0
        self._amplitude = 0.04
        self._speed = 0.6
        self._caption = ""
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._click_request: QPointF | None = None
        self._hover_request: QPointF | None = None
        self._pick_fbo: QOpenGLFramebufferObject | None = None
        self._drag_last: QPointF | None = None
        self._drag_button = None
        self._drag_moved = False
        self._font = QFont()
        self._font.setPointSize(11)

    # ------------------------------------------------------------- public API
    def set_mesh(self, mesh: Mesh | None) -> None:
        self._mesh = mesh
        self._field = None
        self._animated = False
        self._caption = ""
        self._selected = -1
        self._hover = -1
        self.play(False)
        if mesh is not None and mesh.count:
            self._extent = mesh.extent
            self.reset_view()
        self._colors = self._compute_colors()
        self._dirty = {"mesh": True, "colors": True, "field": True}
        self.update()

    def set_color_mode(self, mode: str) -> None:
        if mode != self._color_mode:
            self._color_mode = mode
            self._colors = self._compute_colors()
            self._dirty["colors"] = True
            self.update()

    def set_visible(self, kind: int, on: bool) -> None:
        self._visible[kind] = on
        self.update()

    def set_selected(self, body: int) -> None:
        if body != self._selected:
            self._selected = body
            self.update()

    def set_motion(self, field: np.ndarray | None, animated: bool, caption: str = "") -> None:
        """Show a displacement field, animated as a harmonic oscillation or held static."""
        self._field = None if field is None else np.ascontiguousarray(field, dtype=np.float32)
        self._animated = animated and field is not None
        self._caption = caption if field is not None else ""
        self._dirty["field"] = True
        if self._animated:
            self._phase = math.pi / 2
            self.play(True)
        else:
            self.play(False)
        self.update()

    def clear_motion(self) -> None:
        self.set_motion(None, False)

    def has_motion(self) -> bool:
        return self._field is not None

    def is_animated(self) -> bool:
        return self._animated

    def play(self, on: bool) -> None:
        on = bool(on and self._animated)
        if on == self._playing:
            return
        self._playing = on
        if on:
            self._clock.restart()
            self._timer.start()
        else:
            self._timer.stop()
        self.playingChanged.emit(on)
        self.update()

    def is_playing(self) -> bool:
        return self._playing

    def set_amplitude(self, fraction: float) -> None:
        self._amplitude = float(fraction)
        self.update()

    def amplitude(self) -> float:
        return self._amplitude

    def set_speed(self, hertz: float) -> None:
        self._speed = float(hertz)

    def speed(self) -> float:
        return self._speed

    def reset_view(self) -> None:
        radius = self._extent * 0.6
        if self._mesh is not None and self._mesh.count:
            low, high = self._mesh.bounds
            self._extent = self._mesh.extent
            center = (low + high) / 2.0
            self._target = np.array([center[0], center[1], high[2] - 0.35 * (high[2] - low[2])], dtype=np.float64)
            radius = float(np.linalg.norm(high - low)) / 2.0
        aspect = max(self.width(), 1) / max(self.height(), 1)
        half = math.atan(math.tan(math.radians(self._fov) / 2.0) * min(aspect, 1.0))
        self._distance = radius / math.sin(half) * 1.02
        self._yaw = 0.55
        self._pitch = 0.55
        self.update()

    def grab_image(self):
        return self.grabFramebuffer()

    # ---------------------------------------------------------------- colours
    def _compute_colors(self) -> np.ndarray | None:
        mesh = self._mesh
        if mesh is None or mesh.count == 0:
            return None
        colors = np.empty((mesh.count, 3), dtype=np.float32)
        for kind in range(5):
            colors[mesh.kind == kind] = theme.GL_KIND_COLORS[kind]
        if self._color_mode == "body":
            structural = np.isin(mesh.kind, (KIND_DEVICE, KIND_BOX, KIND_HANDLE))
            bodies = np.unique(mesh.body[structural])
            for body in bodies:
                mask = structural & (mesh.body == body)
                tint = np.array(theme.body_rgb(int(body)), dtype=np.float32)
                kind = mesh.kind[mask]
                shade = np.where(kind == KIND_HANDLE, 0.55, np.where(kind == KIND_BOX, 0.8, 1.0))[:, None]
                colors[mask] = tint * shade
        return colors

    # --------------------------------------------------------------- GL setup
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
        for name in ("u_proj", "u_view", "u_scale", "u_selected", "u_hover", "u_pick", "u_select_color"):
            self._uniforms[name] = GL.glGetUniformLocation(program, name)
        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        for name in ("geometry", "field", "colors", "body"):
            self._buffers[name] = GL.glGenBuffers(1)
        GL.glBindVertexArray(0)
        self._gl_ready = True
        self._dirty = {"mesh": True, "colors": True, "field": True}

    def _upload(self) -> None:
        mesh = self._mesh
        GL.glBindVertexArray(self._vao)
        if self._dirty["mesh"]:
            self._dirty["mesh"] = False
            if mesh is not None and mesh.count:
                geometry = np.ascontiguousarray(np.column_stack([mesh.positions, mesh.normals]).astype(np.float32))
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers["geometry"])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, geometry.nbytes, geometry, GL.GL_STATIC_DRAW)
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(0))
                GL.glEnableVertexAttribArray(1)
                GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))
                body = np.ascontiguousarray(mesh.body.astype(np.float32))
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers["body"])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, body.nbytes, body, GL.GL_STATIC_DRAW)
                GL.glEnableVertexAttribArray(4)
                GL.glVertexAttribPointer(4, 1, GL.GL_FLOAT, GL.GL_FALSE, 4, ctypes.c_void_p(0))
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers["field"])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, mesh.count * 12, np.zeros((mesh.count, 3), np.float32), GL.GL_DYNAMIC_DRAW)
                GL.glEnableVertexAttribArray(2)
                GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers["colors"])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, mesh.count * 12, None, GL.GL_DYNAMIC_DRAW)
                GL.glEnableVertexAttribArray(3)
                GL.glVertexAttribPointer(3, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, ctypes.c_void_p(0))
                self._dirty["colors"] = True
                self._dirty["field"] = True
        if mesh is not None and mesh.count:
            if self._dirty["colors"]:
                self._dirty["colors"] = False
                colors = self._colors if self._colors is not None else self._compute_colors()
                if colors is not None:
                    colors = np.ascontiguousarray(colors, dtype=np.float32)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers["colors"])
                    GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, colors.nbytes, colors)
            if self._dirty["field"]:
                self._dirty["field"] = False
                field = self._field
                if field is None or len(field) != mesh.count:
                    field = np.zeros((mesh.count, 3), dtype=np.float32)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers["field"])
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, field.nbytes, field)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    # ---------------------------------------------------------------- camera
    def _matrices(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        cp, sp = math.cos(self._pitch), math.sin(self._pitch)
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        offset = np.array([cp * sy, -cp * cy, sp]) * self._distance
        eye = self._target + offset
        view = look_at(eye.astype(np.float64), self._target, np.array([0.0, 0.0, 1.0]))
        aspect = max(width, 1) / max(height, 1)
        near = max(self._distance * 0.01, self._extent * 1e-4)
        far = self._distance * 10.0 + self._extent * 4.0
        proj = perspective(self._fov, aspect, near, far)
        return proj.astype(np.float32), view.astype(np.float32)

    def _current_scale(self) -> float:
        if self._field is None:
            return 0.0
        amplitude = self._amplitude * self._extent
        if self._animated:
            return amplitude * math.sin(self._phase)
        return amplitude

    def _tick(self) -> None:
        dt = self._clock.restart() / 1000.0
        self._phase = (self._phase + 2.0 * math.pi * self._speed * dt) % (2.0 * math.pi)
        self.update()

    # --------------------------------------------------------------- painting
    def paintGL(self) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.beginNativePainting()
        dpr = self.devicePixelRatioF()
        width, height = int(self.width() * dpr), int(self.height() * dpr)
        if self._dirty["mesh"] or self._dirty["colors"] or self._dirty["field"]:
            self._upload()
        if self._mesh is not None and self._mesh.count and (self._click_request is not None or self._hover_request is not None):
            self._resolve_picks(width, height, dpr)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(*theme.GL_BACKGROUND, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._mesh is not None and self._mesh.count:
            self._draw(width, height, pick=False)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        painter.endNativePainting()
        self._draw_overlay(painter)
        painter.end()

    def _draw(self, width: int, height: int, pick: bool) -> None:
        proj, view = self._matrices(width, height)
        GL.glUseProgram(self._program)
        GL.glUniformMatrix4fv(self._uniforms["u_proj"], 1, GL.GL_TRUE, proj)
        GL.glUniformMatrix4fv(self._uniforms["u_view"], 1, GL.GL_TRUE, view)
        GL.glUniform1f(self._uniforms["u_scale"], self._current_scale())
        GL.glUniform1f(self._uniforms["u_selected"], float(self._selected))
        GL.glUniform1f(self._uniforms["u_hover"], float(self._hover))
        GL.glUniform1f(self._uniforms["u_pick"], 1.0 if pick else 0.0)
        GL.glUniform3f(self._uniforms["u_select_color"], *theme.SELECTION_RGB)
        GL.glBindVertexArray(self._vao)
        assert self._mesh is not None
        for kind in DRAW_ORDER:
            if not self._visible.get(kind, True):
                continue
            start, count = self._mesh.ranges.get(kind, (0, 0))
            if count:
                GL.glDrawArrays(GL.GL_TRIANGLES, start, count)

    def _resolve_picks(self, width: int, height: int, dpr: float) -> None:
        if self._pick_fbo is None or self._pick_fbo.size() != QSize(width, height):
            self._pick_fbo = QOpenGLFramebufferObject(QSize(width, height), QOpenGLFramebufferObject.Attachment.Depth)
        fbo = self._pick_fbo
        fbo.bind()
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._draw(width, height, pick=True)
        GL.glFinish()

        def read(pos: QPointF) -> int:
            x = int(pos.x() * dpr)
            y = height - 1 - int(pos.y() * dpr)
            if not (0 <= x < width and 0 <= y < height):
                return -1
            data = GL.glReadPixels(x, y, 1, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            if isinstance(data, (bytes, bytearray)):
                pixel = np.frombuffer(bytes(data), dtype=np.uint8)
            else:
                pixel = np.asarray(data, dtype=np.uint8).ravel()
            ident = int(pixel[0]) + 256 * int(pixel[1])
            return ident - 1

        if self._hover_request is not None:
            body = read(self._hover_request)
            self._hover_request = None
            if body != self._hover:
                self._hover = body
                self.hovered.emit(body)
        if self._click_request is not None:
            body = read(self._click_request)
            self._click_request = None
            self.bodyClicked.emit(body)
        fbo.release()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        if self._mesh is None or self._mesh.count == 0:
            painter.setPen(QColor(theme.MUTED))
            font = QFont(self._font)
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Extract a structure to build the 3D model")
            return
        # Axis gizmo.
        _, view = self._matrices(self.width(), self.height())
        rotation = view[:3, :3]
        origin = QPointF(44.0, self.height() - 44.0)
        for axis, color, label in ((0, "#f87171", "X"), (1, "#4ade80", "Y"), (2, "#60a5fa", "Z")):
            e = np.zeros(3)
            e[axis] = 1.0
            v = rotation @ e
            tip = origin + QPointF(v[0] * 28.0, -v[1] * 28.0)
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawLine(origin, tip)
            painter.drawText(tip + QPointF(v[0] * 8.0 - 4.0, -v[1] * 8.0 + 4.0), label)
        if self._caption:
            painter.setPen(QColor(theme.TEXT))
            rect = self.rect().adjusted(12, 10, -12, -10)
            painter.drawText(rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, self._caption)
            if self._animated:
                painter.setPen(QColor(theme.MUTED))
                state = "playing" if self._playing else "paused"
                painter.drawText(rect.adjusted(0, 18, 0, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"{state} · amplitude {self._amplitude * 100:.1f}% of extent")

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._drag_last = event.position()
        self._drag_button = event.button()
        self._drag_moved = False
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        pos = event.position()
        if self._drag_last is None:
            self._hover_request = pos
            self.update()
            return
        delta = pos - self._drag_last
        self._drag_last = pos
        if abs(delta.x()) + abs(delta.y()) > 0:
            self._drag_moved = True
        pan = self._drag_button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or bool(
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        )
        if pan:
            _, view = self._matrices(self.width(), self.height())
            right, up = view[0, :3].astype(np.float64), view[1, :3].astype(np.float64)
            k = 2.0 * self._distance * math.tan(math.radians(self._fov) / 2.0) / max(self.height(), 1)
            self._target = self._target - right * delta.x() * k + up * delta.y() * k
        else:
            self._yaw += delta.x() * 0.008
            self._pitch = float(np.clip(self._pitch + delta.y() * 0.008, -1.55, 1.55))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._drag_last is not None and not self._drag_moved and event.button() == Qt.MouseButton.LeftButton:
            self._click_request = event.position()
            self.update()
        self._drag_last = None
        self._drag_button = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        if delta:
            self._distance = float(np.clip(self._distance * (0.9 ** (delta / 120.0)), self._extent * 0.02, self._extent * 50.0))
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._hover != -1:
            self._hover = -1
            self.hovered.emit(-1)
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Space and self._animated:
            self.play(not self._playing)
        elif event.key() == Qt.Key.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if self._playing:
            self._clock.restart()
            self._timer.start()
