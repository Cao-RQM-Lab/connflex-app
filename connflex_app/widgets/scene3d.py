"""3D view of the extruded structure with live mode animation, orbit camera, and picking.

The mesh is uploaded once as indexed geometry. Rigid-body motion of every mode is applied in
the vertex shader from a small per-body transform texture, so selecting a mode, a deflection
shape, or a frame of a transient never touches the vertex buffers. Colours are also resolved
in the shader: by layer, by body, or from a table over paint groups (electrodes and nets),
which is how nets, selections, and voltages are shown and picked.
"""

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
from connflex_app.geometry import (
    KIND_BEAM,
    KIND_BOX,
    KIND_DEVICE,
    KIND_HANDLE,
    KIND_REJECTED,
    SCENE_UNIT,
    VERTEX_FLOATS,
    Mesh,
    motion_peak,
    trajectory_peak,
)

GROUP_TEXTURE_WIDTH = 1024  # group colours wrap into rows so any number of groups fits a texture

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_bodies;
layout(location = 3) in float a_blend;
layout(location = 4) in float a_kind;
layout(location = 5) in float a_group;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_scale;
uniform sampler2D u_motion;
uniform sampler2D u_body_colors;
uniform sampler2D u_group_colors;
uniform vec3 u_kind_colors[5];
uniform float u_color_mode;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
flat out float v_body;
flat out float v_group;

vec3 rigid(int body, vec3 p) {
    if (body < 0) return vec3(0.0);
    vec3 u = texelFetch(u_motion, ivec2(0, body), 0).xyz;
    vec3 w = texelFetch(u_motion, ivec2(1, body), 0).xyz;
    vec3 c = texelFetch(u_motion, ivec2(2, body), 0).xyz;
    return u + cross(w, p - c);
}

void main() {
    int a = int(floor(a_bodies.x + 0.5));
    int b = int(floor(a_bodies.y + 0.5));
    int kind = int(floor(a_kind + 0.5));
    vec3 d = mix(rigid(a, a_position), rigid(b, a_position), a_blend);
    vec3 p = a_position + u_scale * d;
    vec4 pv = u_view * vec4(p, 1.0);
    v_position = pv.xyz;
    v_normal = mat3(u_view) * a_normal;
    vec3 color = u_kind_colors[kind];
    if (u_color_mode > 1.5) {
        if (kind == 2) color *= 0.8;
        if (a_group > -0.5) {
            int g = int(floor(a_group + 0.5));
            vec4 paint = texelFetch(u_group_colors, ivec2(g % {width}, g / {width}), 0);
            color = mix(color, paint.rgb, paint.a);
        } else if (kind != 1) {
            color *= 0.7;
        }
    } else if (u_color_mode > 0.5 && kind <= 2 && a >= 0) {
        float shade = (kind == 2) ? 0.55 : ((kind == 1) ? 0.8 : 1.0);
        color = texelFetch(u_body_colors, ivec2(0, a), 0).rgb * shade;
    }
    v_color = color;
    v_body = (kind == 3) ? -1.0 : float(a);
    v_group = a_group;
    gl_Position = u_proj * pv;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;
flat in float v_body;
flat in float v_group;
uniform float u_selected;
uniform float u_hover;
uniform float u_pick;
uniform vec3 u_select_color;
out vec4 frag;
void main() {
    if (u_pick > 0.5) {
        int id = int((u_pick > 1.5 ? v_group : v_body) + 1.5);
        frag = vec4(float(id & 255) / 255.0, float((id >> 8) & 255) / 255.0, float((id >> 16) & 255) / 255.0, 1.0);
        return;
    }
    vec3 n = normalize(v_normal);
    vec3 v = normalize(-v_position);
    float facing = dot(n, v);
    float back = facing < 0.0 ? 0.55 : 1.0;   // faces seen from behind (e.g. through a hole) are darker
    if (facing < 0.0) n = -n;
    vec3 l = normalize(vec3(0.35, 0.45, 1.0));
    float diffuse = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float specular = pow(max(dot(n, h), 0.0), 40.0) * 0.22;
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.12;
    vec3 color = v_color;
    if (v_body >= 0.0 && abs(v_body - u_selected) < 0.5) color = mix(color, u_select_color, 0.65);
    else if (v_body >= 0.0 && abs(v_body - u_hover) < 0.5) color = mix(color, vec3(1.0), 0.22);
    vec3 shaded = color * back * (0.32 + 0.68 * diffuse) + vec3(specular) * back + vec3(rim);
    frag = vec4(shaded, 1.0);
}
"""

VERTEX_SHADER = VERTEX_SHADER.replace("{width}", str(GROUP_TEXTURE_WIDTH))

DRAW_ORDER = (KIND_HANDLE, KIND_BOX, KIND_DEVICE, KIND_BEAM, KIND_REJECTED)
STRIDE = 4 * VERTEX_FLOATS  # float32 per vertex: position, normal, bodies, blend, kind, group
LOOP_SECONDS = 8.0  # wall-clock time one pass through a transient takes at speed 1


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


def _texture_2d(width: int, height: int, data: np.ndarray | None, existing: int = 0) -> int:
    """(Re)create an RGBA32F texture used as a lookup table."""
    texture = existing or GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    payload = None if data is None else np.ascontiguousarray(data, dtype=np.float32)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, width, max(height, 1), 0, GL.GL_RGBA, GL.GL_FLOAT, payload)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture


class Scene3D(QOpenGLWidget):
    """Renders the mesh of a project and animates rigid-body motions on the GPU."""

    bodyClicked = Signal(int)
    hovered = Signal(int)
    groupClicked = Signal(int, int)  # paint group (-1 for none) and the keyboard modifiers held
    groupHovered = Signal(int)
    playingChanged = Signal(bool)
    timeChanged = Signal(float)  # simulation time shown, while a trajectory is loaded

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 200)
        self._mesh: Mesh | None = None
        self._centers = np.zeros((0, 3), dtype=np.float32)
        self._motion = np.zeros((0, 3, 4), dtype=np.float32)
        self._peak = 1.0
        self._dirty = {"mesh": False, "motion": False, "colors": False, "groups": False}
        self._gl_ready = False
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._textures = {"motion": 0, "colors": 0, "groups": 0}
        self._uniforms: dict[str, int] = {}
        self._visible = {kind: True for kind in range(5)}
        self._color_mode = "layer"
        self._group_table: np.ndarray | None = None
        self._pick_mode = "body"
        self._hover_group = -1
        self._click_modifiers = 0
        self._trajectory: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._time = 0.0
        self._rate = 1.0
        self._true_scale = False
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
        self._exporting = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._click_request: QPointF | None = None
        self._hover_request: QPointF | None = None
        self._pick_fbo: QOpenGLFramebufferObject | None = None
        self._pick_state: tuple | None = None
        self._drag_last: QPointF | None = None
        self._drag_button = None
        self._drag_moved = False
        self._font = QFont()
        self._font.setPointSize(11)

    # ------------------------------------------------------------- public API
    def set_mesh(self, mesh: Mesh | None, centers: np.ndarray | None = None) -> None:
        """Show ``mesh``; ``centers`` are the bodies' centres of mass in scene units."""
        self._mesh = mesh
        count = 0 if centers is None else len(centers)
        self._centers = np.zeros((count, 3), dtype=np.float32) if centers is None else np.asarray(centers, dtype=np.float32)
        self._motion = np.zeros((count, 3, 4), dtype=np.float32)
        self._motion[:, 2, :3] = self._centers
        self._peak = 1.0
        self._animated = False
        self._caption = ""
        self._trajectory = None
        self._group_table = None
        self._selected = -1
        self._hover = -1
        self._hover_group = -1
        self._pick_state = None
        self.play(False)
        if mesh is not None and mesh.count:
            self._extent = mesh.extent
            self.reset_view()
        self._dirty = {"mesh": True, "motion": True, "colors": True, "groups": True}
        self.update()

    def set_color_mode(self, mode: str) -> None:
        """``"layer"``, ``"body"``, or ``"group"`` (colours from :meth:`set_group_colors`)."""
        if mode != self._color_mode:
            self._color_mode = mode
            self.update()

    def set_group_colors(self, table: np.ndarray | None) -> None:
        """RGBA per paint group, mixed over the layer colour by alpha; shown in colour mode ``"group"``."""
        self._group_table = None if table is None else np.ascontiguousarray(table, dtype=np.float32)
        self._dirty["groups"] = True
        self.update()

    def set_pick_mode(self, mode: str) -> None:
        """What a click and the pointer pick: ``"body"`` or ``"group"`` (an electrode or a net's beams)."""
        if mode != self._pick_mode:
            self._pick_mode = mode
            self._pick_state = None
            self._hover = -1
            self._hover_group = -1
            self.update()

    def set_visible(self, kind: int, on: bool) -> None:
        self._visible[kind] = on
        self.update()

    def set_selected(self, body: int) -> None:
        if body != self._selected:
            self._selected = body
            self.update()

    def set_motion(self, u: np.ndarray | None, theta: np.ndarray | None, animated: bool, caption: str = "") -> None:
        """Show the rigid motion ``u`` (m) and ``theta`` (rad) per body, animated or held static."""
        self._trajectory = None
        self._motion[:, 0, :] = 0.0
        self._motion[:, 1, :] = 0.0
        if u is not None and theta is not None and self._mesh is not None:
            n = min(len(self._motion), len(u))
            self._motion[:n, 0, :3] = np.asarray(u[:n], dtype=np.float64) * SCENE_UNIT
            self._motion[:n, 1, :3] = np.asarray(theta[:n], dtype=np.float64)
            self._peak = motion_peak(self._mesh, u[:n], theta[:n])
            self._animated = animated
            self._caption = caption
        else:
            self._animated = False
            self._caption = ""
        self._dirty["motion"] = True
        if self._animated:
            self._phase = math.pi / 2
            self.play(True)
        else:
            self.play(False)
        self.update()

    def clear_motion(self) -> None:
        self.set_motion(None, None, False)

    def set_trajectory(self, times: np.ndarray, u: np.ndarray, theta: np.ndarray, caption: str = "") -> None:
        """Play back a transient: translation ``u`` (m) and rotation ``theta`` (rad) of every
        body at every entry of ``times`` (s), shapes (times, bodies, 3)."""
        if self._mesh is None or len(times) < 2:
            self.clear_motion()
            return
        bodies = min(len(self._motion), u.shape[1])
        self._trajectory = (
            np.asarray(times, dtype=np.float64),
            (np.asarray(u, dtype=np.float64)[:, :bodies] * SCENE_UNIT).astype(np.float32),
            np.asarray(theta, dtype=np.float64)[:, :bodies].astype(np.float32),
        )
        self._peak = trajectory_peak(self._mesh, u[:, :bodies], theta[:, :bodies])
        self._animated = True
        self._caption = caption or "Transient"
        self._time = float(times[0])
        self._apply_time()
        self.play(True)
        self.update()

    def has_trajectory(self) -> bool:
        return self._trajectory is not None

    def set_time(self, moment: float) -> None:
        """Show the trajectory at simulation time ``moment`` (clamped to the run)."""
        if self._trajectory is None:
            return
        times = self._trajectory[0]
        self._time = float(np.clip(moment, times[0], times[-1]))
        self._apply_time()
        self.update()

    def time(self) -> float:
        return self._time

    def set_rate(self, rate: float) -> None:
        """Playback speed of a trajectory; at 1 one pass takes ``LOOP_SECONDS``."""
        self._rate = float(rate)

    def set_true_scale(self, on: bool) -> None:
        """Play a trajectory at its real size instead of exaggerated to the amplitude setting."""
        self._true_scale = bool(on)
        self.update()

    def exaggeration(self) -> float:
        """Factor the shown motion of a trajectory is magnified by."""
        if self._trajectory is None or self._true_scale:
            return 1.0
        return self._amplitude * self._extent / self._peak

    def set_caption(self, text: str) -> None:
        """Replace the caption of the motion on show without restarting it."""
        if self._caption and text:
            self._caption = text
            self.update()

    def _apply_time(self) -> None:
        times, u, theta = self._trajectory
        position = float(np.interp(self._time, times, np.arange(len(times))))
        low = min(int(position), len(times) - 2)
        weight = position - low
        bodies = u.shape[1]
        self._motion[:bodies, 0, :3] = (1.0 - weight) * u[low] + weight * u[low + 1]
        self._motion[:bodies, 1, :3] = (1.0 - weight) * theta[low] + weight * theta[low + 1]
        self._dirty["motion"] = True
        self.timeChanged.emit(self._time)

    def has_motion(self) -> bool:
        return bool(self._caption)

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

    def capture_frame(self, phase: float):
        """Render the current motion at ``phase`` (radians) and return the framebuffer image.

        For a trajectory the phase runs through the whole transient: 0 is its first time and
        2π its last.
        """
        previous = (self._phase, self._exporting, self._time)
        self._phase = phase
        self._exporting = True
        try:
            if self._trajectory is not None:
                times = self._trajectory[0]
                self._time = float(times[0] + (times[-1] - times[0]) * phase / (2.0 * math.pi))
                self._apply_time()
            return self.grabFramebuffer()
        finally:
            self._phase, self._exporting, self._time = previous
            if self._trajectory is not None:
                self._apply_time()

    def frame_size(self) -> tuple[int, int]:
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr), int(self.height() * dpr)

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
        names = ("u_proj", "u_view", "u_scale", "u_motion", "u_body_colors", "u_group_colors", "u_kind_colors",
                 "u_color_mode", "u_selected", "u_hover", "u_pick", "u_select_color")
        self._uniforms = {name: GL.glGetUniformLocation(program, name) for name in names}
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        for location, size, offset in ((0, 3, 0), (1, 3, 12), (2, 2, 24), (3, 1, 32), (4, 1, 36), (5, 1, 40)):
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(location, size, GL.GL_FLOAT, GL.GL_FALSE, STRIDE, ctypes.c_void_p(offset))
        GL.glBindVertexArray(0)
        GL.glUseProgram(program)
        colors = np.array(theme.GL_KIND_COLORS, dtype=np.float32)
        GL.glUniform3fv(self._uniforms["u_kind_colors"], 5, colors)
        GL.glUniform1i(self._uniforms["u_motion"], 0)
        GL.glUniform1i(self._uniforms["u_body_colors"], 1)
        GL.glUniform1i(self._uniforms["u_group_colors"], 2)
        GL.glUseProgram(0)
        self._gl_ready = True
        self._dirty = {"mesh": True, "motion": True, "colors": True, "groups": True}

    def _upload(self) -> None:
        mesh = self._mesh
        if self._dirty["mesh"]:
            self._dirty["mesh"] = False
            GL.glBindVertexArray(self._vao)
            if mesh is not None and mesh.count:
                vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
                indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
                GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
            GL.glBindVertexArray(0)
        if self._dirty["colors"]:
            self._dirty["colors"] = False
            count = len(self._centers)
            table = np.ones((max(count, 1), 1, 4), dtype=np.float32)
            for index in range(count):
                table[index, 0, :3] = theme.body_rgb(index)
            self._textures["colors"] = _texture_2d(1, max(count, 1), table, self._textures["colors"])
        if self._dirty["groups"]:
            self._dirty["groups"] = False
            table = self._group_table
            count = 0 if table is None else len(table)
            rows = max(-(-count // GROUP_TEXTURE_WIDTH), 1)
            data = np.zeros((rows * GROUP_TEXTURE_WIDTH, 4), dtype=np.float32)
            if count:
                data[:count] = table
            self._textures["groups"] = _texture_2d(GROUP_TEXTURE_WIDTH, rows, data.reshape(rows, GROUP_TEXTURE_WIDTH, 4), self._textures["groups"])
        if self._dirty["motion"]:
            self._dirty["motion"] = False
            count = len(self._motion)
            data = self._motion if count else np.zeros((1, 3, 4), dtype=np.float32)
            self._textures["motion"] = _texture_2d(3, max(count, 1), data, self._textures["motion"])

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
        if not self._caption:
            return 0.0
        if self._trajectory is not None:
            return self.exaggeration()
        amplitude = self._amplitude * self._extent / self._peak
        if self._animated or self._exporting:
            return amplitude * math.sin(self._phase)
        return amplitude

    def _tick(self) -> None:
        dt = self._clock.restart() / 1000.0
        if self._trajectory is not None:
            times = self._trajectory[0]
            span = float(times[-1] - times[0])
            self._time = float(times[0] + (self._time - times[0] + dt * self._rate * span / LOOP_SECONDS) % span)
            self._apply_time()
        else:
            self._phase = (self._phase + 2.0 * math.pi * self._speed * dt) % (2.0 * math.pi)
        self.update()

    # --------------------------------------------------------------- painting
    def paintGL(self) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.beginNativePainting()
        dpr = self.devicePixelRatioF()
        width, height = int(self.width() * dpr), int(self.height() * dpr)
        if any(self._dirty.values()):
            self._upload()
        has_mesh = self._mesh is not None and self._mesh.count > 0
        if has_mesh and (self._click_request is not None or self._hover_request is not None):
            self._resolve_picks(width, height, dpr)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(*theme.GL_BACKGROUND, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if has_mesh:
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
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures["motion"])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures["colors"])
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures["groups"])
        GL.glActiveTexture(GL.GL_TEXTURE0)
        by_group = self._pick_mode == "group"
        GL.glUniformMatrix4fv(self._uniforms["u_proj"], 1, GL.GL_TRUE, proj)
        GL.glUniformMatrix4fv(self._uniforms["u_view"], 1, GL.GL_TRUE, view)
        GL.glUniform1f(self._uniforms["u_scale"], self._current_scale())
        GL.glUniform1f(self._uniforms["u_color_mode"], {"body": 1.0, "group": 2.0}.get(self._color_mode, 0.0))
        # Picking by group shows selection and hover through the colour table instead.
        GL.glUniform1f(self._uniforms["u_selected"], -1.0 if by_group else float(self._selected))
        GL.glUniform1f(self._uniforms["u_hover"], -1.0 if by_group else float(self._hover))
        GL.glUniform1f(self._uniforms["u_pick"], (2.0 if by_group else 1.0) if pick else 0.0)
        GL.glUniform3f(self._uniforms["u_select_color"], *theme.SELECTION_RGB)
        GL.glBindVertexArray(self._vao)
        assert self._mesh is not None
        for kind in DRAW_ORDER:
            if not self._visible.get(kind, True):
                continue
            start, count = self._mesh.ranges.get(kind, (0, 0))
            if count:
                GL.glDrawElements(GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(start * 4))

    def _resolve_picks(self, width: int, height: int, dpr: float) -> None:
        if self._pick_fbo is None or self._pick_fbo.size() != QSize(width, height):
            self._pick_fbo = QOpenGLFramebufferObject(QSize(width, height), QOpenGLFramebufferObject.Attachment.Depth)
            self._pick_state = None
        fbo = self._pick_fbo
        fbo.bind()
        state = (
            id(self._mesh),
            width,
            height,
            self._distance,
            self._yaw,
            self._pitch,
            tuple(self._target),
            self._current_scale(),
            self._time if self._trajectory is not None else None,
            self._pick_mode,
            tuple(sorted(kind for kind, on in self._visible.items() if on)),
        )
        if state != self._pick_state:
            # The ID buffer only needs re-rendering when the camera, the motion, or what is shown changed.
            GL.glViewport(0, 0, width, height)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._draw(width, height, pick=True)
            GL.glFinish()
            self._pick_state = state

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
            ident = int(pixel[0]) + 256 * int(pixel[1]) + 65536 * int(pixel[2])
            return ident - 1

        by_group = self._pick_mode == "group"
        if self._hover_request is not None:
            picked = read(self._hover_request)
            self._hover_request = None
            if by_group:
                if picked != self._hover_group:
                    self._hover_group = picked
                    self.groupHovered.emit(picked)
            elif picked != self._hover:
                self._hover = picked
                self.hovered.emit(picked)
        if self._click_request is not None:
            picked = read(self._click_request)
            self._click_request = None
            if by_group:
                self.groupClicked.emit(picked, self._click_modifiers)
            else:
                self.bodyClicked.emit(picked)
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
        rect = self.rect().adjusted(12, 10, -12, -10)
        if self._caption:
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, self._caption)
            if self._animated and not self._exporting:
                painter.setPen(QColor(theme.MUTED))
                state = "playing" if self._playing else "paused"
                if self._trajectory is None:
                    detail = f"amplitude {self._amplitude * 100:.1f}% of extent"
                elif self._true_scale:
                    detail = "true scale"
                else:
                    detail = f"motion × {self.exaggeration():,.3g}"
                painter.drawText(rect.adjusted(0, 18, 0, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"{state} · {detail}")

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._drag_last = event.position()
        self._drag_button = event.button()
        self._drag_moved = False
        self.setFocus()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        pos = event.position()
        if self._drag_last is None:
            if not self._playing:  # picking re-renders the ID buffer; skip it while animating
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
            self._click_modifiers = int(event.modifiers().value)
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
        if self._hover_group != -1:
            self._hover_group = -1
            self.groupHovered.emit(-1)
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
