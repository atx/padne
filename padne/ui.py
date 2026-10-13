#!/usr/bin/env python3

import contextlib
import logging
import math
import numpy as np
import sys
import warnings
import OpenGL.GL as gl
import time

from typing import Optional, Callable, ClassVar
from dataclasses import dataclass, field

import abc
from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt, Signal, Slot, QRect
from PySide6.QtGui import QSurfaceFormat, QPainter, QPen, QColor, QAction, QActionGroup
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QToolBar, QToolButton, QMenu, QMessageBox, QLineEdit, QSlider, QCheckBox
)
from PySide6.QtCore import QTimer

import shapely.geometry
from scipy.spatial import cKDTree

from . import mesh, solver, units, colormaps, parallel
from .context import stage_timer

# In this file, there are some cursed naming conventions due to the fact
# that we are mixing Python and Qt together.
# Ad hoc rules:
# * objects that inherit from QObject should use Qt naming conventions for methods
#   * member variables should use snake_case anyway
# * other object should normally follow PEP 8


log = logging.getLogger(__name__)


# Define shader source code
VERTEX_SHADER_MESH = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in float color;
out float frag_value;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 0.0, 1.0);
    frag_value = color;
}
"""

FRAGMENT_SHADER_MESH = """
#version 330 core
in float frag_value;
out vec4 out_color;

#define COLOR_COUNT 256
uniform float v_max = 1.0;
uniform float v_min = 0.0;
uniform vec3 color_map[COLOR_COUNT];

void main() {
    float t = (frag_value - v_min) / (v_max - v_min);
    float rescaled = t * COLOR_COUNT;
    int idx = clamp(int(rescaled), 0, COLOR_COUNT - 1);

    out_color = vec4(color_map[idx], 1.0);
}
"""

VERTEX_SHADER_DISCONNECTED = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in float color;  // We still have the color attribute but ignore it
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_DISCONNECTED = """
#version 330 core
out vec4 out_color;

void main() {
    // Render disconnected copper in a subdued gray
    out_color = vec4(0.1, 0.1, 0.1, 1.0);
}
"""

VERTEX_SHADER_EDGES = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;
out vec3 frag_color;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 0.0, 1.0);
    frag_color = color;
}
"""

FRAGMENT_SHADER_EDGES = """
#version 330 core
in vec3 frag_color;
out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
"""

VERTEX_SHADER_POINTS = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 vertex_color;
out vec3 frag_color;
uniform mat4 mvp;
uniform float point_size = 5.0;

void main() {
    gl_Position = mvp * vec4(position, 0.0, 1.0);
    gl_PointSize = point_size;
    frag_color = vertex_color; // Pass color to fragment shader
}
"""

FRAGMENT_SHADER_POINTS = """
#version 330 core
in vec3 frag_color; // Input color from vertex shader
out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
"""


# Small meshes spend most of their time allocating/wrapping arrays under the
# GIL. Even batches of them regress with threads. Only dispatch substantial
# native work, and preserve input order when adding the small serial results.
_UI_PARALLEL_MIN_SIZE = 32768


def _map_ui_work(fn, items, sizes):
    large = [i for i, size in enumerate(sizes) if size >= _UI_PARALLEL_MIN_SIZE]
    if parallel.config().jobs == 1 or len(large) < 2:
        return [fn(item) for item in items]
    results = dict(zip(large, parallel.thread_map(fn, [items[i] for i in large])))
    return [results[i] if i in results else fn(item) for i, item in enumerate(items)]


@dataclass
class BaseSpatialIndex:
    tree: Optional[cKDTree]
    values: np.ndarray
    shape: shapely.geometry.MultiPolygon

    @classmethod
    def _extract_points_and_values(cls, layer_solution: solver.LayerSolution) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("This method should be implemented in subclasses")

    @classmethod
    def from_layer_data(cls, layer: solver.problem.Layer,
                        layer_solution: solver.LayerSolution,
                        value_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
                        ) -> "BaseSpatialIndex":
        vertices, values = cls._extract_points_and_values(layer_solution)

        if value_transform is not None:
            values = value_transform(values)

        # cKDTree is not happy with empty arrays, so we just return an empty index
        if len(vertices) == 0:
            return cls(None, values, layer.shape)

        tree = cKDTree(vertices)

        return cls(tree, values, layer.shape)

    def query_nearest(self, x: float, y: float) -> Optional[float]:
        """Find nearest value to given coordinates."""
        if not self.tree:
            return None

        # Check if point is within layer geometry
        point = shapely.geometry.Point(x, y)
        if not self.shape.contains(point):
            return None

        # Query nearest vertex
        distance, index = self.tree.query([x, y])

        # Return value if distance is reasonable
        if distance < float('inf'):
            return float(self.values[index])

        return None


class VertexSpatialIndex(BaseSpatialIndex):
    """Spatial index for fast vertex value lookups within a layer."""

    @classmethod
    def _extract_points_and_values(cls, layer_solution: solver.LayerSolution) -> tuple[np.ndarray, np.ndarray]:
        """Extract vertex coordinates and their values from the layer solution."""
        pairs = list(zip(layer_solution.meshes, layer_solution.potentials))
        if not pairs:
            return np.empty((0, 2)), np.empty(0)
        return (np.concatenate([msh.positions() for msh, _ in pairs]),
                np.concatenate([values.values for _, values in pairs]))


class FaceSpatialIndex(BaseSpatialIndex):
    """Spatial index for fast face value lookups within a layer."""

    @classmethod
    def _extract_points_and_values(cls, layer_solution: solver.LayerSolution) -> tuple[np.ndarray, np.ndarray]:
        """Extract face coordinates and their values from the layer solution."""
        pairs = list(zip(layer_solution.meshes, layer_solution.power_densities))
        if not pairs:
            return np.empty((0, 2)), np.empty(0)
        return (np.concatenate([
                    msh.positions()[msh.triangles()].mean(axis=1)
                    for msh, _ in pairs]),
                np.concatenate([values.values for _, values in pairs]))


class BaseTool(abc.ABC):
    def __init__(self, mesh_viewer: 'MeshViewer', tool_manager: 'ToolManager'):
        self.mesh_viewer = mesh_viewer
        self.tool_manager = tool_manager

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the display name of the tool."""

    @property
    @abc.abstractmethod
    def status_tip(self) -> str:
        """Returns the status tip for the tool."""

    @property
    def shortcut(self) -> Optional[tuple[Qt.Key, Qt.KeyboardModifier]]:
        return None

    def on_shortcut_press(self, world_point: mesh.Point):
        """Handles a shortcut press event."""
        pass

    def on_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        """Handles a click event on the mesh."""
        pass

    def on_screen_drag(self, dx: float, dy: float, event: QtGui.QMouseEvent):
        """Handles a screen drag event."""
        pass


class PanTool(BaseTool):

    @property
    def name(self) -> str:
        return "Pan"

    @property
    def status_tip(self) -> str:
        return "Pan and zoom the view"

    def on_screen_drag(self, dx: float, dy: float, event: QtGui.QMouseEvent):
        if event.buttons() & (Qt.LeftButton | Qt.MiddleButton):
            self.mesh_viewer.panViewByScreenDelta(dx, dy)


class SetMinValueTool(PanTool):

    @property
    def name(self) -> str:
        return "Min"

    @property
    def status_tip(self) -> str:
        return "Set minimum value for color scale from cursor (M)"

    @property
    def shortcut(self):
        return (Qt.Key_M, Qt.NoModifier)

    def on_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mesh_viewer.setMinValueFromWorldPoint(world_point)

    def on_shortcut_press(self, world_point: mesh.Point):
        log.debug(f"SetMinValueTool: Shortcut pressed at {world_point}")
        self.mesh_viewer.setMinValueFromWorldPoint(world_point)


class SetMaxValueTool(PanTool):
    @property
    def name(self) -> str:
        return "Max"

    @property
    def status_tip(self) -> str:
        return "Set maximum value for color scale from cursor (Shift+M)"

    @property
    def shortcut(self):
        return (Qt.Key_M, Qt.ShiftModifier)

    def on_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mesh_viewer.setMaxValueFromWorldPoint(world_point)

    def on_shortcut_press(self, world_point: mesh.Point):
        log.debug(f"SetMaxValueTool: Shortcut pressed at {world_point}")
        self.mesh_viewer.setMaxValueFromWorldPoint(world_point)


class ToolManager(QtCore.QObject):
    def __init__(self, mesh_viewer: 'MeshViewer', parent=None):
        super().__init__(parent)
        self.mesh_viewer = mesh_viewer

        self.available_tools: list[BaseTool] = [
            PanTool(self.mesh_viewer, self),
            SetMinValueTool(self.mesh_viewer, self),
            SetMaxValueTool(self.mesh_viewer, self)
        ]

        self.active_tool: BaseTool = self.available_tools[0]

    @Slot(BaseTool)
    def activate_tool(self, tool_to_activate: BaseTool):
        log.debug(f"Activating Tool: {tool_to_activate.name}")
        self.active_tool = tool_to_activate

    @Slot(object, QtGui.QMouseEvent)
    def handle_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        log.debug(f"ToolManager: Mesh clicked at {world_point} with tool {self.active_tool.name}. Button: {event.button()}")
        self.active_tool.on_mesh_click(world_point, event)

    @Slot(float, float, QtGui.QMouseEvent)
    def handle_screen_drag(self, dx: float, dy: float, event: QtGui.QMouseEvent):
        log.debug(f"ToolManager: Screen dragged by ({dx}, {dy}) with tool {self.active_tool.name}. Buttons: {event.buttons()}")
        self.active_tool.on_screen_drag(dx, dy, event)

    @Slot(mesh.Point, int, Qt.KeyboardModifiers)
    def handle_key_press_in_mesh(self,
                                 world_point: mesh.Point,
                                 key: Qt.Key,
                                 modifiers: Qt.KeyboardModifiers):
        for tool in self.available_tools:
            shortcut_def = tool.shortcut
            if not shortcut_def:
                continue
            shortcut_key, shortcut_modifier = shortcut_def
            if key == shortcut_key and modifiers == shortcut_modifier:
                log.debug(f"Shortcut {key} with modifiers {modifiers} matched for tool {tool.name} at {world_point}")
                tool.on_shortcut_press(world_point)


class AppToolBar(QToolBar):
    def __init__(self, tool_manager: ToolManager, mesh_viewer: 'MeshViewer', parent=None):
        super().__init__("Main Toolbar", parent)
        self.tool_manager = tool_manager
        self.mesh_viewer = mesh_viewer
        self._setupActions()

    def _setupActions(self):
        self._setupToolActions()
        self.addSeparator()
        self._setupViewMenu()
        self.addSeparator()
        self._setupLayersButton()
        self._setupModesButton()
        self.addSeparator()
        self._setupViewControlActions()

    def _makeAction(self, text: str, tip: str, slot, checked: Optional[bool] = None) -> QAction:
        action = QAction(text, self)
        action.setStatusTip(tip)
        action.setToolTip(tip)
        if checked is not None:
            action.setCheckable(True)
            action.setChecked(checked)
        action.triggered.connect(slot)
        return action

    def _setupToolActions(self):
        """Setup tool selection actions."""
        tool_action_group = QActionGroup(self)
        tool_action_group.setExclusive(True)

        for tool_instance in self.tool_manager.available_tools:
            action = self._makeAction(
                tool_instance.name, tool_instance.status_tip,
                lambda checked, t=tool_instance: self.tool_manager.activate_tool(t),
                checked=self.tool_manager.active_tool == tool_instance,
            )
            self.addAction(action)
            tool_action_group.addAction(action)

    def _setupViewMenu(self):
        """Setup the View menu with visibility toggles."""
        # Create the "View" QToolButton
        view_menu_button = QToolButton(self)
        view_menu_button.setText("View")
        view_menu_button.setToolTip("View options")
        view_menu_button.setPopupMode(QToolButton.InstantPopup)

        # Create the menu that will be shown by the QToolButton
        view_menu = QMenu(view_menu_button)

        self.show_edges_action = self._makeAction(
            "Show Edges", "Toggle visibility of mesh edges (E)",
            self.mesh_viewer.setEdgesVisible, checked=True)
        self.show_outline_action = self._makeAction(
            "Show Outline", "Toggle visibility of mesh outline (Shift+E)",
            self.mesh_viewer.setOutlineVisible, checked=True)
        self.show_connection_points_action = self._makeAction(
            "Show Connection Points", "Toggle visibility of connection points (C)",
            self.mesh_viewer.setConnectionPointsVisible, checked=True)
        view_menu.addAction(self.show_edges_action)
        view_menu.addAction(self.show_outline_action)
        view_menu.addAction(self.show_connection_points_action)

        # Set the menu for the QToolButton
        view_menu_button.setMenu(view_menu)
        self.addWidget(view_menu_button)

        # Connect visibility changes to sync menu checkboxes
        self.mesh_viewer.visibilityChanged.connect(self._syncViewMenuCheckboxes)

    def _setupLayersButton(self):
        """Setup the Layers dropdown button."""
        self.layers_button = QToolButton(self)
        self.layers_button.setText("Layers")
        self.layers_button.setToolTip("Select active layer (V/Shift+V)")
        self.layers_button.setPopupMode(QToolButton.InstantPopup)

        self.layers_menu = QMenu(self.layers_button)
        self.layer_action_group = QActionGroup(self)
        self.layer_action_group.setExclusive(True)

        self.layers_button.setMenu(self.layers_menu)
        self.addWidget(self.layers_button)

    def _setupModesButton(self):
        """Setup the Modes dropdown button."""
        self.modes_button = QToolButton(self)
        self.modes_button.setText("Modes")
        self.modes_button.setToolTip("Select rendering mode")
        self.modes_button.setPopupMode(QToolButton.InstantPopup)

        self.modes_menu = QMenu(self.modes_button)
        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)

        # Create mode actions statically based on mesh_viewer.modes
        for mode in self.mesh_viewer.modes:
            action = QAction(mode.name, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, name=mode.name: self.mesh_viewer.setCurrentModeByName(name)
            )
            self.modes_menu.addAction(action)
            self.mode_action_group.addAction(action)

        # Set initial mode as checked
        initial_mode = self.mesh_viewer.modes[self.mesh_viewer.current_mode_index]
        for action in self.mode_action_group.actions():
            if action.text() == initial_mode.name:
                action.setChecked(True)
                break

        self.modes_button.setMenu(self.modes_menu)
        self.addWidget(self.modes_button)

    def _setupViewControlActions(self):
        """Setup view control actions (Reset View, Full Scale)."""
        self.addAction(self._makeAction(
            "Reset View", "Reset view to fit all content (F)", self.mesh_viewer.autoscaleXY))
        self.addAction(self._makeAction(
            "Full Scale", "Reset color scale to full range (A)", self.mesh_viewer.autoscaleValue))

    def _syncViewMenuCheckboxes(self):
        """Sync View menu checkbox states with MeshViewer visibility states."""
        self.show_edges_action.setChecked(self.mesh_viewer.edges_visible)
        self.show_outline_action.setChecked(self.mesh_viewer.outline_visible)
        self.show_connection_points_action.setChecked(self.mesh_viewer.connection_points_visible)

    @Slot(list)
    def updateLayerSelectionMenu(self, layer_names: list[str]):
        self.layers_menu.clear()
        # Clear actions from group. QActionGroup doesn't have a clear method.
        for action in self.layer_action_group.actions():
            self.layer_action_group.removeAction(action)
            # QActionGroup does not take ownership, so actions are not deleted.
            # If they were added to the menu, menu.clear() handles their deletion.

        for layer_name in layer_names:
            action = QAction(layer_name, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, name=layer_name: self.mesh_viewer.setCurrentLayerByName(name)
            )
            self.layers_menu.addAction(action)
            self.layer_action_group.addAction(action)

        # Ensure the currently active layer in mesh_viewer is checked
        if self.mesh_viewer.visible_layers and \
                self.mesh_viewer.current_layer_index < len(self.mesh_viewer.visible_layers):
            active_layer_name = self.mesh_viewer.current_layer_name
            self.updateActiveLayerInMenu(active_layer_name)

    @Slot(str)
    def updateActiveLayerInMenu(self, active_layer_name: str):
        for action in self.layers_menu.actions():
            if action.text() == active_layer_name:
                action.setChecked(True)
                break

    @Slot(str)
    def updateActiveModeInMenu(self, active_mode_name: str):
        """Update which mode action is checked."""
        for action in self.mode_action_group.actions():
            action.setChecked(action.text() == active_mode_name)


def _create_vao(vertices: np.ndarray, colors: np.ndarray, color_components: int) -> int:
    """Create a VAO with a 2D vertex VBO (attribute 0) and a color VBO (attribute 1)."""
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    vbo_vertices = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_vertices)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(0)

    vbo_colors = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_colors)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, colors, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(1, color_components, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(1)

    return vao


@dataclass
class ShaderProgram:
    shader_program: QOpenGLShaderProgram

    @classmethod
    def from_source(cls, vertex_source, fragment_source):
        shader_program = QOpenGLShaderProgram()
        if not (shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_source)
                and shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_source)
                and shader_program.link()):
            raise RuntimeError(f"Failed to build shader program: {shader_program.log()}")

        return cls(shader_program)

    def set_mvp(self, mvp: np.ndarray):
        gl.glUniformMatrix4fv(self.shader_program.uniformLocation("mvp"), 1, gl.GL_TRUE, mvp.flatten())

    @contextlib.contextmanager
    def use(self):
        self.shader_program.bind()
        yield
        self.shader_program.release()


@dataclass
class RenderedMesh:
    vao_triangles: int
    triangle_count: int
    vao_edges: int
    edge_count: int
    vao_boundary: int
    boundary_count: int

    @dataclass(frozen=True)
    class PreparedData:
        triangle_vertices: np.ndarray[np.float32]
        triangle_colors: np.ndarray[np.float32]
        edge_vertices: np.ndarray[np.float32]
        edge_colors: np.ndarray[np.float32]
        boundary_vertices: np.ndarray[np.float32]
        boundary_colors: np.ndarray[np.float32]

    @classmethod
    def from_prepared_data(cls, data: 'RenderedMesh.PreparedData') -> 'RenderedMesh':
        vao_triangles = _create_vao(data.triangle_vertices, data.triangle_colors, 1)
        vao_edges = _create_vao(data.edge_vertices, data.edge_colors, 3)
        vao_boundary = _create_vao(data.boundary_vertices, data.boundary_colors, 3)
        gl.glBindVertexArray(0)

        return cls(vao_triangles,
                   len(data.triangle_vertices) // 2,
                   vao_edges,
                   len(data.edge_vertices) // 2,
                   vao_boundary,
                   len(data.boundary_vertices) // 2)

    @dataclass(frozen=True)
    class PreparedGeometry:
        """Read-only CPU geometry shared by the two field rendering modes."""
        triangles: np.ndarray
        triangle_vertices: np.ndarray
        edge_vertices: np.ndarray
        edge_colors: np.ndarray
        boundary_vertices: np.ndarray
        boundary_colors: np.ndarray

        def with_colors(self, colors: np.ndarray) -> 'RenderedMesh.PreparedData':
            colors.setflags(write=False)
            return RenderedMesh.PreparedData(
                self.triangle_vertices, colors,
                self.edge_vertices, self.edge_colors,
                self.boundary_vertices, self.boundary_colors,
            )

    @classmethod
    def prepare_geometry(cls, msh: mesh.Mesh) -> 'RenderedMesh.PreparedGeometry':
        """Extract face-ordered geometry without Python half-edge traversal.

        Interior edges deliberately occur twice, matching the original drawing
        order. Boundary edges include both the exterior and any hole rims.
        Native extraction releases the GIL; each task owns its output arrays.
        """
        triangles = msh.triangles()
        corners = msh.positions()[triangles].astype(np.float32)
        boundary = msh.triangle_boundary_mask().reshape(-1).astype(bool)
        edges = np.stack((corners, np.roll(corners, -1, axis=1)), axis=2).reshape(-1, 4)
        edge_vertices = edges[~boundary].reshape(-1)
        boundary_vertices = edges[boundary].reshape(-1)
        geometry = cls.PreparedGeometry(
            triangles, corners.reshape(-1),
            edge_vertices, np.full(edge_vertices.size // 4 * 6, 0.9, dtype=np.float32),
            boundary_vertices, np.full(boundary_vertices.size // 4 * 6, 0.9, dtype=np.float32),
        )
        for array in (geometry.triangles, geometry.triangle_vertices,
                      geometry.edge_vertices, geometry.edge_colors,
                      geometry.boundary_vertices, geometry.boundary_colors):
            array.setflags(write=False)
        return geometry

    @classmethod
    def prepare_zero_form(cls, msh: mesh.Mesh, values: mesh.ZeroForm,
                          geometry: Optional['RenderedMesh.PreparedGeometry'] = None
                          ) -> 'RenderedMesh.PreparedData':
        if geometry is None:
            geometry = cls.prepare_geometry(msh)
        colors = values.values[geometry.triangles].astype(np.float32).reshape(-1)
        return geometry.with_colors(colors)

    @classmethod
    def prepare_two_form(cls, msh: mesh.Mesh, values: mesh.TwoForm,
                         geometry: Optional['RenderedMesh.PreparedGeometry'] = None
                         ) -> 'RenderedMesh.PreparedData':
        if geometry is None:
            geometry = cls.prepare_geometry(msh)
        colors = np.repeat(values.values.astype(np.float32), 3)
        return geometry.with_colors(colors)

    def render_triangles(self):
        gl.glBindVertexArray(self.vao_triangles)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.triangle_count)

    def render_edges(self):
        gl.glBindVertexArray(self.vao_edges)
        gl.glDrawArrays(gl.GL_LINES, 0, self.edge_count)

    def render_boundary(self):
        gl.glBindVertexArray(self.vao_boundary)
        gl.glDrawArrays(gl.GL_LINES, 0, self.boundary_count)

    @classmethod
    def prepare_mesh(cls, msh: mesh.Mesh,
                     geometry: Optional['RenderedMesh.PreparedGeometry'] = None
                     ) -> "RenderedMesh.PreparedData":
        """Prepare disconnected copper, with a zero-valued field."""
        if geometry is None:
            geometry = cls.prepare_geometry(msh)
        return geometry.with_colors(np.zeros(len(msh.faces) * 3, dtype=np.float32))


@dataclass
class RenderedPoints:
    vao_points: int
    point_count: int

    @classmethod
    def from_points(cls, points_data: list[tuple[tuple[float, float], tuple[float, float, float]]]):
        coords = np.array([p for p, _ in points_data], dtype=np.float32).reshape(-1)
        colors = np.array([c for _, c in points_data], dtype=np.float32).reshape(-1)
        vao_points = _create_vao(coords, colors, 3)
        gl.glBindVertexArray(0)
        return cls(vao_points, len(points_data))

    def render(self):
        gl.glBindVertexArray(self.vao_points)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.point_count)


class SliderScale(abc.ABC):
    """Maps a 0..1 slider fraction to a value in [lo, hi] and back."""

    @abc.abstractmethod
    def value_at(self, fraction: float, lo: float, hi: float) -> float:
        ...

    @abc.abstractmethod
    def fraction_of(self, value: float, lo: float, hi: float) -> float:
        ...


@dataclass(frozen=True)
class LinearScale(SliderScale):

    def value_at(self, fraction: float, lo: float, hi: float) -> float:
        return lo + (hi - lo) * fraction

    def fraction_of(self, value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return (value - lo) / (hi - lo)


@dataclass(frozen=True)
class LogScale(SliderScale):
    """
    Fraction 0 is exactly `lo`, (0, 1] spans `decades` decades below `hi`
    logarithmically, so one slider step is a constant ratio.
    """
    decades: float = 4.0

    def value_at(self, fraction: float, lo: float, hi: float) -> float:
        if fraction <= 0.0:
            return lo
        return hi * 10.0 ** (-self.decades * (1.0 - fraction))

    def fraction_of(self, value: float, lo: float, hi: float) -> float:
        if value <= hi * 10.0 ** -self.decades:
            return 0.0
        return 1.0 + math.log10(value / hi) / self.decades


@dataclass(frozen=True)
class ColorScaleState:
    """Snapshot of a rendering mode's color scale, as shown by ColorScaleWidget."""
    min_value: float
    max_value: float
    slider_lo: float
    slider_hi: float
    slider_scale: SliderScale
    capped: bool
    cap_percentile: float
    unit: str
    color_map: colormaps.UniformColorMap


class MeshViewer(QOpenGLWidget):

    @dataclass
    class BaseRenderingMode:
        unit: str
        name: str
        color_map: colormaps.UniformColorMap
        slider_scale: ClassVar[SliderScale] = LinearScale()
        cap_percentile: ClassVar[float] = 99.9

        # The color range, always kept within slider_range
        min_value: float = 0.0
        max_value: float = 1.0
        # When capped, the top of the slider range is percentile_max
        capped: bool = False
        # Global (all layers) data range and cap_percentile of the values,
        # fixed once the solution is set
        data_range: tuple[float, float] = (0.0, 1.0)
        percentile_max: float = 1.0
        solution: Optional[solver.Solution] = None
        spatial_indices: dict[str, BaseSpatialIndex] = field(default_factory=dict)

        rendered_meshes: dict[str, list[RenderedMesh]] = field(default_factory=dict)
        disconnected_rendered_meshes: dict[str, list[RenderedMesh]] = field(default_factory=dict)

        _prepared_rendered_meshes: dict[str, list[RenderedMesh.PreparedData]] = \
            field(default_factory=dict)
        _prepared_disconnected_rendered_meshes: dict[str, list[RenderedMesh.PreparedData]] = \
            field(default_factory=dict)

        def _compute_min_max(self) -> tuple[float, float]:
            """Compute min and max values across all spatial indices."""
            min_val = float('inf')
            max_val = float('-inf')

            for index in self.spatial_indices.values():
                if len(index.values) == 0:
                    continue
                min_val = min(min_val, float(index.values.min()))
                max_val = max(max_val, float(index.values.max()))

            if min_val == float('inf'):
                min_val, max_val = 0.0, 1.0
            elif min_val == max_val:
                min_val, max_val = min_val, min_val + 1.0

            return min_val, max_val

        def _compute_percentile_max(self) -> float:
            """The cap_percentile of all values, falling back to the data max."""
            values = [index.values for index in self.spatial_indices.values()]
            data_min, data_max = self.data_range
            if not any(len(v) for v in values):
                return data_max
            cap = float(np.percentile(np.concatenate(values), self.cap_percentile))
            # A cap at the bottom of the range would collapse the slider range
            return cap if cap > data_min else data_max

        @property
        def slider_range(self) -> tuple[float, float]:
            data_min, data_max = self.data_range
            return data_min, self.percentile_max if self.capped else data_max

        def _clamp_to_slider_range(self, value: float) -> float:
            lo, hi = self.slider_range
            return min(max(value, lo), hi)

        def autoscale(self) -> None:
            """Set the color range to the whole slider range."""
            self.min_value, self.max_value = self.slider_range

        def set_min(self, value: float) -> None:
            """Set the color minimum, pushing the maximum up if needed."""
            self.min_value = self._clamp_to_slider_range(value)
            self.max_value = max(self.max_value, self.min_value)

        def set_max(self, value: float) -> None:
            """Set the color maximum, pushing the minimum down if needed."""
            self.max_value = self._clamp_to_slider_range(value)
            self.min_value = min(self.min_value, self.max_value)

        def set_capped(self, capped: bool) -> None:
            """Toggle the cap; only clamps the color range, never rescales it."""
            self.capped = capped
            self.min_value = self._clamp_to_slider_range(self.min_value)
            self.max_value = self._clamp_to_slider_range(self.max_value)

        def color_scale_state(self) -> ColorScaleState:
            return ColorScaleState(
                min_value=self.min_value,
                max_value=self.max_value,
                slider_lo=self.slider_range[0],
                slider_hi=self.slider_range[1],
                slider_scale=self.slider_scale,
                capped=self.capped,
                cap_percentile=self.cap_percentile,
                unit=self.unit,
                color_map=self.color_map,
            )

        def _build_spatial_indices(self):
            raise NotImplementedError("This method should be implemented in subclasses")

        @abc.abstractmethod
        def set_solution(self, solution: solver.Solution,
                         geometries: Optional[dict[mesh.Mesh, RenderedMesh.PreparedGeometry]] = None,
                         disconnected: Optional[dict[str, list[RenderedMesh.PreparedData]]] = None):
            """Initialize this mode with solution data (build indices + meshes)."""
            self.solution = solution
            self.spatial_indices.clear()

            # We have to delay this until the OpenGL context is properly initialized.
            self.rendered_meshes.clear()
            self.disconnected_rendered_meshes.clear()

            # The OpenGL-independent part can be done right away
            self._prepared_rendered_meshes = {
                layer.name: self._prepare_rendered_meshes_for_layer(layer.name, geometries)
                for layer in self.solution.problem.layers
            }
            self._prepared_disconnected_rendered_meshes = disconnected if disconnected is not None else {
                layer.name: self._prepare_disconnected_rendered_meshes_for_layer(layer.name)
                for layer in self.solution.problem.layers
            }

            self._build_spatial_indices()
            self.data_range = self._compute_min_max()
            self.percentile_max = self._compute_percentile_max()
            self.autoscale()

        def _prepare_rendered_meshes_for_layer(self, layer_name, geometries=None) -> list[RenderedMesh.PreparedData]:
            """Create RenderedMesh objects for a specific layer."""
            raise NotImplementedError("This method should be implemented in subclasses")

        def pick_nearest_value(self, layer_name: str, world_x: float, world_y: float) -> Optional[float]:
            """Pick value at coordinates using spatial index."""
            if layer_name in self.spatial_indices:
                return self.spatial_indices[layer_name].query_nearest(world_x, world_y)
            return None

        def get_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Get pre-built rendered meshes for a layer."""
            if layer_name in self.rendered_meshes:
                # This means that everything is ready for rendering
                return self.rendered_meshes[layer_name]

            # The prepared data has not yet been inserted into the OpenGL
            # context. Which is something we have to do in our main thread,
            # meaning here.

            # Also note: This function is not only called from the main thread,
            # it is also called from paintGL. This means that it is also
            # properly holding the OpenGL context.
            # If this changes, it is necessary to figure something out
            # with mesh_viewer.makeCurrent() and doneCurrent().

            prepared_meshes = self._prepared_rendered_meshes[layer_name]
            self.rendered_meshes[layer_name] = [
                RenderedMesh.from_prepared_data(data)
                for data in prepared_meshes
            ]

            return self.rendered_meshes[layer_name]

        def _prepare_disconnected_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh.PreparedData]:
            """Create RenderedMesh objects for disconnected copper on a specific layer."""
            rendered_meshes = []
            if not self.solution:
                return rendered_meshes

            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                for msh in layer_solution.disconnected_meshes:
                    rendered_meshes.append(RenderedMesh.prepare_mesh(msh))
            return rendered_meshes

        def get_disconnected_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Get pre-built disconnected rendered meshes for a layer."""
            # TODO: Deduplicate this with get_rendered_meshes_for_layer
            if layer_name in self.disconnected_rendered_meshes:
                # This means that everything is ready for rendering
                return self.disconnected_rendered_meshes[layer_name]

            # The prepared data has not yet been inserted into the OpenGL
            # context. Which is something we have to do in our main thread,
            # meaning here.

            prepared_meshes = self._prepared_disconnected_rendered_meshes[layer_name]
            self.disconnected_rendered_meshes[layer_name] = [
                RenderedMesh.from_prepared_data(data)
                for data in prepared_meshes
            ]
            return self.disconnected_rendered_meshes[layer_name]

    @dataclass
    class VoltageRenderingMode(BaseRenderingMode):
        unit: str = "V"
        name: str = "Potential"
        color_map: colormaps.UniformColorMap = colormaps.PLASMA

        def _build_spatial_indices(self):
            """Build spatial indices for fast vertex lookups."""
            layers = list(zip(self.solution.problem.layers, self.solution.layer_solutions))
            indices = _map_ui_work(
                lambda item: VertexSpatialIndex.from_layer_data(*item), layers,
                [sum(len(msh.vertices) for msh in ls.meshes) for _, ls in layers],
            )
            self.spatial_indices = {layer.name: index for (layer, _), index in zip(layers, indices)}

        def _prepare_rendered_meshes_for_layer(self, layer_name: str, geometries=None) -> list[RenderedMesh.PreparedData]:
            """Create RenderedMesh objects for a specific layer."""
            prepared_meshes = []
            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                for msh, values in zip(layer_solution.meshes, layer_solution.potentials):
                    prepared_meshes.append(RenderedMesh.prepare_zero_form(
                        msh, values, None if geometries is None else geometries[msh]))

            return prepared_meshes

    @dataclass
    class PowerDensityRenderingMode(BaseRenderingMode):
        unit: str = "W/mm²"
        name: str = "Power Density"
        color_map: colormaps.UniformColorMap = colormaps.INFERNO
        slider_scale: ClassVar[SliderScale] = LogScale()
        # The raw maximum is usually a near-singular hot spot
        capped: bool = True

        def _compute_min_max(self) -> tuple[float, float]:
            _, max_val = super()._compute_min_max()
            # Usually, we would get a value that is very close to zero anyway,
            # this makes it a bit prettier
            return 0.0, max_val

        def _build_spatial_indices(self):
            """Build spatial indices for fast face lookups."""
            layers = list(zip(self.solution.problem.layers, self.solution.layer_solutions))
            indices = _map_ui_work(
                lambda item: FaceSpatialIndex.from_layer_data(*item), layers,
                [sum(len(msh.faces) for msh in ls.meshes) for _, ls in layers],
            )
            self.spatial_indices = {layer.name: index for (layer, _), index in zip(layers, indices)}

        def _prepare_rendered_meshes_for_layer(self, layer_name: str, geometries=None) -> list[RenderedMesh.PreparedData]:
            """Create RenderedMesh objects for a specific layer."""
            prepared_meshes = []
            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                for msh, values in zip(layer_solution.meshes, layer_solution.power_densities):
                    prepared_meshes.append(RenderedMesh.prepare_two_form(
                        msh, values, None if geometries is None else geometries[msh]))
            return prepared_meshes

    @dataclass
    class CurrentDensityRenderingMode(PowerDensityRenderingMode):
        unit: str = "A/mm²"
        name: str = "Current Density"
        color_map: colormaps.UniformColorMap = colormaps.VIRIDIS

        # Set in set_solution: if any layer lacks a thickness, the whole mode
        # falls back to sheet current (A/mm) so the single/global unit is honest.
        _sheet_fallback: bool = False

        def set_solution(self, solution: solver.Solution,
                         geometries: Optional[dict[mesh.Mesh, RenderedMesh.PreparedGeometry]] = None,
                         disconnected: Optional[dict[str, list[RenderedMesh.PreparedData]]] = None):
            missing = [
                layer.name for layer in solution.problem.layers
                if layer.thickness is None
            ]
            self._sheet_fallback = bool(missing)
            if self._sheet_fallback:
                self.unit = "A/mm"
                log.warning(
                    "Layer(s) %s have no thickness; current density is shown as "
                    "sheet current (A/mm) for the whole mode.", ", ".join(missing))
            else:
                self.unit = "A/mm²"
            super().set_solution(solution, geometries, disconnected)

        def _current_density_factor(self, layer: solver.problem.Layer) -> float:
            """
            Factor c such that |J| = c * sqrt(power_density).

            With thickness: |J| = sqrt(conductance * P) / thickness (A/mm²).
            Sheet fallback (any layer missing thickness): sqrt(conductance * P)
            (A/mm).
            """
            if self._sheet_fallback:
                return float(np.sqrt(layer.conductance))
            return float(np.sqrt(layer.conductance) / layer.thickness)

        def _build_spatial_indices(self):
            """Build spatial indices for fast face lookups."""
            def build(item):
                layer, layer_solution = item
                factor = self._current_density_factor(layer)
                return FaceSpatialIndex.from_layer_data(
                    layer, layer_solution,
                    value_transform=lambda values: factor * np.sqrt(np.maximum(values, 0.0)))

            layers = list(zip(self.solution.problem.layers, self.solution.layer_solutions))
            indices = _map_ui_work(
                build, layers,
                [sum(len(msh.faces) for msh in ls.meshes) for _, ls in layers],
            )
            self.spatial_indices = {layer.name: index for (layer, _), index in zip(layers, indices)}

        def _prepare_rendered_meshes_for_layer(self, layer_name: str, geometries=None) -> list[RenderedMesh.PreparedData]:
            """Create RenderedMesh objects for a specific layer."""
            prepared_meshes = []
            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                factor = self._current_density_factor(layer)
                for msh, values in zip(layer_solution.meshes, layer_solution.power_densities):
                    current_density = mesh.TwoForm(msh)
                    current_density.values[:] = factor * np.sqrt(np.maximum(0.0, values.values))
                    prepared_meshes.append(RenderedMesh.prepare_two_form(
                        msh, current_density, None if geometries is None else geometries[msh]))
            return prepared_meshes

    # Signal to notify when the color scale (range, cap, unit, map) changes
    colorScaleChanged = Signal(object)  # object is ColorScaleState
    # Signal to notify when the current layer changes
    currentLayerChanged = Signal(str)
    # Signal to notify when the list of available layers changes
    availableLayersChanged = Signal(list)
    # Signal to notify when the current rendering mode changes
    currentModeChanged = Signal(str)
    # Signals for tools
    meshClicked = Signal(mesh.Point, QtGui.QMouseEvent)
    screenDragged = Signal(float, float, QtGui.QMouseEvent)
    keyPressedInMesh = Signal(mesh.Point, int, Qt.KeyboardModifiers)
    # Signal for mouse position and probed value
    mousePositionChanged = Signal(mesh.Point, object)  # object can be float or None
    # Signal for visibility changes
    visibilityChanged = Signal()

    @classmethod
    def default_modes(cls) -> list["MeshViewer.BaseRenderingMode"]:
        """The rendering modes in menu order, constructible without a widget."""
        return [
            cls.VoltageRenderingMode(),
            cls.PowerDensityRenderingMode(),
            cls.CurrentDensityRenderingMode(),
        ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.solution: None | solver.Solution = None
        self.rendered_connection_points: dict[str, RenderedPoints] = {}
        self.connection_points_visible: bool = True

        # Rendering modes and current mode tracking
        self.modes = self.default_modes()
        self.current_mode_index = 0  # Start with voltage mode

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.needs_initial_autoscale = False
        self.last_mouse_screen_pos: Optional[QtCore.QPointF] = None
        self.last_mouse_position_change_ts = time.monotonic()
        self.setMouseTracking(True)

        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Layer management
        self.current_layer_index = 0
        self.visible_layers = []  # Will hold names of layers in order

        # OpenGL objects
        self.mesh_shader = None
        self.disconnected_shader = None
        self.edge_shader = None
        self.points_shader = None

        self.edges_visible = True
        self.outline_visible = True

    @property
    def current_rendering_mode(self) -> BaseRenderingMode:
        """Get the currently active rendering mode."""
        return self.modes[self.current_mode_index]

    @property
    def current_layer_name(self) -> str:
        """Get the name of the currently active layer."""
        return self.visible_layers[self.current_layer_index]

    @property
    def aspect_ratio(self) -> float:
        """Get the current aspect ratio (width/height)."""
        return self.width() / self.height() if self.height() > 0 else 1.0

    def _compute_mesh_bounds(self) -> tuple[float, float, float, float] | None:
        """Bounding box (min_x, min_y, max_x, max_y) of all meshes, or None if there are no vertices."""
        if not self.solution:
            return None

        positions = [msh.positions() for ls in self.solution.layer_solutions for msh in ls.meshes]
        pts = np.concatenate(positions) if positions else np.empty((0, 2))
        if len(pts) == 0:
            return None

        (min_x, min_y), (max_x, max_y) = pts.min(axis=0), pts.max(axis=0)
        return float(min_x), float(min_y), float(max_x), float(max_y)

    def _getNearestValue(self, world_x: float, world_y: float) -> Optional[float]:
        """
        Find the value closest to the specified world coordinates using the current rendering mode.

        Uses spatial indexing for fast O(log n) lookups.

        Args:
            world_x: X-coordinate in world space
            world_y: Y-coordinate in world space

        Returns:
            The value at the nearest point, or None if no values are found
            or if the point is outside the layer's geometries.
        """
        if not self.solution or not self.visible_layers:
            return None

        current_layer_name = self.current_layer_name

        # Delegate to current rendering mode
        return self.current_rendering_mode.pick_nearest_value(current_layer_name, world_x, world_y)

    def autoscaleValue(self) -> None:
        """
        Automatically adjust the min/max values for color scaling using the current rendering mode.
        """
        if not self.solution or not self.solution.layer_solutions:
            return  # Nothing to scale if no solution is loaded

        self.current_rendering_mode.autoscale()
        self._emitColorScale()
        self.update()

    def _emitColorScale(self) -> None:
        self.colorScaleChanged.emit(self.current_rendering_mode.color_scale_state())

    @Slot(bool)
    def setCapped(self, capped: bool) -> None:
        """Toggle the percentile cap of the current rendering mode."""
        self.current_rendering_mode.set_capped(capped)
        self._emitColorScale()
        self.update()

    def autoscaleXY(self) -> None:
        """
        Automatically adjust the offset and scale to fit all meshes in the view.
        Sets the view to display all meshes with a small margin around them.
        """
        bounds = self._compute_mesh_bounds()
        if bounds is None:
            return  # No vertices found

        min_x, min_y, max_x, max_y = bounds

        # Calculate center point and dimensions
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        solution_width = max_x - min_x
        solution_height = max_y - min_y

        if solution_width < 1e-6 or solution_height < 1e-6:
            log.warning("Mesh bounds are suspiciously small, refusing to autoscale.")
            return

        # Set view center (negative offset to move view)
        self.offset_x = -center_x
        self.offset_y = -center_y

        margin_factor = 0.9
        aspect = self.aspect_ratio

        # Okay, so:
        # * the y axis is scaled to 1.0
        # * the x axis is scaled to however much is `aspect`
        scale_for_height = 2.0 / solution_height
        scale_for_width = 2.0 * aspect / solution_width
        self.scale = min(scale_for_height, scale_for_width) * margin_factor

        # Refresh the display
        self.update()

    def setSolution(self, prepared: "PreparedUI"):
        """Install a solution prepared by `prepare_ui_data`."""
        self.modes = prepared.modes
        self.solution = prepared.solution

        # Initialize the list of layers from the solution
        self.visible_layers = [layer.name for layer in self.solution.problem.layers]
        self.current_layer_index = 0

        if self.visible_layers:
            self.availableLayersChanged.emit(self.visible_layers)
            self.currentLayerChanged.emit(self.current_layer_name)

        current_mode = self.current_rendering_mode
        self.currentModeChanged.emit(current_mode.name)
        self._emitColorScale()

        # We can't just do autoscaleXY here, since we may be in some
        # semi-initialized state and the widget may not have reached a valid
        # size yet.
        # Unfortunately, the resizeGL method gets called repeatedly with
        # random sizes until it converges to the final size, so we can't
        # even rely on the first call being reliable.
        self.needs_initial_autoscale = True

        if self.mesh_shader is not None:
            self.setupConnectionPointsData()

        self.update()

    def setupConnectionPointsData(self) -> None:
        """Set up the connection points data for rendering."""
        self.rendered_connection_points.clear()

        if not self.solution or not self.solution.problem:
            return

        # Store list of (coordinates, color) tuples for each layer
        points_by_layer: dict[str, list[tuple[tuple[float, float], tuple[float, float, float]]]] = {}

        for network in self.solution.problem.networks:
            # Determine color based on whether the network has a source
            if network.has_source:
                color = (1.0, 0.0, 0.0)  # Red for networks with a source
            else:
                color = (0.5, 0.5, 0.5)  # Gray for networks without a source

            for connection in network.connections:
                layer_name = connection.layer.name
                point_coords = (connection.point.x, connection.point.y)

                if layer_name not in points_by_layer:
                    points_by_layer[layer_name] = []

                # Append a tuple of (coordinates, color)
                points_by_layer[layer_name].append((point_coords, color))

        for layer_name, collected_points_data in points_by_layer.items():
            # We want to render the _red_ points over the gray ones,
            # so we draw them _last_. This is a hack to order them, it
            # depends on the fact that (1.0, 0.0, 0.0) > (0.5, 0.5, 0.5)
            # _This will break if the colors change!_
            # Deduplicate exact (coordinates, colour) duplicates, e.g. the pad
            # centre connection also appearing as a region vertex.
            seen = set()
            deduplicated = []
            for point_data in collected_points_data:
                if point_data in seen:
                    continue
                seen.add(point_data)
                deduplicated.append(point_data)
            collected_points_data = deduplicated
            collected_points_data.sort(key=lambda x: x[1])
            # Pass the list of (coordinates, color) tuples
            rendered_obj = RenderedPoints.from_points(collected_points_data)
            self.rendered_connection_points[layer_name] = rendered_obj

    def initializeGL(self) -> None:
        """Initialize OpenGL settings."""
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Background
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Create and compile shaders
        self.mesh_shader = ShaderProgram.from_source(
            VERTEX_SHADER_MESH, FRAGMENT_SHADER_MESH
        )

        self.disconnected_shader = ShaderProgram.from_source(
            VERTEX_SHADER_DISCONNECTED, FRAGMENT_SHADER_DISCONNECTED
        )

        self.edge_shader = ShaderProgram.from_source(
            VERTEX_SHADER_EDGES, FRAGMENT_SHADER_EDGES
        )

        self.points_shader = ShaderProgram.from_source(
            VERTEX_SHADER_POINTS, FRAGMENT_SHADER_POINTS
        )

        # Set the color map uniform
        self._updateShaderColorMap()

        # If meshes are already set, setup the mesh data
        if self.solution:
            self.setupConnectionPointsData()

    def resizeGL(self, width: int, height: int) -> None:
        """Handle window resizing."""
        gl.glViewport(0, 0, width, height)

        # Perform autoscaling on resize until user manually interacts
        if self.needs_initial_autoscale and width > 0 and height > 0:
            self.autoscaleXY()
            self.update()

    def _computeMVP(self) -> np.ndarray:
        aspect = self.aspect_ratio

        # Create a 2D orthographic projection matrix
        ortho_scale = 1.0 / self.scale
        left = -ortho_scale * aspect
        right = ortho_scale * aspect
        bottom = -ortho_scale
        top = ortho_scale
        near = -1.0
        far = 1.0

        # Define the matrix components with Y-axis flip
        # Change the row for Y projection to add the flip
        proj_matrix = np.array([
            [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, -2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],  # Note the negative sign here
            [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Create translation matrix
        trans_matrix = np.array([
            [1, 0, 0, self.offset_x],
            [0, 1, 0, self.offset_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Combine matrices: projection * translation
        return np.dot(proj_matrix, trans_matrix)

    def _renderMeshTriangles(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]) -> None:
        """Renders the triangles of the meshes for the current layer."""
        with self.mesh_shader.use():
            self.mesh_shader.set_mvp(mvp)

            # Set the min/max value uniforms for color scaling
            gl.glUniform1f(
                self.mesh_shader.shader_program.uniformLocation("v_min"),
                self.current_rendering_mode.min_value
            )
            gl.glUniform1f(
                self.mesh_shader.shader_program.uniformLocation("v_max"),
                self.current_rendering_mode.max_value
            )

            # Draw triangles for current layer only
            for rmesh in rendered_mesh_list:
                rmesh.render_triangles()

    def _renderMeshEdges(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]) -> None:
        """Renders the edges of the meshes for the current layer."""
        if not self.edges_visible:
            return

        with self.edge_shader.use():
            self.edge_shader.set_mvp(mvp)

            # Draw edges for current layer only
            for rmesh in rendered_mesh_list:
                rmesh.render_edges()

    def _renderBoundaryEdges(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]) -> None:
        """Renders the boundary edges of the meshes for the current layer."""
        if not self.outline_visible:
            return

        with self.edge_shader.use():
            self.edge_shader.set_mvp(mvp)

            # Draw boundary edges for current layer only
            for rmesh in rendered_mesh_list:
                rmesh.render_boundary()

    def _renderDisconnectedMeshes(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]) -> None:
        """Renders disconnected copper meshes in gray."""
        if not rendered_mesh_list:
            return

        with self.disconnected_shader.use():
            self.disconnected_shader.set_mvp(mvp)

            # Draw triangles for disconnected meshes
            for rmesh in rendered_mesh_list:
                rmesh.render_triangles()

            # Notably we do not render edges for disconnected meshes.
            # They provide no additional information and look messy anyway...
            # It can be useful to render them when debugging the code

    def _renderConnectionPoints(self, mvp: np.ndarray, rendered_points_obj: RenderedPoints) -> None:
        """Renders the connection points for the current layer."""
        if not self.connection_points_visible:
            return

        with self.points_shader.use():
            self.points_shader.set_mvp(mvp)

            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            rendered_points_obj.render()
            gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)

    def paintGL(self) -> None:
        """Render the mesh using shaders."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not self.mesh_shader or not self.visible_layers:
            log.debug("No shader program or meshes to render")
            return

        mvp = self._computeMVP()

        # Get current layer name
        current_layer_name = self.current_layer_name

        # Render disconnected copper first (behind everything else)
        disconnected_mesh_list = \
            self.current_rendering_mode.get_disconnected_rendered_meshes_for_layer(current_layer_name)
        self._renderDisconnectedMeshes(mvp, disconnected_mesh_list)

        # Get rendered meshes directly from current mode
        current_layer_mesh_list = \
            self.current_rendering_mode.get_rendered_meshes_for_layer(current_layer_name)
        self._renderMeshTriangles(mvp, current_layer_mesh_list)
        self._renderMeshEdges(mvp, current_layer_mesh_list)
        self._renderBoundaryEdges(mvp, current_layer_mesh_list)

        # Do note that layers that do not have any rendered points are not
        # represented in the rendered_connection_points dict.
        if current_layer_name in self.rendered_connection_points:
            rendered_points = self.rendered_connection_points[current_layer_name]
            self._renderConnectionPoints(mvp, rendered_points)

        gl.glBindVertexArray(0)

    def _screenToWorld(self, screen_pos: QtCore.QPointF) -> mesh.Point:
        if self.width() <= 0 or self.height() <= 0:
            log.warning("MeshViewer not sized, cannot convert screen to world coordinates.")
            return mesh.Point(0.0, 0.0)

        viewport_x = screen_pos.x()
        viewport_y = screen_pos.y()

        # Convert to normalized device coordinates (NDC)
        # Qt screen Y is 0 at top, self.height() at bottom.
        # This calculation results in NDC where Y is -1 at top, 1 at bottom.
        ndc_x = (2.0 * viewport_x / self.width()) - 1.0
        ndc_y = (2.0 * viewport_y / self.height()) - 1.0

        aspect = self.aspect_ratio

        # Inverse transformation based on the projection and view matrices
        world_x = (ndc_x * aspect / self.scale) - self.offset_x
        world_y = (ndc_y / self.scale) - self.offset_y

        return mesh.Point(world_x, world_y)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press events."""
        if event.buttons() & (Qt.LeftButton | Qt.MiddleButton):
            self.last_mouse_screen_pos = event.position()

        self.setFocus()  # Ensure the widget gets focus when clicked

        # Emit meshClicked signal regardless of button for potential right-click tools etc.
        # The tool itself can check event.button()
        world_point = self._screenToWorld(event.position())
        self.meshClicked.emit(world_point, event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse movement."""
        if event.buttons() & (Qt.LeftButton | Qt.MiddleButton) and self.last_mouse_screen_pos is not None:
            delta = event.position() - self.last_mouse_screen_pos
            dx = float(delta.x())
            dy = float(delta.y())

            self.screenDragged.emit(dx, dy, event)

            self.last_mouse_screen_pos = event.position()

        if time.monotonic() - self.last_mouse_position_change_ts < 0.1:
            # Avoid too frequent updates
            return

        # Always emit mouse position for status bar updates
        world_point = self._screenToWorld(event.position())
        value = self._getNearestValue(world_point.x, world_point.y)
        self.mousePositionChanged.emit(world_point, value)
        self.last_mouse_position_change_ts = time.monotonic()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() in (Qt.LeftButton, Qt.MiddleButton) and self.last_mouse_screen_pos is not None:
            # TODO: Potentially emit a clickReleased signal if tools need it
            # Clear drag state
            self.last_mouse_screen_pos = None

    def panViewByScreenDelta(self, dx_screen: float, dy_screen: float) -> None:
        """
        Pans the view based on a screen delta.

        Args:
            dx_screen: Change in x screen coordinate.
            dy_screen: Change in y screen coordinate.
        """
        if self.width() <= 0 or self.height() <= 0:
            return

        # User manually panned - disable automatic scaling
        self.needs_initial_autoscale = False

        aspect = self.aspect_ratio

        # Convert screen delta to world delta
        # Horizontal movement (adjusted for aspect ratio)
        dx_world = (dx_screen / self.width()) * (2.0 / self.scale) * aspect

        # Vertical movement (note: Qt's y axis points down, OpenGL Y-axis was flipped in projection)
        # A positive dy_screen (mouse down) should result in a positive dy_world (content moves down)
        dy_world = (dy_screen / self.height()) * (2.0 / self.scale)

        self.offset_x += dx_world
        self.offset_y += dy_world
        self.update()

    def _zoomToScreenPoint(self, screen_x: float, screen_y: float, zoom_by: float) -> None:
        """
        Zoom the viewport, keeping the specified screen point fixed.

        Args:
            screen_x: X coordinate in screen/widget pixels
            screen_y: Y coordinate in screen/widget pixels
            zoom_by: Zoom factor to apply (>1 zooms in, <1 zooms out)
        """
        screen_pos = QtCore.QPointF(screen_x, screen_y)
        world_before = self._screenToWorld(screen_pos)

        self.scale *= zoom_by

        world_after = self._screenToWorld(screen_pos)

        # Adjust offset to keep world_before at the same screen position
        self.offset_x += (world_after.x - world_before.x)
        self.offset_y += (world_after.y - world_before.y)

    @Slot(float)
    def setMinValue(self, value: float) -> None:
        """Sets the minimum of the color scale; clamps max upward if needed."""
        self.current_rendering_mode.set_min(value)
        self._emitColorScale()
        self.update()

    @Slot(float)
    def setMaxValue(self, value: float) -> None:
        """Sets the maximum of the color scale; clamps min downward if needed."""
        self.current_rendering_mode.set_max(value)
        self._emitColorScale()
        self.update()

    def setMinValueFromWorldPoint(self, world_point: mesh.Point) -> None:
        """
        Sets the minimum value of the color scale from a world point.
        If the selected value is greater than the current maximum, both min and max
        are set to the selected value.

        Args:
            world_point: The point in world coordinates.
        """
        value = self._getNearestValue(world_point.x, world_point.y)
        if value is None:
            return
        self.setMinValue(value)

    def setMaxValueFromWorldPoint(self, world_point: mesh.Point) -> None:
        """
        Sets the maximum value of the color scale from a world point.
        If the selected value is less than the current minimum, both min and max
        are set to the selected value.

        Args:
            world_point: The point in world coordinates.
        """
        value = self._getNearestValue(world_point.x, world_point.y)
        if value is None:
            return
        self.setMaxValue(value)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle mouse wheel for zooming towards cursor position."""
        # User manually zoomed - disable automatic scaling
        self.needs_initial_autoscale = False

        cursor_pos = event.position()
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self._zoomToScreenPoint(cursor_pos.x(), cursor_pos.y(), zoom_factor)

        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard events."""
        # Get current mouse position in widget coordinates
        screen_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        # Check if mouse is within widget bounds; if not, world_point might be less meaningful
        # but _screenToWorld should still compute a value.
        # Alternatively, could use center of view if mouse is outside. For now, use cursor.
        world_point = self._screenToWorld(screen_pos)

        # Emit signal for ToolManager to handle general shortcuts
        self.keyPressedInMesh.emit(world_point, event.key(), event.modifiers())

        if event.key() == Qt.Key_V:
            direction = -1 if event.modifiers() & Qt.ShiftModifier else 1
            self.switchLayerBy(direction)
        elif event.key() == Qt.Key_E:
            if event.modifiers() & Qt.ShiftModifier:
                self.setOutlineVisible(not self.outline_visible)
            else:
                self.setEdgesVisible(not self.edges_visible)
        elif event.key() == Qt.Key_C:
            self.setConnectionPointsVisible(not self.connection_points_visible)
        elif event.key() == Qt.Key_F:
            self.autoscaleXY()
        elif event.key() == Qt.Key_A:
            self.autoscaleValue()
        else:
            # Allow other key events to be processed if not handled by shortcuts or specific keys
            super().keyPressEvent(event)

    def switchLayerBy(self, direction: int = 1) -> None:
        """Switch to the next or previous layer in the cycle.

        Args:
            direction: 1 for next layer, -1 for previous layer
        """
        if not self.visible_layers:
            return

        # Move to next/previous layer index
        self.current_layer_index = (self.current_layer_index + direction) % len(self.visible_layers)
        current_layer = self.current_layer_name

        # Emit signal with the current layer name
        self.currentLayerChanged.emit(current_layer)

        # Refresh the display
        self.update()

    @Slot(bool)
    def setEdgesVisible(self, visible: bool):
        """Slot to set the visibility of mesh edges."""
        if self.edges_visible == visible:
            return

        self.edges_visible = visible

        # If we're showing edges but outline is hidden, also show the outline
        if visible and not self.outline_visible:
            self.outline_visible = True
            log.debug("Also showing outline since internal edges are being shown")

        log.debug(f"Mesh edges visibility set to: {self.edges_visible}")
        self.visibilityChanged.emit()
        self.update()

    @Slot(bool)
    def setOutlineVisible(self, visible: bool):
        """Slot to set the visibility of outline edges."""
        if self.outline_visible == visible:
            return

        self.outline_visible = visible

        # If we're hiding the outline and edges are visible, also hide the edges
        if not visible and self.edges_visible:
            self.edges_visible = False
            log.debug("Also hiding internal edges since outline is being hidden")

        log.debug(f"Outline visibility set to: {self.outline_visible}")
        self.visibilityChanged.emit()
        self.update()

    @Slot(bool)
    def setConnectionPointsVisible(self, visible: bool):
        """Slot to set the visibility of connection points."""
        if self.connection_points_visible == visible:
            return

        self.connection_points_visible = visible
        log.debug(f"Connection points visibility set to: {self.connection_points_visible}")
        self.visibilityChanged.emit()
        self.update()

    @Slot(str)
    def setCurrentLayerByName(self, layer_name: str):
        """Sets the current layer by its name."""
        self.current_layer_index = self.visible_layers.index(layer_name)
        self.currentLayerChanged.emit(layer_name)
        self.update()

    @Slot(str)
    def setCurrentModeByName(self, mode_name: str):
        """Sets the current rendering mode by its name."""
        for index, mode in enumerate(self.modes):
            if mode.name != mode_name:
                continue

            old_mode_index = self.current_mode_index
            self.current_mode_index = index

            if old_mode_index == index:
                # Note that this is a _return_
                return

            # Update shader color map for new mode
            self._updateShaderColorMap()

            # Emit signals
            self.currentModeChanged.emit(mode.name)
            self._emitColorScale()
            self.update()

    def _updateShaderColorMap(self) -> None:
        """Update the shader color map uniform with the current mode's color map."""
        if not self.mesh_shader:
            return

        current_color_map = self.current_rendering_mode.color_map
        with self.mesh_shader.use():
            color_map_uniform = self.mesh_shader.shader_program.uniformLocation("color_map")
            # Render 256 colors from the color map
            colors = np.array([current_color_map(i / 255)[0:3] for i in range(256)],
                              dtype=np.float32)
            gl.glUniform3fv(color_map_uniform, 256, colors)


class EditableValueLabel(QLabel):
    """A QLabel that turns into a QLineEdit on double-click for in-place value editing."""

    valueEdited = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.unit = ""
        self._editor: Optional[QLineEdit] = None
        self.setCursor(Qt.IBeamCursor)
        self.setToolTip("Double-click to edit")
        self._refreshText()

    def setValue(self, value: float, unit: str) -> None:
        self.value = value
        self.unit = unit
        self._refreshText()

    def _refreshText(self) -> None:
        self.setText(units.Value(self.value, self.unit).pretty_format())

    def _editorText(self) -> str:
        # pretty_format uses "μ" but units.Value.parse only knows "u"; substitute so the
        # pre-filled text round-trips through the parser if the user just hits Enter.
        return units.Value(self.value, self.unit).pretty_format().replace("μ", "u")

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._editor is not None:
            return
        editor = QLineEdit(self._editorText(), self.parentWidget())
        editor.setAlignment(self.alignment())
        editor.setFont(self.font())
        editor.setGeometry(self.geometry())
        editor.selectAll()
        editor.editingFinished.connect(self._commitEditor)
        editor.installEventFilter(self)
        self._editor = editor
        self.hide()
        editor.show()
        editor.setFocus(Qt.MouseFocusReason)

    def eventFilter(self, watched, event):
        if watched is self._editor and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == Qt.Key_Escape:
                self._cancelEditor()
                return True
        return super().eventFilter(watched, event)

    def _commitEditor(self) -> None:
        if self._editor is None:
            return
        text = self._editor.text()
        self._destroyEditor()
        try:
            parsed = units.Value.parse(text)
        except ValueError:
            return
        self.valueEdited.emit(parsed.value)

    def _cancelEditor(self) -> None:
        self._destroyEditor()

    def _destroyEditor(self) -> None:
        if self._editor is None:
            return
        editor = self._editor
        self._editor = None
        # Avoid re-entering _commitEditor when the editor loses focus during teardown.
        editor.editingFinished.disconnect(self._commitEditor)
        editor.removeEventFilter(self)
        editor.deleteLater()
        self.show()


class ColorScaleWidget(QWidget):
    """Color scale with min/max sliders and a percentile cap toggle."""

    # Signal to notify when unit is changed manually
    unitChanged = Signal(str)
    minValueEdited = Signal(float)
    maxValueEdited = Signal(float)
    capToggled = Signal(bool)

    _SLIDER_STEPS = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        # Replaced by MeshViewer.colorScaleChanged once a solution is set
        self.state = ColorScaleState(
            min_value=0.0, max_value=1.0, slider_lo=0.0, slider_hi=1.0,
            slider_scale=LinearScale(), capped=False, cap_percentile=99.9,
            unit="V", color_map=colormaps.PLASMA)

        self.setMinimumWidth(110)
        self.setMinimumHeight(200)

        self.delta_label: Optional[QLabel] = None
        self.max_label: Optional[EditableValueLabel] = None
        self.min_label: Optional[EditableValueLabel] = None
        self.min_slider: Optional[QSlider] = None
        self.max_slider: Optional[QSlider] = None
        self.percentile_checkbox: Optional[QCheckBox] = None

        self.setupUI()

    def setupUI(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)  # Add a little vertical spacing

        # Delta label at the top of the stretch area
        self.delta_label = QLabel(f"Δ = 0 {self.state.unit}")
        self.delta_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.delta_label)

        # This stretch is where we'll paint our gradient
        layout.addStretch(10)

        # Range labels at the bottom: max above arrow above min, each editable.
        smaller_font = self.font()
        smaller_font.setPointSize(smaller_font.pointSize() - 1)

        self.max_label = EditableValueLabel(self)
        self.max_label.setAlignment(Qt.AlignCenter)
        self.max_label.setFont(smaller_font)
        self.max_label.valueEdited.connect(self.maxValueEdited)
        layout.addWidget(self.max_label)

        arrow_label = QLabel("↑")
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setFont(smaller_font)
        layout.addWidget(arrow_label)

        self.min_label = EditableValueLabel(self)
        self.min_label.setAlignment(Qt.AlignCenter)
        self.min_label.setFont(smaller_font)
        self.min_label.valueEdited.connect(self.minValueEdited)
        layout.addWidget(self.min_label)

        self.min_slider = QSlider(Qt.Horizontal, self)
        self.max_slider = QSlider(Qt.Horizontal, self)
        for slider, tip, slot in (
            (self.min_slider, "Minimum color-scale value",
             self._onMinSliderChanged),
            (self.max_slider, "Maximum color-scale value",
             self._onMaxSliderChanged),
        ):
            slider.setRange(0, self._SLIDER_STEPS)
            slider.setToolTip(tip)
            slider.setFocusPolicy(Qt.NoFocus)
            slider.valueChanged.connect(slot)
            layout.addWidget(slider)

        self.percentile_checkbox = QCheckBox(self)
        self.percentile_checkbox.setFocusPolicy(Qt.NoFocus)
        self.percentile_checkbox.toggled.connect(self.capToggled)
        layout.addWidget(self.percentile_checkbox)

        self.setState(self.state)

    @Slot(object)
    def setState(self, state: ColorScaleState) -> None:
        """Show `state`; does not re-emit any edit signals."""
        self.state = state
        self.updateLabels()

        for slider, value in ((self.min_slider, state.min_value),
                              (self.max_slider, state.max_value)):
            fraction = state.slider_scale.fraction_of(value, state.slider_lo, state.slider_hi)
            fraction = min(max(fraction, 0.0), 1.0)
            slider.blockSignals(True)
            slider.setValue(round(fraction * self._SLIDER_STEPS))
            slider.blockSignals(False)

        self.percentile_checkbox.blockSignals(True)
        self.percentile_checkbox.setChecked(state.capped)
        self.percentile_checkbox.blockSignals(False)
        self.percentile_checkbox.setText(f"Cap {state.cap_percentile:g}%")
        self.percentile_checkbox.setToolTip(
            f"Cap the sliders to the {state.cap_percentile:g}th percentile of all layers")

        self.update()

    @Slot(int)
    def _onMinSliderChanged(self, position: int) -> None:
        self.minValueEdited.emit(self._value_for_position(position))

    @Slot(int)
    def _onMaxSliderChanged(self, position: int) -> None:
        self.maxValueEdited.emit(self._value_for_position(position))

    def _value_for_position(self, position: int) -> float:
        return self.state.slider_scale.value_at(position / self._SLIDER_STEPS,
                                                self.state.slider_lo, self.state.slider_hi)

    def updateLabels(self) -> None:
        """Update the delta and range labels."""
        state = self.state
        delta = state.max_value - state.min_value
        delta_str = units.Value(delta, state.unit).pretty_format(decimal_places=2)

        self.delta_label.setText(f"Δ = {delta_str}")
        self.max_label.setValue(state.max_value, state.unit)
        self.min_label.setValue(state.min_value, state.unit)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the color gradient scale."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Find the rectangle where we should draw the gradient
        # This should be between the delta_label and the max_label of the range stack
        content_rect = self.rect()
        top_margin = self.delta_label.y() + self.delta_label.height() + 2  # +2 for spacing
        bottom_margin = self.height() - self.max_label.y() + 2  # +2 for spacing

        # Calculate the gradient bar rectangle centered horizontally
        bar_width = 20
        gradient_height = content_rect.height() - top_margin - bottom_margin
        # Ensure gradient height is not negative if labels overlap somehow
        gradient_height = max(0, gradient_height)

        gradient_rect = QRect(
            content_rect.left() + (content_rect.width() - bar_width) // 2,  # Center horizontally
            top_margin,
            bar_width,
            gradient_height
        )

        # Draw gradient bar border only if height is positive
        if gradient_rect.height() == 0:
            return
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(gradient_rect)

        # Draw the gradient
        for i in range(gradient_rect.height()):
            # Map position to color
            t = 1.0 - (i / gradient_rect.height())
            color = self.state.color_map(t)

            # Convert to QColor
            qcolor = QColor(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255)
            )

            painter.setPen(qcolor)
            painter.drawLine(
                gradient_rect.left() + 1,
                gradient_rect.top() + i,
                gradient_rect.right() - 1,
                gradient_rect.top() + i
            )


@dataclass
class PreparedUI:
    """GL-free UI preparation for a solution (see `prepare_ui_data`)."""

    solution: solver.Solution
    modes: list["MeshViewer.BaseRenderingMode"]


@stage_timer
def prepare_ui_data(solution: solver.Solution) -> PreparedUI:
    """
    Build the GL-free UI data for `solution`: per-mode spatial indices and the
    prepared render arrays.

    CPU-only native/numpy work -- it never touches Qt or the OpenGL context, so it can
    run before the window exists. The GL side (VAO upload, shader compilation)
    still happens later, on the render thread.
    """
    # Mesh geometry is identical for every field mode. Prepare it once, using
    # threads for large meshes, and share only read-only CPU arrays. GL objects remain local
    # to each mode and are created later with the rendering context current.
    meshes = list(dict.fromkeys(
        msh for ls in solution.layer_solutions
        for msh in [*ls.meshes, *ls.disconnected_meshes]
    ))
    geometries = dict(zip(meshes, _map_ui_work(
        RenderedMesh.prepare_geometry, meshes, [len(msh.faces) for msh in meshes],
    )))
    disconnected = {
        layer.name: [RenderedMesh.prepare_mesh(msh, geometries[msh])
                     for msh in ls.disconnected_meshes]
        for layer, ls in zip(solution.problem.layers, solution.layer_solutions)
    }
    modes = MeshViewer.default_modes()
    for mode in modes:
        mode.set_solution(solution, geometries, disconnected)
    return PreparedUI(solution=solution, modes=modes)


class MainWindow(QMainWindow):

    def __init__(self, prepared: PreparedUI,
                 warnings_list: Optional[list[warnings.WarningMessage]] = None):
        super().__init__()

        self.project_file_name = prepared.solution.problem.project_name or "unknown"
        self.warnings_list = warnings_list if warnings_list else []
        self.warnings_shown = False

        # Should be overwritten soon
        self.setWindowTitle("padne")
        self.setGeometry(100, 100, 900, 600)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create the mesh viewer
        self.mesh_viewer = MeshViewer(self)

        # Create ToolManager
        self.tool_manager = ToolManager(self.mesh_viewer, self)

        # Create color scale widget
        self.color_scale = ColorScaleWidget(self)
        self.color_scale.setFixedWidth(120)

        # Add widgets to layout
        main_layout.addWidget(self.mesh_viewer)
        main_layout.addWidget(self.color_scale)

        # Set the main widget as central widget
        self.setCentralWidget(main_widget)

        # Create and add the AppToolBar
        self.app_toolbar = AppToolBar(self.tool_manager, self.mesh_viewer, self)
        self.addToolBar(Qt.TopToolBarArea, self.app_toolbar)

        self._setupStatusBar()
        self._connectSignals()

        self.mesh_viewer.setSolution(prepared)

    def _setupStatusBar(self) -> None:
        # Add status bar widgets with fixed widths
        self.layer_status_label = QLabel("Layer: -")
        self.layer_status_label.setMinimumWidth(120)

        self.x_position_label = QLabel("X: -")
        self.x_position_label.setMinimumWidth(80)

        self.y_position_label = QLabel("Y: -")
        self.y_position_label.setMinimumWidth(80)

        self.value_label = QLabel("?: ?")
        self.value_label.setMinimumWidth(80)

        self.delta_label = QLabel("Δ: ?")
        self.delta_label.setMinimumWidth(80)

        # Add a small spacer at the beginning
        spacer_label = QLabel("  ")  # Small margin
        self.statusBar().addWidget(spacer_label)

        self.statusBar().addWidget(self.layer_status_label)
        self.statusBar().addWidget(QLabel(" | "))  # Separator
        self.statusBar().addWidget(self.x_position_label)
        self.statusBar().addWidget(QLabel(" | "))  # Separator
        self.statusBar().addWidget(self.y_position_label)
        self.statusBar().addWidget(QLabel(" | "))  # Separator
        self.statusBar().addWidget(self.value_label)
        self.statusBar().addWidget(QLabel(" | "))  # Separator
        self.statusBar().addWidget(self.delta_label)

    def _connectSignals(self) -> None:
        # Connect the MeshViewer
        self.mesh_viewer.colorScaleChanged.connect(self.color_scale.setState)
        self.color_scale.minValueEdited.connect(self.mesh_viewer.setMinValue)
        self.color_scale.maxValueEdited.connect(self.mesh_viewer.setMaxValue)
        self.color_scale.capToggled.connect(self.mesh_viewer.setCapped)
        self.mesh_viewer.currentLayerChanged.connect(self.updateCurrentLayer)
        self.mesh_viewer.availableLayersChanged.connect(self.app_toolbar.updateLayerSelectionMenu)
        self.mesh_viewer.currentLayerChanged.connect(self.app_toolbar.updateActiveLayerInMenu)
        self.mesh_viewer.currentModeChanged.connect(self.app_toolbar.updateActiveModeInMenu)

        # Connect the ToolManager
        self.mesh_viewer.meshClicked.connect(self.tool_manager.handle_mesh_click)
        self.mesh_viewer.screenDragged.connect(self.tool_manager.handle_screen_drag)
        self.mesh_viewer.keyPressedInMesh.connect(self.tool_manager.handle_key_press_in_mesh)

        # Connect mouse position updates
        self.mesh_viewer.mousePositionChanged.connect(self.updateMousePosition)

    def updateCurrentLayer(self, layer_name: str) -> None:
        """Update the window title to show the current layer."""
        self.setWindowTitle(f"padne: {self.project_file_name} - {layer_name}")
        self.layer_status_label.setText(f"Layer: {layer_name}")

    @Slot(mesh.Point, object)
    def updateMousePosition(self, world_point: mesh.Point, value):
        """Update status bar with mouse position and value."""
        self.x_position_label.setText(f"X: {world_point.x:.3f}")
        self.y_position_label.setText(f"Y: {world_point.y:.3f}")

        current_unit = self.mesh_viewer.current_rendering_mode.unit
        if value is None:
            self.value_label.setText(f"{current_unit}: ?")
            self.delta_label.setText("Δ: ?")
            return

        value_str = units.Value(value, current_unit).pretty_format(3)
        self.value_label.setText(f"{current_unit}: {value_str}")

        # Calculate delta from the minimum value of the color scale
        delta_value = value - self.mesh_viewer.current_rendering_mode.min_value
        delta_str = units.Value(delta_value, current_unit).pretty_format(3)
        self.delta_label.setText(f"Δ: {delta_str}")

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Override showEvent to display warnings after window is visible."""
        super().showEvent(event)
        if self.warnings_list and not self.warnings_shown:
            self.warnings_shown = True
            # Use QTimer with 0ms to defer until after the window is fully painted
            # --- we want to avoid showing the dialog before the main window is
            # constructed (since it would block the main window from appearing)
            QTimer.singleShot(0, self._show_warnings_dialog)

    def _show_warnings_dialog(self) -> None:
        """Show the warnings dialog."""
        warning_text = "The solver encountered the following warnings:\n\n"
        for idx, warning_msg in enumerate(self.warnings_list, 1):
            warning_text += f"{idx}. {warning_msg.message}\n"

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Solver Warnings")
        msg_box.setText("The solver encountered warnings during execution.")
        msg_box.setDetailedText(warning_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()


def configure_opengl() -> None:
    """Configure OpenGL settings for the application."""
    # Create OpenGL format
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)  # Use OpenGL 3.3
    gl_format.setProfile(QSurfaceFormat.CoreProfile)  # Use core profile
    gl_format.setSamples(4)  # Enable 4x antialiasing
    QSurfaceFormat.setDefaultFormat(gl_format)


def main(prepared: PreparedUI,
         warnings_list: Optional[list[warnings.WarningMessage]] = None) -> int:
    """Main entry point for the UI application.

    `prepared` is the result of `prepare_ui_data`, which the caller runs (and
    times) before the window is created.
    """
    # Configure OpenGL
    configure_opengl()

    app = QApplication(sys.argv)
    window = MainWindow(prepared, warnings_list)

    window.show()
    return app.exec()
