#!/usr/bin/env python3

import contextlib
import logging
import numpy as np
import sys
import OpenGL.GL as gl
import time

from typing import Optional
from dataclasses import dataclass, field

import abc
from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt, Signal, Slot, QRect
from PySide6.QtGui import QSurfaceFormat, QPainter, QPen, QColor, QAction, QActionGroup
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QToolBar, QToolButton, QMenu
)

import shapely.geometry
from scipy.spatial import cKDTree

from . import mesh, solver, units, colormaps

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


@dataclass
class BaseSpatialIndex:
    tree: Optional[cKDTree]
    values: list[float]
    shape: shapely.geometry.MultiPolygon

    @classmethod
    def _extract_points_and_values(cls, layer_solution: solver.LayerSolution):
        raise NotImplementedError("This method should be implemented in subclasses")

    @classmethod
    def from_layer_data(cls, layer: solver.problem.Layer, layer_solution: solver.LayerSolution) -> "BaseSpatialIndex":
        vertices, values = cls._extract_points_and_values(layer_solution)

        # cKDTree is not happy with empty arrays, so we just return an empty index
        if not vertices:
            return cls(None, [], layer.shape)

        vertex_array = np.array(vertices)
        tree = cKDTree(vertex_array)

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
            return self.values[index]

        return None


class VertexSpatialIndex(BaseSpatialIndex):
    """Spatial index for fast vertex value lookups within a layer."""

    @classmethod
    def _extract_points_and_values(cls, layer_solution: solver.LayerSolution):
        """Extract vertex coordinates and their values from the layer solution."""
        all_vertices = []
        all_values = []

        for msh, values in zip(layer_solution.meshes, layer_solution.potentials):
            for vertex in msh.vertices:
                all_vertices.append([vertex.p.x, vertex.p.y])
                all_values.append(values[vertex])

        return all_vertices, all_values


class FaceSpatialIndex(BaseSpatialIndex):
    """Spatial index for fast face value lookups within a layer."""

    @classmethod
    def _extract_points_and_values(cls, layer_solution: solver.LayerSolution):
        """Extract face coordinates and their values from the layer solution."""
        all_faces = []
        all_values = []

        for msh, values in zip(layer_solution.meshes, layer_solution.power_densities):
            for face in msh.faces:
                # Use the centroid of the face as the representative point
                centroid = face.centroid
                all_faces.append([centroid.x, centroid.y])
                all_values.append(values[face])

        return all_faces, all_values


class BaseTool(abc.ABC):
    def __init__(self, mesh_viewer: 'MeshViewer', tool_manager: 'ToolManager'):
        self.mesh_viewer = mesh_viewer
        self.tool_manager = tool_manager

    @property
    def name(self) -> str:
        """Returns the display name of the tool."""
        pass

    @property
    def status_tip(self) -> str:
        """Returns the status tip for the tool."""
        pass

    def on_activate(self):
        """Called when the tool becomes active."""
        pass

    def on_deactivate(self):
        """Called when the tool becomes inactive."""
        pass

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
        if event.buttons() & Qt.LeftButton:
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
            # Optional: Switch back to Pan tool after action
            # self.tool_manager.activate_tool(self.tool_manager.available_tools[0]) # Assuming Pan is first

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
            # Optional: Switch back to Pan tool after action
            # self.tool_manager.activate_tool(self.tool_manager.available_tools[0]) # Assuming Pan is first

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

        # Activate the first tool by default, but don't call on_activate yet
        # as the tool might not be fully ready (e.g. UI elements)
        # on_activate will be called by the first explicit activate_tool call
        self.active_tool: Optional[BaseTool] = self.available_tools[0]

    @Slot(BaseTool)
    def activate_tool(self, tool_to_activate: Optional[BaseTool]):
        if self.active_tool == tool_to_activate:
            return

        # At the moment, there should always be an active tool we are switching
        # away from. But let's be safe and check.
        if self.active_tool:
            log.debug(f"Deactivating Tool: {self.active_tool.name}")
            self.active_tool.on_deactivate()

        self.active_tool = tool_to_activate

        if self.active_tool:
            log.debug(f"Activating Tool: {self.active_tool.name}")
            self.active_tool.on_activate()

    @Slot(object, QtGui.QMouseEvent)
    def handle_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        if not self.active_tool:
            return

        log.debug(f"ToolManager: Mesh clicked at {world_point} with tool {self.active_tool.name}. Button: {event.button()}")
        self.active_tool.on_mesh_click(world_point, event)

    @Slot(float, float, QtGui.QMouseEvent)
    def handle_screen_drag(self, dx: float, dy: float, event: QtGui.QMouseEvent):
        if not self.active_tool:
            return

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

    def _setupToolActions(self):
        """Setup tool selection actions."""
        tool_action_group = QActionGroup(self)
        tool_action_group.setExclusive(True)

        for tool_instance in self.tool_manager.available_tools:
            action = QAction(tool_instance.name, self)
            action.setStatusTip(tool_instance.status_tip)
            action.setToolTip(tool_instance.status_tip)
            action.setCheckable(True)

            action.triggered.connect(
                lambda checked, t=tool_instance: self.tool_manager.activate_tool(t)
            )

            self.addAction(action)
            tool_action_group.addAction(action)

            # Set the default tool as checked
            if self.tool_manager.active_tool == tool_instance:
                action.setChecked(True)

    def _setupViewMenu(self):
        """Setup the View menu with visibility toggles."""
        # Create the "View" QToolButton
        view_menu_button = QToolButton(self)
        view_menu_button.setText("View")
        view_menu_button.setToolTip("View options")
        view_menu_button.setPopupMode(QToolButton.InstantPopup)

        # Create the menu that will be shown by the QToolButton
        view_menu = QMenu(view_menu_button)

        # Create "Show Edges" action for the menu
        show_edges_action_in_menu = QAction("Show Edges", self)
        show_edges_action_in_menu.setStatusTip("Toggle visibility of mesh edges (E)")
        show_edges_action_in_menu.setToolTip("Toggle visibility of mesh edges (E)")
        show_edges_action_in_menu.setCheckable(True)
        show_edges_action_in_menu.setChecked(True)  # Default to visible
        show_edges_action_in_menu.triggered.connect(self.mesh_viewer.setEdgesVisible)
        view_menu.addAction(show_edges_action_in_menu)

        # Create "Show Outline" action for the menu
        show_outline_action_in_menu = QAction("Show Outline", self)
        show_outline_action_in_menu.setStatusTip("Toggle visibility of mesh outline (Shift+E)")
        show_outline_action_in_menu.setToolTip("Toggle visibility of mesh outline (Shift+E)")
        show_outline_action_in_menu.setCheckable(True)
        show_outline_action_in_menu.setChecked(True)  # Default to visible
        show_outline_action_in_menu.triggered.connect(self.mesh_viewer.setOutlineVisible)
        view_menu.addAction(show_outline_action_in_menu)

        # Create "Show Connection Points" action for the menu
        show_connection_points_action = QAction("Show Connection Points", self)
        show_connection_points_action.setStatusTip("Toggle visibility of connection points (C)")
        show_connection_points_action.setToolTip("Toggle visibility of connection points (C)")
        show_connection_points_action.setCheckable(True)
        show_connection_points_action.setChecked(True)  # Default to visible
        show_connection_points_action.triggered.connect(self.mesh_viewer.setConnectionPointsVisible)
        view_menu.addAction(show_connection_points_action)

        # Set the menu for the QToolButton
        view_menu_button.setMenu(view_menu)
        self.addWidget(view_menu_button)

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
        # Add Reset View button
        fit_view_action = QAction("Reset View", self)
        fit_view_action.setStatusTip("Reset view to fit all content (F)")
        fit_view_action.setToolTip("Reset view to fit all content (F)")
        fit_view_action.triggered.connect(self.mesh_viewer.autoscaleXY)
        self.addAction(fit_view_action)

        # Add Full Scale button
        full_scale_action = QAction("Full Scale", self)
        full_scale_action.setStatusTip("Reset color scale to full range (A)")
        full_scale_action.setToolTip("Reset color scale to full range (A)")
        full_scale_action.triggered.connect(self.mesh_viewer.autoscaleValue)
        self.addAction(full_scale_action)

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


@dataclass
class ShaderProgram:
    shader_program: QOpenGLShaderProgram = field(default_factory=QOpenGLShaderProgram)

    @classmethod
    def from_source(cls, vertex_source, fragment_source):
        shader_program = QOpenGLShaderProgram()
        shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_source)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_source)
        linked = shader_program.link()
        if not linked:
            raise Exception("Failed to link shader program")

        return cls(shader_program)

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

    @classmethod
    def _from_common(cls,
                     triangle_vertices: list[float],
                     triangle_colors: list[float],
                     edge_vertices: list[float],
                     edge_colors: list[float],
                     boundary_vertices: list[float],
                     boundary_colors: list[float]) -> 'RenderedMesh':

        def create_vao(vertices: list[float], colors: list[float], color_components: int) -> int:
            """Create a VAO with vertex and color VBOs."""
            vao = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(vao)

            # VBO for vertices (attribute 0, 2D coordinates)
            vbo_vertices = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_vertices)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                np.array(vertices, dtype=np.float32),
                gl.GL_STATIC_DRAW
            )
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(0)

            # VBO for colors (attribute 1, 1D or 3D components)
            vbo_colors = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_colors)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                np.array(colors, dtype=np.float32),
                gl.GL_STATIC_DRAW
            )
            gl.glVertexAttribPointer(1, color_components, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(1)

            return vao

        # Create VAOs for each mesh component
        vao_triangles = create_vao(triangle_vertices, triangle_colors, 1)
        vao_edges = create_vao(edge_vertices, edge_colors, 3)
        vao_boundary = create_vao(boundary_vertices, boundary_colors, 3)

        gl.glBindVertexArray(0)

        return cls(vao_triangles,
                   len(triangle_vertices) // 2,
                   vao_edges,
                   len(edge_vertices) // 2,
                   vao_boundary,
                   len(boundary_vertices) // 2)

    @staticmethod
    def _serialize_edges_from_mesh(msh: mesh.Mesh):
        edge_vertices = []
        edge_colors = []
        boundary_vertices = []
        boundary_colors = []

        for face in msh.faces:

            for edge in face.edges:
                v1 = edge.origin
                v2 = edge.next.origin

                vertices_data = [v1.p.x, v1.p.y, v2.p.x, v2.p.y]
                # TODO: It would make sense for the color to be configurable
                # and/or based on some property of the edge
                color_data = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
                if edge.twin.is_boundary:
                    boundary_vertices.extend(vertices_data)
                    boundary_colors.extend(color_data)
                else:
                    edge_vertices.extend(vertices_data)
                    edge_colors.extend(color_data)

        return edge_vertices, edge_colors, boundary_vertices, boundary_colors

    @classmethod
    def from_zero_form(cls, msh: mesh.Mesh, values: mesh.ZeroForm):
        triangle_vertices = []
        triangle_colors = []

        for face in msh.faces:
            # Note that we assume the face is a triangle. This should be already
            # checked by other layers

            # Vertex data
            for vertex in face.vertices:
                triangle_vertices.extend([vertex.p.x, vertex.p.y])
                # This is where we differ from from_two_form
                triangle_colors.extend([values[vertex]])

        edge_vertices, edge_colors, boundary_vertices, boundary_colors = \
            cls._serialize_edges_from_mesh(msh)

        return cls._from_common(
            triangle_vertices,
            triangle_colors,
            edge_vertices,
            edge_colors,
            boundary_vertices,
            boundary_colors
        )

    @classmethod
    def from_two_form(cls, msh: mesh.Mesh, values: mesh.TwoForm):
        triangle_vertices = []
        triangle_colors = []

        for face in msh.faces:
            # Note that we assume the face is a triangle. This should be already
            # checked by other layers

            # Vertex data
            for vertex in face.vertices:
                triangle_vertices.extend([vertex.p.x, vertex.p.y])
                # This is where we differ from from_zero_form
                triangle_colors.extend([values[face]])

        edge_vertices, edge_colors, boundary_vertices, boundary_colors = \
            cls._serialize_edges_from_mesh(msh)

        return cls._from_common(
            triangle_vertices,
            triangle_colors,
            edge_vertices,
            edge_colors,
            boundary_vertices,
            boundary_colors
        )

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
    def from_mesh(cls, msh: mesh.Mesh) -> 'RenderedMesh':
        """Create a RenderedMesh from a mesh with zero values.
        Used for disconnected copper regions that will be rendered in gray."""
        # Create a ZeroForm with all values set to zero
        zero_values = mesh.ZeroForm(msh)
        for vertex in msh.vertices:
            zero_values[vertex] = 0.0

        # Use the existing from_zero_form method
        return cls.from_zero_form(msh, zero_values)


@dataclass
class RenderedPoints:
    vao_points: int
    point_count: int

    @classmethod
    def from_points(cls, points_data: list[tuple[tuple[float, float], tuple[float, float, float]]]):
        if not points_data:
            # Handle empty list to avoid errors with glBufferData
            vao_points = gl.glGenVertexArrays(1)
            # No need to create VBOs if there's no data
            return cls(vao_points, 0)

        flat_points_coords = []
        flat_points_colors = []
        for (p_x, p_y), (r, g, b) in points_data:
            flat_points_coords.extend([p_x, p_y])
            flat_points_colors.extend([r, g, b])

        vao_points = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_points)

        # VBO for point coordinates
        vbo_point_coords = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_point_coords)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            np.array(flat_points_coords, dtype=np.float32),
            gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # VBO for point colors
        vbo_point_colors = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_point_colors)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            np.array(flat_points_colors, dtype=np.float32),
            gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)
        # The number of points is the length of the original points_data list
        return cls(vao_points, len(points_data))

    def render(self):
        if self.point_count > 0:
            gl.glBindVertexArray(self.vao_points)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.point_count)


class MeshViewer(QOpenGLWidget):

    @dataclass
    class BaseRenderingMode:
        unit: str
        name: str
        color_map: colormaps.UniformColorMap
        min_value: float = 0.0
        max_value: float = 1.0
        solution: Optional[solver.Solution] = None
        spatial_indices: dict[str, BaseSpatialIndex] = field(default_factory=dict)
        rendered_meshes: dict[str, list[RenderedMesh]] = field(default_factory=dict)
        disconnected_rendered_meshes: dict[str, list[RenderedMesh]] = field(default_factory=dict)

        def _compute_min_max(self) -> tuple[float, float]:
            """Compute min and max values across all spatial indices."""
            min_val = float('inf')
            max_val = float('-inf')

            for index in self.spatial_indices.values():
                if not index.values:
                    continue
                min_val = min(min_val, min(index.values))
                max_val = max(max_val, max(index.values))

            if min_val == float('inf'):
                min_val, max_val = 0.0, 1.0
            elif min_val == max_val:
                min_val, max_val = min_val, min_val + 1.0

            return min_val, max_val

        def autoscale_values(self, solution: solver.Solution):
            """Autoscale values for the rendering mode."""
            self.min_value, self.max_value = self._compute_min_max()

        def _build_spatial_indices(self):
            raise NotImplementedError("This method should be implemented in subclasses")

        @abc.abstractmethod
        def set_solution(self, solution: solver.Solution):
            """Initialize this mode with solution data (build indices + meshes)."""
            self.solution = solution
            self.spatial_indices.clear()
            self._build_spatial_indices()
            # We have to delay this until the OpenGL context is properly initialized.
            self.rendered_meshes.clear()
            self.disconnected_rendered_meshes.clear()

        def _create_rendered_meshes_for_layer(self, layer_name) -> list[RenderedMesh]:
            """Create RenderedMesh objects for a specific layer."""
            raise NotImplementedError("This method should be implemented in subclasses")

        def pick_nearest_value(self, layer_name: str, world_x: float, world_y: float) -> Optional[float]:
            """Pick value at coordinates using spatial index."""
            if layer_name in self.spatial_indices:
                return self.spatial_indices[layer_name].query_nearest(world_x, world_y)
            return None

        def get_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Get pre-built rendered meshes for a layer."""
            if layer_name not in self.rendered_meshes:
                # If not already built, create them
                # Ideally, we would do this in .set_solution, but that
                # is getting called before OpenGL context is ready
                self.rendered_meshes[layer_name] = \
                    self._create_rendered_meshes_for_layer(layer_name)
            return self.rendered_meshes[layer_name]

        def _create_disconnected_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Create RenderedMesh objects for disconnected copper on a specific layer."""
            rendered_meshes = []
            if not self.solution:
                return rendered_meshes

            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                for msh in layer_solution.disconnected_meshes:
                    rendered_meshes.append(RenderedMesh.from_mesh(msh))
            return rendered_meshes

        def get_disconnected_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Get pre-built disconnected rendered meshes for a layer."""
            if layer_name not in self.disconnected_rendered_meshes:
                # If not already built, create them
                self.disconnected_rendered_meshes[layer_name] = \
                    self._create_disconnected_rendered_meshes_for_layer(layer_name)
            return self.disconnected_rendered_meshes[layer_name]

    @dataclass
    class VoltageRenderingMode(BaseRenderingMode):
        unit: str = "V"
        name: str = "Potential"
        color_map: colormaps.UniformColorMap = colormaps.PLASMA

        def _build_spatial_indices(self):
            """Build spatial indices for fast vertex lookups."""
            self.spatial_indices.clear()
            for layer, layer_solution in zip(self.solution.problem.layers, self.solution.layer_solutions):
                spatial_index = VertexSpatialIndex.from_layer_data(layer, layer_solution)
                self.spatial_indices[layer.name] = spatial_index

        def _create_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Create RenderedMesh objects for a specific layer."""
            rendered_meshes = []
            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                for msh, values in zip(layer_solution.meshes, layer_solution.potentials):
                    rendered_meshes.append(RenderedMesh.from_zero_form(msh, values))
            return rendered_meshes

    @dataclass
    class PowerDensityRenderingMode(BaseRenderingMode):
        unit: str = "W/mm²"
        name: str = "Power Density"
        color_map: colormaps.UniformColorMap = colormaps.INFERNO

        def _compute_min_max(self) -> tuple[float, float]:
            _, max_val = super()._compute_min_max()
            # Usually, we would get a value that is very close to zero anyway,
            # this makes it a bit prettier
            return 0.0, max_val

        def _build_spatial_indices(self):
            """Build spatial indices for fast face lookups."""
            self.spatial_indices.clear()
            for layer, layer_solution in zip(self.solution.problem.layers, self.solution.layer_solutions):
                spatial_index = FaceSpatialIndex.from_layer_data(layer, layer_solution)
                self.spatial_indices[layer.name] = spatial_index

        def _create_rendered_meshes_for_layer(self, layer_name: str) -> list[RenderedMesh]:
            """Create RenderedMesh objects for a specific layer."""
            rendered_meshes = []
            for layer, layer_solution in zip(self.solution.problem.layers,
                                             self.solution.layer_solutions):
                if layer.name != layer_name:
                    continue
                for msh, values in zip(layer_solution.meshes, layer_solution.power_densities):
                    rendered_meshes.append(RenderedMesh.from_two_form(msh, values))
            return rendered_meshes

    # Signal to notify when the value range changes
    valueRangeChanged = Signal(float, float)
    # Signal to notify when the current layer changes
    currentLayerChanged = Signal(str)
    # Signal to notify when the list of available layers changes
    availableLayersChanged = Signal(list)
    # Signal to notify when the current rendering mode changes
    currentModeChanged = Signal(str)
    # Signal to notify when the unit changes
    unitChanged = Signal(str)
    # Signal to notify when the color map changes
    colorMapChanged = Signal(object)  # object is colormaps.UniformColorMap
    # Signals for tools
    meshClicked = Signal(mesh.Point, QtGui.QMouseEvent)
    screenDragged = Signal(float, float, QtGui.QMouseEvent)
    keyPressedInMesh = Signal(mesh.Point, int, Qt.KeyboardModifiers)
    # Signal for mouse position and voltage probing
    mousePositionChanged = Signal(mesh.Point, object)  # object can be float or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.solution: None | solver.Solution = None
        # Layer name -> RenderedMesh
        self.rendered_meshes: dict[str, list] = {}
        self.rendered_connection_points: dict[str, RenderedPoints] = {}
        self.connection_points_visible: bool = True

        # Rendering modes and current mode tracking
        self.modes = [
            self.VoltageRenderingMode(),
            self.PowerDensityRenderingMode()
        ]
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
        """
        Compute the bounding box of all meshes across all layers.

        Returns:
            A tuple of (min_x, min_y, max_x, max_y) or None if no vertices found.
        """
        if not self.solution or not self.solution.layer_solutions:
            return None

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for layer_solution in self.solution.layer_solutions:
            for msh in layer_solution.meshes:
                for vertex in msh.vertices:
                    x, y = vertex.p.x, vertex.p.y
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

        # Check if we found any vertices
        if min_x == float('inf'):
            return None

        return min_x, min_y, max_x, max_y

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

    def autoscaleValue(self):
        """
        Automatically adjust the min/max values for color scaling using the current rendering mode.
        """
        if not self.solution or not self.solution.layer_solutions:
            return  # Nothing to scale if no solution is loaded

        # Delegate to current rendering mode
        self.current_rendering_mode.autoscale_values(self.solution)

        # Emit signal to notify about the new value range
        self.valueRangeChanged.emit(self.current_rendering_mode.min_value, self.current_rendering_mode.max_value)
        self.update()

    def autoscaleXY(self):
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

    @Slot(solver.Solution)
    def setSolution(self, solution: solver.Solution):
        """Set the solution for the mesh viewer."""
        self.solution = solution

        # Initialize the list of layers from the solution
        self.visible_layers = [layer.name for layer in solution.problem.layers]
        self.current_layer_index = 0

        # Emit signal with available layers
        if self.visible_layers:
            self.availableLayersChanged.emit(self.visible_layers)

        # Emit signal with initial layer
        if self.visible_layers:
            self.currentLayerChanged.emit(self.current_layer_name)

        # Initialize all modes and emit mode signals
        current_mode = self.current_rendering_mode

        # Initialize all modes with solution data (spatial indices + rendered meshes)
        for mode in self.modes:
            mode.set_solution(solution)
            mode.autoscale_values(solution)

        # Emit mode-related signals
        self.currentModeChanged.emit(current_mode.name)
        self.unitChanged.emit(current_mode.unit)
        self.colorMapChanged.emit(current_mode.color_map)
        self.valueRangeChanged.emit(
            self.current_rendering_mode.min_value, self.current_rendering_mode.max_value
        )

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

    def setupConnectionPointsData(self):
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
            if not collected_points_data:
                continue
            # We want to render the _red_ points over the gray ones,
            # so we draw them _last_. This is a hack to order them, it
            # depends on the fact that (1.0, 0.0, 0.0) > (0.5, 0.5, 0.5)
            # _This will break if the colors change!_
            collected_points_data.sort(key=lambda x: x[1])
            # Pass the list of (coordinates, color) tuples
            rendered_obj = RenderedPoints.from_points(collected_points_data)
            self.rendered_connection_points[layer_name] = rendered_obj

    def initializeGL(self):
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

    def resizeGL(self, width, height):
        """Handle window resizing."""
        gl.glViewport(0, 0, width, height)

        # Perform autoscaling on resize until user manually interacts
        if self.needs_initial_autoscale and width > 0 and height > 0:
            self.autoscaleXY()
            self.update()

    def _computeMVP(self):
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

    def _renderMeshTriangles(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]):
        """Renders the triangles of the meshes for the current layer."""
        with self.mesh_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.mesh_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )

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

    def _renderMeshEdges(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]):
        """Renders the edges of the meshes for the current layer."""
        if not self.edges_visible or not self.edge_shader:
            return

        with self.edge_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.edge_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )

            # Draw edges for current layer only
            for rmesh in rendered_mesh_list:
                rmesh.render_edges()

    def _renderBoundaryEdges(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]):
        """Renders the boundary edges of the meshes for the current layer."""
        if not self.outline_visible or not self.edge_shader:
            return

        with self.edge_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.edge_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )

            # Draw boundary edges for current layer only
            for rmesh in rendered_mesh_list:
                rmesh.render_boundary()

    def _renderDisconnectedMeshes(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]):
        """Renders disconnected copper meshes in gray."""
        if not self.disconnected_shader or not rendered_mesh_list:
            return

        with self.disconnected_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.disconnected_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )

            # Draw triangles for disconnected meshes
            for rmesh in rendered_mesh_list:
                rmesh.render_triangles()

            # Notably we do not render edges for disconnected meshes.
            # They provide no additional information and look messy anyway...
            # It can be useful to render them when debugging the code

    def _renderConnectionPoints(self, mvp: np.ndarray, rendered_points_obj: RenderedPoints):
        """Renders the connection points for the current layer."""
        if not self.connection_points_visible or not self.points_shader:
            return

        if rendered_points_obj.point_count == 0:
            return

        with self.points_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.points_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )

            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            rendered_points_obj.render()
            gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)

    def paintGL(self):
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
        # These formulas were implicitly used in _getValueFromCursor and worked for picking.
        world_x = (ndc_x * aspect / self.scale) - self.offset_x
        world_y = (ndc_y / self.scale) - self.offset_y

        return mesh.Point(world_x, world_y)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse press events."""
        if event.buttons() & Qt.LeftButton:  # Typically, tools operate on left click
            self.last_mouse_screen_pos = event.position()

        self.setFocus()  # Ensure the widget gets focus when clicked

        # Emit meshClicked signal regardless of button for potential right-click tools etc.
        # The tool itself can check event.button()
        world_point = self._screenToWorld(event.position())
        self.meshClicked.emit(world_point, event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse movement."""
        if event.buttons() & Qt.LeftButton and self.last_mouse_screen_pos is not None:
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
        voltage = self._getNearestValue(world_point.x, world_point.y)
        self.mousePositionChanged.emit(world_point, voltage)
        self.last_mouse_position_change_ts = time.monotonic()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton and self.last_mouse_screen_pos is not None:
            # TODO: Potentially emit a clickReleased signal if tools need it
            # Clear drag state
            self.last_mouse_screen_pos = None

    def panViewByScreenDelta(self, dx_screen: float, dy_screen: float):
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

    def setMinValueFromWorldPoint(self, world_point: mesh.Point):
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

        self.current_rendering_mode.min_value = value
        # Enforce min_value <= max_value
        if value > self.current_rendering_mode.max_value:
            self.current_rendering_mode.max_value = value

        self.valueRangeChanged.emit(self.current_rendering_mode.min_value, self.current_rendering_mode.max_value)
        self.update()

    def setMaxValueFromWorldPoint(self, world_point: mesh.Point):
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

        self.current_rendering_mode.max_value = value
        # Enforce min_value <= max_value
        if value < self.current_rendering_mode.min_value:
            self.current_rendering_mode.min_value = value

        self.valueRangeChanged.emit(self.current_rendering_mode.min_value, self.current_rendering_mode.max_value)
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        # User manually zoomed - disable automatic scaling
        self.needs_initial_autoscale = False

        zoom_factor = 1.2
        if event.angleDelta().y() > 0:
            self.scale *= zoom_factor
        else:
            self.scale /= zoom_factor
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
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

    def switchLayerBy(self, direction: int = 1):
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

    def switchToNextLayer(self):
        """Switch to the next layer in the cycle."""
        self.switchLayerBy(1)

    def switchToPreviousLayer(self):
        """Switch to the previous layer in the cycle."""
        self.switchLayerBy(-1)

    @Slot(bool)
    def setEdgesVisible(self, visible: bool):
        """Slot to set the visibility of mesh edges."""
        self.edges_visible = visible

        # If we're showing edges but outline is hidden, also show the outline
        if visible and not self.outline_visible:
            self.outline_visible = True
            log.debug("Also showing outline since internal edges are being shown")

        log.debug(f"Mesh edges visibility set to: {self.edges_visible}")
        self.update()

    @Slot(bool)
    def setOutlineVisible(self, visible: bool):
        """Slot to set the visibility of outline edges."""
        self.outline_visible = visible

        # If we're hiding the outline and edges are visible, also hide the edges
        if not visible and self.edges_visible:
            self.edges_visible = False
            log.debug("Also hiding internal edges since outline is being hidden")

        log.debug(f"Outline visibility set to: {self.outline_visible}")
        self.update()

    @Slot(bool)
    def setConnectionPointsVisible(self, visible: bool):
        """Slot to set the visibility of connection points."""
        self.connection_points_visible = visible
        log.debug(f"Connection points visibility set to: {self.connection_points_visible}")
        self.update()

    @Slot(str)
    def setCurrentLayerByName(self, layer_name: str):
        """Sets the current layer by its name."""
        if layer_name in self.visible_layers:
            self.current_layer_index = self.visible_layers.index(layer_name)
            self.currentLayerChanged.emit(layer_name)
            self.update()
        else:
            log.error(f"Attempted to set current layer to unknown layer: {layer_name}")

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
            self.unitChanged.emit(mode.unit)
            self.colorMapChanged.emit(mode.color_map)
            self.valueRangeChanged.emit(self.current_rendering_mode.min_value, self.current_rendering_mode.max_value)
            self.update()

    def _updateShaderColorMap(self):
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


class ColorScaleWidget(QWidget):
    """Widget that displays a color scale with delta and absolute range."""

    # Signal to notify when unit is changed manually
    unitChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.v_min = 0.0
        self.v_max = 1.0
        self.unit = "V"  # Default unit
        self.color_map = colormaps.PLASMA  # Default color map

        self.setMinimumWidth(110)
        self.setMinimumHeight(200)

        # New labels
        self.delta_label = None
        self.range_label = None

        self.setupUI()

    def setupUI(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)  # Add a little vertical spacing

        # Delta label at the top of the stretch area
        self.delta_label = QLabel(f"Δ = 0 {self.unit}")
        self.delta_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.delta_label)

        # This stretch is where we'll paint our gradient
        layout.addStretch(10)

        # Range label at the bottom showing absolute min/max values
        self.range_label = QLabel(f"Range: 0 {self.unit} - 0 {self.unit}")
        self.range_label.setAlignment(Qt.AlignCenter)
        # Make range label slightly smaller font
        font = self.range_label.font()
        font.setPointSize(font.pointSize() - 1)
        self.range_label.setFont(font)
        layout.addWidget(self.range_label)

    @Slot(float, float)
    def setRange(self, v_min, v_max):
        """Set the minimum and maximum values for the scale."""
        self.v_min = v_min
        self.v_max = v_max
        self.updateLabels()
        self.update()

    @Slot(str)
    def setUnit(self, unit):
        """Set the unit for the scale."""
        self.unit = unit
        self.updateLabels()

    @Slot(object)
    def setColorMap(self, color_map):
        """Set the color map for the scale."""
        self.color_map = color_map
        self.update()  # Trigger a repaint

    def updateLabels(self):
        """Update the delta and range labels."""
        delta = self.v_max - self.v_min
        delta_str = units.Value(delta, self.unit).pretty_format(decimal_places=2)
        min_str = units.Value(self.v_min, self.unit).pretty_format()
        max_str = units.Value(self.v_max, self.unit).pretty_format()

        self.delta_label.setText(f"Δ = {delta_str}")
        self.range_label.setText(f"{max_str}\n  ↑\n{min_str}")

    def paintEvent(self, event):
        """Paint the color gradient scale."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Find the rectangle where we should draw the gradient
        # This should be between the delta_label and range_label
        content_rect = self.rect()
        top_margin = self.delta_label.y() + self.delta_label.height() + 2  # +2 for spacing
        bottom_margin = self.height() - self.range_label.y() + 2  # +2 for spacing

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
            color = self.color_map(t)

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


class MainWindow(QMainWindow):

    projectLoaded = Signal(solver.Solution)

    def __init__(self, solution: solver.Solution, project_name: Optional[str]):
        super().__init__()

        self.project_file_name = project_name if project_name else "Loaded Solution"

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

        # Connect signals/slots
        self.mesh_viewer.valueRangeChanged.connect(self.color_scale.setRange)
        self.mesh_viewer.unitChanged.connect(self.color_scale.setUnit)
        self.mesh_viewer.colorMapChanged.connect(self.color_scale.setColorMap)
        self.projectLoaded.connect(self.mesh_viewer.setSolution)
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

        self.projectLoaded.emit(solution)

    # Removed loadProject method

    def updateCurrentLayer(self, layer_name):
        """Update the window title to show the current layer."""
        self.setWindowTitle(f"padne: {self.project_file_name} - {layer_name}")
        self.layer_status_label.setText(f"Layer: {layer_name}")

    @Slot(mesh.Point, object)
    def updateMousePosition(self, world_point: mesh.Point, value):
        """Update status bar with mouse position and value."""
        self.x_position_label.setText(f"X: {world_point.x:.3f}")
        self.y_position_label.setText(f"Y: {world_point.y:.3f}")

        if value is not None:
            current_unit = self.mesh_viewer.current_rendering_mode.unit
            value_str = units.Value(value, current_unit).pretty_format(3)
            self.value_label.setText(f"{current_unit}: {value_str}")

            # Calculate delta from the minimum value of the color scale
            delta_value = value - self.mesh_viewer.current_rendering_mode.min_value
            delta_str = units.Value(delta_value, current_unit).pretty_format(3)
            self.delta_label.setText(f"Δ: {delta_str}")
        else:
            current_unit = self.mesh_viewer.current_rendering_mode.unit
            self.value_label.setText(f"{current_unit}: ?")
            self.delta_label.setText("Δ: ?")


def configure_opengl():
    """Configure OpenGL settings for the application."""
    # Create OpenGL format
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)  # Use OpenGL 3.3
    gl_format.setProfile(QSurfaceFormat.CoreProfile)  # Use core profile
    gl_format.setSamples(4)  # Enable 4x antialiasing
    QSurfaceFormat.setDefaultFormat(gl_format)


def main(solution: solver.Solution, project_name: Optional[str]):
    """Main entry point for the UI application."""
    # Configure OpenGL
    configure_opengl()

    # Create and run application
    # Try to get existing instance, or create one if not present.
    # Using sys.argv can be problematic in some contexts (e.g. pytest),
    # but is standard for standalone Qt apps. Using [] if no Qt-specific args are needed.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if hasattr(sys, 'argv') else [])

    window = MainWindow(solution, project_name)

    window.show()
    return app.exec()
