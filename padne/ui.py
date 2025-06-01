#!/usr/bin/env python3

import contextlib
import logging
import numpy as np
import sys
import OpenGL.GL as gl

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

from . import mesh, solver, units, colormaps


log = logging.getLogger(__name__)


# Define shader source code
VERTEX_SHADER_MESH = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in float color;
out float frag_value;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 0.0, 1.0);  // Explicitly set z=0.0
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
    out_color = vec4(frag_color, 1.0);  // Pass through the color (white)
}
"""

VERTEX_SHADER_POINTS = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 vertex_color; // Added color attribute
out vec3 frag_color;                       // Added varying for fragment shader
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
// uniform vec3 point_color = vec3(1.0, 0.0, 0.0); // Removed uniform color

void main() {
    out_color = vec4(frag_color, 1.0); // Use varying color
}
"""


COLOR_MAP = colormaps.PLASMA


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
    def shortcut(self) -> tuple[Qt.Key, Qt.KeyboardModifier] | None:
        return None

    def on_shortcut_press(self, world_point: mesh.Point):
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
            self.mesh_viewer.pan_view_by_screen_delta(dx, dy)


class SetMinValueTool(PanTool):

    @property
    def name(self) -> str:
        return "Min"

    @property
    def status_tip(self) -> str:
        return "Set minimum value for color scale from cursor"

    @property
    def shortcut(self):
        return (Qt.Key_M, Qt.NoModifier)

    def on_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mesh_viewer.set_min_value_from_world_point(world_point)
            # Optional: Switch back to Pan tool after action
            # self.tool_manager.activate_tool(self.tool_manager.available_tools[0]) # Assuming Pan is first

    def on_shortcut_press(self, world_point: mesh.Point):
        log.debug(f"SetMinValueTool: Shortcut pressed at {world_point}")
        self.mesh_viewer.set_min_value_from_world_point(world_point)


class SetMaxValueTool(PanTool):
    @property
    def name(self) -> str:
        return "Max"

    @property
    def status_tip(self) -> str:
        return "Set maximum value for color scale from cursor"

    @property
    def shortcut(self):
        return (Qt.Key_M, Qt.ShiftModifier)

    def on_mesh_click(self, world_point: mesh.Point, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mesh_viewer.set_max_value_from_world_point(world_point)
            # Optional: Switch back to Pan tool after action
            # self.tool_manager.activate_tool(self.tool_manager.available_tools[0]) # Assuming Pan is first

    def on_shortcut_press(self, world_point: mesh.Point):
        log.debug(f"SetMaxValueTool: Shortcut pressed at {world_point}")
        self.mesh_viewer.set_max_value_from_world_point(world_point)


class ToolManager(QtCore.QObject):
    def __init__(self, mesh_viewer: 'MeshViewer', parent=None):
        super().__init__(parent)
        self.mesh_viewer = mesh_viewer
        
        self.available_tools: list[BaseTool] = [
            PanTool(self.mesh_viewer, self),
            SetMinValueTool(self.mesh_viewer, self),
            SetMaxValueTool(self.mesh_viewer, self)
        ]
        
        self.active_tool: Optional[BaseTool] = None
        if self.available_tools:
            # Activate the first tool by default, but don't call on_activate yet
            # as the tool might not be fully ready (e.g. UI elements)
            # on_activate will be called by the first explicit activate_tool call
            self.active_tool = self.available_tools[0]

    @Slot(BaseTool)
    def activate_tool(self, tool_to_activate: BaseTool):
        if self.active_tool == tool_to_activate:
            return

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
        self._setup_actions()

    def _setup_actions(self):
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

            # Set the default tool (first tool in the list) as checked
            if self.tool_manager.active_tool == tool_instance:
                action.setChecked(True)
                # self.tool_manager.activate_tool(tool_instance) # Already active by default in ToolManager

        # Add a separator after the tool actions
        self.addSeparator()

        # Create the "View" QToolButton
        view_menu_button = QToolButton(self)
        view_menu_button.setText("View")
        view_menu_button.setToolTip("View options")
        # This makes it into a popup menu
        view_menu_button.setPopupMode(QToolButton.InstantPopup)

        # Create the menu that will be shown by the QToolButton
        view_menu = QMenu(view_menu_button)

        # Create "Show Edges" action for the menu
        show_edges_action_in_menu = QAction("Show Edges", self)
        show_edges_action_in_menu.setStatusTip("Toggle visibility of mesh edges")
        show_edges_action_in_menu.setToolTip("Toggle visibility of mesh edges")
        show_edges_action_in_menu.setCheckable(True)
        show_edges_action_in_menu.setChecked(True)  # Default to visible
        
        # Connect to MeshViewer's slot
        show_edges_action_in_menu.triggered.connect(self.mesh_viewer.set_edges_visible)
        
        # Add the action to the menu
        view_menu.addAction(show_edges_action_in_menu)
        
        # Create "Show Connection Points" action for the menu
        show_connection_points_action = QAction("Show Connection Points", self)
        show_connection_points_action.setStatusTip("Toggle visibility of connection points")
        show_connection_points_action.setToolTip("Toggle visibility of connection points")
        show_connection_points_action.setCheckable(True)
        show_connection_points_action.setChecked(True) # Default to visible
        show_connection_points_action.triggered.connect(self.mesh_viewer.set_connection_points_visible)
        view_menu.addAction(show_connection_points_action)


        # Set the menu for the QToolButton
        view_menu_button.setMenu(view_menu)
        
        # Add the QToolButton to the toolbar
        self.addWidget(view_menu_button)

        # Add a separator
        self.addSeparator()

        # Create the "Layers" QToolButton
        self.layers_button = QToolButton(self)
        self.layers_button.setText("Layers")
        self.layers_button.setToolTip("Select active layer")
        self.layers_button.setPopupMode(QToolButton.InstantPopup)

        self.layers_menu = QMenu(self.layers_button)
        self.layer_action_group = QActionGroup(self)
        self.layer_action_group.setExclusive(True)

        self.layers_button.setMenu(self.layers_menu)
        self.addWidget(self.layers_button)

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
        if self.mesh_viewer.visible_layers and self.mesh_viewer.current_layer_index < len(self.mesh_viewer.visible_layers):
            active_layer_name = self.mesh_viewer.visible_layers[self.mesh_viewer.current_layer_index]
            self.updateActiveLayerInMenu(active_layer_name)


    @Slot(str)
    def updateActiveLayerInMenu(self, active_layer_name: str):
        for action in self.layers_menu.actions():
            if action.text() == active_layer_name:
                action.setChecked(True)
                break


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

    @classmethod
    def from_mesh(cls, msh: mesh.Mesh, values: mesh.ZeroForm):
        triangle_vertices = []
        triangle_colors = []
        edge_vertices = []
        edge_colors = []

        for face in msh.faces:
            # Note that we assume the face is a triangle. This should be already
            # checked by other layers

            # Vertex data
            for vertex in face.vertices:
                triangle_vertices.extend([vertex.p.x, vertex.p.y])
                triangle_colors.extend([values[vertex]])

            # Edge data
            for edge in face.edges:
                # TODO: I am not quite sure how this differs from the previous case
                # Maybe we can be sneaky and just use the same data, potentially
                # inserting some spacing in between?
                for e in [edge, edge.next]:
                    edge_vertices.extend([e.origin.p.x, e.origin.p.y])
                    edge_colors.extend([0.9, 0.9, 0.9])

        # VAO for triangles
        
        vao_triangles = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_triangles)

        # VBO for triangle vertices
        vbo_triangle_vertices = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_triangle_vertices)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            np.array(triangle_vertices, dtype=np.float32),
            gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # VBO for triangle colors
        vbo_triangle_colors = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_triangle_colors)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            np.array(triangle_colors, dtype=np.float32),
            gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        #  VAO for edges
        vao_edges = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_edges)

        # VBO for edge vertices
        vbo_edge_vertices = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_edge_vertices)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            np.array(edge_vertices, dtype=np.float32),
            gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # VBO for edge colors
        vbo_edge_colors = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_edge_colors)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            np.array(edge_colors, dtype=np.float32),
            gl.GL_STATIC_DRAW
        )
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

        return cls(vao_triangles,
                   len(triangle_vertices) // 2,
                   vao_edges,
                   len(edge_vertices))

    def render_triangles(self):
        gl.glBindVertexArray(self.vao_triangles)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.triangle_count)

    def render_edges(self):
        gl.glBindVertexArray(self.vao_edges)
        gl.glDrawArrays(gl.GL_LINES, 0, self.edge_count)


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
    # Signal to notify when the value range changes
    valueRangeChanged = Signal(float, float)
    # Signal to notify when the current layer changes
    currentLayerChanged = Signal(str)
    # Signal to notify when the list of available layers changes
    availableLayersChanged = Signal(list)
    # Signals for tools
    meshClicked = Signal(mesh.Point, QtGui.QMouseEvent)
    screenDragged = Signal(float, float, QtGui.QMouseEvent)
    keyPressedInMesh = Signal(mesh.Point, int, Qt.KeyboardModifiers)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.solution: None | solver.Solution = None
        # Layer name -> RenderedMesh
        self.rendered_meshes: dict[str, list] = {}
        self.rendered_connection_points: dict[str, RenderedPoints] = {}
        self.connection_points_visible: bool = True

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.last_mouse_screen_pos: Optional[QtCore.QPointF] = None
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

    def _getNearestValue(self, world_x: float, world_y: float) -> Optional[float]:
        """
        Find the value at the vertex closest to the specified world coordinates,
        only if the world coordinates are within the layer's defined geometries.
        
        Args:
            world_x: X-coordinate in world space
            world_y: Y-coordinate in world space
            
        Returns:
            The value at the nearest vertex, or None if no vertices are found
            or if the point is outside the layer's geometries.
        """
        if not self.solution or not self.visible_layers or self.current_layer_index >= len(self.visible_layers):
            return None

        current_layer_name = self.visible_layers[self.current_layer_index]

        # Find the corresponding problem.Layer for the point-in-polygon check
        problem_layer_for_check: Optional[solver.problem.Layer] = None
        layer_index_for_solution = -1

        for idx, p_layer in enumerate(self.solution.problem.layers):
            if p_layer.name == current_layer_name:
                problem_layer_for_check = p_layer
                # Store index for consistent LayerSolution access
                layer_index_for_solution = idx
                break
        
        if not problem_layer_for_check:
            log.debug(f"Layer {current_layer_name} not found in problem definition.")
            return None

        # Perform point-in-polygon check using Shapely
        shapely_point = shapely.geometry.Point(world_x, world_y)
        
        if not problem_layer_for_check.shape.contains(shapely_point):
            # Click is outside the defined MultiPolygon for this layer
            log.debug(f"Point ({world_x}, {world_y}) is outside defined shape for layer {current_layer_name}.")
            return None

        # If the point is inside, proceed to find the nearest vertex value
        target_point = mesh.Point(world_x, world_y)
        
        closest_distance = float('inf')
        closest_value = None
        
        # Access the LayerSolution using the stored index
        if layer_index_for_solution == -1 or layer_index_for_solution >= len(self.solution.layer_solutions):
            log.warning(f"Could not find matching LayerSolution for {current_layer_name} using index {layer_index_for_solution}.")
            return None
            
        layer_solution = self.solution.layer_solutions[layer_index_for_solution]
        
        # Check each mesh in the current layer's solution
        for mesh_index, msh in enumerate(layer_solution.meshes):
            values = layer_solution.values[mesh_index]
            
            for vertex in msh.vertices:
                distance = vertex.p.distance(target_point)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_value = values[vertex]
        
        if closest_value is not None:
            log.debug(f"Nearest value for point ({world_x}, {world_y}) in layer {current_layer_name} is {closest_value}.")
        else:
            log.debug(f"Point ({world_x}, {world_y}) was inside geometry for layer {current_layer_name}, but no nearest vertex value found.")
            
        return closest_value

    def autoscaleValue(self):
        """
        Automatically adjust the min/max values for color scaling.
        Finds the minimum and maximum values across all layer solutions.
        """
        if not self.solution or not self.solution.layer_solutions:
            return  # Nothing to scale if no solution is loaded
        
        # Initialize min and max values
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        # Go through all layer solutions to find min/max values
        for layer_solution in self.solution.layer_solutions:
            # Each layer solution has multiple meshes and their corresponding values
            for msh, values in zip(layer_solution.meshes, layer_solution.values):
                # Check values at each vertex
                for vertex in msh.vertices:
                    value = values[vertex]
                    self.min_value = min(self.min_value, value)
                    self.max_value = max(self.max_value, value)
        
        # If no values were found or if all values are the same
        if self.min_value == float('inf') or self.min_value == self.max_value:
            self.min_value = 0.0
            self.max_value = 1.0
        
        # Emit signal to notify about the new value range
        self.valueRangeChanged.emit(self.min_value, self.max_value)

    def autoscaleXY(self):
        """
        Automatically adjust the offset and scale to fit all meshes in the view.
        Sets the view to display all meshes with a small margin around them.
        """
        if not self.solution or not self.solution.layer_solutions:
            return  # Nothing to scale if no solution is loaded
        
        # Find the bounds of all meshes across all layers
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for layer_solution in self.solution.layer_solutions:
            # Iterate through all meshes in the layer solution
            for msh in layer_solution.meshes:
                for vertex in msh.vertices:
                    x, y = vertex.p.x, vertex.p.y
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
        
        # Check if we found any vertices
        if min_x == float('inf'):
            return  # No vertices found
        
        # Calculate center point and dimensions
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        # Set view center (negative offset to move view)
        self.offset_x = -center_x
        self.offset_y = -center_y
        
        # Set scale to fit everything with a 10% margin
        # Consider the aspect ratio of the viewport
        margin_factor = 0.9  # 10% margin
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        
        if width > 0 and height > 0:
            # Scale based on the larger dimension, accounting for aspect ratio
            if width * aspect > height:
                self.scale = margin_factor * 2.0 / width
            else:
                self.scale = margin_factor * 2.0 / (height / aspect)

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
            self.currentLayerChanged.emit(self.visible_layers[self.current_layer_index])

        self.autoscaleValue()
        self.autoscaleXY()

        if self.mesh_shader is not None:
            self.setupMeshData()
            self.setupConnectionPointsData()

        self.update()

    def setupMeshData(self):
        """Set up the mesh data for rendering."""

        # TODO: Clear previously rendered meshes
        assert self.solution is not None

        for layer, lsol in zip(self.solution.problem.layers, self.solution.layer_solutions):
            if layer.name not in self.rendered_meshes:
                self.rendered_meshes[layer.name] = []
            
            for msh, values in zip(lsol.meshes, lsol.values):
                # Create a RenderedMesh object for each mesh
                self.rendered_meshes[layer.name].append(
                    RenderedMesh.from_mesh(msh, values)
                )

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
        
        self.edge_shader = ShaderProgram.from_source(
            VERTEX_SHADER_EDGES, FRAGMENT_SHADER_EDGES
        )
        
        self.points_shader = ShaderProgram.from_source(
            VERTEX_SHADER_POINTS, FRAGMENT_SHADER_POINTS
        )

        # Set the color map uniform to Gradient.rainbow
        with self.mesh_shader.use():
            color_map_uniform = self.mesh_shader.shader_program.uniformLocation("color_map")
            # Render 256 colors from the color map
            colors = np.array([COLOR_MAP(i / 255)[0:3] for i in range(256)],
                              dtype=np.float32)
            gl.glUniform3fv(color_map_uniform, 256, colors)
        
        # If meshes are already set, setup the mesh data
        if self.solution:
            self.setupMeshData()
            self.setupConnectionPointsData()

    def resizeGL(self, width, height):
        """Handle window resizing."""
        gl.glViewport(0, 0, width, height)

    def _computeMVP(self):
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        
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

    def _render_mesh_triangles(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]):
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
                self.min_value
            )
            gl.glUniform1f(
                self.mesh_shader.shader_program.uniformLocation("v_max"),
                self.max_value
            )

            # Draw triangles for current layer only
            for rmesh in rendered_mesh_list:
                rmesh.render_triangles()

    def _render_mesh_edges(self, mvp: np.ndarray, rendered_mesh_list: list[RenderedMesh]):
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

    def _render_connection_points(self, mvp: np.ndarray, rendered_points_obj: Optional[RenderedPoints]):
        """Renders the connection points for the current layer."""
        if not self.connection_points_visible or not self.points_shader:
            return
        
        if not rendered_points_obj or rendered_points_obj.point_count == 0:
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
        
        if not self.mesh_shader or not self.rendered_meshes or not self.visible_layers:
            log.debug("No shader program or meshes to render")
            return
        
        mvp = self._computeMVP()
        
        # Get current layer name
        current_layer_name = self.visible_layers[self.current_layer_index]
        
        # Ensure current_layer_name exists in rendered_meshes before accessing
        if current_layer_name not in self.rendered_meshes:
            log.warning(f"Current layer {current_layer_name} not found in rendered_meshes.")
            return
            
        current_layer_mesh_list = self.rendered_meshes[current_layer_name]
        self._render_mesh_triangles(mvp, current_layer_mesh_list)
        self._render_mesh_edges(mvp, current_layer_mesh_list)

        # Use .get() for safer access, as a layer might not have connection points
        rendered_points_obj = self.rendered_connection_points.get(current_layer_name)
        self._render_connection_points(mvp, rendered_points_obj)
        
        gl.glBindVertexArray(0)

    def _screen_to_world(self, screen_pos: QtCore.QPointF) -> mesh.Point:
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

        aspect = self.width() / self.height()
        
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
        world_point = self._screen_to_world(event.position())
        self.meshClicked.emit(world_point, event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse movement."""
        if event.buttons() & Qt.LeftButton and self.last_mouse_screen_pos is not None:
            delta = event.position() - self.last_mouse_screen_pos
            dx = float(delta.x())
            dy = float(delta.y())
            
            self.screenDragged.emit(dx, dy, event)
            
            self.last_mouse_screen_pos = event.position()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton and self.last_mouse_screen_pos is not None:
            # TODO: Potentially emit a clickReleased signal if tools need it
            # Clear drag state
            self.last_mouse_screen_pos = None

    def pan_view_by_screen_delta(self, dx_screen: float, dy_screen: float):
        """
        Pans the view based on a screen delta.
        
        Args:
            dx_screen: Change in x screen coordinate.
            dy_screen: Change in y screen coordinate.
        """
        if self.width() <= 0 or self.height() <= 0:
            return

        aspect = self.width() / self.height()
        
        # Convert screen delta to world delta
        # Horizontal movement (adjusted for aspect ratio)
        dx_world = (dx_screen / self.width()) * (2.0 / self.scale) * aspect
        
        # Vertical movement (note: Qt's y axis points down, OpenGL Y-axis was flipped in projection)
        # A positive dy_screen (mouse down) should result in a positive dy_world (content moves down)
        dy_world = (dy_screen / self.height()) * (2.0 / self.scale)
        
        self.offset_x += dx_world
        self.offset_y += dy_world
        self.update()

    def set_min_value_from_world_point(self, world_point: mesh.Point):
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
        
        if value > self.max_value:
            self.min_value = value
            self.max_value = value
        else:
            self.min_value = value
        self.valueRangeChanged.emit(self.min_value, self.max_value)
        self.update()

    def set_max_value_from_world_point(self, world_point: mesh.Point):
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
        
        if value < self.min_value:
            self.max_value = value
            self.min_value = value
        else:
            self.max_value = value
        self.valueRangeChanged.emit(self.min_value, self.max_value)
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
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
        # but _screen_to_world should still compute a value.
        # Alternatively, could use center of view if mouse is outside. For now, use cursor.
        world_point = self._screen_to_world(screen_pos)
        
        # Emit signal for ToolManager to handle general shortcuts
        self.keyPressedInMesh.emit(world_point, event.key(), event.modifiers())

        if event.key() == Qt.Key_V:
            self.switchToNextLayer()
        # else: # Allow other key events to be processed if not handled by shortcuts or specific keys
            # super().keyPressEvent(event) # This might not be needed if all keys are handled via signal or specific checks
        
        # If the event was not accepted by any tool via the signal or specific key checks,
        # then call superclass implementation.
        if not event.isAccepted():
            super().keyPressEvent(event)
            
    def switchToNextLayer(self):
        """Switch to the next layer in the cycle."""
        if not self.visible_layers:
            return
            
        # Move to next layer index
        self.current_layer_index = (self.current_layer_index + 1) % len(self.visible_layers)
        current_layer = self.visible_layers[self.current_layer_index]
        
        # Emit signal with the current layer name
        self.currentLayerChanged.emit(current_layer)
        
        # Refresh the display
        self.update()

    @Slot(bool)
    def set_edges_visible(self, visible: bool):
        """Slot to set the visibility of mesh edges."""
        self.edges_visible = visible
        log.debug(f"Mesh edges visibility set to: {self.edges_visible}")
        self.update()

    @Slot(bool)
    def set_connection_points_visible(self, visible: bool):
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
            log.warning(f"Attempted to set current layer to unknown layer: {layer_name}")


class ColorScaleWidget(QWidget):
    """Widget that displays a color scale with delta and absolute range."""
    
    # Signal to notify when unit is changed manually
    unitChanged = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.v_min = 0.0
        self.v_max = 1.0
        self.unit = "V"  # Default unit
        
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
    
    def updateLabels(self):
        """Update the delta and range labels."""
        delta = self.v_max - self.v_min
        delta_str = units.Value(delta, self.unit).pretty_format()
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
            color = COLOR_MAP(t)
            
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
        
        # Connect signals/slots
        self.mesh_viewer.valueRangeChanged.connect(self.color_scale.setRange)
        self.projectLoaded.connect(self.mesh_viewer.setSolution)
        self.mesh_viewer.currentLayerChanged.connect(self.updateCurrentLayer)
        self.mesh_viewer.availableLayersChanged.connect(self.app_toolbar.updateLayerSelectionMenu)
        self.mesh_viewer.currentLayerChanged.connect(self.app_toolbar.updateActiveLayerInMenu)
        
        # Connect the ToolManager
        self.mesh_viewer.meshClicked.connect(self.tool_manager.handle_mesh_click)
        self.mesh_viewer.screenDragged.connect(self.tool_manager.handle_screen_drag)
        self.mesh_viewer.keyPressedInMesh.connect(self.tool_manager.handle_key_press_in_mesh)
        
        self.projectLoaded.emit(solution)

    # Removed loadProject method

    def updateCurrentLayer(self, layer_name):
        """Update the window title to show the current layer."""
        self.setWindowTitle(f"padne: {self.project_file_name} - {layer_name}")


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
