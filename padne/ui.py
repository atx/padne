#!/usr/bin/env python3

import contextlib
import logging
import numpy as np
import matplotlib
import sys
import OpenGL.GL as gl

from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

from PySide6 import QtGui
from PySide6.QtCore import Qt, Signal, Slot, QRect
from PySide6.QtGui import QSurfaceFormat, QPainter, QPen, QColor
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout

from . import kicad, mesh, solver


def pretty_format_si_number(value: float, unit: str):
    """Pretty format a number with SI prefix and unit.
    
    Args:
        value: The numeric value to format
        unit: The unit symbol/abbreviation
        
    Returns:
        A formatted string with the value, appropriate SI prefix, and unit
    
    Examples:
        >>> pretty_format_si_number(0.000001, "A")
        '1.000 μA'
        >>> pretty_format_si_number(1500, "V")
        '1.500 kV'
    """
    if value == 0:
        return f"0 {unit}"
        
    # Define SI prefixes and their corresponding powers of 10
    prefixes = {
        -12: "p",  # pico
        -9: "n",   # nano
        -6: "μ",   # micro
        -3: "m",   # milli
        0: "",     # base unit
        3: "k",    # kilo
        6: "M",    # mega
        9: "G",    # giga
        12: "T"    # tera
    }
    
    # Determine the appropriate prefix for the value
    abs_value = abs(value)
    exponent = 0
    
    if abs_value < 1e-10:
        return f"0 {unit}"  # Treat very small values as zero
        
    if abs_value >= 1:
        while abs_value >= 1000 and exponent < 12:
            abs_value /= 1000
            exponent += 3
    else:
        while abs_value < 1 and exponent > -12:
            abs_value *= 1000
            exponent -= 3
    
    # Format the value with the appropriate precision
    # Use fewer decimal places for larger numbers
    if abs_value >= 100:
        formatted_value = f"{abs_value:.1f}"
    elif abs_value >= 10:
        formatted_value = f"{abs_value:.2f}"
    else:
        formatted_value = f"{abs_value:.3f}"
    
    # Remove trailing zeros after decimal point
    if "." in formatted_value:
        formatted_value = formatted_value.rstrip("0").rstrip(".")
    
    # Apply the sign from the original value
    if value < 0:
        formatted_value = "-" + formatted_value
    
    # Return the formatted string with prefix and unit
    return f"{formatted_value} {prefixes[exponent]}{unit}"

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


COLOR_MAP = matplotlib.colormaps["viridis"]


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


class MeshViewer(QOpenGLWidget):
    # Signal to notify when the value range changes
    valueRangeChanged = Signal(float, float)
    # Signal to notify when the current layer changes
    currentLayerChanged = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.solution: None | solver.Solution = None
        # Layer name -> RenderedMesh
        self.rendered_meshes: dict[str, list] = {}

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.last_pos = None
        self.setMouseTracking(True)
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Layer management
        self.current_layer_index = 0
        self.visible_layers = []  # Will hold names of layers in order
        
        # OpenGL objects
        self.mesh_shader = None
        self.edge_shader = None

    def _getNearestValue(self, world_x: float, world_y: float) -> Optional[float]:
        """
        Find the value at the vertex closest to the specified world coordinates.
        
        Args:
            world_x: X-coordinate in world space
            world_y: Y-coordinate in world space
            
        Returns:
            The value at the nearest vertex, or None if no vertices are found
        """
        if not self.solution or not self.visible_layers or self.current_layer_index >= len(self.visible_layers):
            return None

        # TODO: Eventually, we want to have a spatial index for this
        
        # Get current layer name
        current_layer = self.visible_layers[self.current_layer_index]
        
        # Check if this layer has rendered meshes
        if current_layer not in self.rendered_meshes:
            return None
        
        # Create a point at the specified world coordinates
        target_point = mesh.Point(world_x, world_y)
        
        # Initialize variables to track the closest vertex
        closest_distance = float('inf')
        closest_value = None
        
        # Find the corresponding layer solution
        layer_index = None
        for i, layer in enumerate(self.solution.problem.layers):
            if layer.name == current_layer:
                layer_index = i
                break
        
        if layer_index is None:
            return None
        
        layer_solution = self.solution.layer_solutions[layer_index]
        
        # Check each mesh in the current layer
        for mesh_index, msh in enumerate(layer_solution.meshes):
            values = layer_solution.values[mesh_index]
            
            # Check each vertex in the mesh
            for vertex in msh.vertices:
                # Calculate distance to this vertex
                distance = vertex.p.distance(target_point)
                
                # Update closest vertex if this one is closer
                if distance < closest_distance:
                    closest_distance = distance
                    closest_value = values[vertex]
        
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

    def setSolution(self, solution: solver.Solution):
        """Set the solution for the mesh viewer."""
        self.solution = solution

        # Initialize the list of layers from the solution
        self.visible_layers = [layer.name for layer in solution.problem.layers]
        self.current_layer_index = 0
        
        # Emit signal with initial layer
        if self.visible_layers:
            self.currentLayerChanged.emit(self.visible_layers[self.current_layer_index])

        self.autoscaleValue()
        self.autoscaleXY()

        if self.mesh_shader is not None:
            self.setupMeshData()

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

    def initializeGL(self):
        """Initialize OpenGL settings."""
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Background
        gl.glDisable(gl.GL_CULL_FACE)
        
        # Create and compile shaders
        self.mesh_shader = ShaderProgram.from_source(
            VERTEX_SHADER_MESH, FRAGMENT_SHADER_MESH
        )
        
        self.edge_shader = ShaderProgram.from_source(
            VERTEX_SHADER_EDGES, FRAGMENT_SHADER_EDGES
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

    def paintGL(self):
        """Render the mesh using shaders."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        if not self.mesh_shader or not self.rendered_meshes or not self.visible_layers:
            print("No shader program or meshes to render")
            return
        
        mvp = self._computeMVP()
        
        # Get current layer name
        current_layer = self.visible_layers[self.current_layer_index]
        
        # Only proceed if the current layer has rendered meshes
        if current_layer not in self.rendered_meshes:
            return
        
        # Draw triangles with mesh shader
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
            for rmesh in self.rendered_meshes[current_layer]:
                rmesh.render_triangles()
        
        # Draw edges with edge shader
        with self.edge_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.edge_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )
            
            # Draw edges for current layer only
            for rmesh in self.rendered_meshes[current_layer]:
                rmesh.render_edges()
        
        gl.glBindVertexArray(0)

    def mousePressEvent(self, event):
        """Handle mouse press events for panning."""
        self.last_pos = event.position()
        self.setFocus()  # Ensure the widget gets focus when clicked

    def _handlePanning(self, current_pos):
        """
        Handle the panning logic when the user drags with the left mouse button.
        
        Args:
            current_pos: The current mouse position
        """
        # Calculate the current aspect ratio
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        
        # Convert screen coordinates to world coordinates
        delta = current_pos - self.last_pos
        
        # The conversion factor should consider:
        # 1. The current scale
        # 2. The viewport size
        # 3. The orthographic projection bounds
        
        # Horizontal movement (adjusted for aspect ratio)
        dx_world = (delta.x() / self.width()) * (2.0 / self.scale) * aspect
        
        # Vertical movement (note: Qt's y axis points down, now consistent with flipped OpenGL Y-axis)
        dy_world = (delta.y() / self.height()) * (2.0 / self.scale)
        
        # Update offsets
        self.offset_x += dx_world
        self.offset_y += dy_world
        
        # Update the last position
        self.last_pos = current_pos

    def mouseMoveEvent(self, event):
        """Handle mouse movement for panning."""
        if event.buttons() & Qt.LeftButton and self.last_pos is not None:
            self._handlePanning(event.position())
            self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_factor = 1.2
        if event.angleDelta().y() > 0:
            self.scale *= zoom_factor
        else:
            self.scale /= zoom_factor
        self.update()
        
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_V:
            self.switchToNextLayer()
        elif event.key() == Qt.Key_M:
            # Shift+M: Set maximum value from cursor position
            if event.modifiers() & Qt.ShiftModifier:
                self.setMaxValueFromCursor()
            # m: Set minimum value from cursor position
            else:
                self.setMinValueFromCursor()
        else:
            super().keyPressEvent(event)
            
    def _getValueFromCursor(self) -> float:
        """
        Helper method to get the value at the cursor position in world coordinates.
        
        Returns:
            The value at the nearest vertex, or None if no value could be found
        """
        # Get cursor position in screen coordinates
        cursor_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        
        # Convert to OpenGL viewport coordinates
        viewport_x = cursor_pos.x()
        viewport_y = cursor_pos.y()  # No need to flip y-coordinate since we flipped the projection matrix
        
        # Convert to normalized device coordinates (NDC)
        ndc_x = 2.0 * viewport_x / self.width() - 1.0
        ndc_y = 2.0 * viewport_y / self.height() - 1.0
        
        # Convert to world coordinates
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        world_x = (ndc_x * aspect / self.scale) - self.offset_x
        world_y = (ndc_y / self.scale) - self.offset_y
        
        # Get nearest value at this position
        return self._getNearestValue(world_x, world_y)
        
    def setMinValueFromCursor(self):
        """Set the minimum value of the color scale from the cursor position.
        If the selected value is greater than the current maximum, both min and max
        are set to the selected value.
        """
        value = self._getValueFromCursor()
        
        # Update minimum value if a valid value was found
        if value is not None:
            if value > self.max_value:
                # If new min is above current max, clamp range to this value
                self.min_value = value
                self.max_value = value
            else:
                # Otherwise, just set the new minimum
                self.min_value = value
            # Emit signal to notify about the new value range
            self.valueRangeChanged.emit(self.min_value, self.max_value)
            self.update()

    def setMaxValueFromCursor(self):
        """Set the maximum value of the color scale from the cursor position.
        If the selected value is less than the current minimum, both min and max
        are set to the selected value.
        """
        value = self._getValueFromCursor()
        
        # Update maximum value if a valid value was found
        if value is not None:
            if value < self.min_value:
                # If new max is below current min, clamp range to this value
                self.max_value = value
                self.min_value = value
            else:
                # Otherwise, just set the new maximum
                self.max_value = value
            # Emit signal to notify about the new value range
            self.valueRangeChanged.emit(self.min_value, self.max_value)
            self.update()

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


class ColorScaleWidget(QWidget):
    """Widget that displays a color scale with delta and absolute range."""
    
    # Signal to notify when unit is changed manually
    unitChanged = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.v_min = 0.0
        self.v_max = 1.0
        self.unit = "V"  # Default unit
        
        self.setMinimumWidth(110)  # Increased minimum width for range label
        self.setMinimumHeight(200)  # Set a reasonable minimum height
        
        # New labels
        self.delta_label = None
        self.range_label = None
        
        self.setupUI()

    
    def setupUI(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)  # Add a little vertical spacing
        
        # Delta label at the top of the stretch area
        self.delta_label = QLabel(f"Δ = 0 {self.unit}")  # Placeholder text
        self.delta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.delta_label)
        
        # This stretch is where we'll paint our gradient
        layout.addStretch(10)
        
        # Range label at the bottom showing absolute min/max values
        self.range_label = QLabel(f"Range: 0 {self.unit} - 0 {self.unit}")  # Placeholder text
        self.range_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
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
        self.update()  # Trigger repaint
    
    @Slot(str)
    def setUnit(self, unit):
        """Set the unit for the scale."""
        self.unit = unit
        self.updateLabels()
    
    def updateLabels(self):
        """Update the delta and range labels."""
        delta = self.v_max - self.v_min
        delta_str = pretty_format_si_number(delta, self.unit)
        min_str = pretty_format_si_number(self.v_min, self.unit)
        max_str = pretty_format_si_number(self.v_max, self.unit)
        
        self.delta_label.setText(f"Δ = {delta_str}")
        self.range_label.setText(f"({min_str} → {max_str})")  # Show absolute range context
    
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
        if gradient_rect.height() > 0:
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
    def __init__(self, kicad_pro_path):
        super().__init__()

        self.setWindowTitle("PDN Simulator Viewer")
        self.setGeometry(100, 100, 900, 600)  # Slightly wider to accommodate color scale
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Set zero padding/margin
        main_layout.setSpacing(0)  # Remove spacing between widgets
        
        # Create the mesh viewer
        self.mesh_viewer = MeshViewer(self)
        
        # Create color scale widget
        self.color_scale = ColorScaleWidget(self)
        self.color_scale.setFixedWidth(120)
        
        # Add widgets to layout
        main_layout.addWidget(self.mesh_viewer)  # Mesh viewer takes remaining space
        main_layout.addWidget(self.color_scale)  # Color scale has fixed width
        
        # Set the main widget as central widget
        self.setCentralWidget(main_widget)
        
        # Connect signals/slots
        self.mesh_viewer.valueRangeChanged.connect(self.color_scale.setRange)
        
        # Load and mesh the KiCad project
        self._configureLogging()
        self.loadProject(kicad_pro_path)

    def _configureLogging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
        
    def loadProject(self, kicad_pro_path):
        """Load a KiCad project and display the F.Cu layer."""
        # Load the KiCad project
        print(f"Loading project: {kicad_pro_path}")
        prob = kicad.load_kicad_project(Path(kicad_pro_path))
        
        # Solve the problem to get the values for visualization
        solution = solver.solve(prob)

        self.mesh_viewer.setSolution(solution)
        
        # Connect the layer change signal to update the window title
        self.mesh_viewer.currentLayerChanged.connect(self.updateCurrentLayer)
        
        # Set an appropriate unit (assuming voltage for now)
        self.color_scale.setUnit("V")
            
    def updateCurrentLayer(self, layer_name):
        """Update the window title to show the current layer."""
        self.setWindowTitle(f"PDN Simulator Viewer - Layer: {layer_name}")


def configure_opengl():
    """Configure OpenGL settings for the application."""
    # Create OpenGL format
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)  # Use OpenGL 3.3
    gl_format.setProfile(QSurfaceFormat.CoreProfile)  # Use core profile
    gl_format.setSamples(4)  # Enable 4x antialiasing
    QSurfaceFormat.setDefaultFormat(gl_format)


def main():
    """Main entry point for the UI application."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python -m padne.ui <path_to_kicad_pro_file>")
        return 1
    
    kicad_pro_path = sys.argv[1]
    
    # Configure OpenGL
    configure_opengl()
    
    # Create and run application
    app = QApplication(sys.argv)
    window = MainWindow(kicad_pro_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
