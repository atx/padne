#!/usr/bin/env python3

import sys
import numpy as np
import contextlib
import OpenGL.GL as gl
from pathlib import Path
from dataclasses import dataclass, field

from PySide6 import QtGui
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, Signal, Slot, QRect
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QPainter, QPen, QColor
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader

from padne import kicad, mesh, solver
from padne.mesh import ZeroForm


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


@dataclass
class Gradient:
    colors: np.ndarray
    steps: list

    @classmethod
    def construct(cls, input: list[tuple[tuple[float, float, float], float]]):
        colors = np.array([np.array(c) for c, _ in input], dtype=np.float32)
        steps = [s for _, s in input]
        return cls(colors, steps)

    def __post_init__(self):
        assert len(self.colors) == len(self.steps)
        assert all(0.0 <= s <= 1.0 for s in self.steps)
        for i in range(len(self.steps) - 1):
            assert self.steps[i] < self.steps[i + 1]

    def value(self, x: float):
        if x <= self.steps[0]:
            return self.colors[0]
        if x >= self.steps[-1]:
            return self.colors[-1]

        for i in range(len(self.steps) - 1):
            if self.steps[i] <= x < self.steps[i + 1]:
                t = (x - self.steps[i]) / (self.steps[i + 1] - self.steps[i])
                return self.colors[i] * (1 - t) + self.colors[i + 1] * t

        assert False, "Should not reach here"


Gradient.rainbow = Gradient.construct([
    [[0.0, 0.0, 1.0], 0.00],
    [[0.0, 0.5, 1.0], 0.15],
    [[0.0, 1.0, 0.0], 0.35],
    [[0.8, 0.8, 0.0], 0.60],
    [[1.0, 0.5, 0.0], 0.80],
    [[1.0, 0.0, 0.0], 1.00],
])


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
    def from_mesh(cls, msh: mesh.Mesh, values: ZeroForm):
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
                   len(edge_vertices) // 2)

    def render_triangles(self):
        gl.glBindVertexArray(self.vao_triangles)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.triangle_count)

    def render_edges(self):
        gl.glBindVertexArray(self.vao_edges)
        gl.glDrawArrays(gl.GL_LINES, 0, self.edge_count)


class MeshViewer(QOpenGLWidget):
    # Signal to notify when the value range changes
    valueRangeChanged = Signal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.meshes = []
        self.mesh_values = None
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.last_pos = None
        self.setMouseTracking(True)
        
        # OpenGL objects
        self.mesh_shader = None
        self.edge_shader = None

    def setMeshes(self, meshes, values=None):
        self.meshes = meshes
        self.mesh_values = values
        
        # Calculate min and max values for scaling
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        if values:
            for zeroform in values:
                # Get min/max from all vertices in the mesh
                for vertex in zeroform.mesh.vertices:
                    value = zeroform[vertex]
                    self.min_value = min(self.min_value, value)
                    self.max_value = max(self.max_value, value)
        
        # If min_value is still infinity, set defaults
        if self.min_value == float('inf'):
            self.min_value = 0.0
            self.max_value = 1.0
            
        # Emit signal with the new value range
        self.valueRangeChanged.emit(self.min_value, self.max_value)
        
        # Calculate the bounds of all meshes to set initial view
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for m in meshes:
            for vertex in m.vertices:
                x, y = vertex.p.x, vertex.p.y
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        # Set initial view to show all meshes
        if min_x != float('inf'):
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            
            self.offset_x = -center_x
            self.offset_y = -center_y
            
            # Set scale to fit everything with a small margin
            if width > 0 and height > 0:
                self.scale = min(1.8 / width, 1.8 / height)
        
        # If OpenGL is initialized, setup the mesh data in GPU
        if self.mesh_shader is not None:
            self.setup_mesh_data()
        
        self.update()

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
            color_map = Gradient.rainbow
            color_map_uniform = self.mesh_shader.shader_program.uniformLocation("color_map")
            # Render 256 colors from the color map
            colors = np.array([color_map.value(i / 255) for i in range(256)], dtype=np.float32)
            gl.glUniform3fv(color_map_uniform, 256, colors)
        
        # If meshes are already set, setup the mesh data
        if self.meshes:
            self.setup_mesh_data()
            
    def setup_mesh_data(self):
        """Set up the VAOs and VBOs for rendering."""
        # TODO: Clean up previous rendered meshes
        self.rendered_meshes = []
        
        print(f"Setting up mesh data for {len(self.meshes)} meshes")
        
        for m, mesh_values in zip(self.meshes, self.mesh_values):
            # Process each face to create triangle data
            self.rendered_meshes.append(RenderedMesh.from_mesh(m, mesh_values))

    def resizeGL(self, width, height):
        """Handle window resizing."""
        gl.glViewport(0, 0, width, height)

    def _compute_mvp(self):
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        
        # Create a 2D orthographic projection matrix
        ortho_scale = 1.0 / self.scale
        left = -ortho_scale * aspect
        right = ortho_scale * aspect
        bottom = -ortho_scale
        top = ortho_scale
        near = -1.0
        far = 1.0
        
        # Define the matrix components
        proj_matrix = np.array([
            [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
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
        
        if not self.mesh_shader or not self.meshes:
            print("No shader program or meshes to render")
            return
        
        mvp = self._compute_mvp()
        
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
            
            # Draw triangles
            for rendered_mesh in self.rendered_meshes:
                rendered_mesh.render_triangles()
        
        # Draw edges with edge shader
        with self.edge_shader.use():
            # Set the MVP uniform
            gl.glUniformMatrix4fv(
                self.edge_shader.shader_program.uniformLocation("mvp"),
                1, gl.GL_TRUE, mvp.flatten()
            )
            
            # Draw edges
            for rendered_mesh in self.rendered_meshes:
                rendered_mesh.render_edges()
        
        gl.glBindVertexArray(0)

    def mousePressEvent(self, event):
        """Handle mouse press events for panning."""
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for panning."""
        if event.buttons() & Qt.LeftButton and self.last_pos is not None:
            delta = event.position() - self.last_pos
            self.offset_x += delta.x() / self.scale / self.width() * 2
            self.offset_y -= delta.y() / self.scale / self.height() * 2
            self.last_pos = event.position()
            self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_factor = 1.2
        if event.angleDelta().y() > 0:
            self.scale *= zoom_factor
        else:
            self.scale /= zoom_factor
        self.update()


class ColorScaleWidget(QWidget):
    """Widget that displays a color scale with min/max values and units."""
    
    # Signal to notify when unit is changed manually
    unitChanged = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gradient = Gradient.rainbow
        self.v_min = 0.0
        self.v_max = 1.0
        self.unit = "V"  # Default unit
        
        self.setMinimumWidth(80)  # Set a reasonable minimum width
        self.setMinimumHeight(200)  # Set a reasonable minimum height
        
        self.setupUI()

    
    def setupUI(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Max value label at the top of the stretch area
        self.max_label = QLabel(f"{self.v_max:.3f} {self.unit}")
        self.max_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.max_label)
        
        # This stretch is where we'll paint our gradient
        layout.addStretch(10)
        
        # Min value label at the bottom of the stretch area
        self.min_label = QLabel(f"{self.v_min:.3f} {self.unit}")
        self.min_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.min_label)
    
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
        """Update the min/max labels with current values and unit."""
        self.min_label.setText(pretty_format_si_number(self.v_min, self.unit))
        self.max_label.setText(pretty_format_si_number(self.v_max, self.unit))
    
    def paintEvent(self, event):
        """Paint the color gradient scale."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Find the rectangle where we should draw the gradient
        # This should be between the max_label and min_label
        content_rect = self.rect()
        top_margin = self.max_label.y() + self.max_label.height()
        bottom_margin = self.height() - self.min_label.y()
        
        # Calculate the gradient bar rectangle centered horizontally
        bar_width = 20
        gradient_rect = QRect(
            content_rect.left() + (content_rect.width() - bar_width) // 2,  # Center horizontally
            top_margin,
            bar_width,
            content_rect.height() - top_margin - bottom_margin
        )
        
        # Draw gradient bar border
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(gradient_rect)
        
        # Draw the gradient
        for i in range(gradient_rect.height()):
            # Map position to color
            t = 1.0 - (i / gradient_rect.height())
            color = self.gradient.value(t)
            
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
        self.color_scale.setFixedWidth(100)  # Fixed width of 100 pixels
        
        # Add widgets to layout
        main_layout.addWidget(self.mesh_viewer)  # Mesh viewer takes remaining space
        main_layout.addWidget(self.color_scale)  # Color scale has fixed width
        
        # Set the main widget as central widget
        self.setCentralWidget(main_widget)
        
        # Connect signals/slots
        self.mesh_viewer.valueRangeChanged.connect(self.color_scale.setRange)
        
        # Load and mesh the KiCad project
        self.loadProject(kicad_pro_path)
        
    def loadProject(self, kicad_pro_path):
        """Load a KiCad project and display the F.Cu layer."""
        try:
            # Load the KiCad project
            print(f"Loading project: {kicad_pro_path}")
            prob = kicad.load_kicad_project(Path(kicad_pro_path))
            
            # Solve the problem to get the values for visualization
            print("Solving problem...")
            solution = solver.solve(prob)
            
            # Find the F.Cu layer and its solution
            f_cu_layer = None
            f_cu_solution = None
            for i, layer in enumerate(prob.layers):
                if layer.name == "F.Cu":
                    f_cu_layer = layer
                    f_cu_solution = solution.layer_solutions[i]
                    break
            
            if f_cu_layer is None:
                print("Error: F.Cu layer not found in the project")
                return
            
            # Display the meshes from the solution
            print(f"Displaying {len(f_cu_solution.meshes)} mesh regions")
            # Pass both meshes and solution values to the mesh viewer
            self.mesh_viewer.setMeshes(f_cu_solution.meshes, f_cu_solution.values)
            
            # Set an appropriate unit (assuming voltage for now)
            self.color_scale.setUnit("V")
            
        except Exception as e:
            print(f"Error loading project: {str(e)}")
            import traceback
            traceback.print_exc()


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
