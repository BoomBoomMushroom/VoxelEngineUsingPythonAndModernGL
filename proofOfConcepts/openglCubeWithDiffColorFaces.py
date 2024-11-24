import pygame
import moderngl
import numpy as np
from pygame.locals import *
from PIL import Image
from glm import mat4, translate, rotate, perspective, lookAt

# Initialize pygame and create window
pygame.init()
window_size = (800, 600)
window = pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)

# Create ModernGL context
ctx = moderngl.create_context()

# Enable depth testing to fix face rendering order
ctx.enable(moderngl.DEPTH_TEST)


# Shader program (vertex and fragment shader)
vertex_shader = """
#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_vert;
in vec2 in_texcoord;
out vec2 frag_texcoord;

void main() {
    frag_texcoord = in_texcoord;
    gl_Position = projection * view * model * vec4(in_vert, 1.0);
}
"""

fragment_shader = """
#version 330
in vec2 frag_texcoord;
out vec4 color;
uniform sampler2D tex;

void main() {
    color = texture(tex, frag_texcoord);
}
"""

# Create shader program
program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader,
)

# Cube vertices (positions, UV coordinates)
vertices = np.array([
    # Front face
    -0.5, -0.5, -0.5,  0.0, 0.0,
     0.5, -0.5, -0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5,  0.5, -0.5,  0.0, 1.0,

    # Back face
    -0.5, -0.5,  0.5,  0.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
    -0.5,  0.5,  0.5,  0.0, 1.0,
], dtype='f4')

# Indices for each face (2 triangles per face)
face_indices = [
    np.array([0, 1, 2, 0, 2, 3], dtype='i4'),  # Front face
    np.array([4, 5, 6, 4, 6, 7], dtype='i4'),  # Back face
    np.array([0, 1, 5, 0, 5, 4], dtype='i4'),  # Bottom face
    np.array([2, 3, 7, 2, 7, 6], dtype='i4'),  # Top face
    np.array([0, 3, 7, 0, 7, 4], dtype='i4'),  # Left face
    np.array([1, 2, 6, 1, 6, 5], dtype='i4')   # Right face
]

# Vertex buffer object (VBO)
vbo = ctx.buffer(vertices.tobytes())

# Load textures
textures = []
for i in range(1, 7):  # texture_1.png to texture_6.png
    img_path = f"./demo_textures/texture_{i}.png"
    img = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure compatibility
    texture = ctx.texture(img.size, 3, img.tobytes())
    texture.build_mipmaps()  # Optional for better quality at smaller sizes
    textures.append(texture)

# Set up the view and projection matrices for 3D rendering
projection = perspective(45.0, window_size[0] / window_size[1], 0.1, 100.0)
view = lookAt((2.0, 2.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
model = mat4(1.0)

# Set uniforms for model, view, and projection matrices
program['projection'].write(projection)
program['view'].write(view)

# Create one VAO per face
vaos = []
for face_index in face_indices:
    ibo = ctx.buffer(face_index.tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'in_vert', 'in_texcoord', index_buffer=ibo)
    vaos.append(vao)

# Main render loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    ctx.clear()

    # Set model matrix (no rotation, no translation)
    program['model'].write(model)

    # Bind each texture and render the corresponding face
    for i in range(6):
        textures[i].use()  # Bind the texture
        vaos[i].render(moderngl.TRIANGLES)  # Render the face with its VAO

    # Swap buffers
    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
