import pygame
import moderngl
import numpy as np
from pygame.locals import *

# Initialize pygame and create window
pygame.init()
window_size = (800, 600)
#window = pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)
window = pygame.display.set_mode(window_size)

# Create ModernGL context
#ctx = moderngl.create_context()
ctx = moderngl.create_context(standalone=True)

# Shader program (vertex and fragment shader)
with open("./shaders/demo_vertex.vert", "r") as vertFile:
    vertex_shader = vertFile.read()

with open("./shaders/demo_fragment.frag", "r") as fragmentFile:
    fragment_shader = fragmentFile.read()

# Create shader program
program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader,
)


def drawFrameFromFrameBuffer(frameBuffer):
    frameBufferSurface = pygame.image.frombuffer(frameBuffer.read(), fbo.size, "RGB")
    window.blit(frameBufferSurface, (0,0))
    

# Main render loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    ctx.clear()

    x = np.linspace(-1.0, 1.0, 50)
    y = np.random.rand(50) - 0.5
    r = np.ones(50)
    g = np.zeros(50)
    b = np.zeros(50)

    vertices = np.dstack([x, y, r, g, b])
    vbo = ctx.buffer(vertices.astype('f4').tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'in_vert', 'in_color')

    fbo = ctx.simple_framebuffer(window_size)
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.LINE_STRIP)

    
    drawFrameFromFrameBuffer(fbo)

    # Swap buffers
    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()
