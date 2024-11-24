from __future__ import annotations
import pygame
import moderngl
import numpy as np
from pygame.locals import *
from PIL import Image
from glm import mat4, translate, rotate, perspective, lookAt, vec3
import math
import json
import os
import functools
import uuid
import threading

# Initialize pygame and create window
pygame.init()
window_size = (800, 600)
window = pygame.display.set_mode(window_size, DOUBLEBUF | OPENGL)

# Create ModernGL context
ctx = moderngl.create_context(share=True)

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
    gl_Position = projection * view * model * vec4(in_vert, 1);
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

textureCache = {}

@functools.cache
def getTexture(img_path, maxWidth=-1, maxHeight=-1, animationIndex=0):
    img = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure compatibility
    
    # Calculate the crop box for the top-right corner
    if maxWidth > 0 and maxHeight > 0:
        #xStart = animationIndex * maxWidth
        xStart = 0
        yStart = animationIndex * maxHeight
        crop_box = (xStart, yStart, xStart+maxWidth, yStart+maxHeight)
        img = img.crop(crop_box)
    
    texture = ctx.texture(img.size, 3, img.tobytes())
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    #texture.build_mipmaps()  # Optional for better quality at smaller sizes
    return texture

@functools.cache
def getAndSaveTextureToCache(img_path):
    texture = getTexture(img_path, 16, 16)
    textureCache[img_path] = texture

class Block:
    def __init__(self, x=0, y=0, z=0, texturePaths: dict={"all": "./texture.png"}, modernGLContext=ctx, modernGlProgram=program, fullFaceArray=[False]*6):
        self.defaultTexturePath = "./texture.png"
        
        self.fullFaceInt = 0b0000_0000
        self.setFullFacesInt(*fullFaceArray)
        
        self.x = x
        self.y = y
        self.z = z
        self.texturePaths = texturePaths
        
        self.ctx = modernGLContext
        self.program = modernGlProgram
        
        self.uuid = uuid.uuid4()
        
        self.vertices = []
        self.face_indices = []
        self.orderOfFaces = ["all"]
        
        self.vbo = None
        self.objectDataVAOs = None
        self.activeTextures = {
            "front": self.defaultTexturePath,
            "back": self.defaultTexturePath,
            "top": self.defaultTexturePath,
            "bottom": self.defaultTexturePath,
            "left": self.defaultTexturePath,
            "right": self.defaultTexturePath,
        }
        
        self.generateMesh()

    # Make this its own function incase we want to redo the whole mesh later. ex. for some reason we change something like its position
    def generateMesh(self):
        self.generateVertices()
        self.generateFaceIndices()
        
        self.generateRenderingBuffersAndArrays()

    # This is its own function because some times we need to update the vertices and which faces are visible.
    # ex if we place or break a block nearby
    def generateRenderingBuffersAndArrays(self):
        self.createVertexBufferObject()
        self.createObjectDataVertexArrayObject()

    def setFullFacesInt(self, front=False, back=False, top=False, bottom=False, left=False, right=False):
        # Basically an int where certain bits are flags; Like the cpu flags on the NES (my emulator)
        # 8 Bits  |  0 = False & 1 = True (duh)
        # 00 = Two unused bit (Most Significant Bit)
        # 0 = Is Front
        # 0 = Is Back
        # 0 = Is Top
        # 0 = Is Bottom
        # 0 = Is Left
        # 0 = Is Right (Least Significant Bit)
        self.fullFaceInt = 0b0000_0000
        if front:   self.fullFaceInt |= 0b0010_0000
        if back:    self.fullFaceInt |= 0b0001_0000
        if top:     self.fullFaceInt |= 0b0000_1000
        if bottom:  self.fullFaceInt |= 0b0000_0100
        if left:    self.fullFaceInt |= 0b0000_0010
        if right:   self.fullFaceInt |= 0b0000_0001
        
        #print(bin(self.fullFaceInt).split("0b")[1].zfill(8))

    def getFullFacesFromInt(self, fullFaceInt=None):
        if fullFaceInt == None: fullFaceInt = self.fullFaceInt
        
        return {
            "front": fullFaceInt & 0b0010_0000,
            "back": fullFaceInt & 0b0001_0000,
            "top": fullFaceInt & 0b0000_1000,
            "bottom": fullFaceInt & 0b0000_0100,
            "left": fullFaceInt & 0b0000_0010,
            "right": fullFaceInt & 0b0000_0001,
        }

    def removeFace(self, faceName):
        """
        try:
            index = self.orderOfFaces.index(faceName)
        except:
            return
            #raise BaseException("Face already removed or non-existent; Theoretically this shouldn't happen right?")
        
        self.face_indices.pop(index)
        self.orderOfFaces.pop(index)
        
        # TODO: Possible problem, it will always regenerate buffers & array when removing a face, in `removeFacesBasedOnNeighbors`
        self.generateRenderingBuffersAndArrays()
        """
        
        vaoIndex = -1
        for i in range(len(self.objectDataVAOs)):
            dataVAO = self.objectDataVAOs[i]
            if dataVAO["face"] == faceName:
                vaoIndex = i
                break
        if vaoIndex != -1:
            self.objectDataVAOs.pop(vaoIndex)

    def removeFacesBasedOnNeighbors(self, neighbors: list[Block]):
        if type(neighbors) == Block: neighbors = [neighbors]
        #self.generateMesh()
        
        #fullFacesOfMe = self.getFullFacesFromInt(self.fullFaceInt)
        
        for neighbor in neighbors:
            if self.uuid == neighbor.uuid: continue
            
            numberOfDiffsInPosition = 0
            diffX = neighbor.x - self.x
            diffY = neighbor.y - self.y
            diffZ = neighbor.z - self.z
            
            if diffX != 0: numberOfDiffsInPosition += 1
            if diffY != 0: numberOfDiffsInPosition += 1
            if diffZ != 0: numberOfDiffsInPosition += 1
            
            if (diffX*diffX + diffY*diffY + diffZ*diffZ) > 2**2:
                continue
            
            if numberOfDiffsInPosition == 0:
                raise RuntimeError("Hey! There are two blocks in the same space! This is not allowed, fix it?")
            elif numberOfDiffsInPosition > 1:
                continue # None of our faces are touching this block, so we can safely skip it
            
            fullFacesOfNeighbor = self.getFullFacesFromInt(neighbor.fullFaceInt)
            
            if diffX == 1 and fullFacesOfNeighbor["right"]: # Block on our Right
                self.removeFace("right")
            if diffX == -1 and fullFacesOfNeighbor["left"]: # Block on our Left
                self.removeFace("left")
            
            if diffY == 1 and fullFacesOfNeighbor["top"]: # Block on Top
                self.removeFace("top")
            if diffY == -1 and fullFacesOfNeighbor["bottom"]: # Block on our Bottom
                self.removeFace("bottom")
            
            if diffZ == 1 and fullFacesOfNeighbor["back"]: # Block on Front
                self.removeFace("back")
            if diffZ == -1 and fullFacesOfNeighbor["front"]: # Block on our Back
                self.removeFace("front")
        
        #self.generateRenderingBuffersAndArrays() # Should happen in `removeFace`
            

    def generateVertices(self):
        x, y, z = (self.x, self.y, self.z)
        
        # Cube vertices (positions, UV coordinates)
        self.vertices = np.array([
            # For some reason the UVs on the Front face & The Back face are swapped? Ex. you modify back's UVs and front changes but back doesn't
            
            # Front face
            0+x, 0+y, 0+z,  1, 1,
            1+x, 0+y, 0+z,  0, 1,
            1+x, 1+y, 0+z,  0, 0,
            0+x, 1+y, 0+z,  1, 0,

            # Back face
            0+x, 0+y, 1+z,  0, 1,
            1+x, 0+y, 1+z,  1, 1,
            1+x, 1+y, 1+z,  1, 0,
            0+x, 1+y, 1+z,  0, 0,
            
            # Top face
            0+x, 1+y, 0+z,  0, 0,
            1+x, 1+y, 0+z,  1, 0,
            0+x, 1+y, 1+z,  0, 1,
            1+x, 1+y, 1+z,  1, 1,
            
            # Right face
            1+x, 0+y, 0+z,  1, 1,
            1+x, 0+y, 1+z,  0, 1,
            1+x, 1+y, 0+z,  1, 0,
            1+x, 1+y, 1+z,  0, 0,
            
            # Left face
            0+x, 0+y, 0+z,  0, 1,
            0+x, 0+y, 1+z,  1, 1,
            0+x, 1+y, 0+z,  0, 0,
            0+x, 1+y, 1+z,  1, 0,
            
            # Bottom face
            0+x, 0+y, 0+z,  0, 1,
            1+x, 0+y, 0+z,  1, 1,
            0+x, 0+y, 1+z,  0, 0,
            1+x, 0+y, 1+z,  1, 0,
        ], dtype='f4')

    def generateFaceIndices(self):
        if len(self.vertices) == 0:
            raise BaseException("Vertices not generated yet! We need to do that before generating the face indices")
        
        # Indices for each face (2 triangles per face)
        self.face_indices = [
            np.array([0, 1, 2, 0, 2, 3], dtype='i4'),       # Front face
            np.array([4, 5, 6, 4, 6, 7], dtype='i4'),       # Back face
            np.array([20, 21, 23, 20, 22, 23], dtype='i4'), # Bottom face
            np.array([8, 9, 11, 8, 10, 11], dtype='i4'),    # Top face
            np.array([16, 17, 19, 16, 18, 19], dtype='i4'), # Left face
            np.array([12, 13, 15, 12, 14, 15], dtype='i4'), # Right face
        ]
        self.orderOfFaces = ["front", "back", "bottom", "top", "left", "right"]
        
        if len(self.face_indices) != len(self.orderOfFaces):
            raise ValueError("So like... why aren't theses matching up? They need to be in order to properly assign textures!")

    def createVertexBufferObject(self):
        verticesBytes = self.vertices.tobytes()
        self.vbo = self.ctx.buffer(verticesBytes)

    def createObjectDataVertexArrayObject(self):
        self.objectDataVAOs = []
        
        for i in range(0, len(self.face_indices)):
            face_index = self.face_indices[i]
            
            ibo = self.ctx.buffer(face_index.tobytes())
            #vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert', 'in_texcoord', index_buffer=ibo)
            vao = self.ctx.vertex_array(self.program, self.vbo, 'in_vert', 'in_texcoord', index_buffer=ibo)
            
            currentFace = self.orderOfFaces[i] 
            
            texturePath = self.defaultTexturePath
            try:
                texturePath = self.texturePaths[ currentFace ]
            except:
                try:
                    texturePath = self.texturePaths["all"]
                except:
                    print("Something went wrong fetching the texture path for this block; Falling back to the default texture path")
            
            if texturePath not in textureCache:
                #print(f"{texturePath} is not cached, adding to cache now")
                getAndSaveTextureToCache(texturePath)
            
            self.activeTextures[currentFace] = texturePath
            
            self.objectDataVAOs.append({
                #"texturePath": texturePath,
                "face": currentFace,
                "vao": vao,
            })
        

# Camera class
class Camera:
    def __init__(self, position, yaw, pitch, tilt):
        self.position = vec3(position)
        
        self.yaw = yaw
        self.pitch = pitch
        self.tilt = tilt
        
        self.fov = 45
        self.aspectRatio = window_size[0] / window_size[1]
        self.nearPlane = 0.1
        self.farPlane = 100
        
        self.projection = perspective(self.fov, self.aspectRatio, self.nearPlane, self.farPlane)

    def get_view_matrix(self):
        # Start with an identity matrix
        view = mat4(1)
        
        # Apply rotations (yaw, pitch, tilt)
        view = rotate(view, np.radians(self.tilt), vec3(0, 0, 1))
        view = rotate(view, np.radians(self.pitch), vec3(1, 0, 0))
        view = rotate(view, np.radians(self.yaw), vec3(0, 1, 0))
        
        # Translate to the camera position
        view = translate(view, -self.position)
        
        return view

    def move_forward(self, speed):
        # Move forward relative to the yaw direction
        self.position.x += speed * math.sin(math.radians(self.yaw))
        self.position.z -= speed * math.cos(math.radians(self.yaw))

    def move_backward(self, speed):
        # Move backward relative to the yaw direction
        self.position.x -= speed * math.sin(math.radians(self.yaw))
        self.position.z += speed * math.cos(math.radians(self.yaw))

    def strafe_left(self, speed):
        # Move left relative to the yaw direction (perpendicular to forward)
        self.position.x -= speed * math.cos(math.radians(self.yaw))
        self.position.z -= speed * math.sin(math.radians(self.yaw))

    def strafe_right(self, speed):
        # Move right relative to the yaw direction (perpendicular to forward)
        self.position.x += speed * math.cos(math.radians(self.yaw))
        self.position.z += speed * math.sin(math.radians(self.yaw))

# Initialize camera
camera = Camera(position=(2, 2, 5), yaw=0, pitch=0, tilt=0)

# Set up the projection matrix for 3D rendering
program['projection'].write(camera.projection)

@functools.cache
def getFilesStartingWithString(path, string):
    if os.path.isfile(f"{path}/{string}.png"): return [f"{string}.png"]
    if string in blockTextureOverride.keys(): return [blockTextureOverride[string]]
    
    files = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and string in i:
            files.append(i)
            break
    return files

specialBlocksToFullFaceArray = {
    "air": [False]*6,
    "lava": [False, False, False, True, False, False],
}
blockTextureOverride = {
    "lava": "lava_still.png"
}
blocksToSkip = [
    "air",
]

blocks = [
    Block(0, 0, 0, {"all": "./texture.png"}, ctx, program, [True]*6),
]

def loadSubChunkFromJson(subChunk):
    blocksAsIndices = subChunk[0] # 1 Sub Chunk is 16x16x16
    blockPalette = subChunk[1]
    startPosition = subChunk[2]
    
    for y in range(16):
        for z in range(16):
            for x in range(16):
                blockIndex = x + (z * 16) + (y * 16 * 16)
                paletteIndex = blocksAsIndices[blockIndex]
                blockFromPalette = blockPalette[paletteIndex]
                blockName = blockFromPalette["Name"]
                blockName = blockName.split("minecraft:")[1]
                #print(blockName)
                
                if blockName in blocksToSkip:
                    #print(f"Skipping {blockName}")
                    continue
                
                textureFolder = "./BlockTextures1_21_3"
                possibleFiles = getFilesStartingWithString(textureFolder, blockName)
                if len(possibleFiles) == 0: continue
                
                texturePath = f"{textureFolder}/{possibleFiles[0]}"
                textures = {
                    "all": texturePath
                }
                
                # 
                
                fullFaceArray = [True]*6
                if blockName in specialBlocksToFullFaceArray:
                    fullFaceArray = specialBlocksToFullFaceArray[blockName]
                
                newBlock = Block(
                    x + startPosition[0],
                    y + startPosition[1],
                    z + startPosition[2],
                    textures,
                    ctx,
                    program,
                    fullFaceArray
                )
                blocks.append(newBlock)
                print(blockIndex, 4096)
    
    print(blockPalette)

with open("SubChunk.json", "r") as subChunkFile:
    blocks = []
    subChunk = json.loads(subChunkFile.read())[0]
    
    print(ctx)
    #loadSubChunkThread = threading.Thread(None, lambda: loadSubChunkFromJson(subChunk))
    #loadSubChunkThread.start()
    
    loadSubChunkFromJson(subChunk)

def removeUnseenFaces():
    for i in range(0, len(blocks)):
        block = blocks[i]
        block.removeFacesBasedOnNeighbors(blocks)

def removePortionOfUnseenFaces(jobNumber=0, splits=1):
    amountToDo = math.ceil(len(blocks) / splits)
    start = jobNumber * amountToDo
    end = start + amountToDo
    
    for i in range(start, end):
        block = blocks[i]
        block.removeFacesBasedOnNeighbors(blocks)

removeUnseenFaces()
"""
#removeUnseenFacesThread = threading.Thread(None, removeUnseenFaces)
howManyThreadsToRemoveUnseen = 50
for i in range(0, howManyThreadsToRemoveUnseen):
    removeUnseenFacesThread = threading.Thread(
        None, 
        lambda: removePortionOfUnseenFaces(i, howManyThreadsToRemoveUnseen)
    )
    removeUnseenFacesThread.start()
"""


# Main render loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

    # Handle keyboard input for camera movement
    keys = pygame.key.get_pressed()
    speed = 0.1  # movement speed

    if keys[K_w]:  # Move forward relative to yaw
        camera.move_forward(speed)
    if keys[K_s]:  # Move backward relative to yaw
        camera.move_backward(speed)
    if keys[K_a]:  # Move left (strafe)
        camera.strafe_left(speed)
    if keys[K_d]:  # Move right (strafe)
        camera.strafe_right(speed)
    if keys[K_SPACE]:  # Move up
        camera.position.y += speed
    if keys[K_LSHIFT]:  # Move down
        camera.position.y -= speed

    # Rotate camera
    if keys[K_LEFT]:  # Rotate left
        camera.yaw -= 1
    if keys[K_RIGHT]:  # Rotate right
        camera.yaw += 1
    if keys[K_UP]:  # Rotate up
        camera.pitch -= 1
    if keys[K_DOWN]:  # Rotate down
        camera.pitch += 1

    # Calculate view matrix
    view = camera.get_view_matrix()
    program['view'].write(view)

    # Clear the screen
    ctx.clear()

    # Set model matrix (identity for now)
    model = mat4(1)
    program['model'].write(model)

    # Bind each texture and render the corresponding face
    for block in blocks:
        for vaoData in block.objectDataVAOs:
            #texturePath = vaoData["texturePath"]
            faceName = vaoData["face"]
            texturePath = block.activeTextures[faceName]
            
            textureCache[texturePath].use()
            vaoData["vao"].render(moderngl.TRIANGLES)

    # Swap buffers
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
