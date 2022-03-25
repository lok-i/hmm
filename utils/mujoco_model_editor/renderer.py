from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import numpy as np
from common import *
import glm

class Renderer(object):
    def __init__(self, ):
        self.quad = gluNewQuadric()
        self.vao = glGenVertexArrays(1)

    def render_cylinder(self, bottom, top, radius):
        glPushMatrix()
        glTranslated(bottom[0], bottom[1], bottom[2])
        z = np.array([0, 0, 1])
        p = top - bottom
        if norm(p) > 1e-8:
            t = np.cross(z, p)
            angle = math.degrees(math.acos(z.dot(p) / norm(p)))
            glRotated(angle, t[0], t[1], t[2])
        glScaled(radius, radius, norm(p))

        gluCylinder(self.quad, 1, 1, 1, 32, 5)

        glPushMatrix()
        glRotated(180, 1, 0, 0)
        gluDisk(self.quad, 0, 1, 32, 5)
        glPopMatrix()
        glTranslated(0, 0, 1)
        gluDisk(self.quad, 0, 1, 32, 5)

        glPopMatrix()

    def render_cone(self, bottom, top, radius):
        glPushMatrix()
        glTranslated(bottom[0], bottom[1], bottom[2])
        z = np.array([0, 0, 1])
        p = top - bottom
        if norm(p) > 1e-8:
            t = np.cross(z, p)
            angle = math.degrees(math.acos(z.dot(p) / norm(p)))
            glRotated(angle, t[0], t[1], t[2])
        glScaled(radius, radius, norm(p))

        glutSolidCone(1, 1, 32, 5)
        glRotated(180, 1, 0, 0)
        gluDisk(self.quad, 0, 1, 32, 5)
        glPopMatrix()

    def render_arrow(self, bottom, top, radius):
        axis = normalize(top-bottom)
        arrow_len = 2 * radius
        mid = top - arrow_len*axis
        self.render_cylinder(bottom, mid, radius)
        self.render_cone(mid, top, 1.5*radius)

    def render_point(self, pos, radius):
        glPushMatrix()
        glTranslated(pos[0], pos[1], pos[2])
        glScaled(radius, radius, radius)
        glutSolidSphere(1.0, 32, 16)
        glPopMatrix()

    def render_cube(self, pos, size):
        glPushMatrix()
        glTranslated(pos[0], pos[1], pos[2])
        glScaled(size, size, size)
        glutSolidCube(1.0)
        glPopMatrix()

    def render_capsule(self, bottom, top, radius):
        self.render_cylinder(bottom, top, radius)
        self.render_point(bottom, radius)
        self.render_point(top, radius)

    def render_text(self, text, pos, scale, dir):
        glActiveTexture(GL_TEXTURE0)
        # glBindVertexArray(self.vao)
        angle_rad    = math.atan2(dir[1], dir[0])
        rotateM      = glm.rotate(glm.mat4(1), angle_rad, glm.vec3(0, 0, 1))
        transOriginM = glm.translate(glm.mat4(1), glm.vec3(pos[0],pos[1], 0))

        char_x = 0
        for c in text:
            c = ord(c)
            ch          = self.characters[c]
            w, h        = ch[1][0] * scale, ch[1][1] * scale
            xrel, yrel  = char_x + ch[2][0] * scale, (ch[1][1] - ch[2][1]) * scale
            char_x     += (ch[3] >> 6) * scale 
            scaleM      = glm.scale(glm.mat4(1), glm.vec3(w, h, 1))
            transRelM   = glm.translate(glm.mat4(1), glm.vec3(xrel, yrel, 0))
            modelM      = transOriginM * rotateM * transRelM * scaleM
            
            glUniformMatrix4fv(0, 1, GL_FALSE, glm.value_ptr(modelM))
            glBindTexture(GL_TEXTURE_2D, ch[0])
            glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)