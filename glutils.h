#pragma once

#include <glad/glad.h>
#include <X11/Xlib.h>
#include <GL/glx.h>
#include <GL/gl.h>

#include <stdlib.h>
#include <stdio.h>

#include "geometry.h"

extern Display *display;
extern Window window;
extern Atom wmDelete;

void initWindow(vec2 size);
void cleanupWindow();

void renderScreenQuad();

GLuint createTexture(int width, int height);
GLuint createTextureArray(int width, int height, int layer);
GLuint createTextureArrayRG(int width, int height, int layer);
GLuint createFramebuffer(GLuint texture);

GLuint createIndexedVAO(const vec3 *vertices, int vertexCount, const int *indices, int indexCount);
GLuint createVAO(const vec3 *vertices, int vertexCount);

void checkOpenGLError();
void cleanupUtils();