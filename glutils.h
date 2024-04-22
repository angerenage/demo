#pragma once

#include <glad/glad.h>
#include <stdio.h>

#include "geometry.h"

void renderScreenQuad();

GLuint createTexture(int width, int height);
GLuint createTextureArray(int width, int height, int layer);
GLuint createTextureArrayRG(int width, int height, int layer);
GLuint createFramebuffer(GLuint texture);

GLuint createIndexedVAO(const vec3 *vertices, int vertexCount, const int *indices, int indexCount);
GLuint createVAO(const vec3 *vertices, int vertexCount);

void checkOpenGLError();
void cleanupUtils();