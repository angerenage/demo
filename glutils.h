#pragma once

#include <glad/glad.h>
#include <stdio.h>

#include "geometry.h"

GLuint createTexture(int width, int height);
GLuint createFramebuffer(GLuint texture);

GLuint createIndexedVAO(const vec3 *vertices, int vertexCount, const int *indices, int indexCount);
GLuint createVAO(const vec3 *vertices, int vertexCount);