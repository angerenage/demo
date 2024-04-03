#pragma once

#include <glad/glad.h>
#include <stdio.h>

#include "geometry.h"

GLuint createTexture(int width, int height);
GLuint createFramebuffer(GLuint texture);

GLuint createPositionVAO(const vec3* vertices, int vertexCount, const int* indices, int indexCount);