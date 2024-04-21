#pragma once

#include <glad/glad.h>
#include <string.h>
#include <stdlib.h>
#include "glutils.h"
#include "shader.h"

extern GLuint spectrum;

GLuint generateGrid(vec2 size, int subdivision, int *vertexNbr);
GLuint createParticles(int pointCount, float radius);

void initWater();
void updateSpectrum(float time);

void cleanupWater();