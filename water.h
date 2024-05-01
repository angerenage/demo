#pragma once

#include <glad/glad.h>
#include <string.h>
#include <stdlib.h>
#include "glutils.h"
#include "shader.h"

extern GLuint displacementTextures;
extern GLuint slopeTextures;

Mesh generateGrid(vec2 size, int subdivision);
Mesh createParticles(int pointCount, float radius);

void initWater();
void updateSpectrum(float time);

void cleanupWater();