#pragma once

#include <glad/glad.h>
#include <string.h>
#include <stdlib.h>
#include "glutils.h"
#include "shader.h"

extern GLuint displacementTextures;
extern GLuint slopeTextures;
extern GLuint underwaterDepthTexture;
extern GLuint underwaterColorTexture;
extern GLuint underwaterFBO;

Mesh generateGrid(vec2 size, int subdivision);
Mesh createParticles(int pointCount, float radius);

void initWater();
void updateSpectrum(float time);
void updateUnderwaterTextures(vec2 screenSize);

void cleanupWater();