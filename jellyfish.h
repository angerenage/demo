#pragma once

#include "glutils.h"

void halfBezier(vec3 P0, vec3 P1, vec3 P2, vec3 P3, int resolution, vec3* points);

Mesh generateDome(vec2 size, float inset);
Mesh genarateTentacles(vec3 pos, vec3 size, int resolution);

void initJellyfish();
void renderJellyfish(mat4 projection, mat4 view, vec3 camPos, float time);
void cleanupJellyfish();