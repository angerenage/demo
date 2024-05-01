#pragma once

#include "glutils.h"

void halfBezier(vec3 P0, vec3 P1, vec3 P2, vec3 P3, int resolution, vec3* points);

Mesh generateDome(vec2 size, float inset);