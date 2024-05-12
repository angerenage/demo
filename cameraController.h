#pragma once

#include <stdio.h>
#include "geometry.h"

//extern float cameraAngleX, cameraAngleY;

vec3 bezier(float t, vec3 p0, vec3 p1, vec3 p2, vec3 p3);
vec3 bezierDerivative(float t, vec3 p0, vec3 p1, vec3 p2, vec3 p3);
float dynamicArcLength(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t);
float findTForNormalizedS(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float s);
vec3 bezierUniform(float s, vec3 p0, vec3 p1, vec3 p2, vec3 p3);
vec3 bezierDerivativeUniform(float s, vec3 p0, vec3 p1, vec3 p2, vec3 p3);

void defaultCameraTransforms(vec3 *pos, vec3 *dir, float distance, vec2 angles);
vec3 initializeCameraPosition();
mat4 getCameraMatrix(vec3 *camPos, float time);

float getTime(int step);