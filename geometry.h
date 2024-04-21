#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <float.h>

#ifndef M_PI
	#define M_PI 3.1415926535897932384626433832795
#endif

typedef struct vec2_s {
	float x, y;
} vec2;

typedef struct vec3_s {
	float x, y, z;
} vec3;

typedef struct {
	float m[16];
} mat4;

mat4 projectionMatrix(float fov, float aspectRatio, float nearPlane, float farPlane);
mat4 viewMatrix(vec3 position, vec3 focus, vec3 up);

vec3 substracteVector(vec3 v1, vec3 v2);
vec3 normalize(vec3 v);
vec3 crossProduct(vec3 v1, vec3 v2);
float length(vec3 p);

float lerp(float a, float b, float t);
float gaussianBulge(float x, float y, float A, float sigma);