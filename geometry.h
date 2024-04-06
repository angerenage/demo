#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <float.h>

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

float lerp(float a, float b, float t);
float polarDistance(float r1, float theta1, float r2, float theta2);
float segment_distance(const vec3* point);
float gaussianBulge(float x, float y, float A, float sigma);

float length(vec3 p);