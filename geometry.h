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
	float m[4][4];
} mat4;

mat4 getIdentity();
mat4 projectionMatrix(float fov, float aspectRatio, float nearPlane, float farPlane);
mat4 viewMatrix(vec3 position, vec3 focus, vec3 up);
void translationMatrix(mat4 *matrix, vec3 translation);
void rotationMatrix(mat4 *matrix, vec3 rotation);
void scaleMatrix(mat4 *matrix, vec3 scale);
mat4 generateTransformationMatrix(vec3 pos, vec3 rot, vec3 scl);

vec3 vec3_add(vec3 a, vec3 b);
vec3 vec3_subtract(vec3 v1, vec3 v2);
vec3 vec3_scale(vec3 v, float s);
vec3 vec3_lerp(vec3 a, vec3 v, float t);
vec3 normalize(vec3 v);
vec3 crossProduct(vec3 v1, vec3 v2);
float length(vec3 p);

float lerp(float a, float b, float t);
float degreesToRadians(float degrees);
float gaussianBulge(float x, float y, float A, float sigma);