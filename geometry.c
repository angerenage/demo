#include "geometry.h"

mat4 projectionMatrix(float fov, float aspectRatio, float nearPlane, float farPlane) {
	mat4 mat = {0};
	
	float yScale = 1.0f / tan(fov / 2.0f);
	float xScale = yScale / aspectRatio;
	float frustumLength = farPlane - nearPlane;

	mat.m[0] = xScale;
	mat.m[5] = yScale;
	mat.m[10] = -((farPlane + nearPlane) / frustumLength);
	mat.m[11] = -1;
	mat.m[14] = -((2 * nearPlane * farPlane) / frustumLength);
	
	return mat;
}

mat4 viewMatrix(vec3 position, vec3 focus, vec3 up) {
	vec3 f = normalize(substracteVector(focus, position));
	vec3 r = normalize(crossProduct(f, up));
	vec3 u = crossProduct(r, f);

	mat4 mat = {0};
	mat.m[0] = r.x;
	mat.m[4] = r.y;
	mat.m[8] = r.z;
	mat.m[1] = u.x;
	mat.m[5] = u.y;
	mat.m[9] = u.z;
	mat.m[2] = -f.x;
	mat.m[6] = -f.y;
	mat.m[10] = -f.z;
	mat.m[15] = 1;

	mat.m[12] = -r.x * position.x - r.y * position.y - r.z * position.z;
	mat.m[13] = -u.x * position.x - u.y * position.y - u.z * position.z;
	mat.m[14] = f.x * position.x + f.y * position.y + f.z * position.z;

	return mat;
}

vec3 substracteVector(vec3 v1, vec3 v2) {
	vec3 result;
	result.x = v1.x - v2.x;
	result.y = v1.y - v2.y;
	result.z = v1.z - v2.z;
	return result;
}

vec3 normalize(vec3 v) {
	float length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	vec3 result;
	result.x = v.x / length;
	result.y = v.y / length;
	result.z = v.z / length;
	return result;
}

vec3 crossProduct(vec3 v1, vec3 v2) {
	vec3 result;
	result.x = v1.y * v2.z - v1.z * v2.y;
	result.y = v1.z * v2.x - v1.x * v2.z;
	result.z = v1.x * v2.y - v1.y * v2.x;
	return result;
}

float lerp(float a, float b, float t) {
	return a + (t) * (b - a);
}

float gaussianBulge(float x, float y, float A, float sigma) {
	return A * exp(-(x*x + y*y) / (2 * sigma * sigma));
}

float length(vec3 p) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
