#include "geometry.h"

mat4 getIdentity() {
	mat4 matrix = {0};
	for (int i = 0; i < 4; i++) {
		matrix.m[i][i] = 1.0f;
	}
	return matrix;
}

mat4 projectionMatrix(float fov, float aspectRatio, float nearPlane, float farPlane) {
	mat4 matrix = {0};
	
	float yScale = 1.0f / tan(fov / 2.0f);
	float xScale = yScale / aspectRatio;
	float frustumLength = farPlane - nearPlane;

	matrix.m[0][0] = xScale;
	matrix.m[1][1] = yScale;
	matrix.m[2][2] = -((farPlane + nearPlane) / frustumLength);
	matrix.m[2][3] = -1;
	matrix.m[3][2] = -((2 * nearPlane * farPlane) / frustumLength);
	
	return matrix;
}

mat4 viewMatrix(vec3 position, vec3 focus, vec3 up) {
	vec3 f = normalize(vec3_subtract(focus, position));
	vec3 r = normalize(crossProduct(f, up));
	vec3 u = crossProduct(r, f);

	mat4 matrix = {0};
	matrix.m[0][0] = r.x;
	matrix.m[1][0] = r.y;
	matrix.m[2][0] = r.z;
	matrix.m[0][1] = u.x;
	matrix.m[1][1] = u.y;
	matrix.m[2][1] = u.z;
	matrix.m[0][2] = -f.x;
	matrix.m[1][2] = -f.y;
	matrix.m[2][2] = -f.z;
	matrix.m[3][3] = 1;

	matrix.m[3][0] = -r.x * position.x - r.y * position.y - r.z * position.z;
	matrix.m[3][1] = -u.x * position.x - u.y * position.y - u.z * position.z;
	matrix.m[3][2] = f.x * position.x + f.y * position.y + f.z * position.z;

	return matrix;
}

void translationMatrix(mat4 *matrix, vec3 translation) {
	matrix->m[3][0] = translation.x;
	matrix->m[3][1] = translation.y;
	matrix->m[3][2] = translation.z;
}

void rotationMatrix(mat4 *matrix, vec3 rotation) {
	float cp = cos(rotation.x);
	float sp = sin(rotation.x);
	float cy = cos(rotation.y);
	float sy = sin(rotation.y);
	float cr = cos(rotation.z);
	float sr = sin(rotation.z);

	matrix->m[0][0] = cr * cy;
	matrix->m[0][1] = cr * sy * sp - sr * cp;
	matrix->m[0][2] = cr * sy * cp + sr * sp;
	matrix->m[1][0] = sr * cy;
	matrix->m[1][1] = sr * sy * sp + cr * cp;
	matrix->m[1][2] = sr * sy * cp - cr * sp;
	matrix->m[2][0] = -sy;
	matrix->m[2][1] = cy * sp;
	matrix->m[2][2] = cy * cp;
}

void scaleMatrix(mat4 *matrix, vec3 scale) {
	matrix->m[0][0] *= scale.x;
	matrix->m[1][1] *= scale.y;
	matrix->m[2][2] *= scale.z;
}

mat4 generateTransformationMatrix(vec3 pos, vec3 rot, vec3 scl) {
	mat4 matrix = getIdentity();
	translationMatrix(&matrix, pos);
	rotationMatrix(&matrix, rot);
	scaleMatrix(&matrix, scl);
	return matrix;
}

vec3 vec3_add(vec3 a, vec3 b) {
	return (vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 vec3_subtract(vec3 v1, vec3 v2) {
	vec3 result;
	result.x = v1.x - v2.x;
	result.y = v1.y - v2.y;
	result.z = v1.z - v2.z;
	return result;
}

vec3 vec3_scale(vec3 v, float s) {
	return (vec3){v.x * s, v.y * s, v.z * s};
}

vec3 vec3_lerp(vec3 a, vec3 b, float t) {
	return vec3_add(vec3_scale(a, 1 - t), vec3_scale(b, t));
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

float length(vec3 p) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float lerp(float a, float b, float t) {
	return a + (t) * (b - a);
}

float degreesToRadians(float degrees) {
	return degrees * (M_PI / 180);
}

float gaussianBulge(float x, float y, float A, float sigma) {
	return A * exp(-(x*x + y*y) / (2 * sigma * sigma));
}
