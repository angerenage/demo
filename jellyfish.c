#include "jellyfish.h"

#define NUM_POINTS 16
#define NUM_SLICES 7

static const float a = 3.0f, b = 1.5f;

vec3 bezierPoints[NUM_SLICES];

void halfBezier(vec3 P0, vec3 P1, vec3 P2, vec3 P3, int resolution, vec3* points) {
	for (int i = 0; i <= resolution; i++) {
		float t = i / (float)resolution / 2;
		
		vec3 A = vec3_lerp(P0, P1, t);
		vec3 B = vec3_lerp(P1, P2, t);
		vec3 C = vec3_lerp(P2, P3, t);
		
		vec3 D = vec3_lerp(A, B, t);
		vec3 E = vec3_lerp(B, C, t);
		
		points[i] = vec3_lerp(D, E, t);
	}
}

Mesh generateDome(vec2 size, float inset) {
	halfBezier((vec3){size.x, 0, 0}, (vec3){size.x - inset, 2 * size.y, 0}, (vec3){-size.x + inset, 2 * size.y, 0}, (vec3){-size.x, 0, 0}, NUM_SLICES - 1, bezierPoints);

	const int vertexNumber = (NUM_POINTS * (NUM_SLICES - 1));

	vec3 domeVertices[vertexNumber + 1];
	int indices[vertexNumber * 6];

	// Generate dome vertices
	for (int i = 0; i < NUM_SLICES - 1; i++) {
		double radius = bezierPoints[i].x;
		double height = bezierPoints[i].y;

		for (int j = 0; j < NUM_POINTS; j++) {
			double angle = 2 * M_PI * (float)j / NUM_POINTS;

			domeVertices[j + NUM_POINTS * i].x = radius * cos(angle);
			domeVertices[j + NUM_POINTS * i].y = height;
			domeVertices[j + NUM_POINTS * i].z = radius * sin(angle);
		}
	}

	domeVertices[vertexNumber] = (vec3){bezierPoints[NUM_SLICES - 1].x, bezierPoints[NUM_SLICES - 1].y, 0.0};

	// Generate indices for triangles
	for (int i = 0; i < NUM_SLICES - 1; i++) {
		for (int j = 0; j < NUM_POINTS; j++) {
			int baseIndex = j + NUM_POINTS * i;
			int baseIndexAbove = baseIndex + NUM_POINTS;

			int nextIndex = (j + 1) % NUM_POINTS + i * NUM_POINTS;
			int nextIndexAbove = (j + 1) % NUM_POINTS + (i + 1) * NUM_POINTS;

			if (baseIndexAbove >= vertexNumber) baseIndexAbove = vertexNumber;
			if (nextIndexAbove >= vertexNumber) nextIndexAbove = vertexNumber;

			// First triangle
			indices[baseIndex * 6 + 0] = baseIndex;
			indices[baseIndex * 6 + 1] = baseIndexAbove;
			indices[baseIndex * 6 + 2] = nextIndexAbove;

			// Second triangle
			indices[baseIndex * 6 + 3] = baseIndex;
			indices[baseIndex * 6 + 4] = nextIndexAbove;
			indices[baseIndex * 6 + 5] = nextIndex;
		}
	}

	GLuint vao = createIndexedVAO(domeVertices, vertexNumber + 1, indices, vertexNumber * 6);
	
	return (Mesh){vao, vertexNumber + 1, vertexNumber * 6};
}