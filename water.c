#include "water.h"

GLuint generateGrid(vec2 size, int subdivision, int *indexNbr) {
	const int width = subdivision + 1;
	const int vertexNbr = (subdivision + 2) * (subdivision + 2);
	*indexNbr = width * width * 6;

	GLuint vao = 0;
	vec3 *positions = (vec3*)malloc(sizeof(vec3) * vertexNbr);
	int *indices = (int*)malloc(sizeof(int) * (*indexNbr));

	if (positions && indices) {
		int index = 0;

		for (int y = 0; y <= width; y++) {
			for (int x = 0; x <= width; x++) {
				float posX = size.x * ((float)x / (float)width);
				float posZ = size.y * ((float)y / (float)width);

				positions[x + y * (width + 1)] = (vec3){posX - size.x / 2.0, 0.0, posZ - size.y / 2.0};

				if (x < width && y < width) {
					int topLeft = x + y * (width + 1);
					int topRight = (x + 1) + y * (width + 1);
					int bottomLeft = x + (y + 1) * (width + 1);
					int bottomRight = (x + 1) + (y + 1) * (width + 1);

					// First triangle
					indices[index++] = topLeft;
					indices[index++] = bottomLeft;
					indices[index++] = topRight;

					// Second triangle
					indices[index++] = topRight;
					indices[index++] = bottomLeft;
					indices[index++] = bottomRight;
				}
			}
		}

		vao = createIndexedVAO(positions, vertexNbr, indices, *indexNbr);

		free(positions);
		free(indices);
	}

	return vao;
}

GLuint createParticles(int pointCount, float radius) {
	vec3 *points = (vec3*)malloc(pointCount * sizeof(vec3));
	GLuint vao = 0;

	if (points) {
		for (int i = 0; i < pointCount; ++i) {
			float u = (float)rand() / RAND_MAX;
			float v = (float)rand() / RAND_MAX;
			float theta = u * 2.0f * M_PI;
			float phi = acos(2.0f * v - 1.0f);
			float r = cbrtf((float)rand() / RAND_MAX) * radius;

			points[i].x = r * sin(phi) * cos(theta);
			points[i].y = r * sin(phi) * sin(theta);
			points[i].z = r * cos(phi);
		}

		vao = createVAO(points, pointCount);

		free(points);
	}

	return vao;
}