#include "sphere.h"

#define X .525731112119133606
#define Z .850650808352039932

// Vertices of a unit icosahedron
static const vec3 icosahedron_positions[] = {
	{-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
	{0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
	{Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0} 
};

// Indices to define the triangles of the icosahedron
static const int icosahedron_indices[] = {
	0, 4, 1, 0, 9, 4, 9, 5, 4, 4, 5, 8, 4, 8, 1,
	8, 10, 1, 8, 3, 10, 5, 3, 8, 5, 2, 3, 2, 7, 3,
	7, 10, 3, 7, 6, 10, 7, 11, 6, 11, 0, 6, 0, 1, 6,
	6, 1, 10, 9, 0, 11, 9, 11, 2, 9, 2, 5, 7, 2, 11
};

static vec3 midpoint(vec3 v1, vec3 v2) {
	return (vec3){(v1.x + v2.x) * 0.5f, (v1.y + v2.y) * 0.5f, (v1.z + v2.z) * 0.5f};
}

Mesh generateIcosphere(int subdivisions) {
	vec3* vertices = malloc(sizeof(vec3) * 12);
	int* indices = malloc(sizeof(int) * 60);
	int vertexCount = 12;
	int indexCount = 60;

	memcpy(vertices, icosahedron_positions, vertexCount * sizeof(vec3));
	memcpy(indices, icosahedron_indices, indexCount * sizeof(int));

	for (int i = 0; i < subdivisions; i++) {
		int newIndicesCount = indexCount * 4;
		int* newIndices = malloc(sizeof(int) * newIndicesCount);
		int newVertexCount = 0;

		for (int j = 0; j < indexCount; j += 3) {
			int idx0 = indices[j];
			int idx1 = indices[j + 1];
			int idx2 = indices[j + 2];

			vec3 mid0 = normalize(midpoint(vertices[idx0], vertices[idx1]));
			vec3 mid1 = normalize(midpoint(vertices[idx1], vertices[idx2]));
			vec3 mid2 = normalize(midpoint(vertices[idx2], vertices[idx0]));

			vertices = realloc(vertices, sizeof(vec3) * (vertexCount + 3));
			vertices[vertexCount] = mid0;
			vertices[vertexCount + 1] = mid1;
			vertices[vertexCount + 2] = mid2;
			int midIdx0 = vertexCount++;
			int midIdx1 = vertexCount++;
			int midIdx2 = vertexCount++;

			int baseIdx = j * 4;
			newIndices[baseIdx] = idx0;
			newIndices[baseIdx + 1] = midIdx0;
			newIndices[baseIdx + 2] = midIdx2;

			newIndices[baseIdx + 3] = midIdx0;
			newIndices[baseIdx + 4] = idx1;
			newIndices[baseIdx + 5] = midIdx1;

			newIndices[baseIdx + 6] = midIdx0;
			newIndices[baseIdx + 7] = midIdx1;
			newIndices[baseIdx + 8] = midIdx2;

			newIndices[baseIdx + 9] = midIdx2;
			newIndices[baseIdx + 10] = midIdx1;
			newIndices[baseIdx + 11] = idx2;

			newVertexCount += 3;
		}

		free(indices);
		indices = newIndices;
		indexCount = newIndicesCount;
	}

	GLuint vao = createIndexedVAO(vertices, vertexCount, indices, indexCount);
	free(vertices);
	free(indices);

	return (Mesh){vao, vertexCount, indexCount};
}
