#include "planet.h"

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

Mesh generateIcosphere() {
	GLuint vao = createPositionVAO(icosahedron_positions, 12, icosahedron_indices, 60);
	Mesh mesh = {vao, icosahedron_positions, 12, icosahedron_indices, 60};
	return mesh;
}

void freeMesh(Mesh* mesh) {
	if (mesh->VAO) glDeleteVertexArrays(1, &mesh->VAO);
	//if (mesh->vertices) free(mesh->vertices);
	//if (mesh->indices) free(mesh->indices);
}
