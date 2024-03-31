#pragma once

#include <GL/glew.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <stdio.h>

#include "geometry.h"

typedef struct vertex_s {
	vec3 pos;
	vec3 normal;
} Vertex;

typedef struct mesh_s {
	GLuint VAO;
	Vertex* vertices;
	int vertexCount;
	int* indices;
	int indexCount;
} Mesh;

Mesh generateIcosphere(bool *err);

void freeMesh(Mesh* mesh);
