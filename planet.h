#pragma once

#include <GL/glew.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <stdio.h>

#include "geometry.h"

typedef struct mesh_s {
	GLuint VAO;
	const vec3* vertices;
	int vertexCount;
	const int* indices;
	int indexCount;
} Mesh;

Mesh generateIcosphere();

void freeMesh(Mesh* mesh);
