#pragma once

#include <glad/glad.h>
#include <stdlib.h>
#include <stddef.h>

#include "glutils.h"

typedef struct starPoint_s {
	vec3 position;
	float density;
} StarPoint;

Mesh generateGalaxy(unsigned int num_stars);
GLuint createGalaxyVAO(const StarPoint *stars, unsigned int num_stars);