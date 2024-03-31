#pragma once

#include <GL/glew.h>
#include <stdlib.h>

#include "geometry.h"

typedef struct starPoint_s {
    vec3 position;
    float density;
} StarPoint;

StarPoint *generateGalaxy(unsigned int num_stars);

GLuint createGalaxyVAO(const StarPoint *stars, unsigned int num_stars);