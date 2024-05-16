#pragma once

#include <glad/glad.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <stdio.h>

#include "glutils.h"

extern const vec3 icosahedron_positions[12];

Mesh generateIcosphere(int subdivisions);
