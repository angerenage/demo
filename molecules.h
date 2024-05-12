#pragma once

#include "glutils.h"
#include "sphere.h"

void generateDoubleHelix(int numPoints, float radius, float twist);
void renderDNA(mat4 projection, mat4 view);
void cleanupMolecules();