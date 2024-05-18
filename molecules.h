#pragma once

#include "glutils.h"
#include "sphere.h"

void initMolecules();

void generateDoubleHelix(int numPoints, float radius, float twist);
void renderDNA(mat4 projection, mat4 view, vec3 camPos, float time);

void generateAtom();
void renderAtoms(mat4 projection, mat4 view, float time);

void cleanupMolecules();