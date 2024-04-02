#pragma once

#include <GL/glew.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "geometry.h"

typedef struct glyph_s {
	uint8_t c[5];
} Glyph;

typedef struct charSquare_s {
	vec2 p[4];
	unsigned int i[6];
	int id;
} CharSquare;

Glyph getGlyphForCharacter(char c);

GLuint createText(char *text, int *indiceCount);
CharSquare *createCharacter(Glyph g, int *charId, int *squareNumber);