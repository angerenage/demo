#pragma once

#include <glad/glad.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "glutils.h"

typedef struct glyph_s {
	uint8_t c[5];
} Glyph;

typedef struct charSquare_s {
	vec2 p[4];
	unsigned int i[6];
	int id;
} CharSquare;

Glyph getGlyphForCharacter(wchar_t c);

Mesh createText(wchar_t *text);
CharSquare *createCharacter(Glyph g, int *charId, int *squareNumber);