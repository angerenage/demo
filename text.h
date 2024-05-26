#pragma once

#include <glad/glad.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <wchar.h>

#include "glutils.h"

typedef struct glyph_s {
	uint8_t c[5];
} Glyph;

typedef struct charSquare_s {
	vec2 p[4];
	unsigned int i[6];
	int id;
} CharSquare;

typedef struct text_s {
	wchar_t *text;
	Mesh mesh;
	float width;
	vec3 pos;
	float scale;
} Text;

typedef enum verticalAnchor_e {
	TOP_ANCHOR, MIDDLE_ANCHOR, BOTTOM_ANCHOR
} VerticalAnchor;

typedef enum horizontalAnchor_e {
	RIGHT_ANCHOR, CENTER_ANCHOR, LEFT_ANCHOR
} HorizontalAnchor;

Glyph getGlyphForCharacter(wchar_t c);

Text createText(wchar_t *text, float scale);
CharSquare *createCharacter(Glyph g, int *charId, int *squareNumber);

void fixHorizontal(Text *text, HorizontalAnchor anchor, vec2 screenSize, float distance);
void fixVertical(Text *text, VerticalAnchor anchor, vec2 screenSize, float distance);