#include "text.h"

const static Glyph digits[] = {
	{ // 0
		0b00111110,
		0b01010001,
		0b01001001,
		0b01000101,
		0b00111110,
	},
	{ // 1
		0b00000000,
		0b01000010,
		0b01111111,
		0b01000000,
		0b00000000,
	},
	{ // 2
		0b01000010,
		0b01100001,
		0b01010001,
		0b01001001,
		0b01000110,
	},
	{ // 3
		0b00100001,
		0b01000001,
		0b01000101,
		0b01001011,
		0b00110001,
	},
	{ // 4
		0b00011000,
		0b00010100,
		0b00010010,
		0b01111111,
		0b00010000,
	},
	{ // 5
		0b00100111,
		0b01000101,
		0b01000101,
		0b01000101,
		0b00111001,
	},
	{ // 6
		0b00111100,
		0b01001010,
		0b01001001,
		0b01001001,
		0b00110000,
	},
	{ // 7
		0b00000001,
		0b01110001,
		0b00001001,
		0b00000101,
		0b00000011,
	},
	{ // 8
		0b00110110,
		0b01001001,
		0b01001001,
		0b01001001,
		0b00110110,
	},
	{ // 9
		0b00000110,
		0b01001001,
		0b01001001,
		0b00101001,
		0b00011110,
	}
};

const static Glyph lowercase[] = {
	{ // a
		0b00100000,
		0b01010100,
		0b01010100,
		0b01010100,
		0b01111000,
	},
	{ // b
		0b01111111,
		0b01001000,
		0b01000100,
		0b01000100,
		0b00111000,
	},
	{ // c
		0b00111000,
		0b01000100,
		0b01000100,
		0b01000100,
		0b00100000,
	},
	{ // d
		0b00111000,
		0b01000100,
		0b01000100,
		0b01001000,
		0b01111111,
	},
	{ // e
		0b00111000,
		0b01010100,
		0b01010100,
		0b01010100,
		0b00011000,
	},
	{ // f
		0b00001000,
		0b01111110,
		0b00001001,
		0b00000001,
		0b00000010,
	},
	{ // g
		0b00011000,
		0b10100100,
		0b10100100,
		0b10100100,
		0b01111100,
	},
	{ // h
		0b01111111,
		0b00001000,
		0b00000100,
		0b00000100,
		0b01111000,
	},
	{ // i
		0b00000000,
		0b01000100,
		0b01111101,
		0b01000000,
		0b00000000,
	},
	{ // j
		0b01000000,
		0b10000000,
		0b10000100,
		0b01111101,
		0b00000000,
	},
	{ // k
		0b01111111,
		0b00010000,
		0b00101000,
		0b01000100,
		0b00000000,
	},
	{ // l
		0b00000000,
		0b01000001,
		0b01111111,
		0b01000000,
		0b00000000,
	},
	{ // m
		0b01111100,
		0b00000100,
		0b00011000,
		0b00000100,
		0b01111000,
	},
	{ // n
		0b01111100,
		0b00001000,
		0b00000100,
		0b00000100,
		0b01111000,
	},
	{ // o
		0b00111000,
		0b01000100,
		0b01000100,
		0b01000100,
		0b00111000,
	},
	{ // p
		0b11111100,
		0b00100100,
		0b00100100,
		0b00100100,
		0b00011000,
	},
	{ // q
		0b00011000,
		0b00100100,
		0b00100100,
		0b00101000,
		0b11111100,
	},
	{ // r
		0b01111100,
		0b00001000,
		0b00000100,
		0b00000100,
		0b00001000,
	},
	{ // s
		0b01001000,
		0b01010100,
		0b01010100,
		0b01010100,
		0b00100000,
	},
	{ // t
		0b00000100,
		0b00111111,
		0b01000100,
		0b01000000,
		0b00100000,
	},
	{ // u
		0b00111100,
		0b01000000,
		0b01000000,
		0b00100000,
		0b01111100,
	},
	{ // v
		0b00011100,
		0b00100000,
		0b01000000,
		0b00100000,
		0b00011100,
	},
	{ // w
		0b00111100,
		0b01000000,
		0b00110000,
		0b01000000,
		0b00111100,
	},
	{ // x
		0b01000100,
		0b00101000,
		0b00010000,
		0b00101000,
		0b01000100,
	},
	{ // y
		0b00001100,
		0b01010000,
		0b01010000,
		0b01010000,
		0b00111100,
	},
	{ // z
		0b01000100,
		0b01100100,
		0b01010100,
		0b01001100,
		0b01000100,
	}
};

const static Glyph uppercase[] = {
	{ // A
		0b01111110,
		0b00010001,
		0b00010001,
		0b00010001,
		0b01111110,
	},
	{ // B
		0b01111111,
		0b01001001,
		0b01001001,
		0b01001001,
		0b00110110,
	},
	{ // C
		0b00111110,
		0b01000001,
		0b01000001,
		0b01000001,
		0b00100010,
	},
	{ // D
		0b01111111,
		0b01000001,
		0b01000001,
		0b00100010,
		0b00011100,
	},
	{ // E
		0b01111111,
		0b01001001,
		0b01001001,
		0b01001001,
		0b01000001,
	},
	{ // F
		0b01111111,
		0b00001001,
		0b00001001,
		0b00001001,
		0b00000001,
	},
	{ // G
		0b00111110,
		0b01000001,
		0b01001001,
		0b01001001,
		0b01111010,
	},
	{ // H
		0b01111111,
		0b00001000,
		0b00001000,
		0b00001000,
		0b01111111,
	},
	{ // I
		0b00000000,
		0b01000001,
		0b01111111,
		0b01000001,
		0b00000000,
	},
	{ // J
		0b00100000,
		0b01000000,
		0b01000001,
		0b00111111,
		0b00000001,
	},
	{ // K
		0b01111111,
		0b00001000,
		0b00010100,
		0b00100010,
		0b01000001,
	},
	{ // L
		0b01111111,
		0b01000000,
		0b01000000,
		0b01000000,
		0b01000000,
	},
	{ // M
		0b01111111,
		0b00000010,
		0b00001100,
		0b00000010,
		0b01111111,
	},
	{ // N
		0b01111111,
		0b00000100,
		0b00001000,
		0b00010000,
		0b01111111,
	},
	{ // O
		0b00111110,
		0b01000001,
		0b01000001,
		0b01000001,
		0b00111110,
	},
	{ // P
		0b01111111,
		0b00001001,
		0b00001001,
		0b00001001,
		0b00000110,
	},
	{ // Q
		0b00111110,
		0b01000001,
		0b01010001,
		0b00100001,
		0b01011110,
	},
	{ // R
		0b01111111,
		0b00001001,
		0b00011001,
		0b00101001,
		0b01000110,
	},
	{ // S
		0b01000110,
		0b01001001,
		0b01001001,
		0b01001001,
		0b00110001,
	},
	{ // T
		0b00000001,
		0b00000001,
		0b01111111,
		0b00000001,
		0b00000001,
	},
	{ // U
		0b00111111,
		0b01000000,
		0b01000000,
		0b01000000,
		0b00111111,
	},
	{ // V
		0b00011111,
		0b00100000,
		0b01000000,
		0b00100000,
		0b00011111,
	},
	{ // W
		0b00111111,
		0b01000000,
		0b00111000,
		0b01000000,
		0b00111111,
	},
	{ // X
		0b01100011,
		0b00010100,
		0b00001000,
		0b00010100,
		0b01100011,
	},
	{ // Y
		0b00000111,
		0b00001000,
		0b01110000,
		0b00001000,
		0b00000111,
	},
	{ // Z
		0b01100001,
		0b01010001,
		0b01001001,
		0b01000101,
		0b01000011,
	}
};

Glyph getGlyphForCharacter(wchar_t c) {
	if (isascii(c)) {
		if (isdigit(c)) {
			return digits[c - '0'];
		}
		else if (isalpha(c)) {
			if (islower(c)) {
				return lowercase[c - 'a'];
			}
			else {
				return uppercase[c - 'A'];
			}
		}
		// Punctuation
	}
	else {
		// Special case for é, è, ê, à, ... (if needed)
		if (c == L'é') {
			return (Glyph) {
				{ // é
					0b00111000,
					0b01010100,
					0b01010110,
					0b01010101,
					0b00011000,
				}
			};
		}
		else if (c == L'è') {
			return (Glyph) {
				{ // é
					0b00111000,
					0b01010101,
					0b01010110,
					0b01010100,
					0b00011000,
				}
			};
		}

		printf("Non ascii character\n");
	}

	return (Glyph) {
		{ // space
			0b00000000,
			0b00000000,
			0b00000000,
			0b00000000,
			0b00000000,
		}
	};
}

Mesh createText(wchar_t *text) {
	int charId = 0;
	unsigned int totalSquareCount = 0;

	vec3 *points = NULL;
	unsigned int *indices = NULL;
	unsigned int pointCount = 0;
	unsigned int indexCount = 0;

	while (*text != '\0') {
		Glyph g = getGlyphForCharacter(*text);

		int squareNumber = 0;
		CharSquare *squares = createCharacter(g, &charId, &squareNumber);
		if (squareNumber > 0 && squares) {
			points = realloc(points, (pointCount + squareNumber * 4) * sizeof(vec3));
			indices = realloc(indices, (indexCount + squareNumber * 6) * sizeof(unsigned int));

			for (int i = 0; i < squareNumber; i++) {
				for (int j = 0; j < 4; j++) { // Each square has 4 vertices
					points[pointCount].x = squares[i].p[j].x;
					points[pointCount].y = squares[i].p[j].y;
					points[pointCount].z = squares[i].id;
					pointCount++;
				}
				for (int k = 0; k < 6; k++) { // Each square has 6 indices
					indices[indexCount++] = squares[i].i[k] + totalSquareCount * 4;
				}
				totalSquareCount++;
			}
			free(squares);
		}

		text++;
		charId++;
	}

	GLuint vao = createIndexedVAO(points, pointCount, indices, indexCount);
	free(points);
	free(indices);

	return (Mesh){vao, pointCount, indexCount};
}

CharSquare *createCharacter(Glyph g, int *charId, int *squareNumber) {
	CharSquare *squares = (CharSquare*)malloc(sizeof(CharSquare) * 5 * 8);
	int squareNum = 0;

	if (squares) {
		for (int row = 0; row < 8; row++) {
			for (int column = 0; column < 5; column++) {
				if (g.c[column] & (1 << row)) {
					squares[squareNum].id = (*charId) * 5 * 8 + squareNum;

					vec2 origin = (vec2){(column + 6 * (*charId)) * 0.11f, row * -0.11f};
					squares[squareNum].p[0] = origin;
					squares[squareNum].p[1] = (vec2){origin.x + 0.1f, origin.y};
					squares[squareNum].p[2] = (vec2){origin.x, origin.y - 0.1f};
					squares[squareNum].p[3] = (vec2){origin.x + 0.1f, origin.y - 0.1f};

					squares[squareNum].i[0] = 2;
					squares[squareNum].i[1] = 1;
					squares[squareNum].i[2] = 0;
					squares[squareNum].i[3] = 1;
					squares[squareNum].i[4] = 2;
					squares[squareNum].i[5] = 3;

					squareNum++;
				}
			}
		}
	}

	*squareNumber = squareNum;
	return squares;
}