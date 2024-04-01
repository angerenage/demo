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
	},

	/*{ // base
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
		0b00000000,
	},*/
};

typedef struct textPoint_s {
	vec2 pos;
	int id;
} TextPoint;

static GLuint createTextVAO(const TextPoint* points, int pointCount, const int* indices, int indexCount) {
	GLuint vao, vbo, ebo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	glBufferData(GL_ARRAY_BUFFER, pointCount * sizeof(TextPoint), points, GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(int), indices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_INT, GL_FALSE, sizeof(int), (void*)0);

	glBindVertexArray(0);

	return vao;
}

Glyph getGlyphForCharacter(char c) {
	if (isdigit(c)) {
		return digits[c - '0'];
	}

	1/0;
}

GLuint createText(char *text, int *baseId, int *indiceCount) {
	int charId = *baseId;
    unsigned int totalSquareCount = 0;

    TextPoint *points = NULL;
    unsigned int *indices = NULL;
    unsigned int pointCount = 0;
    unsigned int indexCount = 0;

    while (*text != '\0') {
        Glyph g = getGlyphForCharacter(*text);

        int squareNumber = 0;
        CharSquare *squares = createCharacter(g, &charId, &squareNumber);
        if (squareNumber > 0 && squares) {
            points = realloc(points, (pointCount + squareNumber * 4) * sizeof(TextPoint));
            indices = realloc(indices, (indexCount + squareNumber * 6) * sizeof(unsigned int));

            for (int i = 0; i < squareNumber; i++) {
                for (int j = 0; j < 4; j++) { // Each square has 4 vertices
                    points[pointCount].pos = squares[i].p[j];
                    points[pointCount].id = squares[i].id;
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

	*indiceCount = indexCount;
    GLuint vao = createTextVAO(points, pointCount, indices, indexCount);

    free(points);
    free(indices);

    return vao;
}

CharSquare *createCharacter(Glyph g, int *charId, int *squareNumber) {
	CharSquare *squares = (CharSquare*)malloc(sizeof(CharSquare) * 5 * 8);
	int squareNum = 0;

	if (squares) {
		for (int row = 0; row < 8; row++) {
			for (int column = 0; column < 5; column++) {
				if (g.c[column] & (1 << row)) {
					squares[squareNum].id = (*charId) + squareNum;
					vec2 *pos = squares[squareNum].p;

					vec2 origin = (vec2){row * 0.11f, column * 0.11f};
					pos[0] = origin;
					pos[1] = (vec2){origin.x + 0.1f, origin.y};
					pos[2] = (vec2){origin.x, origin.y + 0.1f};
					pos[3] = (vec2){origin.x + 0.1f, origin.y + 0.1f};

					unsigned int *indices = squares[squareNum].i;
					indices[0] = 0;
					indices[1] = 1;
					indices[2] = 2;
					indices[3] = 1;
					indices[4] = 3;
					indices[5] = 2;

					squareNum++;
				}
			}
		}
	}

	*squareNumber = squareNum;
	return squares;
}