#include "jellyfish.h"

#define NUM_POINTS 64
#define NUM_SLICES 25

vec3 bezierPoints[NUM_SLICES];

void halfBezier(vec3 P0, vec3 P1, vec3 P2, vec3 P3, int resolution, vec3* points) {
	for (int i = 0; i <= resolution; i++) {
		float t = i / (float)resolution / 2;
		
		vec3 A = vec3_lerp(P0, P1, t);
		vec3 B = vec3_lerp(P1, P2, t);
		vec3 C = vec3_lerp(P2, P3, t);
		
		vec3 D = vec3_lerp(A, B, t);
		vec3 E = vec3_lerp(B, C, t);
		
		points[i] = vec3_lerp(D, E, t);
	}
}

Mesh generateDome(vec2 size, float inset) {
	halfBezier((vec3){size.x, 0, 0}, (vec3){size.x - inset, 2 * size.y, 0}, (vec3){-size.x + inset, 2 * size.y, 0}, (vec3){-size.x, 0, 0}, NUM_SLICES - 1, bezierPoints);

	const int vertexNumber = (NUM_POINTS * (NUM_SLICES - 1));

	vec3 domeVertices[vertexNumber + 1];
	unsigned int indices[vertexNumber * 6];

	// Generate dome vertices
	for (int i = 0; i < NUM_SLICES - 1; i++) {
		double radius = bezierPoints[i].x;
		double height = bezierPoints[i].y;

		for (int j = 0; j < NUM_POINTS; j++) {
			double angle = 2 * M_PI * (float)j / NUM_POINTS;

			domeVertices[j + NUM_POINTS * i].x = radius * cos(angle);
			domeVertices[j + NUM_POINTS * i].y = height;
			domeVertices[j + NUM_POINTS * i].z = radius * sin(angle);
		}
	}

	domeVertices[vertexNumber] = (vec3){bezierPoints[NUM_SLICES - 1].x, bezierPoints[NUM_SLICES - 1].y, 0.0};

	// Generate indices for triangles
	for (int i = 0; i < NUM_SLICES - 1; i++) {
		for (int j = 0; j < NUM_POINTS; j++) {
			int baseIndex = j + NUM_POINTS * i;
			int baseIndexAbove = baseIndex + NUM_POINTS;

			int nextIndex = (j + 1) % NUM_POINTS + i * NUM_POINTS;
			int nextIndexAbove = (j + 1) % NUM_POINTS + (i + 1) * NUM_POINTS;

			if (baseIndexAbove >= vertexNumber) baseIndexAbove = vertexNumber;
			if (nextIndexAbove >= vertexNumber) nextIndexAbove = vertexNumber;

			// First triangle
			indices[baseIndex * 6 + 0] = baseIndex;
			indices[baseIndex * 6 + 1] = baseIndexAbove;
			indices[baseIndex * 6 + 2] = nextIndexAbove;

			// Second triangle
			indices[baseIndex * 6 + 3] = baseIndex;
			indices[baseIndex * 6 + 4] = nextIndexAbove;
			indices[baseIndex * 6 + 5] = nextIndex;
		}
	}

	GLuint vao = createIndexedVAO(domeVertices, vertexNumber + 1, (unsigned int *)indices, vertexNumber * 6);
	
	return (Mesh){vao, vertexNumber + 1, vertexNumber * 6};
}

Mesh genarateTentacles(vec3 pos, vec3 size, int resolution) {
	const int vertexCount = 2 * (resolution + 1);
	const int indiciesCount = 6 * resolution;

	vec3 *vertices = (vec3*)malloc(sizeof(vec3) * vertexCount);
	unsigned int *indices = (unsigned int*)malloc(sizeof(int) * indiciesCount);

	for (int i = 0; i < vertexCount; i += 2) {
		float x = (float)i / ((float)resolution * 2.0f);
		float width = 1 - exp((x - 1) * 10);

		vertices[i].x = pos.x - (size.x / 2) * width;
		vertices[i].y = pos.y - size.y * x;
		vertices[i].z = pos.z - (size.z / 2) * width;

		vertices[i + 1].x = pos.x + (size.x / 2) * width;
		vertices[i + 1].y = pos.y - size.y * x;
		vertices[i + 1].z = pos.z + (size.z / 2) * width;
	}

	int j = 0;
	for (int i = 0; i < indiciesCount; i += 6) {
		indices[i] = j;
		indices[i + 1] = j + 1;
		indices[i + 2] = j + 3;

		indices[i + 3] = j;
		indices[i + 4] = j + 3;
		indices[i + 5] = j + 2;

		j += 2;
	}

	GLuint vao = createIndexedVAO(vertices, vertexCount, indices, indiciesCount);
	free(vertices);
	free(indices);
	
	return (Mesh){vao, vertexCount, indiciesCount};
}

static const int resolution = 20;
static const vec2 pos = {0.75, 1.25};
static const vec2 size = {0.75, 5.5};

static Mesh extDome, intDome;
static Mesh tentacles[4];

void initJellyfish() {
	extDome = generateDome((vec2){3.0, 1.25}, 0.0);
	intDome = generateDome((vec2){3.0, 1.0}, 0.2);

	tentacles[0] = genarateTentacles((vec3){pos.x, pos.y, 0}, (vec3){size.x, size.y, 0}, resolution);
	tentacles[1] = genarateTentacles((vec3){0, pos.y, pos.x}, (vec3){0, size.y, size.x}, resolution);
	tentacles[2] = genarateTentacles((vec3){-pos.x, pos.y, 0}, (vec3){size.x, size.y, 0}, resolution);
	tentacles[3] = genarateTentacles((vec3){0, pos.y, -pos.x}, (vec3){0, size.y, size.x}, resolution);
}

void renderJellyfish(mat4 projection, mat4 view, vec3 camPos, float time) {
	glEnable(GL_BLEND);
	glDisable(GL_CULL_FACE);
	glDepthFunc(GL_ALWAYS);

	glUseProgram(jellyfishShader);

	glUniformMatrix4fv(glGetUniformLocation(jellyfishShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
	glUniformMatrix4fv(glGetUniformLocation(jellyfishShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

	glUniform1f(glGetUniformLocation(jellyfishShader, "time"), time);
	glUniform3fv(glGetUniformLocation(jellyfishShader, "camPos"), 1, (GLfloat*)&camPos);
	glUniform1f(glGetUniformLocation(jellyfishShader, "camDist"), length(camPos) - 3.0);

	for (int i = 0; i < 4; i++) {
		glBindVertexArray(tentacles[i].VAO);
		glDrawElements(GL_TRIANGLES, tentacles[i].indexCount, GL_UNSIGNED_INT, NULL);
	}

	glBindVertexArray(intDome.VAO);
	glDrawElements(GL_TRIANGLES, intDome.indexCount, GL_UNSIGNED_INT, NULL);

	glBindVertexArray(extDome.VAO);
	glDrawElements(GL_TRIANGLES, extDome.indexCount, GL_UNSIGNED_INT, NULL);

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
}

void cleanupJellyfish() {
	freeMesh(extDome);
	freeMesh(intDome);
	for (int i = 0; i < 4; i++) {
		freeMesh(tentacles[i]);
	}
}
