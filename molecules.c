#include "molecules.h"

static Mesh atomSphere = {0};
static GLuint dnaInstancedVBO = 0;
static int posNumber = 0;

void generateDoubleHelix(int numPoints, float radius, float twist) {
	if (atomSphere.VAO == 0) {
		atomSphere = generateIcosphere(3);
	}

	vec3 *positions = (vec3*)malloc(numPoints * 2 * sizeof(vec3));

	if (positions) {
		float angleIncrement = twist / numPoints;
		float heightIncrement = 1.0f;

		for (int i = 0; i < numPoints; i++) {
			float t = i * angleIncrement;
			float z = i * heightIncrement;
			
			// Helix 1
			positions[2 * i].x = radius * cos(t);
			positions[2 * i].y = z;
			positions[2 * i].z = radius * sin(t);
			
			// Helix 2
			positions[2 * i + 1].x = radius * cos(t + M_PI);
			positions[2 * i + 1].y = z;
			positions[2 * i + 1].z = radius * sin(t + M_PI);
		}
		
		posNumber = numPoints * 2;
		dnaInstancedVBO = setupInstanceBuffer(atomSphere.VAO, positions, posNumber);
		free(positions);
	}
}

void renderDNA(mat4 projection, mat4 view) {
	glUseProgram(dnaShader);

	mat4 model = getIdentity();
	translationMatrix(&model, (vec3){0.0, -posNumber / 4.0f, 0.0});

	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

	glBindVertexArray(atomSphere.VAO);
	glDrawElementsInstanced(GL_TRIANGLES, atomSphere.indexCount, GL_UNSIGNED_INT, NULL, posNumber);
	glBindVertexArray(0);
}

void cleanupMolecules() {
	glDeleteBuffers(1, &dnaInstancedVBO);
}