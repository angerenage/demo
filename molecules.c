#include "molecules.h"

static Mesh adnSphere = {0};
static Mesh atomSphere = {0};

static GLuint dnaInstancedVBO = 0;
static int posNumber = 0;

static GLuint atomInstancedVBO = 0;
static const int nucleusNumber = sizeof(icosahedron_positions) / sizeof(vec3);

static const vec3 atomPos = {-1.0, 0.0, 0.0};

void initMolecules() {
	adnSphere = generateIcosphere(3);
	atomSphere = generateIcosphere(3);
}

void generateDoubleHelix(int numPoints, float radius, float twist) {
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
		dnaInstancedVBO = setupInstanceBuffer(adnSphere.VAO, positions, posNumber);
		free(positions);
	}
}

void renderDNA(mat4 projection, mat4 view, vec3 camPos, float time) {
	mat4 model = getIdentity();
	translationMatrix(&model, (vec3){0.0, -posNumber / 4.0f, 0.0});

	glEnable(GL_BLEND);
	glUseProgram(dnaShader);

	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

	float camDist = length(vec3_subtract(atomPos, camPos)) - 0.3;
	glUniform1f(glGetUniformLocation(dnaShader, "camDist"), camDist);
	float scale = lerp(0.3, 1.0, fmin(1.0, time / 3));
	glUniform1f(glGetUniformLocation(dnaShader, "scale"), scale);

	glBindVertexArray(adnSphere.VAO);
	glDrawElementsInstanced(GL_TRIANGLES, adnSphere.indexCount, GL_UNSIGNED_INT, NULL, posNumber);
	glBindVertexArray(0);

	glDisable(GL_BLEND);
}

void generateAtom() {
	atomInstancedVBO = setupInstanceBuffer(atomSphere.VAO, icosahedron_positions, nucleusNumber);
}

void renderAtoms(mat4 projection, mat4 view) {
	mat4 model = getIdentity();
	translationMatrix(&model, atomPos);
	scaleMatrix(&model, (vec3){0.002, 0.002, 0.002});

	glUseProgram(atomShader);

	glUniformMatrix4fv(glGetUniformLocation(atomShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
	glUniformMatrix4fv(glGetUniformLocation(atomShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
	glUniformMatrix4fv(glGetUniformLocation(atomShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

	glBindVertexArray(atomSphere.VAO);
	glDrawElementsInstanced(GL_TRIANGLES, atomSphere.indexCount, GL_UNSIGNED_INT, NULL, nucleusNumber);
	glBindVertexArray(0);
}

void cleanupMolecules() {
	glDeleteBuffers(1, &dnaInstancedVBO);
	glDeleteBuffers(1, &atomInstancedVBO);
}