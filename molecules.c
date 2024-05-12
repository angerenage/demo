#include "molecules.h"

static Mesh atomSphere = {0};
static GLuint dnaInstancedVBO = 0;
static int numPoints = 0;

void generateDoubleHelix(int numPoints, float radius, float twist) {
	if (atomSphere.VAO == 0) {
		atomSphere = generateIcosphere(3);
	}

	vec3 *positions = (vec3*)malloc(numPoints * 2 * sizeof(vec3));

	float angleIncrement = twist / numPoints;
	float heightIncrement = 1.0f / numPoints;

	for (int i = 0; i < numPoints; i++) {
		float t = i * angleIncrement;
		float z = i * heightIncrement;
		
		// Helix 1
		positions[2 * i].x = 0.0;//radius * cos(t);
		positions[2 * i].y = 0.0;//z;
		positions[2 * i].z = 0.0;//radius * sin(t);

		//printf("test : {%f, %f, %f}\n", radius * cos(t), radius * sin(t), z);
		
		// Helix 2
		positions[2 * i + 1].x = 0.0;//radius * cos(t + M_PI);
		positions[2 * i + 1].y = 0.0;//z;
		positions[2 * i + 1].z = 0.0;//radius * sin(t + M_PI);
	}

	numPoints *= 2;
	dnaInstancedVBO = createVAO(positions, numPoints);//setupInstanceBuffer(atomSphere.VAO, positions, numPoints);
	free(positions);
}

void renderDNA(mat4 projection, mat4 view) {
	glPointSize(10.0f);
	glDisable(GL_DEPTH_TEST);

	glUseProgram(dnaShader);

	mat4 model = getIdentity();
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
	glUniformMatrix4fv(glGetUniformLocation(dnaShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

	//glBindVertexArray(atomSphere.VAO);
	glBindVertexArray(dnaInstancedVBO);
	glDrawArrays(GL_POINTS, 0, numPoints);
	//glDrawElements(GL_TRIANGLES, atomSphere.indexCount, GL_UNSIGNED_INT, NULL);
	//glDrawElementsInstanced(GL_POINTS, atomSphere.indexCount, GL_UNSIGNED_INT, NULL, numPoints);
	glBindVertexArray(0);
}

void cleanupMolecules() {
	glDeleteBuffers(1, &dnaInstancedVBO);
}