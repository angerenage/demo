#include "galaxy.h"

Mesh generateGalaxy(unsigned int num_stars) {
	srand(8);

	StarPoint *stars = (StarPoint*)malloc(sizeof(StarPoint) * num_stars);
	if (stars) {
		float r_max = 5.0f;		// Galaxy radius
		float A = 0.1f;			// Amplitude for the bulge
		float sigma = 0.5f;		// Spread for the bulge
		float C = 0.05f;		// Initial height of the arms

		for (int i = 0; i < num_stars - 1; i++) {
			float angle = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
			float r = sqrt((float)rand() / (float)RAND_MAX) * r_max;

			float x = r * cos(angle);
			float z = r * sin(angle);

			float thickness = gaussianBulge(x, z, A, sigma) + (r_max - r) * C;
			float height = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f);
			float y = height * thickness;

			StarPoint point = {(vec3){angle, r, y}, height};
			stars[i] = point;
		}

		stars[num_stars - 1] = (StarPoint){(vec3){0.0, 0.0, 0.0}, NAN}; // Adding quasar
	}

	GLuint vao = createGalaxyVAO(stars, num_stars);
	free(stars);

	return (Mesh){vao, num_stars, 0};
}

GLuint createGalaxyVAO(const StarPoint *stars, unsigned int num_stars) {
	GLuint vao, vbo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, num_stars * sizeof(StarPoint), stars, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(StarPoint), (void*)(offsetof(StarPoint, position)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(StarPoint), (void*)(offsetof(StarPoint, density)));

	glBindVertexArray(0);

	return vao;
}