#include "galaxy.h"

static float distanceFromSpiral(const vec2* polarPoint, float a, float b, float theta_min, float theta_max) {
	float r = polarPoint->x;

	float theta = fmod(polarPoint->y, 2 * M_PI);
	if (theta < 0) {
		theta += 2 * M_PI;
	}

	const float theta_step = 0.01;
	float minDistance = FLT_MAX;

	// Iterate over the spiral to find the closest point
	for (float t = theta_min; t <= theta_max; t += theta_step) {
		float r_spiral = a * exp(b * t);

		float distance = polarDistance(r, theta, r_spiral, t);
		minDistance = fmin(minDistance, distance);
	}

	return minDistance;
}

StarPoint *generateGalaxy(unsigned int num_stars) {
	srand(8);

	StarPoint *stars = (StarPoint*)malloc(sizeof(StarPoint) * num_stars);
	if (stars) {
		float r_max = 5.0f;						// Galaxy radius
		float threshold_r = 4.0f;				// Density radius threshold
		float b = log(r_max) / (4.0f * M_PI);	// Thigthness of spiral 1
		float a2 = 1.0 / exp(b * M_PI);			// Thigthness of spiral 2

		float A = 0.1f;							// Amplitude for the bulge
		float sigma = 0.5f;						// Spread for the bulge
		float C = 0.05f;						// Initial height of the arms

		float theta_max = 5.0f * M_PI;

		for (int i = 0; i < num_stars - 1; i++) {
			float angle = ((float)rand() / (float)RAND_MAX) * 2 * M_PI;
			float r = sqrt((float)rand() / (float)RAND_MAX) * r_max;

			float x = r * cos(angle);
			float z = r * sin(angle);

			float thickness = gaussianBulge(x, z, A, sigma) + (r_max - r) * C;
			float height = ((((float)rand() / (float)RAND_MAX) * 2.0f) - 1.0f);
			float y = height * thickness;

			float distance = distanceFromSpiral(&(vec2){r, angle}, 1.0f, b, 0.0f, theta_max) * 0.5f;
			distance = fmin(distance, distanceFromSpiral(&(vec2){r, angle}, a2, b, M_PI, theta_max + M_PI) * 0.5f);
			distance = fmin(distance, segment_distance(&(vec3){x, y, z}) / 2.0f);

			float density = lerp(0.0f, 0.5f, distance * 2.0f);
			density *= fmax(height, 0.3);

			StarPoint point = {(vec3){x, y, z}, density};
			stars[i] = point;
		}

		stars[num_stars - 1] = (StarPoint){(vec3){0.0, 0.0, 0.0}, -1.0}; // Adding quasar
	}
	return stars;
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