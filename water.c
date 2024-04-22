#include "water.h"

GLuint generateGrid(vec2 size, int subdivision, int *indexNbr) {
	const int width = subdivision + 1;
	const int vertexNbr = (subdivision + 2) * (subdivision + 2);
	*indexNbr = width * width * 6;

	GLuint vao = 0;
	vec3 *positions = (vec3*)malloc(sizeof(vec3) * vertexNbr);
	int *indices = (int*)malloc(sizeof(int) * (*indexNbr));

	if (positions && indices) {
		int index = 0;

		for (int y = 0; y <= width; y++) {
			for (int x = 0; x <= width; x++) {
				float posX = size.x * ((float)x / (float)width);
				float posZ = size.y * ((float)y / (float)width);

				positions[x + y * (width + 1)] = (vec3){posX - size.x / 2.0, 0.0, posZ - size.y / 2.0};

				if (x < width && y < width) {
					int topLeft = x + y * (width + 1);
					int topRight = (x + 1) + y * (width + 1);
					int bottomLeft = x + (y + 1) * (width + 1);
					int bottomRight = (x + 1) + (y + 1) * (width + 1);

					// First triangle
					indices[index++] = topLeft;
					indices[index++] = bottomLeft;
					indices[index++] = topRight;

					// Second triangle
					indices[index++] = topRight;
					indices[index++] = bottomLeft;
					indices[index++] = bottomRight;
				}
			}
		}

		vao = createIndexedVAO(positions, vertexNbr, indices, *indexNbr);

		free(positions);
		free(indices);
	}

	return vao;
}

GLuint createParticles(int pointCount, float radius) {
	vec3 *points = (vec3*)malloc(pointCount * sizeof(vec3));
	GLuint vao = 0;

	if (points) {
		for (int i = 0; i < pointCount; ++i) {
			float u = (float)rand() / RAND_MAX;
			float v = (float)rand() / RAND_MAX;
			float theta = u * 2.0f * M_PI;
			float phi = acos(2.0f * v - 1.0f);
			float r = cbrtf((float)rand() / RAND_MAX) * radius;

			points[i].x = r * sin(phi) * cos(theta);
			points[i].y = r * sin(phi) * sin(theta);
			points[i].z = r * cos(phi);
		}

		vao = createVAO(points, pointCount);

		free(points);
	}

	return vao;
}

typedef struct spectrumParameters_s {
	float scale;
	float windDirection;
	float spreadBlend;
	float swell;
	float windSpeed;
	float fetch;
	float peakEnhancement;
	float shortWavesFade;
} SpectrumParameters;

static const SpectrumParameters params[] = {
	//scale, windDirection, spreadBlend, swell, windSpeed, fetch,       peakEnhancement, shortWavesFade
	{0.1,    22.0,          0.642,       1.0,   2.0,       100000.0,    1.0,             0.025},
	{0.07,   59.0,          0.0,         1.0,   2.0,       1000.0,      1.0,             0.01},
	{0.25,   97.0,          0.14,        1.0,   20.0,      100000000.0, 1.0,             0.5},
	{0.25,   67.0,          0.47,        1.0,   20.0,      1000000.0,   1.0,             0.5},
	{0.15,   105.0,         0.2,         1.0,   5.0,       1000000.0,   1.0,             0.5},
	{0.1,    19.0,          0.298,       1.0,   1.0,       10000.0,     1.0,             0.5},
	{1.0,    209.0,         0.56,        0.695, 1.0,       200000.0,    1.0,             0.0001},
	{0.23,   0.0,           0.0,         1.0,   1.0,       1000.0,      1.0,             0.0001},
};

static const int frequencySize = 1024;
static const float gravity = 9.81f;

GLuint displacementTextures = 0;
GLuint slopeTextures = 0;
/*static*/ GLuint spectrumTextures = 0;
static GLuint spectrumFBO = 0;
static GLuint initialSpectrumTex = 0;

void initWater() {
	//computing initial spectrum
	initialSpectrumTex = createTextureArray(frequencySize, frequencySize, 4);

	GLuint initialSpectrumFBO;
	glGenFramebuffers(1, &initialSpectrumFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, initialSpectrumFBO);

	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, initialSpectrumTex, 0, 0);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, initialSpectrumTex, 0, 1);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, initialSpectrumTex, 0, 2);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, initialSpectrumTex, 0, 3);

	GLenum initDrawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
	glDrawBuffers(4, initDrawBuffers);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		printf("Framebuffer is not complete!\n");

	glViewport(0, 0, frequencySize, frequencySize);
	glUseProgram(initialSpectrumShader);

	glUniform1ui(glGetUniformLocation(initialSpectrumShader, "_Seed"), 0);
	glUniform1ui(glGetUniformLocation(initialSpectrumShader, "_N"), frequencySize);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_LengthScale0"), 94.0f);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_LengthScale1"), 128.0f);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_LengthScale2"), 64.0f);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_LengthScale3"), 32.0f);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_LowCutoff"), 0.0001f);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_HighCutoff"), 9000.0f);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_Gravity"), gravity);
	glUniform1f(glGetUniformLocation(initialSpectrumShader, "_Depth"), 20.0f);

	for (int i = 0; i < 8; i++) {
		char buffer[14];
		char buffer2[50];
		sprintf(buffer, "_Spectrums[%d]", i);
		strcpy(buffer2, buffer);

		float jonswapAlpha = 0.076f * pow(gravity * params[i].fetch / params[i].windSpeed / params[i].windSpeed, -0.22f);
		float jonswapPeakFrequency = 22.0f * pow(params[i].windSpeed * params[i].fetch / gravity / gravity, -0.33f);

		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".scale")), params[i].scale);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".angle")), (params[i].windDirection / 180.0) * M_PI);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".spreadBlend")), params[i].spreadBlend);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".swell")), params[i].swell);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".alpha")), jonswapAlpha);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".peakOmega")), jonswapPeakFrequency);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".gamma")), params[i].peakEnhancement);
		strcpy(buffer2, buffer);
		glUniform1f(glGetUniformLocation(initialSpectrumShader, strcat(buffer2, ".shortWavesFade")), params[i].shortWavesFade);
	}

	renderScreenQuad();

	

	// setup for actual spectrum
	spectrumTextures = createTextureArray(frequencySize, frequencySize, 8);
	glBindImageTexture(0, spectrumTextures, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA16F);

	glGenFramebuffers(1, &spectrumFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, spectrumFBO);

	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, spectrumTextures, 0, 0);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, spectrumTextures, 0, 1);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, spectrumTextures, 0, 2);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, spectrumTextures, 0, 3);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, spectrumTextures, 0, 4);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, spectrumTextures, 0, 5);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT6, spectrumTextures, 0, 6);
	glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT7, spectrumTextures, 0, 7);

	GLenum upDrawBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5, GL_COLOR_ATTACHMENT6, GL_COLOR_ATTACHMENT7};
	glDrawBuffers(8, upDrawBuffers);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		printf("Framebuffer is not complete!\n");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// setup for displacement texture
	displacementTextures = createTextureArray(frequencySize, frequencySize, 4);
	glBindImageTexture(1, displacementTextures, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA16F);

	slopeTextures = createTextureArrayRG(frequencySize, frequencySize, 4);
	glBindImageTexture(2, slopeTextures, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG16F);

	updateSpectrum(0.0f);

	glDeleteFramebuffers(1, &initialSpectrumFBO);
}

void updateSpectrum(float time) {
	glBindFramebuffer(GL_FRAMEBUFFER, spectrumFBO);
	glViewport(0, 0, frequencySize, frequencySize);
	glUseProgram(spectrumUpdateShader);

	glUniform1ui(glGetUniformLocation(spectrumUpdateShader, "_N"), frequencySize);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_LengthScale0"), 94.0f);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_LengthScale1"), 128.0f);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_LengthScale2"), 64.0f);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_LengthScale3"), 32.0f);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_RepeatTime"), 200.0f);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_FrameTime"), time);
	glUniform1f(glGetUniformLocation(spectrumUpdateShader, "_Gravity"), gravity);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, initialSpectrumTex);
	glUniform1i(glGetUniformLocation(spectrumUpdateShader, "initialSpectrum"), 0);
	
	renderScreenQuad();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// IFFT
	glUseProgram(horizontalFFTShader);
	glDispatchCompute(1, frequencySize, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glUseProgram(verticalFFTShader);
	glDispatchCompute(1, frequencySize, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Map assembly
	glUseProgram(assembleMapsShader);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, spectrumTextures);
	glUniform1i(glGetUniformLocation(assembleMapsShader, "_SpectrumTextures"), 0);

	glDispatchCompute(frequencySize / 8, frequencySize / 8, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void cleanupWater() {
	glDeleteTextures(1, &displacementTextures);
	glDeleteTextures(1, &slopeTextures);
	glDeleteTextures(1, &initialSpectrumTex);
	glDeleteTextures(1, &spectrumTextures);
	glDeleteFramebuffers(1, &spectrumFBO);
}
