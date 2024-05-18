#pragma once

#include <glad/glad.h>
#include <stdio.h>

extern GLuint textShader;
extern GLuint snoiseShader;

extern GLuint galaxyShader;

extern GLuint starShader;
extern GLuint bloomShader;
extern GLuint planetShader;

extern GLuint jellyfishShader;
extern GLuint particleShader;
extern GLuint underwaterPostProcessShader;
extern GLuint initialSpectrumShader;
extern GLuint spectrumUpdateShader;
extern GLuint waterShader;
extern GLuint atmospherePostProcessShader;

extern GLuint assembleMapsShader;
extern GLuint horizontalFFTShader;
extern GLuint verticalFFTShader;

extern GLuint cellShader;
extern GLuint dnaShader;
extern GLuint atomShader;
extern GLuint electronShader;

extern GLuint debugShader;

void initShaders();

GLuint compileShader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode);
GLuint compileComputeShader(const char *shaderCode);