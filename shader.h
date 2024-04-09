#pragma once

#include <glad/glad.h>
#include <stdio.h>

extern unsigned int galaxyShader;
extern unsigned int starShader;
extern unsigned int planetShader;
extern unsigned int textShader;
extern unsigned int snoiseShader;
extern unsigned int bloomShader;

extern unsigned int debugShader;

void initShaders();

unsigned int compileShader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode);