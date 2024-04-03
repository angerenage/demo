#pragma once

#include <glad/glad.h>
#include <stdio.h>

extern unsigned int galaxyShader;
extern unsigned int planeteShader;
extern unsigned int textShader;
extern unsigned int snoiseShader;

void initShaders();

unsigned int compileShader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode);