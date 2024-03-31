#pragma once

#include <GL/glew.h>
#include <stdio.h>

extern const char galaxyVertShaderSrc[];
extern const char galaxyFragShaderSrc[];

extern const char planeteVertShaderSrc[];
extern const char planeteGemoShaderSrc[];
extern const char planeteFragShaderSrc[];

extern unsigned int galaxyShader;
extern unsigned int planeteShader;

void initShaders();

unsigned int compileShader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode);