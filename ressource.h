#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#include <assert.h>

#define CHUNK 4096

int loadRessource(const char *sourceFile, void** data, size_t *data_size);
void zerr(int ret);