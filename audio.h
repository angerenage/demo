#pragma once

#include <mikmod.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "ressource.h"

#define PCM_DEVICE "default"

extern pthread_t audioThread;

typedef struct memReader_s {
	MREADER core;
	const void *buffer;
	long len;
	long pos;
} MemReader;

void initAudio();
void playMod(const char *filename);
void stopSound();
void cleanupAudio();