#pragma once

#include <mikmod.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define PCM_DEVICE "default"

extern pthread_t audioThread;

void initAudio();
void playMod(const char *filename);
void stopSound();
void cleanupAudio();