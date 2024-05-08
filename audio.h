#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>
#include <math.h>
//#include "geometry.h"

#define PCM_DEVICE "default"

void initAudio();
void cleanupAudio();