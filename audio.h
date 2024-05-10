#pragma once

#include <mikmod.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define PCM_DEVICE "default"

void initAudio();

void send_midi_note(int portid, int note, int velocity, int channel, bool on_off);

void cleanupAudio();