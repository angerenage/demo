#include "audio.h"

static snd_pcm_t *pcm_handle;

void initAudio() {
	unsigned int pcm, tmp, dir;
	int rate, channels, seconds;
	snd_pcm_hw_params_t *params;
	snd_pcm_uframes_t frames;
	float freq;

	rate = 44100; // Sample rate
	channels = 1; // Mono sound
	seconds = 4; // Duration of the tone
	freq = 440.0; // Frequency of the tone

	if (snd_pcm_open(&pcm_handle, PCM_DEVICE, SND_PCM_STREAM_PLAYBACK, 0) < 0) {
		printf("ERROR: Can't open \"%s\" PCM device.\n", PCM_DEVICE);
		exit(EXIT_FAILURE);
	}

	snd_pcm_hw_params_alloca(&params);
	snd_pcm_hw_params_any(pcm_handle, params);

	if (snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
		printf("ERROR: Can't set interleaved mode.\n");
		exit(EXIT_FAILURE);
	}

	if (snd_pcm_hw_params_set_format(pcm_handle, params, SND_PCM_FORMAT_FLOAT_LE) < 0) {
		printf("ERROR: Can't set format.\n");
		exit(EXIT_FAILURE);
	}

	if (snd_pcm_hw_params_set_channels(pcm_handle, params, channels) < 0) {
		printf("ERROR: Can't set channels number.\n");
		exit(EXIT_FAILURE);
	}

	if (snd_pcm_hw_params_set_rate_near(pcm_handle, params, &rate, 0) < 0) {
		printf("ERROR: Can't set rate.\n");
		exit(EXIT_FAILURE);
	}

	if (snd_pcm_hw_params(pcm_handle, params) < 0) {
		printf("ERROR: Can't set harware parameters.\n");
		exit(EXIT_FAILURE);
	}

	if (snd_pcm_hw_params_set_buffer_size_near(pcm_handle, params, &frames) < 0) {
		printf("ERROR: Can't set buffer size.\n");
		exit(EXIT_FAILURE);
	}
}

void cleanupAudio() {
	snd_pcm_drain(pcm_handle);
	snd_pcm_close(pcm_handle);
}
