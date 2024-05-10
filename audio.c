#include "audio.h"

void initAudio() {
	MikMod_RegisterAllDrivers();
	MikMod_RegisterAllLoaders();

	md_mode |= DMODE_SOFT_MUSIC;
	if (MikMod_Init("")) {
		fprintf(stderr, "Could not initialize sound, reason: %s\n", MikMod_strerror(MikMod_errno));
		exit(EXIT_FAILURE);
	}

	MODULE *module = Player_Load("./.build/2ND_PM.S3M", 64, 0); // 64 voices
	if (module) {
		Player_Start(module);
		while (Player_Active()) {
			MikMod_Update();
			//usleep(10000); // Sleep to prevent CPU hogging
		}
		Player_Stop();
		Player_Free(module);
	}
	else {
		fprintf(stderr, "Could not load module, reason: %s\n", MikMod_strerror(MikMod_errno));
	}
}

void cleanupAudio() {
	MikMod_Exit();
}
