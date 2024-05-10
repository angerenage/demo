#include "audio.h"

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static int keep_playing = 1;

pthread_t audioThread;

static void* audioThreadRoutine(void *arg) {
	const char* module_path = (const char*)arg;

	MODULE *module = Player_Load(module_path, 64, 0);
	if (module) {
		Player_Start(module);
		pthread_mutex_lock(&mutex);
        while (Player_Active() && keep_playing) {
            pthread_mutex_unlock(&mutex);
            MikMod_Update();
            usleep(10000);
            pthread_mutex_lock(&mutex);
        }
        pthread_mutex_unlock(&mutex);
		Player_Stop();
		Player_Free(module);
	}
	else {
		fprintf(stderr, "Could not load module, reason: %s\n", MikMod_strerror(MikMod_errno));
	}
}

void initAudio() {
	MikMod_RegisterAllDrivers();
	MikMod_RegisterAllLoaders();

	md_mode |= DMODE_SOFT_MUSIC;
	if (MikMod_Init("")) {
		fprintf(stderr, "Could not initialize sound, reason: %s\n", MikMod_strerror(MikMod_errno));
		exit(EXIT_FAILURE);
	}
}

void playMod(const char *filename) {
	if (pthread_create(&audioThread, NULL, audioThreadRoutine, (void*)filename) != 0) {
		fprintf(stderr, "Failed to create thread\n");
		exit(EXIT_FAILURE);
	}
}

void stopSound() {
    pthread_mutex_lock(&mutex);
    keep_playing = 0;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
}

void cleanupAudio() {
	pthread_join(audioThread, NULL);
	MikMod_Exit();
}
