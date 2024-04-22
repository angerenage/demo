#include <glad/glad.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "glutils.h"
#include "shader.h"
#include "galaxy.h"
#include "sphere.h"
#include "water.h"
#include "text.h"

vec2 screenSize = {600.0, 800.0};

bool running = true;
int displayedScene = 0;
bool mousePressed = false;
int lastX = 0, lastY = 0;
float cameraAngleX = -0.75f, cameraAngleY = 0.0f;

mat4 projection = {0};

void handleEvents(Display *display, Atom wmDelete) {
	XEvent event;
	while (XPending(display)) {
		XNextEvent(display, &event);
		switch (event.type) {
			case ClientMessage:
				if (event.xclient.data.l[0] == wmDelete) {
					running = false;
				}
				break;

			case ConfigureNotify:
				{
					XConfigureEvent xce = event.xconfigure;
					glViewport(0, 0, xce.width, xce.height);

					screenSize = (vec2){(float)xce.width, (float)xce.height};

					projection = projectionMatrix(M_PI / 4.0, (float)xce.width / (float)xce.height, 0.1f, 1000.0f);
				}
				break;

			case KeyPress:
				KeySym key = XLookupKeysym(&event.xkey, 0);
				if (key == XK_Escape) {
					running = false;
				}
				else if (key == XK_Tab) {
					displayedScene++;
					if (displayedScene >= 5) displayedScene = 0;
				}
				break;

			case ButtonPress:
				if (event.xbutton.button == Button1) {
					mousePressed = true;
					lastX = event.xbutton.x;
					lastY = event.xbutton.y;
				}
				break;

			case ButtonRelease:
				if (event.xbutton.button == Button1) {
					mousePressed = false;
				}
				break;

			case MotionNotify:
				if (mousePressed) {
					int dx = event.xmotion.x - lastX;
					int dy = event.xmotion.y - lastY;
					lastX = event.xmotion.x;
					lastY = event.xmotion.y;

					cameraAngleX += (float)dy * 0.001f;
					cameraAngleY -= (float)dx * 0.001f;
				}
				break;
		}
	}
}

int main() {
	Display *display = XOpenDisplay(NULL);
	if (display == NULL) {
		fprintf(stderr, "Cannot open display\n");
		return 1;
	}

	int screen = DefaultScreen(display);
	Window root = RootWindow(display, screen);

	int fbAttribs[] = {
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_STENCIL_SIZE, 8,
		GLX_DOUBLEBUFFER, True,
		GLX_SAMPLE_BUFFERS, 1,
		GLX_SAMPLES, 4,
		None
	};

	int fbcount;
	GLXFBConfig *fbConfigs = glXChooseFBConfig(display, screen, fbAttribs, &fbcount);
	if (!fbConfigs || fbcount == 0) {
		fprintf(stderr, "Failed to retrieve framebuffer config\n");
		return 1;
	}

	GLXFBConfig fbConfig = fbConfigs[0]; // Choisissez le premier de la liste
	XFree(fbConfigs); // Libérer la liste des configurations

	XVisualInfo *vi = glXGetVisualFromFBConfig(display, fbConfig);
	if (vi == NULL) {
		fprintf(stderr, "No appropriate visual found\n");
		return 1;
	}

	XSetWindowAttributes swa;
	swa.colormap = XCreateColormap(display, root, vi->visual, AllocNone);
	swa.border_pixel = 0;
	swa.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask;

	Window win = XCreateWindow(
		display, root,
		0, 0, 800, 600, 0,
		vi->depth, InputOutput,
		vi->visual,
		CWBorderPixel | CWColormap | CWEventMask, &swa
	);

	// Handle close button
	Atom wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(display, win, &wmDelete, 1);

	XMapWindow(display, win);
	XStoreName(display, win, "Zoom Demo");

	typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
	glXCreateContextAttribsARBProc glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB((const GLubyte *) "glXCreateContextAttribsARB");
	if (glXCreateContextAttribsARB == NULL) {
		fprintf(stderr, "glXCreateContextAttribsARB not found. Exiting.\n");
		exit(1);
	}

	int contextAttribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
		GLX_CONTEXT_MINOR_VERSION_ARB, 3,
		GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
		None
	};

	GLXContext glc = glXCreateContextAttribsARB(display, fbConfig, NULL, True, contextAttribs);
	if (!glc) {
		fprintf(stderr, "Failed to create GL context\n");
		return 1;
	}
	glXMakeCurrent(display, win, glc);

	if (!gladLoadGLLoader((GLADloadproc)glXGetProcAddress)) {
		fprintf(stderr, "Failed to initialize GLAD\n");
		return -1;
	}

	glEnable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_PROGRAM_POINT_SIZE);

	glDepthFunc(GL_LESS);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	initShaders();

	// Simplex noise texture calculation
	glViewport(0, 0, 1024, 1024);
	GLuint noiseTexture = createTexture(1024, 1024);
	GLuint noiseFBO = createFramebuffer(noiseTexture);

	glBindFramebuffer(GL_FRAMEBUFFER, noiseFBO);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(snoiseShader);

	glUniform2f(glGetUniformLocation(snoiseShader, "resolution"), 512.0, 512.0);
	glUniform1f(glGetUniformLocation(snoiseShader, "time"), 0.0);

	renderScreenQuad();

	initWater();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, screenSize.x, screenSize.y);


	projection = projectionMatrix(M_PI / 4.0, 800.0f / 600.0f, 0.01f, 1000.0f);

	unsigned int num_stars = 30000;
	StarPoint *stars = generateGalaxy(num_stars);
	GLuint galaxyVAO = 0;
	if (stars) {
		galaxyVAO = createGalaxyVAO(stars, num_stars);
		free(stars);
	}
	else running = false;

	Mesh star = generateIcosphere(2);
	Mesh planet = generateIcosphere(2);

	int waterIndexNumber = 0;
	GLuint water = generateGrid((vec2){10.0, 10.0}, 1000, &waterIndexNumber);
	const int particleNbr = 100;
	GLuint particles = createParticles(particleNbr, 1.0);

	int indiceCount = 0;
	GLuint t = createText(L"Appuyez sur tab pour changer de scène", &indiceCount);
	
	float camDistance = 10.0f;

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	vec3 lastCamPos = {camDistance * sin(cameraAngleX) * sin(cameraAngleY), camDistance * cos(cameraAngleX), camDistance * sin(cameraAngleX) * cos(cameraAngleY)};

	float lastTime = 0.0;
	while (running) {
		clock_gettime(CLOCK_MONOTONIC, &end);
		float ftime = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
		
		handleEvents(display, wmDelete);

		vec3 camPos = {camDistance * sin(cameraAngleX) * sin(cameraAngleY), camDistance * cos(cameraAngleX), camDistance * sin(cameraAngleX) * cos(cameraAngleY)};
		mat4 view = viewMatrix(camPos, (vec3){0.0, 0.0, 0.0}, (vec3){0.0, 1.0, 0.0});

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(textShader);

		glUniform1f(glGetUniformLocation(textShader, "aspectRatio"), screenSize.x / screenSize.y);
		glUniform1f(glGetUniformLocation(textShader, "time"), ftime);

		glBindVertexArray(t);
		glDrawElements(GL_TRIANGLES, indiceCount, GL_UNSIGNED_INT, NULL);
		
		switch (displayedScene) {
			case 0: // Drawing galaxy
				glEnable(GL_BLEND);

				glUseProgram(galaxyShader);

				glUniformMatrix4fv(glGetUniformLocation(galaxyShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(galaxyShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1f(glGetUniformLocation(galaxyShader, "screenWidth"), screenSize.x);
				glUniform1f(glGetUniformLocation(galaxyShader, "r_max"), 5.0);

				glDepthMask(0x00);
				glBindVertexArray(galaxyVAO);
				glDrawArrays(GL_POINTS, 0, num_stars);
				glDepthMask(0xFF);

				glDisable(GL_BLEND);
				break;

			case 1:
				// Drawing star
				glUseProgram(starShader);

				glUniformMatrix4fv(glGetUniformLocation(starShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(starShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1i(glGetUniformLocation(starShader, "subdivisions"), 6);
				glUniform1f(glGetUniformLocation(starShader, "radius"), 2.0);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, noiseTexture);
				glUniform1i(glGetUniformLocation(starShader, "noiseTexture"), 0);

				glBindVertexArray(star.VAO);
				glDrawElements(GL_TRIANGLES, star.indexCount, GL_UNSIGNED_INT, NULL);


				glEnable(GL_BLEND);

				glUseProgram(bloomShader);

				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "view"), 1, GL_FALSE, (GLfloat*)&view);
				glUniform1f(glGetUniformLocation(bloomShader, "bloomRadius"), 5.5);

				renderScreenQuad();

				glDisable(GL_BLEND);
				break;

			case 2:
				// Drawing planet
				glUseProgram(planetShader);

				glUniformMatrix4fv(glGetUniformLocation(planetShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(planetShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1i(glGetUniformLocation(planetShader, "subdivisions"), 6);
				glUniform1f(glGetUniformLocation(planetShader, "radius"), 1.0);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, noiseTexture);
				glUniform1i(glGetUniformLocation(planetShader, "noiseTexture"), 0);
				glUniform3f(glGetUniformLocation(planetShader, "lightDir"), 0.0, 1.0, 0.0);

				glBindVertexArray(planet.VAO);
				glDrawElements(GL_TRIANGLES, planet.indexCount, GL_UNSIGNED_INT, NULL);
				break;

			case 3:
				// Drawing water
				updateSpectrum(ftime);

				glViewport(0, 0, screenSize.x, screenSize.y);

				glUseProgram(waterSahder);

				glUniformMatrix4fv(glGetUniformLocation(waterSahder, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(waterSahder, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D_ARRAY, displacementTextures);
				glUniform1i(glGetUniformLocation(waterSahder, "_DisplacementTextures"), 0);

				glBindVertexArray(water);
				glDrawElements(GL_TRIANGLES, waterIndexNumber, GL_UNSIGNED_INT, NULL);
				break;

			case 4:
				// Underwater scene
				glEnable(GL_BLEND);

				glUseProgram(particleShader);

				glUniformMatrix4fv(glGetUniformLocation(particleShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(particleShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform3fv(glGetUniformLocation(particleShader, "camPos"), 1, (GLfloat*)&camPos);
				glUniform1f(glGetUniformLocation(particleShader, "radius"), 1.0);
				glUniform1f(glGetUniformLocation(particleShader, "time"), ftime);
				glUniform1f(glGetUniformLocation(particleShader, "deltaTime"), ftime - lastTime);
				glUniform3f(glGetUniformLocation(particleShader, "camDir"), lastCamPos.x - camPos.x, lastCamPos.y - camPos.y, lastCamPos.z - camPos.z);

				glBindVertexArray(particles);
				glDrawArrays(GL_POINTS, 0, particleNbr);

				glDisable(GL_BLEND);
				break;
		}

		checkOpenGLError();

		lastCamPos.x = camPos.x;
		lastCamPos.y = camPos.y;
		lastCamPos.z = camPos.z;
		lastTime = ftime;

		glXSwapBuffers(display, win);
	}

	if (noiseTexture) glDeleteTextures(1, &noiseTexture);
	if (noiseFBO) glDeleteFramebuffers(1, &noiseFBO);
	if (galaxyVAO) glDeleteVertexArrays(1, &galaxyVAO);
	if (water) glDeleteVertexArrays(1, &water);
	if (particles) glDeleteVertexArrays(1, &particles);
	if (t) glDeleteVertexArrays(1, &t);
	freeMesh(&star);
	freeMesh(&planet);
	cleanupWater();
	cleanupUtils();

	glXMakeCurrent(display, None, NULL);
	glXDestroyContext(display, glc);
	XDestroyWindow(display, win);
	XFreeColormap(display, swa.colormap);
	XFree(vi);
	XCloseDisplay(display);

	return 0;
}