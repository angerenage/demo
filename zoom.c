#include <GL/glew.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <stdio.h>
#include <stdbool.h>

#include "shader.h"
#include "galaxy.h"
#include "planete.h"

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

void checkOpenGLError() {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		char *error;

		switch (err) {
			case GL_INVALID_OPERATION:              error = "INVALID_OPERATION"; break;
			case GL_INVALID_ENUM:                   error = "INVALID_ENUM"; break;
			case GL_INVALID_VALUE:                  error = "INVALID_VALUE"; break;
			case GL_OUT_OF_MEMORY:                  error = "OUT_OF_MEMORY"; break;
			case GL_INVALID_FRAMEBUFFER_OPERATION:  error = "INVALID_FRAMEBUFFER_OPERATION"; break;
			default:                                error = "UNKNOWN_ERROR"; break;
		}

		printf("OpenGL Error: %s\n", error);
	}
}

bool running = true;
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

						projection = projectionMatrix(M_PI / 4.0, (float)xce.width / (float)xce.height, 0.1f, 1000.0f);
					}
					break;

				case KeyPress:
					KeySym key = XLookupKeysym(&event.xkey, 0);
					if (key == XK_Escape) {
						running = false;
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

	// Attributs pour le framebuffer config
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
	swa.event_mask = ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask;

	Window win = XCreateWindow(
		display, root,
		0, 0, 800, 600, 0,
		vi->depth, InputOutput,
		vi->visual,
		CWColormap | CWEventMask, &swa
	);

	// Handle close button
	Atom wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", True);
	XSetWMProtocols(display, win, &wmDelete, 1);

	XMapWindow(display, win);
	XStoreName(display, win, "Zoom Demo");

	glXCreateContextAttribsARBProc glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB((const GLubyte *) "glXCreateContextAttribsARB");

	if (glXCreateContextAttribsARB == NULL) {
		fprintf(stderr, "glXCreateContextAttribsARB not found. Exiting.\n");
		exit(1);
	}

	int contextAttribs[] = {
		GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
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

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		return -1;
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);

	glDepthFunc(GL_LESS);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_PROGRAM_POINT_SIZE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	initShaders();

	projection = projectionMatrix(M_PI / 4.0, 800.0f / 600.0f, 0.1f, 1000.0f);

	unsigned int num_stars = 30000;
	StarPoint *stars = generateGalaxy(num_stars);
	GLuint galaxyVAO = 0;
	if (stars) {
		galaxyVAO = createGalaxyVAO(stars, num_stars);
		free(stars);
	}
	else running = false;

	bool error = false;
	Mesh planete = generateIcosphere(&error);
	if (error) {
		printf("error\n");
		running = false;
	}
	
	float camDistance = 10.0f;

	while (running) {
		handleEvents(display, wmDelete);

		mat4 view = viewMatrix((vec3){camDistance  * sin(cameraAngleX) * sin(cameraAngleY), camDistance * cos(cameraAngleX), camDistance  * sin(cameraAngleX) * cos(cameraAngleY)}, (vec3){0.0, 0.0, 0.0}, (vec3){0.0, 1.0, 0.0});

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		// Drawing galaxy
		glUseProgram(galaxyShader);

		glUniformMatrix4fv(glGetUniformLocation(galaxyShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
		glUniformMatrix4fv(glGetUniformLocation(galaxyShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

		
		glDepthMask(0x00);
		glBindVertexArray(galaxyVAO);
		glDrawArrays(GL_POINTS, 0, num_stars);
		glBindVertexArray(0);
		glDepthMask(0xFF);
		


		// Drawing planetes
		glUseProgram(planeteShader);

		glUniformMatrix4fv(glGetUniformLocation(planeteShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
		glUniformMatrix4fv(glGetUniformLocation(planeteShader, "view"), 1, GL_FALSE, (GLfloat*)&view);
		
		glUniform1i(glGetUniformLocation(planeteShader, "subdivisions"), 1);

		glUniform3f(glGetUniformLocation(planeteShader, "lightDir"), 0.0, 1.0, 0.0);

		glBindVertexArray(planete.VAO);
		glDrawElements(GL_TRIANGLES, planete.indexCount, GL_UNSIGNED_INT, NULL);
		glBindVertexArray(0);

		checkOpenGLError();

		glXSwapBuffers(display, win);
	}

	freeMesh(&planete);

	glXMakeCurrent(display, None, NULL);
    glXDestroyContext(display, glc);
    XDestroyWindow(display, win);
    XFreeColormap(display, swa.colormap);
    XFree(vi);
    XCloseDisplay(display);

	return 0;
}