#include <X11/keysym.h>

#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "glutils.h"
#include "shader.h"
#include "galaxy.h"
#include "sphere.h"
#include "water.h"
#include "jellyfish.h"
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

					updateUnderwaterTextures(screenSize);
					projection = projectionMatrix(M_PI / 4.0, (float)xce.width / (float)xce.height, 0.1f, 1000.0f);
				}
				break;

			case KeyPress:
				{
					KeySym key = XLookupKeysym(&event.xkey, 0);
					if (key == XK_Escape) {
						running = false;
					}
					else if (key == XK_Tab) {
						displayedScene++;
						if (displayedScene >= 5) displayedScene = 0;
					}
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
	initWindow(screenSize);

	glEnable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_PROGRAM_POINT_SIZE);

	glDepthFunc(GL_LESS);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	initShaders();
	initNoise();
	initWater(screenSize);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, screenSize.x, screenSize.y);


	Mesh galaxy = generateGalaxy(30000);

	Mesh star = generateIcosphere(2);
	Mesh planet = generateIcosphere(2);

	Mesh water = generateGrid((vec2){50.0, 50.0}, 1000);
	Mesh particles = createParticles(100, 1.0);
	Mesh jellyfish = generateDome((vec2){3.0, 1.5}, 0.0);

	Mesh t = createText(L"Appuyez sur tab pour changer de scène");

	
	projection = projectionMatrix(M_PI / 4.0, 800.0f / 600.0f, 0.01f, 1000.0f);

	float camDistance = 10.0f;
	vec3 lastCamPos = {camDistance * sin(cameraAngleX) * sin(cameraAngleY), camDistance * cos(cameraAngleX), camDistance * sin(cameraAngleX) * cos(cameraAngleY)};
	
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

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

		glBindVertexArray(t.VAO);
		glDrawElements(GL_TRIANGLES, t.indexCount, GL_UNSIGNED_INT, NULL);
		
		switch (displayedScene) {
			case 0: // Drawing galaxy
				glEnable(GL_BLEND);

				glUseProgram(galaxyShader);

				glUniformMatrix4fv(glGetUniformLocation(galaxyShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(galaxyShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1f(glGetUniformLocation(galaxyShader, "screenWidth"), screenSize.x);
				glUniform1f(glGetUniformLocation(galaxyShader, "r_max"), 5.0);

				glDepthMask(0x00);
				glBindVertexArray(galaxy.VAO);
				glDrawArrays(GL_POINTS, 0, galaxy.vertexCount);
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

				glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				glViewport(0, 0, screenSize.x, screenSize.y);

				glUseProgram(waterSahder);

				glUniformMatrix4fv(glGetUniformLocation(waterSahder, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(waterSahder, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D_ARRAY, displacementTextures);
				glUniform1i(glGetUniformLocation(waterSahder, "_DisplacementTextures"), 0);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D_ARRAY, slopeTextures);
				glUniform1i(glGetUniformLocation(waterSahder, "_SlopeTextures"), 1);

				glUniform3fv(glGetUniformLocation(waterSahder, "_WorldSpaceCameraPos"), 1, (GLfloat*)&camPos);

				glBindVertexArray(water.VAO);
				glDrawElements(GL_TRIANGLES, water.indexCount, GL_UNSIGNED_INT, NULL);

				// Water scene post-processing
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glUseProgram(atmospherePostProcessShader);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderDepthTexture);
				glUniform1i(glGetUniformLocation(atmospherePostProcessShader, "renderDepthTexture"), 0);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderColorTexture);
				glUniform1i(glGetUniformLocation(atmospherePostProcessShader, "renderColorTexture"), 1);

				glUniformMatrix4fv(glGetUniformLocation(atmospherePostProcessShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(atmospherePostProcessShader, "view"), 1, GL_FALSE, (GLfloat*)&view);
				glUniform3fv(glGetUniformLocation(atmospherePostProcessShader, "cameraPos"), 1, (GLfloat*)&camPos);

				renderScreenQuad();
				break;

			case 4:
				// Underwater scene
				glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				// Drawing jellyfish
				glUseProgram(debugShader);

				glUniformMatrix4fv(glGetUniformLocation(debugShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(debugShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glBindVertexArray(jellyfish.VAO);
				glDrawElements(GL_TRIANGLES, jellyfish.indexCount, GL_UNSIGNED_INT, NULL);

				// Drawing particles
				glEnable(GL_BLEND);

				glUseProgram(particleShader);

				glUniformMatrix4fv(glGetUniformLocation(particleShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(particleShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform3fv(glGetUniformLocation(particleShader, "camPos"), 1, (GLfloat*)&camPos);
				glUniform1f(glGetUniformLocation(particleShader, "radius"), 1.0);
				glUniform1f(glGetUniformLocation(particleShader, "time"), ftime);
				glUniform1f(glGetUniformLocation(particleShader, "deltaTime"), ftime - lastTime);
				glUniform3f(glGetUniformLocation(particleShader, "camDir"), lastCamPos.x - camPos.x, lastCamPos.y - camPos.y, lastCamPos.z - camPos.z);

				glBindVertexArray(particles.VAO);
				glDrawArrays(GL_POINTS, 0, particles.vertexCount);

				glDisable(GL_BLEND);

				// Underwater scene post-processing
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
				glUseProgram(underwaterPostProcessShader);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderDepthTexture);
				glUniform1i(glGetUniformLocation(underwaterPostProcessShader, "underwaterDepthTexture"), 0);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderColorTexture);
				glUniform1i(glGetUniformLocation(underwaterPostProcessShader, "underwaterColorTexture"), 1);

				glUniformMatrix4fv(glGetUniformLocation(underwaterPostProcessShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(underwaterPostProcessShader, "view"), 1, GL_FALSE, (GLfloat*)&view);
				glUniform3fv(glGetUniformLocation(underwaterPostProcessShader, "cameraPos"), 1, (GLfloat*)&camPos);

				renderScreenQuad();
				break;
		}

		checkOpenGLError();

		lastCamPos.x = camPos.x;
		lastCamPos.y = camPos.y;
		lastCamPos.z = camPos.z;
		lastTime = ftime;

		glXSwapBuffers(display, window);
	}

	freeMesh(galaxy);
	freeMesh(star);
	freeMesh(planet);
	freeMesh(water);
	freeMesh(particles);
	freeMesh(jellyfish);
	freeMesh(t);
	cleanupWater();
	cleanupUtils();

	cleanupWindow();

	return 0;
}