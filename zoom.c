#include <X11/keysym.h>

#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "glutils.h"
#include "audio.h"
#include "shader.h"
#include "cameraController.h"
#include "galaxy.h"
#include "sphere.h"
#include "water.h"
#include "jellyfish.h"
#include "molecules.h"
#include "text.h"

vec2 screenSize = {600.0, 800.0};

bool running = true;
bool launched = false;

struct timespec start;

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

					updatePostProcessTextures(screenSize);
					projection = projectionMatrix(M_PI / 4.0, (float)xce.width / (float)xce.height, 0.1f, 1000.0f);
				}
				break;

			case KeyPress:
				{
					KeySym key = XLookupKeysym(&event.xkey, 0);
					if (key == XK_Escape) {
						running = false;
					}
					else if (key == XK_space && !launched) {
						launched = true;
						clock_gettime(CLOCK_MONOTONIC, &start);
						#ifndef WSL
							playMod("./mods/music.it.gz");
						#endif
					}
				}
				break;
		}
	}
}

int main() {
	#ifndef WSL
		initAudio();
	#endif
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

	vec2 waterSize = {50.0, 50.0};
	Mesh water = generateGrid(waterSize, 500);
	Mesh particles = createParticles(100, 1.0);
	initJellyfish();

	Mesh t = createText(L"Appuyez sur Espace pour commencer");

	initMolecules();
	generateDoubleHelix(100, 1.0, 75.0);
	generateAtom();

	
	projection = projectionMatrix(M_PI / 4.0, 800.0f / 600.0f, 0.001f, 1000.0f);
	vec3 lastCamPos = initializeCameraPosition();
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	float defaultTime = getTime(11);
	float lastTime = defaultTime;

	while (running) {
		handleEvents(display, wmDelete);

		struct timespec end;
		clock_gettime(CLOCK_MONOTONIC, &end);
		float ftime = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
		ftime += defaultTime;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		// Drawing text
		if (!launched) {
			glUseProgram(textShader);

			glUniform1f(glGetUniformLocation(textShader, "aspectRatio"), screenSize.x / screenSize.y);
			glUniform1f(glGetUniformLocation(textShader, "time"), ftime);

			glBindVertexArray(t.VAO);
			glDrawElements(GL_TRIANGLES, t.indexCount, GL_UNSIGNED_INT, NULL);
		}
		else {
			vec3 camPos;
			mat4 view = getCameraMatrix(&camPos, ftime);

			if (ftime < getTime(3)) {
				// Drawing galaxy
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
			}
			else if (ftime < getTime(4)) {
				float sunScale = fmin(1.0, lerp(0.0, 1.0, (ftime - getTime(3)) / 3.0));

				mat4 model = getIdentity();
				scaleMatrix(&model, (vec3){sunScale, sunScale, sunScale});

				// Drawing star
				glUseProgram(starShader);

				glUniformMatrix4fv(glGetUniformLocation(starShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
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

				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "view"), 1, GL_FALSE, (GLfloat*)&view);
				glUniform1f(glGetUniformLocation(bloomShader, "bloomRadius"), 3.0);

				glUniform3f(glGetUniformLocation(bloomShader, "bloomColor"), 1.0, 0.0, 0.0);

				renderScreenQuad();

				glDisable(GL_BLEND);
			}
			else if (ftime < getTime(6)) {
				float planeteScale = fmin(1.0, lerp(0.0, 1.0, (ftime - getTime(4)) / 3.0));
				vec3 planetPos = {0.0, 0.0, 50.0};

				mat4 model = generateTransformationMatrix(planetPos, (vec3){0.0, 0.0, 0.0}, (vec3){planeteScale, planeteScale, planeteScale});

				// Drawing planet
				glUseProgram(planetShader);

				glUniformMatrix4fv(glGetUniformLocation(planetShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(planetShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(planetShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1i(glGetUniformLocation(planetShader, "subdivisions"), 6);
				glUniform1f(glGetUniformLocation(planetShader, "radius"), 1.0);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, noiseTexture);
				glUniform1i(glGetUniformLocation(planetShader, "noiseTexture"), 0);
				glUniform3f(glGetUniformLocation(planetShader, "lightDir"), 0.0, 0.5, -1.0);
				glUniform3fv(glGetUniformLocation(planetShader, "camPos"), 1, (GLfloat*)&camPos);
				glUniform1f(glGetUniformLocation(planetShader, "camDist"), length(vec3_subtract(camPos, planetPos)) - 1.0);

				glBindVertexArray(planet.VAO);
				glDrawElements(GL_TRIANGLES, planet.indexCount, GL_UNSIGNED_INT, NULL);

				glEnable(GL_BLEND);

				glUseProgram(bloomShader);

				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(bloomShader, "view"), 1, GL_FALSE, (GLfloat*)&view);
				glUniform1f(glGetUniformLocation(bloomShader, "bloomRadius"), 1.2);

				glUniform3f(glGetUniformLocation(bloomShader, "bloomColor"), 0.56, 0.8, 1.0);

				renderScreenQuad();

				glDisable(GL_BLEND);
			}
			else if (ftime < getTime(8)) {
				// Drawing water
				updateSpectrum(ftime);

				glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				glViewport(0, 0, screenSize.x, screenSize.y);

				glUseProgram(waterShader);

				glUniformMatrix4fv(glGetUniformLocation(waterShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(waterShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D_ARRAY, displacementTextures);
				glUniform1i(glGetUniformLocation(waterShader, "_DisplacementTextures"), 0);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D_ARRAY, slopeTextures);
				glUniform1i(glGetUniformLocation(waterShader, "_SlopeTextures"), 1);

				glUniform3fv(glGetUniformLocation(waterShader, "_WorldSpaceCameraPos"), 1, (GLfloat*)&camPos);

				glBindVertexArray(water.VAO);

				mat4 model = getIdentity();
				const float tile = 4;
				for (float x = -waterSize.x * (tile / 2); x < waterSize.x * (tile / 2); x += waterSize.x) {
					for (float y = -waterSize.y * (tile / 2); y < waterSize.y * (tile / 2); y += waterSize.y) {
						translationMatrix(&model, (vec3){x, 0.0, y});
						glUniformMatrix4fv(glGetUniformLocation(waterShader, "model"), 1, GL_FALSE, (GLfloat*)&model);

						glDrawElements(GL_TRIANGLES, water.indexCount, GL_UNSIGNED_INT, NULL);
					}
				}

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
			}
			else if (ftime < getTime(10)) {
				// Underwater scene
				glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				// Drawing jellyfish
				renderJellyfish(projection, view, camPos, ftime);
				
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
			}
			else if (ftime < getTime(11)) {
				mat4 model = getIdentity();
				scaleMatrix(&model, (vec3){10.0, 10.0, 10.0});

				// Draw cells
				glUseProgram(cellShader);

				glUniformMatrix4fv(glGetUniformLocation(cellShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(cellShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(cellShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				float scale = lerp(0.01, 1.0, (ftime - getTime(10)) / 3.0);
				glUniform1f(glGetUniformLocation(cellShader, "scale"), scale);
				glUniform1f(glGetUniformLocation(cellShader, "camDist"), length(camPos));

				renderScreenQuad();
			}
			else if (ftime < getTime(13)) {
				renderAtoms(projection, view);
				renderDNA(projection, view, camPos);
			}

			lastCamPos.x = camPos.x;
			lastCamPos.y = camPos.y;
			lastCamPos.z = camPos.z;
		}

		checkOpenGLError();
		
		lastTime = ftime;

		glXSwapBuffers(display, window);
	}

	stopSound();

	freeMesh(galaxy);
	freeMesh(star);
	freeMesh(planet);
	freeMesh(water);
	freeMesh(particles);
	freeMesh(t);
	cleanupWater();
	cleanupJellyfish();

	cleanupMolecules();

	cleanupUtils();

	#ifndef WSL
		cleanupAudio();
	#endif
	cleanupWindow();

	return 0;
}