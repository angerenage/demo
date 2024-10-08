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

vec2 screenSize = {800.0, 600.0};

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

	Text openingText = createText(L"Appuyez sur Espace pour commencer", 0.08);


	Text creditText = createText(L"Démo imaginée et développée par Ange Rollet", 0.065);
	Text musicTitleText = createText(L"- Musique -", 0.08);
	Text musicCreditText = createText(L"\"And I will ever be\" par Elmusho", 0.06);
	Text thanksTitleText = createText(L"- Remerciements spéciaux -", 0.08);
	Text thanksParis8Text = createText(L"Université Paris 8    -    Farès Belhadj     ", 0.055);
	Text thanksWaterText = createText(L"      Gasgiant    -    GarrettGunnell", 0.055);
	Text thanksStackText = createText(L"Stackoverflow    -    Stackexchange", 0.055);
	Text thanksElseText = createText(L"ChatGPT    -    Ronja  ", 0.055);
	Text thanksShadersText = createText(L"Patricio Gonzalez Vivo    -    Jen Lowe              ", 0.055);

	Text *credits[] = {
		&creditText,
		&musicTitleText,
		&musicCreditText,
		&thanksTitleText,
		&thanksParis8Text,
		&thanksWaterText,
		&thanksStackText,
		&thanksElseText,
		&thanksShadersText,
	};


	initMolecules();
	generateDoubleHelix(130, 1.0, 82.0);
	generateAtom();

	
	projection = projectionMatrix(M_PI / 4.0, screenSize.x / screenSize.y, 0.001f, 1000.0f);
	vec3 lastCamPos = initializeCameraPosition();
	
	clock_gettime(CLOCK_MONOTONIC, &start);
	float defaultTime = getTime(0);
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
			fixHorizontal(&openingText, CENTER_ANCHOR, screenSize, 0.0);
			fixVertical(&openingText, BOTTOM_ANCHOR, screenSize, 100.0);

			mat4 model = getIdentity();
			translationMatrix(&model, openingText.pos);
			scaleMatrix(&model, (vec3){openingText.scale, openingText.scale, openingText.scale});
			
			glUseProgram(textShader);

			glUniformMatrix4fv(glGetUniformLocation(textShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
			glUniform1f(glGetUniformLocation(textShader, "aspectRatio"), screenSize.x / screenSize.y);
			glUniform1f(glGetUniformLocation(textShader, "time"), ftime);

			glBindVertexArray(openingText.mesh.VAO);
			glDrawElements(GL_TRIANGLES, openingText.mesh.indexCount, GL_UNSIGNED_INT, NULL);
		}
		else {
			vec3 camPos;
			mat4 view = getCameraMatrix(&camPos, ftime);

			if (currentScene == GALAXY_SCENE) {
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
			else if (currentScene == SUN_SCENE) {
				float sunScale = fmin(1.0, lerp(0.0, 1.0, (ftime - getTime(3)) / 3.0));

				mat4 model = getIdentity();
				scaleMatrix(&model, (vec3){sunScale, sunScale, sunScale});

				// Drawing star
				glUseProgram(starShader);

				glUniformMatrix4fv(glGetUniformLocation(starShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(starShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(starShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1i(glGetUniformLocation(starShader, "subdivisions"), 5);
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
			else if (currentScene == PLANET_SCENE) {
				float planeteScale = fmin(1.0, lerp(0.0, 1.0, (ftime - getTime(4)) / 3.0));
				vec3 planetPos = {0.0, 0.0, 50.0};

				mat4 model = generateTransformationMatrix(planetPos, (vec3){0.0, 0.0, 0.0}, (vec3){planeteScale, planeteScale, planeteScale});

				// Drawing planet
				glUseProgram(planetShader);

				glUniformMatrix4fv(glGetUniformLocation(planetShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(planetShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(planetShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				glUniform1i(glGetUniformLocation(planetShader, "subdivisions"), 5);
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
			else if (currentScene == WATER_SCENE) {
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
				glUniform1i(glGetUniformLocation(waterShader, "displacementTextures"), 0);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D_ARRAY, slopeTextures);
				glUniform1i(glGetUniformLocation(waterShader, "slopeTextures"), 1);

				glUniform3fv(glGetUniformLocation(waterShader, "camPos"), 1, (GLfloat*)&camPos);

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
				glUniform3fv(glGetUniformLocation(atmospherePostProcessShader, "camPos"), 1, (GLfloat*)&camPos);

				renderScreenQuad();
			}
			else if (currentScene == UNDERWATER_SCENE) {
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
				glUniform3fv(glGetUniformLocation(underwaterPostProcessShader, "camPos"), 1, (GLfloat*)&camPos);

				renderScreenQuad();
			}
			else if (currentScene == CELL_SCENE) {
				mat4 model = getIdentity();
				scaleMatrix(&model, (vec3){50.0, 50.0, 50.0});

				// Draw cells
				glUseProgram(cellShader);

				glUniformMatrix4fv(glGetUniformLocation(cellShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
				glUniformMatrix4fv(glGetUniformLocation(cellShader, "projection"), 1, GL_FALSE, (GLfloat*)&projection);
				glUniformMatrix4fv(glGetUniformLocation(cellShader, "view"), 1, GL_FALSE, (GLfloat*)&view);

				float scale = lerp(0.01, 1.0, (ftime - getTime(10) - 1.0) / 5.0);
				glUniform1f(glGetUniformLocation(cellShader, "scale"), scale);
				glUniform1f(glGetUniformLocation(cellShader, "camDist"), length(camPos));

				renderScreenQuad();
			}
			else if (currentScene == MOLECULE_SCENE) {
				// Draw DNA molecule and atom
				renderAtoms(projection, view, ftime);
				renderDNA(projection, view, camPos, ftime - getTime(11));
			}
			else if (currentScene == CREDIT_SCENE) {
				fixHorizontal(&creditText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&creditText, TOP_ANCHOR, screenSize, screenSize.y * 0.175 - 25.0);


				fixHorizontal(&musicTitleText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&musicTitleText, TOP_ANCHOR, screenSize, screenSize.y * 0.35);

				fixHorizontal(&musicCreditText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&musicCreditText, TOP_ANCHOR, screenSize, screenSize.y * 0.55);


				fixHorizontal(&thanksTitleText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&thanksTitleText, TOP_ANCHOR, screenSize, screenSize.y * 0.75);

				fixHorizontal(&thanksParis8Text, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&thanksParis8Text, MIDDLE_ANCHOR, screenSize, 0.0);

				fixHorizontal(&thanksWaterText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&thanksWaterText, MIDDLE_ANCHOR, screenSize, screenSize.y * -0.2);

				fixHorizontal(&thanksStackText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&thanksStackText, MIDDLE_ANCHOR, screenSize, screenSize.y * -0.4);

				fixHorizontal(&thanksElseText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&thanksElseText, MIDDLE_ANCHOR, screenSize, screenSize.y * -0.6);

				fixHorizontal(&thanksShadersText, CENTER_ANCHOR, screenSize, 0.0);
				fixVertical(&thanksShadersText, MIDDLE_ANCHOR, screenSize, screenSize.y * -0.8);


				for (unsigned int i = 0; i < sizeof(credits) / sizeof(Text*); i++) {
					mat4 model = getIdentity();
					translationMatrix(&model, credits[i]->pos);
					scaleMatrix(&model, (vec3){credits[i]->scale, credits[i]->scale, credits[i]->scale});
					
					glUseProgram(textShader);

					glUniformMatrix4fv(glGetUniformLocation(textShader, "model"), 1, GL_FALSE, (GLfloat*)&model);
					glUniform1f(glGetUniformLocation(textShader, "aspectRatio"), screenSize.x / screenSize.y);
					glUniform1f(glGetUniformLocation(textShader, "time"), ftime - getTime(13) - (i * 0.4f));

					glBindVertexArray(credits[i]->mesh.VAO);
					glDrawElements(GL_TRIANGLES, credits[i]->mesh.indexCount, GL_UNSIGNED_INT, NULL);
				}
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
	freeMesh(openingText.mesh);
	for (unsigned int i = 0; i < sizeof(credits) / sizeof(Text*); i++) {
		freeMesh(credits[i]->mesh);
	}
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