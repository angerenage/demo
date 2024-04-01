#include "shader.h"

unsigned int galaxyShader = 0;
unsigned int planeteShader = 0;

void initShaders() {
	galaxyShader = compileShader(galaxyVertShaderSrc, NULL, galaxyFragShaderSrc);
	planeteShader = compileShader(planeteVertShaderSrc, planeteGemoShaderSrc, planeteFragShaderSrc);
}

unsigned int compileShader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode) {
	unsigned int vertex, geometry, fragment;
	int success;
	char infoLog[512];

	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertex, 512, NULL, infoLog);
		printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
	}

	// geometry shader
	if (gShaderCode) {
		geometry = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometry, 1, &gShaderCode, NULL);
		glCompileShader(geometry);
		
		glGetShaderiv(geometry, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(geometry, 512, NULL, infoLog);
			printf("ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n%s\n", infoLog);
		}
	}

	// fragment shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragment, 512, NULL, infoLog);
		printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
	}

	unsigned int ID = glCreateProgram();
	glAttachShader(ID, vertex);
	if (gShaderCode) glAttachShader(ID, geometry);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(ID, 512, NULL, infoLog);
		printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
	}

	glDeleteShader(vertex);
	if (gShaderCode) glDeleteShader(geometry);
	glDeleteShader(fragment);

	return ID;
}

const char galaxyVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;
layout(location = 1) in float densityIn;

uniform mat4 projection;
uniform mat4 view;

out float density;
out vec3 position;
flat out int star;

float rand(vec2 co) {
	return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float smootherRand(vec2 co) {
	float total = 0.0;
	float amplitude = 1.0;
	float frequency = 1.0;
	for (int i = 0; i < 4; i++) { // for 4 octaves
		total += rand(co * frequency) * amplitude;
		amplitude *= 0.5;
		frequency *= 2.0;
	}
	return total;
}

void main() {
	gl_Position = projection * view * vec4(positionIn, 1.0);
	float centerDist = length(positionIn.xz);

	density = densityIn;
	position = positionIn;
	star = 0;

	if (densityIn < 0.0f) { // Detecting quasar
		gl_PointSize = 75.0f;
	}
	else if (centerDist < 4.0f && smootherRand(positionIn.xz) > 1.5) {
		star = 1;
		gl_PointSize = 2.0f;
	}
	else {
		float correctedDensity = densityIn;
		if (centerDist > 4.0f) {
			float falloff = (5.0f - centerDist);
			correctedDensity *= falloff;
		}
		
		float maxSize = 200.0f;
		float minSize = 10.0f;
		gl_PointSize = mix(minSize, maxSize, correctedDensity * 3.0);
	}
}
)glsl";

const char galaxyFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in float density;
in vec3 position;
flat in int star;

float rand(vec2 co) {
	return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float smootherRand(vec2 co) {
	float total = 0.0;
	float amplitude = 1.0;
	float frequency = 1.0;
	for (int i = 0; i < 4; i++) { // for 4 octaves
		total += rand(co * frequency) * amplitude;
		amplitude *= 0.5;
		frequency *= 2.0;
	}
	return total;
}

void main() {
	float dist = length(gl_PointCoord - vec2(0.5, 0.5));
	if (dist > 0.5f) discard;

	if (density < 0.0f) { // Detecting quasar
		float alpha = max(0.0, (1.0 - (dist * dist) * 4.2) + 0.05) * 0.8;
		alpha = min(alpha, 1.0);
		fragColor = vec4(vec3(1.0, 1.0, 1.0), alpha);
	}
	else if (star == 1) {
		fragColor = vec4(1.0f);
	}
	else {
		float centerDist = length(position.xz);

		float minAlpha = 0.01f;
		float maxAlpha = 0.3f;
		float alpha = max(minAlpha, mix(maxAlpha, minAlpha, ((density * 5.0f) / centerDist) * 10.0f));

		if (centerDist > 4.0f) {
			float falloff = (5.0f - centerDist);
			alpha *= falloff;
		}

		vec3 color = mix(vec3(0.46f, 0.54f, 0.54f), vec3(0.70f, 0.80f, 0.85f), density * 10.0f);
		float redThreshold = 0.1f;
		if (density <= redThreshold) {
			color = mix(color, vec3(0.35f, 0.3f, 0.3f), (redThreshold - density) * 7.0f);

			if (smootherRand(position.xz) < 0.6f) {
				color = mix(color, vec3(0.9f, 0.2f, 0.2f), 0.3f);
			}
		}

		if (centerDist < 1.5f) {
			vec3 centerColor = vec3(0.8f, 0.7f, 0.6f);
			float centerFactor = smoothstep(0.0f, 2.0f, centerDist);
			color = mix(centerColor, color, centerFactor);
		}
		fragColor = vec4(color, alpha);
	}
}
)glsl";

// --------------------------- PLANETE SHADERS ---------------------------

const char planeteVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;
layout(location = 1) in vec3 normalIn;

uniform mat4 projection;
uniform mat4 view;

out vec3 normal;

void main() {
	vec3 position = positionIn;
	position.y = -positionIn.y;
	normal = normalIn;
	normal.y = -normalIn.y;

	gl_Position = projection * view * vec4(position, 1.0);
}
)glsl";

const char planeteGemoShaderSrc[] = R"glsl(#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 64) out;

in vec3 normal[];

out vec3 fragNormal;

uniform int subdivisions;

const int MAX_SUBDIVISIONS = 5;

void emitVertex(vec4 position) {
	gl_Position = position;
	EmitVertex();
}

void subdivideAndEmit(vec4 A, vec4 B, vec4 C, int s) {
	if (s <= 1 || s > MAX_SUBDIVISIONS) {
		emitVertex(A);
		emitVertex(B);
		emitVertex(C);
		EndPrimitive();
	}
	else {
		vec4 lastPoints[MAX_SUBDIVISIONS + 1];
		lastPoints[0] = A;

		for (int i = 1; i <= s; i++) {
			vec4 p1 = mix(A, B, float(i) / float(s));
			vec4 p2 = mix(A, C, float(i) / float(s));

			vec4 points[MAX_SUBDIVISIONS + 1];
			points[0] = p1;
			int pointCount = 1;

			for (int j = 1; j <= i; j++) {
				vec4 p3 = mix(p1, p2, float(j) / float(i));
				points[pointCount++] = p3;

				if (j > 1) {
					emitVertex(lastPoints[j - 2]);
					emitVertex(lastPoints[j - 1]);
					emitVertex(points[j - 1]);
					EndPrimitive();
				}
				
				if (j <= i) {
					emitVertex(lastPoints[j - 1]);
					emitVertex(p3);
					emitVertex(p3 - (p2 - p1) / float(i));
					EndPrimitive();
				}
			}

			for (int x = 0; x < pointCount; x++) {
				lastPoints[x] = points[x];
			}
		}
	}
}

void main() {
	vec4 A = gl_in[0].gl_Position;
	vec4 B = gl_in[1].gl_Position;
	vec4 C = gl_in[2].gl_Position;

	subdivideAndEmit(A, B, C, subdivisions);
}
)glsl";

const char planeteFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec3 fragNormal;

uniform vec3 lightDir;

void main() {
	float shadow = max(0.0, dot(lightDir, fragNormal));
	fragColor = vec4(vec3(shadow), 1.0);
}
)glsl";
