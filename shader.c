#include "shader.h"

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

static const char galaxyVertShaderSrc[] = R"glsl(#version 330 core
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
		float minSize = 15.0f;
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

static const char planeteVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

void main() {
	gl_Position = vec4(positionIn, 1.0);
}
)glsl";

static const char planeteGemoShaderSrc[] = R"glsl(#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 128) out;

out vec3 fragPosition;
out vec3 fragNormal;

uniform mat4 projection;
uniform mat4 view;
uniform int subdivisions;

const int MAX_SUBDIVISIONS = 5;

vec3 calculateNormal(vec3 point) {
	return normalize(point);
}

void emitVertex(vec3 position) {
	fragPosition = position;
	fragNormal = calculateNormal(position);
	gl_Position = projection * view * vec4(normalize(position), 1.0);
	EmitVertex();
}

void subdivideAndEmit(vec3 A, vec3 B, vec3 C, int s) {
	if (s <= 1 || s > MAX_SUBDIVISIONS) {
		emitVertex(A);
		emitVertex(B);
		emitVertex(C);
		EndPrimitive();
	} else {
		vec3 lastPoints[MAX_SUBDIVISIONS + 1];
		lastPoints[0] = A;

		for (int i = 1; i <= s; i++) {
			vec3 p1 = mix(A, B, float(i) / float(s));
			vec3 p2 = mix(A, C, float(i) / float(s));

			vec3 points[MAX_SUBDIVISIONS + 1];
			points[0] = p1;
			int pointCount = 1;

			for (int j = 1; j <= i; j++) {
				vec3 p3 = mix(p1, p2, float(j) / float(i));
				points[pointCount++] = p3;

				if (j > 1) {
					emitVertex(lastPoints[j - 2]);
					emitVertex(lastPoints[j - 1]);
					emitVertex(points[j - 1]);
					EndPrimitive();
				}
				
				if (j <= i) {
					vec3 p4 = p3 - (p2 - p1) / float(i);
					emitVertex(lastPoints[j - 1]);
					emitVertex(p3);
					emitVertex(p4);
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
	vec3 A = gl_in[0].gl_Position.xyz;
	vec3 B = gl_in[1].gl_Position.xyz;
	vec3 C = gl_in[2].gl_Position.xyz;

	subdivideAndEmit(A, B, C, subdivisions);
}
)glsl";

// https://www.ronja-tutorials.com/post/010-triplanar-mapping/
static const char planeteFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightDir;
uniform sampler2D noiseTexture;

void main() {
	float scale = 0.75;
	float sharpness = 64.0;

	vec4 col_front = texture(noiseTexture, fragPosition.xy * scale + vec2(0.5, 0.5));
	vec4 col_side = texture(noiseTexture, fragPosition.yz * scale + vec2(0.5, 0.5));
	vec4 col_top = texture(noiseTexture, fragPosition.zx * scale + vec2(0.5, 0.5));

	vec3 blendWeights = pow(abs(fragNormal), vec3(sharpness));
	blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z);

	vec4 color = col_front * blendWeights.z + 
				 col_side * blendWeights.x + 
				 col_top * blendWeights.y;

	float shadow = max(0.0, dot(lightDir, fragNormal));
	fragColor = shadow * color;
}
)glsl";

// --------------------------- TEXT SHADERS ---------------------------

static const char textVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec2 positionIn;
layout(location = 1) in int idIn;

flat out int id;

uniform float aspectRatio;

void main() {
	id = idIn;

	vec2 adjustedPosition = positionIn / 10.0f;
	adjustedPosition.x /= aspectRatio;

	gl_Position = vec4(adjustedPosition, 0.0, 1.0) - vec4(0.90f, 0.80f, 0.0, 0.0);
}
)glsl";

static const char textFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

flat in int id;

uniform float time;

float pseudoRandom(float seed) {
	return fract(sin(seed) * 10000.0);
}

void main() {
	float randomness = pseudoRandom(id * 0.1);
	float bias = id / 100.0;
	float visibilityThreshold = randomness * 0.5 + bias;
	if (visibilityThreshold > time * 7.0) discard;

	fragColor = vec4(1.0); 
}
)glsl";

// --------------------------- NOISE SHADERS ---------------------------

static const char postVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

void main() {
	gl_Position = vec4(positionIn, 1.0);
}
)glsl";

// Algorithm by patriciogonzalezvivo (https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83)
static const char snoise[] = R"glsl(#version 330 core
out vec4 fragColor;

uniform float time;
uniform vec2 resolution;
const float warpAmount = 0.0;

vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v) {
	const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
	vec2 i  = floor(v + dot(v, C.yy));
	vec2 x0 = v -   i + dot(i, C.xx);
	vec2 i1;
	i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
	vec4 x12 = x0.xyxy + C.xxzz;
	x12.xy -= i1;
	i = mod(i, 289.0);
	vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0))
	+ i.x + vec3(0.0, i1.x, 1.0));
	vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
	m = m*m ;
	m = m*m ;
	vec3 x = 2.0 * fract(p * C.www) - 1.0;
	vec3 h = abs(x) - 0.5;
	vec3 ox = floor(x + 0.5);
	vec3 a0 = x - ox;
	m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
	vec3 g;
	g.x  = a0.x  * x0.x  + h.x  * x0.y;
	g.yz = a0.yz * x12.xz + h.yz * x12.yw;
	return 130.0 * dot(m, g);
}

float fractal_noise(vec2 p, int octaves, float persistence, float lacunarity) {
	float total = 0.0;
	float amplitude = 1.0;
	float maxAmplitude = 0.0;
	float frequency = 1.0;

	for (int i = 0; i < octaves; i++) {
		total += snoise(p * frequency) * amplitude;
		maxAmplitude += amplitude;
		amplitude *= persistence;
		frequency *= lacunarity;
	}

	return total / maxAmplitude;
}

vec2 warp(vec2 p) {
	float noise = snoise(p * 0.1 + time * 0.1);
	return p + warpAmount * vec2(cos(noise + time), sin(noise + time));
}

void main() {
	vec2 uv = gl_FragCoord.xy / resolution.xy;
	vec2 warpedPosition = warp(uv * 10.0);
	float noise = fractal_noise(warpedPosition, 4, 0.5, 2.0);
	fragColor = vec4(vec3(noise + 1.0) / 2.0, 1.0);
}
)glsl";

unsigned int galaxyShader = 0;
unsigned int planeteShader = 0;
unsigned int textShader = 0;
unsigned int snoiseShader = 0;

void initShaders() {
	galaxyShader = compileShader(galaxyVertShaderSrc, NULL, galaxyFragShaderSrc);
	planeteShader = compileShader(planeteVertShaderSrc, planeteGemoShaderSrc, planeteFragShaderSrc);
	textShader = compileShader(textVertShaderSrc, NULL, textFragShaderSrc);
	snoiseShader = compileShader(postVertSrc, NULL, snoise);
}