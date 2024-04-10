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

// --------------------------- GALAXY SHADERS ---------------------------

static const char galaxyVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;
layout(location = 1) in float heightIn;

#define M_PI 3.1415926535897932384626433832795
#define FLT_MAX 1.0 / 0.0

out float density;
out vec3 position;
flat out int star;

uniform mat4 projection;
uniform mat4 view;

uniform float screenWidth;
uniform float r_max;

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

float distanceFromSpiral(vec2 polarPoint, float a, float b, float theta_min, float theta_max) {
	float minDistance = FLT_MAX;

	float theta_start = polarPoint.x;
	if (polarPoint.x <= theta_min) theta_start += 2 * M_PI;

	for (float t = theta_start; t <= theta_max; t += 2 * M_PI) {
		float r_spiral = a * exp(b * t);

		float distance = abs(polarPoint.y - r_spiral);
		minDistance = min(minDistance, distance);
	}

	return minDistance;
}

float segment_distance(vec3 point) {
	float x_distance = abs(point.x) - 0.5f;
	float y_distance = abs(point.y) - 0.05f;
	float z_distance = abs(point.z) - 0.05f;

	return max(0.0f, x_distance) + max(0.0f, y_distance) + max(0.0f, z_distance);
}

void main() {
	if (isnan(heightIn)) { // Detecting quasar
		density = -1.0;
		gl_PointSize = 60.0f * (screenWidth / 800.0);
		gl_Position = projection * view * vec4(0.0, 0.0, 0.0, 1.0);
	}
	else {
		float x = positionIn.y * cos(positionIn.x);
		float y = positionIn.z;
		float z = positionIn.y * sin(positionIn.x);

		gl_Position = projection * view * vec4(vec3(x, y, z), 1.0);

		float threshold_r = 4.0f;				// Density radius threshold
		float b = log(r_max) / (4.0f * M_PI);	// Thigthness of spiral 1
		float a2 = 1.0 / exp(b * M_PI);			// Thigthness of spiral 2
		float theta_max = 5.0f * M_PI;

		float distance = distanceFromSpiral(positionIn.xy, 1.0f, b, 0.0, theta_max) * 0.5f;
		distance = min(distance, distanceFromSpiral(positionIn.xy, a2, b, M_PI, theta_max + M_PI) * 0.5f);
		distance = min(distance, segment_distance(vec3(x, y, z) / 2.0f));

		density = mix(0.0f, 0.5f, distance * 2.0f);
		density *= max(heightIn, 0.3);

		position = vec3(x, y, z);
		star = 0;
		
		if (positionIn.y < threshold_r && smootherRand(positionIn.xz) > 1.5) {
			star = 1;
			gl_PointSize = 2.0f;
		}
		else {
			float correctedDensity = density;
			if (positionIn.y > threshold_r) {
				float falloff = (5.0f - positionIn.y);
				correctedDensity *= falloff;
			}
			
			float maxSize = 75.0f;
			float minSize = 8.0f;
			gl_PointSize = min(maxSize, mix(minSize, maxSize, ((correctedDensity * 7.0f) / positionIn.y) * 1.5) * (screenWidth / 800.0));
		}
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
		float alpha = max(0.0, (1.0 - (dist * dist) * 4.2) + 0.05) * 0.9;
		alpha = min(alpha, 1.0);
		fragColor = vec4(vec3(1.0), alpha);
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

// --------------------------- SPHERE SHADERS ---------------------------

static const char sphereVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

void main() {
	gl_Position = vec4(positionIn, 1.0);
}
)glsl";

static const char sphereGemoShaderSrc[] = R"glsl(#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 128) out;

out vec3 fragPosition;
out vec3 fragNormal;

uniform mat4 projection;
uniform mat4 view;
uniform int subdivisions;
uniform float radius;

const int MAX_SUBDIVISIONS = 6;

vec3 calculateNormal(vec3 point) {
	return normalize(point);
}

void emitVertex(vec3 position) {
	fragPosition = position * radius;
	fragNormal = calculateNormal(position);
	gl_Position = projection * view * vec4(normalize(position) * radius, 1.0);
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

// --------------------------- STAR SHADERS ---------------------------

// https://www.ronja-tutorials.com/post/010-triplanar-mapping/
static const char starFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec3 fragPosition;
in vec3 fragNormal;

uniform sampler2D noiseTexture;

void main() {
	float sharpness = 1.0;

	vec4 colFront = texture(noiseTexture, fragPosition.xy / 2.0 + vec2(0.5, 0.5));
	vec4 colSide = texture(noiseTexture, fragPosition.yz / 2.0 + vec2(0.5, 0.5));
	vec4 colTop = texture(noiseTexture, fragPosition.zx / 2.0 + vec2(0.5, 0.5));

	vec3 blendWeights = pow(abs(fragNormal), vec3(sharpness));
	blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z);

	vec4 noise = colFront * blendWeights.z +
				 colSide * blendWeights.x +
				 colTop * blendWeights.y;

	float lum = abs((noise.r * 2.0) - 1.0);

	vec3 hotColor = mix(vec3(1.0, 0.86, 0.73), vec3(1.0, 0.45, 0.0), min(1.0, lum * 10.0 + 0.5));
	vec3 color = mix(hotColor, vec3(0.82, 0.2, 0.01), lum * 2.0);

	fragColor = vec4(color, 1.0);
}
)glsl";

// --------------------------- PLANET SHADERS ---------------------------

static const char planetFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightDir;
uniform sampler2D noiseTexture;

void main() {
	float sharpness = 1.0;

	vec4 colFront = texture(noiseTexture, fragPosition.xy * 2.0 + vec2(0.5, 0.5));
	vec4 colSide = texture(noiseTexture, fragPosition.yz * 2.0 + vec2(0.5, 0.5));
	vec4 colTop = texture(noiseTexture, fragPosition.zx * 2.0 + vec2(0.5, 0.5));

	vec3 blendWeights = pow(abs(fragNormal), vec3(sharpness));
	blendWeights /= (blendWeights.x + blendWeights.y + blendWeights.z);

	vec4 noise = colFront * blendWeights.z +
				 colSide * blendWeights.x +
				 colTop * blendWeights.y;

	fragColor = vec4(vec3(noise) * dot(fragNormal, lightDir), 1.0);
}
)glsl";

// --------------------------- TEXT SHADERS ---------------------------

static const char textVertShaderSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

out float id;

uniform float aspectRatio;

void main() {
	id = positionIn.z;

	vec2 adjustedPosition = positionIn.xy / 10.0f;
	adjustedPosition.x /= aspectRatio;

	gl_Position = vec4(adjustedPosition, 0.0, 1.0) - vec4(0.90f, 0.80f, 0.0, 0.0);
}
)glsl";

static const char textFragShaderSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in float id;

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
	fragColor = vec4(vec3((noise + 1.0) / 2.0), 1.0);
}
)glsl";

// --------------------------- BLOOM SHADERS ---------------------------

static const char bloomVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

out vec2 texCoords;

uniform mat4 projection;
uniform mat4 view;
uniform float bloomRadius;

void main() {
	vec3 viewRight = normalize(vec3(view[0][0], view[1][0], view[2][0]));
	vec3 viewUp = normalize(vec3(view[0][1], view[1][1], view[2][1]));

	float distance = length((view * vec4(vec3(0.0), 1.0)).xyz);
	float scale = bloomRadius / distance;

	vec3 position = (positionIn.x * bloomRadius * viewRight + positionIn.y * bloomRadius * viewUp) * scale;

	gl_Position = projection * view * vec4(position, 1.0);
	texCoords = positionIn.xy;
}
)glsl";

static const char bloomFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec2 texCoords;

void main() {
	if (length(texCoords) > 1.0) discard;
	fragColor = vec4(1.0, 0.0, 0.0, (1.0 - length(texCoords)) * 1.5);
}
)glsl";

// --------------------------- WATER SHADERS ---------------------------

static const char particleVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

uniform mat4 projection;
uniform mat4 view;

uniform vec3 camPos;
uniform float radius;

out float pointSize;

void main() {
	vec3 relativePosition = mod(positionIn + camPos, radius) - vec3(radius / 2.0);
	vec3 worldPosition = relativePosition + camPos;

	vec3 toEdge = radius / 2.0 - abs(relativePosition);
    float minDistanceToEdge = min(min(toEdge.x, toEdge.y), toEdge.z);

    pointSize = max(0.0, 2.0 * minDistanceToEdge / radius);

	gl_Position = projection * view * vec4(worldPosition, 1.0);
	gl_PointSize = 40.0;
}
)glsl";

static const char particleFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in float pointSize;

uniform vec3 camDir;

float pointLineDistance(vec2 point, vec2 lineStart, vec2 lineEnd) {
	vec2 lineDir = lineEnd - lineStart;
	float lineLength = length(lineDir);
	vec2 normalizedDir = lineDir / lineLength;

	float projection = dot(point - lineStart, normalizedDir);
	projection = clamp(projection, 0.0, lineLength);

	vec2 closestPoint = lineStart + normalizedDir * projection;
	return length(point - closestPoint);
}

float gaussian(float x, float mu, float sigma) {
	return exp(-0.5 * pow((x - mu) / sigma, 2.0));
}

void main() {
	vec2 lineStart = normalize(camDir.xy) * (0.5 - pointSize);
	vec2 lineStop = -lineStart;
	float dist = pointLineDistance(gl_PointCoord - vec2(0.5, 0.5), lineStart, lineStop);
	
	if (length(camDir.xy) <= 0.001) {
		dist = length(gl_PointCoord - vec2(0.5, 0.5));
	}

	float focusEffect = 1.0 - gaussian(dist, pointSize / 2.0, 0.1);
	if (dist > pointSize / 2.0) discard;

	fragColor = vec4(vec3(focusEffect), 1.0);//vec4(1.0, 1.0, 1.0, 1.0 - focusEffect);
}
)glsl";

// --------------------------- DEBUG SHADERS ---------------------------

static const char debugVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

uniform mat4 projection;
uniform mat4 view;

uniform float time;

void main() {
	gl_PointSize = 5.0;
	gl_Position = projection * view * vec4(positionIn + vec3(0.0, sin(time * 6.28 + positionIn.x), 0.0), 1.0);
}
)glsl";

static const char debugFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

void main() {
	fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)glsl";

unsigned int textShader = 0;
unsigned int snoiseShader = 0;

unsigned int galaxyShader = 0;

unsigned int starShader = 0;
unsigned int bloomShader = 0;
unsigned int planetShader = 0;

unsigned int particleShader = 0;

unsigned int debugShader = 0;

void initShaders() {
	textShader = compileShader(textVertShaderSrc, NULL, textFragShaderSrc);
	snoiseShader = compileShader(postVertSrc, NULL, snoise);

	galaxyShader = compileShader(galaxyVertShaderSrc, NULL, galaxyFragShaderSrc);

	starShader = compileShader(sphereVertShaderSrc, sphereGemoShaderSrc, starFragShaderSrc);
	bloomShader = compileShader(bloomVertSrc, NULL, bloomFragSrc);
	planetShader = compileShader(sphereVertShaderSrc, sphereGemoShaderSrc, planetFragShaderSrc);
	
	particleShader = compileShader(particleVertSrc, NULL, particleFragSrc);

	debugShader = compileShader(debugVertSrc, NULL, debugFragSrc);
}