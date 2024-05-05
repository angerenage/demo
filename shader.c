#include "shader.h"

GLuint compileShader(const char *vShaderCode, const char *gShaderCode, const char *fShaderCode) {
	GLuint vertex, geometry, fragment;
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

	GLuint ID = glCreateProgram();
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

GLuint compileComputeShader(const char *shaderCode) {
	GLuint compute;
	int success;
	char infoLog[512];

	// vertex shader
	compute = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(compute, 1, &shaderCode, NULL);
	glCompileShader(compute);
	
	glGetShaderiv(compute, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(compute, 512, NULL, infoLog);
		printf("ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n%s\n", infoLog);
	}

	GLuint ID = glCreateProgram();
	glAttachShader(ID, compute);
	glLinkProgram(ID);
	
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(ID, 512, NULL, infoLog);
		printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
	}

	glDeleteShader(compute);

	return ID;
}

// --------------------------- GALAXY SHADERS ---------------------------

static const char galaxyVertSrc[] = R"glsl(#version 330 core
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

const char galaxyFragSrc[] = R"glsl(#version 330 core
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

static const char sphereVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

void main() {
	gl_Position = vec4(positionIn, 1.0);
}
)glsl";

static const char sphereGemoSrc[] = R"glsl(#version 330 core
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
static const char starFragSrc[] = R"glsl(#version 330 core
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

static const char planetFragSrc[] = R"glsl(#version 330 core
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

static const char textVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

out float id;

uniform float aspectRatio;

void main() {
	id = positionIn.z;

	vec2 adjustedPosition = positionIn.xy / 10.0f;
	adjustedPosition.x /= aspectRatio;

	gl_Position = vec4(adjustedPosition, -1.0, 1.0) - vec4(0.90f, 0.80f, 0.0, 0.0);
}
)glsl";

static const char textFragSrc[] = R"glsl(#version 330 core
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

out vec2 fragPos;

void main() {
	gl_Position = vec4(positionIn, 1.0);
	fragPos = positionIn.xy;
}
)glsl";

// Algorithm by patriciogonzalezvivo (https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83)
static const char snoiseFragSrc[] = R"glsl(#version 330 core
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
uniform vec3 camDir;
uniform float time;
uniform float deltaTime;
uniform float radius;

out vec2 stretchFactor;
out float pointSize;

void main() {
	vec3 relativePosition = mod(positionIn - vec3(0.0, time / 40.0, 0.0) - camPos, radius) - vec3(radius / 2.0);
	vec3 worldPosition = relativePosition + camPos;

	vec3 toEdge = radius / 2.0 - abs(relativePosition);
	float minDistanceToEdge = min(min(toEdge.x, toEdge.y), toEdge.z);

	vec3 cameraDirection = normalize(projection * view * vec4(camDir, 0.0)).xyz;
	vec3 velocity = length(cameraDirection) > 0.0 ? cameraDirection : vec3(0.0, deltaTime * 10.0, 0.0);
	stretchFactor = (vec4(velocity, 0.0)).xy;
	pointSize = max(0.0, 2.0 * minDistanceToEdge / radius);

	gl_Position = projection * view * vec4(worldPosition, 1.0);
	gl_PointSize = 40.0;
}
)glsl";

static const char particleFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec2 stretchFactor;
in float pointSize;

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
	vec2 lineStart = stretchFactor * (0.5 - pointSize);
	vec2 lineStop = -lineStart;
	float dist = pointLineDistance(gl_PointCoord - vec2(0.5, 0.5), lineStart, lineStop);

	float focusEffect = 1.0 - gaussian(dist, pointSize / 2.0, 0.1);
	if (focusEffect <= pointSize / 2.0 || dist > pointSize / 2.0) discard;

	fragColor = vec4(1.0, 1.0, 1.0, focusEffect);
}
)glsl";

static const char initialSpectrumFragSrc[] = R"glsl(#version 330 core
layout(location = 0) out vec4 outColor0;
layout(location = 1) out vec4 outColor1;
layout(location = 2) out vec4 outColor2;
layout(location = 3) out vec4 outColor3;

in vec2 fragPos;

#define M_PI 3.1415926535897932384626433832795

struct SpectrumParameters {
	float scale;
	float angle;
	float spreadBlend;
	float swell;
	float alpha;
	float peakOmega;
	float gamma;
	float shortWavesFade;
};

uniform uint _Seed;
uniform uint _N;
uniform float _LengthScale0, _LengthScale1, _LengthScale2, _LengthScale3;
uniform float _LowCutoff, _HighCutoff;
uniform float _Gravity;
uniform float _Depth;
uniform SpectrumParameters _Spectrums[8];

float hash(uint n) {
	n = (n << 13U) ^ n;
	n = n * (n * n * 15731U + 0x789221U) + 0x13763125U;
	return float(n & uint(0x7fffffffU)) / float(0x7fffffff);
}

vec2 uniformToGaussian(float u1, float u2) {
	float R = sqrt(-2.0 * log(u1));
	float theta = 2.0 * M_PI * u2;

	return vec2(R * cos(theta), R * sin(theta));
}

float dispersion(float kMag) {
	return sqrt(_Gravity * kMag * tanh(min(kMag * _Depth, 20)));
}

float dispersionDerivative(float kMag) {
	float th = tanh(min(kMag * _Depth, 20.0));
	float ch = cosh(kMag * _Depth);
	return _Gravity * (_Depth * kMag / (ch * ch) + th) / dispersion(kMag) / 2.0;
}

float TMACorrection(float omega) {
	float omegaH = omega * sqrt(_Depth / _Gravity);
	if (omegaH <= 1.0f)
		return 0.5f * omegaH * omegaH;
	if (omegaH < 2.0f)
		return 1.0f - 0.5f * (2.0f - omegaH) * (2.0f - omegaH);

	return 1.0f;
}

float JONSWAP(float omega, SpectrumParameters spectrum) {
	float sigma = (omega <= spectrum.peakOmega) ? 0.07f : 0.09f;

	float r = exp(-(omega - spectrum.peakOmega) * (omega - spectrum.peakOmega) / 2.0f / sigma / sigma / spectrum.peakOmega / spectrum.peakOmega);
	
	float oneOverOmega = 1.0f / omega;
	float peakOmegaOverOmega = spectrum.peakOmega / omega;
	return spectrum.scale * TMACorrection(omega) * spectrum.alpha * _Gravity * _Gravity
		* oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega
		* exp(-1.25f * peakOmegaOverOmega * peakOmegaOverOmega * peakOmegaOverOmega * peakOmegaOverOmega)
		* pow(abs(spectrum.gamma), r);
}

float NormalizationFactor(float s) {
	float s2 = s * s;
	float s3 = s2 * s;
	float s4 = s3 * s;
	if (s < 5) return -0.000564f * s4 + 0.00776f * s3 - 0.044f * s2 + 0.192f * s + 0.163f;
	else return -4.80e-08f * s4 + 1.07e-05f * s3 - 9.53e-04f * s2 + 5.90e-02f * s + 3.93e-01f;
}

float Cosine2s(float theta, float s) {
	return NormalizationFactor(s) * pow(abs(cos(0.5f * theta)), 2.0f * s);
}

float SpreadPower(float omega, float peakOmega) {
	if (omega > peakOmega)
		return 9.77f * pow(abs(omega / peakOmega), -2.5f);
	else
		return 6.97f * pow(abs(omega / peakOmega), 5.0f);
}

float directionSpectrum(float theta, float omega, SpectrumParameters spectrum) {
	float s = SpreadPower(omega, spectrum.peakOmega) + 16 * tanh(min(omega / spectrum.peakOmega, 20)) * spectrum.swell * spectrum.swell;
	return mix(2.0f / 3.1415f * cos(theta) * cos(theta), Cosine2s(theta - spectrum.angle, s), spectrum.spreadBlend);
}

float shortWavesFade(float kLength, SpectrumParameters spectrum) {
	return exp(-spectrum.shortWavesFade * spectrum.shortWavesFade * kLength * kLength);
}

void main() {
	vec2 id = (fragPos.xy + vec2(1.0)) * _N;

	uint seed = uint(id.x + _N * id.y + _N);
	seed += _Seed;

	float lengthScales[4] = float[4](_LengthScale0, _LengthScale1, _LengthScale2, _LengthScale3);

	vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

	for (uint i = 0U; i < 4U; i++) {
		float deltaK = 2.0 * M_PI / lengthScales[i];
		vec2 K = (id - vec2(_N)) * deltaK;
		float kLength = length(K);

		seed += i + uint(hash(seed) * 10.0);
		
		if (_LowCutoff <= kLength && kLength <= _HighCutoff) {
			vec4 uniformRandSamples = vec4(hash(seed), hash(seed * 2U), hash(seed * 3U), hash(seed * 4U));
			vec2 gauss1 = uniformToGaussian(uniformRandSamples.x, uniformRandSamples.y);
			vec2 gauss2 = uniformToGaussian(uniformRandSamples.z, uniformRandSamples.w);

			float kAngle = atan(K.y, K.x);
			float omega = dispersion(kLength);

			float dOmegadk = dispersionDerivative(kLength);

			float spectrum = JONSWAP(omega, _Spectrums[i * 2U]) * directionSpectrum(kAngle, omega, _Spectrums[i * 2U]) * shortWavesFade(kLength, _Spectrums[i * 2U]);
			spectrum += JONSWAP(omega, _Spectrums[i * 2U + 1U]) * directionSpectrum(kAngle, omega, _Spectrums[i * 2U + 1U]) * shortWavesFade(kLength, _Spectrums[i * 2U + 1U]);

			color = vec4(vec2(gauss2.x, gauss1.y) * sqrt(2.0 * spectrum * abs(dOmegadk) / kLength * deltaK * deltaK), 0.0, 1.0);
		}

		if (i == 0U) outColor0 = color;
		else if (i == 1U) outColor1 = color;
		else if (i == 2U) outColor2 = color;
		else if (i == 3U) outColor3 = color;
	}
}
)glsl";

static const char spectrumUpdateFragSrc[] = R"glsl(#version 330 core
layout(location = 0) out vec4 outColor0;
layout(location = 1) out vec4 outColor1;
layout(location = 2) out vec4 outColor2;
layout(location = 3) out vec4 outColor3;
layout(location = 4) out vec4 outColor4;
layout(location = 5) out vec4 outColor5;
layout(location = 6) out vec4 outColor6;
layout(location = 7) out vec4 outColor7;

#define M_PI 3.1415926535897932384626433832795

in vec2 fragPos;

uniform uint _N;
uniform float _LengthScale0, _LengthScale1, _LengthScale2, _LengthScale3;
uniform float _RepeatTime, _FrameTime;
uniform float _Gravity;

uniform sampler2DArray initialSpectrum;

void writeTo(vec4 color, int i) {
	if (i == 0) outColor0 = color;
	else if (i == 1) outColor1 = color;
	else if (i == 2) outColor2 = color;
	else if (i == 3) outColor3 = color;
	else if (i == 4) outColor4 = color;
	else if (i == 5) outColor5 = color;
	else if (i == 6) outColor6 = color;
	else if (i == 7) outColor7 = color;
}

vec2 eulerFormula(float x) {
	return vec2(cos(x), sin(x));
}

vec2 complexMult(vec2 a, vec2 b) {
	return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
	vec2 pos = (fragPos + vec2(1.0)) / 2.0;
	vec2 id = pos * _N;

	float lengthScales[4] = float[]( _LengthScale0, _LengthScale1, _LengthScale2, _LengthScale3 );

	for (int i = 0; i < 4; ++i) {
		vec2 h0 = texture(initialSpectrum, vec3(pos, i)).rg;
		vec2 h0conj = texture(initialSpectrum, vec3(mod(1.0 - pos, 1.0), i)).rg * vec2(1.0, -1.0);

		float halfN = _N / 2.0f;
		vec2 K = (id - halfN) * 2.0f * M_PI / lengthScales[i];
		float kMag = length(K);
		float kMagRcp = 1.0 / kMag;

		if (kMag < 0.0001f) {
			kMagRcp = 1.0f;
		}

		float w_0 = 2.0f * M_PI / _RepeatTime;
		float dispersion = floor(sqrt(_Gravity * kMag) / w_0) * w_0 * _FrameTime;

		vec2 exponent = eulerFormula(dispersion);

		vec2 htilde = complexMult(h0, exponent) + complexMult(h0conj, vec2(exponent.x, -exponent.y));
		vec2 ih = vec2(-htilde.y, htilde.x);

		vec2 displacementX = ih * K.x * kMagRcp;
		vec2 displacementY = htilde;
		vec2 displacementZ = ih * K.y * kMagRcp;

		vec2 displacementX_dx = -htilde * K.x * K.x * kMagRcp;
		vec2 displacementY_dx = ih * K.x;
		vec2 displacementZ_dx = -htilde * K.x * K.y * kMagRcp;

		vec2 displacementY_dz = ih * K.y;
		vec2 displacementZ_dz = -htilde * K.y * K.y * kMagRcp;

		vec2 htildeDisplacementX = vec2(displacementX.x - displacementZ.y, displacementX.y + displacementZ.x);
		vec2 htildeDisplacementZ = vec2(displacementY.x - displacementZ_dx.y, displacementY.y + displacementZ_dx.x);

		vec2 htildeSlopeX = vec2(displacementY_dx.x - displacementY_dz.y, displacementY_dx.y + displacementY_dz.x);
		vec2 htildeSlopeZ = vec2(displacementX_dx.x - displacementZ_dz.y, displacementX_dx.y + displacementZ_dz.x);

		writeTo(vec4(htildeDisplacementX, htildeDisplacementZ), i * 2);
		writeTo(vec4(htildeSlopeX, htildeSlopeZ), i * 2 + 1);
	}
}
)glsl";

static const char waterVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

uniform mat4 view;
uniform mat4 projection;
uniform sampler2DArray _DisplacementTextures;

out vec3 fragPos;
out vec2 UV;
out float depth;

const float _DisplacementDepthAttenuation = 1.0;

void main() {
	UV = positionIn.xz / 50.0 + 0.5;

	vec3 displacement = vec3(0.0);
	for (int i = 0; i < 4; i++) {
		displacement += texture(_DisplacementTextures, vec3(UV, i)).xyz;
	}

	fragPos = positionIn + displacement;

	vec4 clipPos = projection * view * vec4(fragPos, 1.0);
	depth = clipPos.z / clipPos.w;

	displacement = mix(vec3(0.0), displacement, pow(clamp(depth, 0.0, 1.0), _DisplacementDepthAttenuation));
	gl_Position = projection * view * vec4(positionIn + displacement, 1.0);
}
)glsl";

static const char waterFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

#define M_PI 3.1415926535897932384626433832795

in vec3 fragPos;
in vec2 UV;
in float depth;

uniform sampler2DArray _DisplacementTextures;
uniform sampler2DArray _SlopeTextures;

uniform vec3 _WorldSpaceCameraPos;


const vec3 _SunDirection = vec3(-1.29, -1.0, -4.86);
const float _Roughness = 0.075, _FoamRoughnessModifier = 0.0;
const float _NormalStrength = 1.0;

const vec3 _SunIrradiance = vec3(1.0, 0.694, 0.32);
const vec3 _ScatterColor = vec3(0.016, 0.0736, 0.16), _BubbleColor = vec3(0.0, 0.02, 0.0159), _FoamColor = vec3(0.6, 0.5568, 0.492);
const float _HeightModifier = 1.0, _BubbleDensity = 1.0;

const float _WavePeakScatterStrength = 1.0, _ScatterStrength = 1.0, _ScatterShadowStrength = 0.5, _EnvironmentLightStrength = 0.5;


const float _FoamDepthAttenuation = 1.0;

float DotClamped(vec3 a, vec3 b) {
	return clamp(dot(a, b), 0.0, 1.0);
}

float SchlickFresnel(vec3 normal, vec3 viewDir) {
	return 0.02f + (1 - 0.02f) * (pow(1 - DotClamped(normal, viewDir), 5.0f));
}

float SmithMaskingBeckmann(vec3 H, vec3 S, float roughness) {
	float hdots = max(0.001f, DotClamped(H, S));
	float a = hdots / (roughness * sqrt(1 - hdots * hdots));
	float a2 = a * a;

	return a < 1.6f ? (1.0f - 1.259f * a + 0.396f * a2) / (3.535f * a + 2.181 * a2) : 0.0f;
}

float Beckmann(float ndoth, float roughness) {
	float exp_arg = (ndoth * ndoth - 1) / (roughness * roughness * ndoth * ndoth);

	return exp(exp_arg) / (M_PI * roughness * roughness * ndoth * ndoth * ndoth * ndoth);
}

void main() {
	vec3 lightDir = -normalize(_SunDirection);
	vec3 viewDir = normalize(_WorldSpaceCameraPos - fragPos);
	vec3 halfwayDir = normalize(lightDir + viewDir);
	float LdotH = DotClamped(lightDir, halfwayDir);
	float VdotH = DotClamped(viewDir, halfwayDir);

	vec4 displacementFoam = vec4(0.0);
	vec2 slopes = vec2(0.0);
	for (int i = 0; i < 4; i++) {
		slopes += texture(_SlopeTextures, vec3(UV, i)).rg;
		displacementFoam += texture(_DisplacementTextures, vec3(UV, i));
	}

	slopes *= _NormalStrength;
	float foam = mix(0.0, clamp(displacementFoam.a, 0.0, 1.0), pow(depth, _FoamDepthAttenuation));

	mat3 normalMatrix = mat3(1.0); // No transformation
	normalMatrix = inverse(transpose(normalMatrix));

	vec3 macroNormal = vec3(0, 1, 0);
	vec3 mesoNormal = normalize(vec3(-slopes.x, 1.0f, -slopes.y));
	mesoNormal = normalize(normalMatrix * normalize(mesoNormal));

	float NdotL = DotClamped(mesoNormal, lightDir);

	float a = _Roughness + foam * _FoamRoughnessModifier;
	float ndoth = max(0.0001f, dot(mesoNormal, halfwayDir));

	float viewMask = SmithMaskingBeckmann(halfwayDir, viewDir, a);
	float lightMask = SmithMaskingBeckmann(halfwayDir, lightDir, a);

	float G = 1.0 / (1 + viewMask + lightMask);

	float eta = 1.33f;
	float R = ((eta - 1) * (eta - 1)) / ((eta + 1) * (eta + 1));
	float thetaV = acos(viewDir.y);

	float numerator = pow(1 - dot(mesoNormal, viewDir), 5 * exp(-2.69 * a));
	float F = R + (1 - R) * numerator / (1.0f + 22.7f * pow(a, 1.5f));
	F = clamp(F, 0.0, 1.0);

	vec3 specular = _SunIrradiance * F * G * Beckmann(ndoth, a);
	specular /= 4.0f * max(0.001f, DotClamped(macroNormal, lightDir));
	specular *= DotClamped(mesoNormal, lightDir);

	vec3 envReflection = vec3(0.56, 0.8, 1.0);//texCUBE(_EnvironmentMap, reflect(-viewDir, mesoNormal)).rgb;
	envReflection *= _EnvironmentLightStrength;

	float H = max(0.0f, displacementFoam.y) * _HeightModifier;
	vec3 scatterColor = _ScatterColor;
	vec3 bubbleColor = _BubbleColor;
	float bubbleDensity = _BubbleDensity;

	float k1 = _WavePeakScatterStrength * H * pow(DotClamped(lightDir, -viewDir), 4.0f) * pow(0.5f - 0.5f * dot(lightDir, mesoNormal), 3.0f);
	float k2 = _ScatterStrength * pow(DotClamped(viewDir, mesoNormal), 2.0f);
	float k3 = _ScatterShadowStrength * NdotL;
	float k4 = bubbleDensity;

	vec3 scatter = (k1 + k2) * scatterColor * _SunIrradiance * 1.0 / (1 + lightMask);
	scatter += k3 * scatterColor * _SunIrradiance + k4 * bubbleColor * _SunIrradiance;

	vec3 result = (1 - F) * scatter + specular + F * envReflection;
	result = max(vec3(0.0f), result);
	result = mix(result, _FoamColor, clamp(foam, 0.0, 1.0));

	fragColor = vec4(result, 1.0f);
}
)glsl";

static const char assembleMapsCompSrc[] = R"glsl(#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, binding = 0) uniform image2DArray _SpectrumTextures;
layout(rgba16f, binding = 1) uniform image2DArray _DisplacementTextures;
layout(rg16f, binding = 2) uniform image2DArray _SlopeTextures;

uniform vec2 _Lambda;
uniform float _FoamDecayRate;
uniform float _FoamBias;
uniform float _FoamThreshold;
uniform float _FoamAdd;

vec4 Permute(vec4 data, vec3 id) {
	return data * (1.0f - 2.0f * mod(id.x + id.y, 2.0));
}

void main() {
	for (int i = 0; i < 4; i++) {
		vec4 htildeDisplacement = Permute(imageLoad(_SpectrumTextures, ivec3(gl_GlobalInvocationID.xy, i * 2)), gl_GlobalInvocationID);
		vec4 htildeSlope = Permute(imageLoad(_SpectrumTextures, ivec3(gl_GlobalInvocationID.xy, i * 2 + 1)), gl_GlobalInvocationID);

		vec2 dxdz = htildeDisplacement.rg;
		vec2 dydxz = htildeDisplacement.ba;
		vec2 dyxdyz = htildeSlope.rg;
		vec2 dxxdzz = htildeSlope.ba;
		
		float jacobian = (1.0f + _Lambda.x * dxxdzz.x) * (1.0f + _Lambda.y * dxxdzz.y) - _Lambda.x * _Lambda.y * dydxz.y * dydxz.y;

		vec3 displacement = vec3(_Lambda.x * dxdz.x, dydxz.x, _Lambda.y * dxdz.y);

		vec2 slopes = dyxdyz / (1.0 + abs(dxxdzz * _Lambda));

		float foam = imageLoad(_DisplacementTextures, ivec3(gl_GlobalInvocationID.xy, i)).a;
		foam *= exp(-_FoamDecayRate);
		foam = clamp(foam, 0.0, 1.0);

		float biasedJacobian = max(0.0f, -(jacobian - _FoamBias));

		if (biasedJacobian > _FoamThreshold)
			foam += _FoamAdd * biasedJacobian;


		imageStore(_DisplacementTextures, ivec3(gl_GlobalInvocationID.xy, i), vec4(displacement, foam / 5));
		imageStore(_SlopeTextures, ivec3(gl_GlobalInvocationID.xy, i), vec4(slopes, 0.0, 0.0));
	}
}
)glsl";

// --------------------------- FFT SHADERS ---------------------------

static const char horizontalFFTSrc[] = R"glsl(#version 430 core
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#define SIZE 1024
#define LOG_SIZE 10

layout(rgba16f, binding = 0) uniform image2DArray _FourierTarget;

shared vec4 fftGroupBuffer[2][SIZE];

vec2 ComplexMult(vec2 a, vec2 b) {
	return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void ButterflyValues(uint step, uint index, out uvec2 indices, out vec2 twiddle) {
	const float twoPi = 6.28318530718;
	uint b = SIZE >> (step + 1);
	uint w = b * (index / b);
	uint i = (w + index) % SIZE;
	twiddle.y = sin(-twoPi / SIZE * w);
	twiddle.x = cos(-twoPi / SIZE * w);

	twiddle.y = -twiddle.y;
	indices = uvec2(i, i + b);
}

vec4 FFT(uint threadIndex, vec4 inputValue) {
	fftGroupBuffer[0][threadIndex] = inputValue;
	barrier();
	bool flag = false;

	for (uint step = 0; step < LOG_SIZE; step++) {
		uvec2 inputsIndices;
		vec2 twiddle;
		ButterflyValues(step, threadIndex, inputsIndices, twiddle);

		vec4 v = fftGroupBuffer[int(flag)][inputsIndices.y];
		fftGroupBuffer[int(!flag)][threadIndex] = fftGroupBuffer[int(flag)][inputsIndices.x] + vec4(ComplexMult(twiddle, v.xy), ComplexMult(twiddle, v.zw));

		flag = !flag;
		barrier();
	}

	return fftGroupBuffer[int(flag)][threadIndex];
}

void main() {
	for (int i = 0; i < 8; i++) {
		vec4 data = imageLoad(_FourierTarget, ivec3(gl_GlobalInvocationID.xy, i));
		vec4 result = FFT(gl_GlobalInvocationID.x, data);
		imageStore(_FourierTarget, ivec3(gl_GlobalInvocationID.xy, i), result);
	}
}
)glsl";

static const char verticalFFTSrc[] = R"glsl(#version 430 core
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

#define SIZE 1024
#define LOG_SIZE 10

layout(rgba16f, binding = 0) uniform image2DArray _FourierTarget;

shared vec4 fftGroupBuffer[2][SIZE];

vec2 ComplexMult(vec2 a, vec2 b) {
	return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void ButterflyValues(uint step, uint index, out uvec2 indices, out vec2 twiddle) {
	const float twoPi = 6.28318530718;
	uint b = SIZE >> (step + 1);
	uint w = b * (index / b);
	uint i = (w + index) % SIZE;
	twiddle.y = sin(-twoPi / SIZE * w);
	twiddle.x = cos(-twoPi / SIZE * w);

	twiddle.y = -twiddle.y;
	indices = uvec2(i, i + b);
}

vec4 FFT(uint threadIndex, vec4 inputValue) {
	fftGroupBuffer[0][threadIndex] = inputValue;
	barrier();
	bool flag = false;

	for (uint step = 0; step < LOG_SIZE; step++) {
		uvec2 inputsIndices;
		vec2 twiddle;
		ButterflyValues(step, threadIndex, inputsIndices, twiddle);

		vec4 v = fftGroupBuffer[int(flag)][inputsIndices.y];
		fftGroupBuffer[int(!flag)][threadIndex] = fftGroupBuffer[int(flag)][inputsIndices.x] + vec4(ComplexMult(twiddle, v.xy), ComplexMult(twiddle, v.zw));

		flag = !flag;
		barrier();
	}

	return fftGroupBuffer[int(flag)][threadIndex];
}

void main() {
	for (int i = 0; i < 8; i++) {
		vec4 data = imageLoad(_FourierTarget, ivec3(gl_GlobalInvocationID.yx, i));
		vec4 result = FFT(gl_GlobalInvocationID.x, data);
		imageStore(_FourierTarget, ivec3(gl_GlobalInvocationID.yx, i), result);
	}
}
)glsl";

// --------------------------- ATMOSPHERE SHADERS ---------------------------

static const char atmospherePostFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec2 fragPos;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPos;

uniform sampler2DMS renderDepthTexture;
uniform sampler2DMS renderColorTexture;

mat4 invViewProj = mat4(1.0);

vec3 reconstructWorldPos(float depth, vec2 TexCoords) {
	vec4 clipSpaceLocation = vec4(TexCoords * 2.0 - 1.0, depth, 1.0);
	vec4 worldSpaceLocation = invViewProj * clipSpaceLocation;
	return worldSpaceLocation.xyz / worldSpaceLocation.w;
}

const vec3 _SunDirection = vec3(-1.29, -1.0, -4.86);
const vec3 _SunColor = vec3(3.0117648, 1.945098, 0.8784314);
const float _FogHeight = 218.0;
const float _FogAttenuation = 1.63;
const float _FogDensity = 0.01;
const float _FogOffset = 583.0;
const vec3 _FogColor = vec3(0.83, 0.85, 0.87);

const vec3 skyColor = vec3(0.56, 0.8, 1.0);
const float waterHeight = 150.0;
const float farPlane = 1000.0f;

void main() {
	invViewProj = inverse(projection * view);

	float depth = 0.0;
	for (int i = 0; i < 4; i++) {
		depth += texelFetch(renderDepthTexture, ivec2(gl_FragCoord.xy), i).x;
	}
	depth /= 4.0;

	vec3 col = vec3(0.0);
	for (int i = 0; i < 4; i++) {
		col += texelFetch(renderColorTexture, ivec2(gl_FragCoord.xy), i).xyz;
	}
	col /= 4.0;

	if (depth >= 1) col = skyColor;

	vec3 worldPos = reconstructWorldPos(depth, fragPos / 2 + 0.5);
	vec3 viewDir = normalize(cameraPos - worldPos);

	float height = min(_FogHeight, worldPos.y + waterHeight) / _FogHeight;
	height = pow(clamp(0.0, 1.0, height), 1.0f / _FogAttenuation);

	float viewDistance = depth * farPlane;
	
	float fogFactor = (_FogDensity / sqrt(log(2))) * max(0.0f, viewDistance - _FogOffset);
	fogFactor = exp2(-fogFactor * fogFactor);

	vec3 sunDir = normalize(_SunDirection);
	vec3 sun = _SunColor * pow(clamp(0.0, 1.0, dot(viewDir, sunDir)), 3500.0f);

	vec3 color = mix(_FogColor, col, clamp(0.0, 1.0, height + fogFactor));
	
	fragColor = vec4(color + max(vec3(0.0), sun), 1.0);
}
)glsl";

// --------------------------- UNDERWATER SHADERS ---------------------------

static const char underwaterPostFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec2 fragPos;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPos;

uniform sampler2DMS underwaterDepthTexture;
uniform sampler2DMS underwaterColorTexture;

mat4 invViewProj = mat4(1.0);

vec3 reconstructWorldPos(float depth, vec2 TexCoords) {
	vec4 clipSpaceLocation = vec4(TexCoords * 2.0 - 1.0, depth, 1.0);
	vec4 worldSpaceLocation = invViewProj * clipSpaceLocation;
	return worldSpaceLocation.xyz / worldSpaceLocation.w;
}

void main() {
	invViewProj = inverse(projection * view);

	float depth = 0.0;
	for (int i = 0; i < 4; i++) {
		depth += texelFetch(underwaterDepthTexture, ivec2(gl_FragCoord.xy), i).x;
	}
	depth /= 4.0;

	vec3 color = vec3(0.0);
	for (int i = 0; i < 4; i++) {
		color += texelFetch(underwaterColorTexture, ivec2(gl_FragCoord.xy), i).xyz;
	}
	color /= 4.0;

	float fogDensity = 0.9;
	float fogFactor = exp(-depth * fogDensity);
	fogFactor = clamp(fogFactor, 0.0, 1.0);
	vec3 fogColor = vec3(0.0, 0.1, 0.2);


	vec3 worldPos = reconstructWorldPos(depth, fragPos / 2 + 0.5);

	vec3 viewDirection = normalize(worldPos - cameraPos);
	vec3 upDirection = vec3(0.0, 1.0, 0.0);
	float angle = dot(viewDirection, upDirection);
	float surfaceVisibility = smoothstep(-0.5, 1.0, angle);
	vec3 surfaceColor = vec3(0.66, 0.78, 0.98);
	
	if (depth >= 1) {
		color = mix(color, surfaceColor, surfaceVisibility);
	}
	color = mix(fogColor, color, fogFactor);

	fragColor = vec4(color, 1.0);
}
)glsl";

// --------------------------- JELLYFISH SHADERS ---------------------------

//https://math.stackexchange.com/questions/2341764/asymmetric-periods-in-a-sine-curve
static const char jellyfishVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

out vec3 fragPos;

uniform mat4 projection;
uniform mat4 view;

uniform float time;

void main() {
	const float minRadius = 0.7;
	const float radiusAmpl = 0.4;

	float height = positionIn.y;

	float x = time + height;
	float size = (sin(x - 0.5 * sin(x)) + 1.0) / 2.0;
	vec3 pos = positionIn * (minRadius + radiusAmpl * size) * min(minRadius, exp(1.0 - (height / 1.5)));
	pos.y = height;

	fragPos = positionIn;
	gl_Position = projection * view * vec4(pos, 1.0);
}
)glsl";

static const char jellyfishFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

#define M_PI 3.1415926535897932384626433832795

in vec3 fragPos;

uniform vec3 cameraPos;

float circle(vec2 UV, vec2 pos) {
	return max(0.0, mix(1.0, 0.0, abs(length(UV + pos) * 2 - 1) * 2));
}

void main() {
	float baseAlpha = 0.5;

	float radiusMask = max(0.0, length(fragPos.xz / 2) - 0.35);

	const float shift = 0.75;
	float ringMask = circle(fragPos.xz, vec2(shift));
	ringMask += circle(fragPos.xz, vec2(-shift));
	ringMask += circle(fragPos.xz, vec2(-shift, shift));
	ringMask += circle(fragPos.xz, vec2(shift, -shift));
	ringMask *= radiusMask;

	vec3 ring = vec3(0.9, 0.12, 0.49) * ringMask;

	float heightMask = max(0.0, (1 - fragPos.y * 2.0) - baseAlpha);
	heightMask += max(0.0, fragPos.y / 4.0);

	vec3 viewDirection = normalize(fragPos - cameraPos);
	float fresnel = clamp(0.0, 1.0, dot(viewDirection, normalize(fragPos)) * 2 + 0.5) / 4;

	fragColor = vec4(vec3(0.89, 0.69, 0.7) + ring, baseAlpha + heightMask + fresnel + ringMask);//vec4(vec3(heightMask), 1.0);
}
)glsl";

// --------------------------- DEBUG SHADERS ---------------------------

static const char debugVertSrc[] = R"glsl(#version 330 core
layout(location = 0) in vec3 positionIn;

out vec3 fragPos;

uniform mat4 projection;
uniform mat4 view;

void main() {
	fragPos = positionIn;
	gl_PointSize = 5.0;
	gl_Position = projection * view * vec4(positionIn, 1.0);
}
)glsl";

static const char debugFragSrc[] = R"glsl(#version 330 core
out vec4 fragColor;

in vec3 fragPos;

void main() {
	fragColor = vec4(1.0);
}
)glsl";

GLuint textShader = 0;
GLuint snoiseShader = 0;

GLuint galaxyShader = 0;

GLuint starShader = 0;
GLuint bloomShader = 0;
GLuint planetShader = 0;

GLuint jellyfishShader = 0;
GLuint particleShader = 0;
GLuint underwaterPostProcessShader = 0;
GLuint initialSpectrumShader = 0;
GLuint spectrumUpdateShader = 0;
GLuint waterShader = 0;
GLuint atmospherePostProcessShader = 0;

GLuint assembleMapsShader = 0;
GLuint horizontalFFTShader = 0;
GLuint verticalFFTShader = 0;

GLuint debugShader = 0;

void initShaders() {
	textShader = compileShader(textVertSrc, NULL, textFragSrc);
	snoiseShader = compileShader(postVertSrc, NULL, snoiseFragSrc);

	galaxyShader = compileShader(galaxyVertSrc, NULL, galaxyFragSrc);

	starShader = compileShader(sphereVertSrc, sphereGemoSrc, starFragSrc);
	bloomShader = compileShader(bloomVertSrc, NULL, bloomFragSrc);
	planetShader = compileShader(sphereVertSrc, sphereGemoSrc, planetFragSrc);
	
	jellyfishShader = compileShader(jellyfishVertSrc, NULL, jellyfishFragSrc);
	particleShader = compileShader(particleVertSrc, NULL, particleFragSrc);
	underwaterPostProcessShader = compileShader(postVertSrc, NULL, underwaterPostFragSrc);
	initialSpectrumShader = compileShader(postVertSrc, NULL, initialSpectrumFragSrc);
	spectrumUpdateShader = compileShader(postVertSrc, NULL, spectrumUpdateFragSrc);
	waterShader = compileShader(waterVertSrc, NULL, waterFragSrc);
	atmospherePostProcessShader = compileShader(postVertSrc, NULL, atmospherePostFragSrc);

	assembleMapsShader = compileComputeShader(assembleMapsCompSrc);
	horizontalFFTShader = compileComputeShader(horizontalFFTSrc);
	verticalFFTShader = compileComputeShader(verticalFFTSrc);

	debugShader = compileShader(debugVertSrc, NULL, debugFragSrc);
}