#include "cameraController.h"

vec3 bezier(float t, vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
	vec3 term1 = vec3_scale(p0, (1 - t) * (1 - t) * (1 - t));
	vec3 term2 = vec3_scale(p1, 3 * (1 - t) * (1 - t) * t);
	vec3 term3 = vec3_scale(p2, 3 * (1 - t) * t * t);
	vec3 term4 = vec3_scale(p3, t * t * t);

	return vec3_add(vec3_add(vec3_add(term1, term2), term3), term4);
}

vec3 bezierDerivative(float t, vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
	vec3 p1_p0 = vec3_subtract(p1, p0);
	vec3 p2_p1 = vec3_subtract(p2, p1);
	vec3 p3_p2 = vec3_subtract(p3, p2);

	vec3 term1 = vec3_scale(p1_p0, 3 * (1 - t) * (1 - t));
	vec3 term2 = vec3_scale(p2_p1, 6 * (1 - t) * t);
	vec3 term3 = vec3_scale(p3_p2, 3 * t * t);

	return vec3_add(vec3_add(term1, term2), term3);
}

float dynamicArcLength(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
    int numSegments = 100;
    vec3 prevPoint = bezier(0, p0, p1, p2, p3);
    float curLength = 0.0;
    float dt = t / numSegments;

    for (int i = 1; i <= numSegments; ++i) {
        float ti = i * dt;
        vec3 currentPoint = bezier(ti, p0, p1, p2, p3);
        curLength += length(vec3_subtract(prevPoint, currentPoint));
        prevPoint = currentPoint;
    }

    return curLength;
}

float findTForNormalizedS(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float s) {
    float totalLength = dynamicArcLength(p0, p1, p2, p3, 1.0);
    float targetLength = s * totalLength;
    float lower = 0.0;
    float upper = 1.0;
    float mid;

    while (upper - lower > 0.0001) {
        mid = (lower + upper) / 2.0;
        float midLength = dynamicArcLength(p0, p1, p2, p3, mid);

        if (midLength < targetLength) {
            lower = mid;
        } else {
            upper = mid;
        }
    }

    return mid;
}


vec3 bezierUniform(float s, vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
    float t = findTForNormalizedS(p0, p1, p2, p3, s);
    return bezier(t, p0, p1, p2, p3);
}

vec3 bezierDerivativeUniform(float s, vec3 p0, vec3 p1, vec3 p2, vec3 p3) {
    float t = findTForNormalizedS(p0, p1, p2, p3, s);
    return bezierDerivative(t, p0, p1, p2, p3);
}


Scene currentScene;

typedef struct s_curveDef {
	vec3 P1, P2, P3, P4;
} CurveDef;

typedef CurveDef (*CurveDefFun)(vec3, vec3);

typedef struct s_bezierParams {
	float length;
	Scene scene;
	CurveDefFun def;
} BezierParams;

static const float camDistance = 12.0f;
static const float cameraAngleX = -0.75f, cameraAngleY = 0.0f;
static vec3 currentStartPos;
static vec3 currentStartDir;

void defaultCameraTransforms(vec3 *pos, vec3 *dir, float distance, vec2 angles) {
	vec3 defaultPos;
	defaultPos.x = distance * sin(angles.x) * sin(angles.y);
	defaultPos.y = distance * cos(angles.x);
	defaultPos.z = distance * sin(angles.x) * cos(angles.y);

	vec3 defaultDir = normalize(defaultPos);

	*pos = defaultPos;
	*dir = defaultDir;
}

vec3 initializeCameraPosition() {
	defaultCameraTransforms(&currentStartPos, &currentStartDir, camDistance, (vec2){cameraAngleX, cameraAngleY});
	return currentStartPos;
}

// Curve definition functions

static CurveDef straightGalaxy(vec3 startPos, vec3 startDir) {
	vec3 endPos = vec3_add(vec3_scale(startPos, -0.2), startPos);
	vec3 endDir = normalize(vec3_subtract(startPos, endPos));
	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 0.1), startPos), vec3_add(vec3_scale(endDir, 0.1), endPos), endPos};
}

static const vec3 galaxyPosition = {0.0, 0.0, -4.0};
static CurveDef galaxyZoom(vec3 startPos, vec3 startDir) {
	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 3.0), startPos), vec3_add((vec3){0.0, 0.0, -2.0}, galaxyPosition), galaxyPosition};
}

static const vec3 sunPos = {0.0, -0.5, -2.0};
static CurveDef hideGalaxy(vec3 startPos, vec3 startDir) {
	return (CurveDef){startPos, vec3_add(startDir, startPos), (vec3){sunPos.x, startPos.y, sunPos.z}, sunPos};
}

static const vec3 planetPos = {0.0, 0., 50.0};
static CurveDef sunScene(vec3 startPos, vec3 startDir) {
	vec3 sunSceneStartPos = {0.0, 0.0, -100.0};

	vec3 endPos = {3.0, 0.0, 0.0};
	vec3 endDir = normalize(vec3_subtract(endPos, planetPos));
	return (CurveDef){sunSceneStartPos, vec3_add((vec3){0.0, 0.0, 5.0}, sunSceneStartPos), vec3_add(vec3_scale(endDir, 50.0), endPos), endPos};
}

static CurveDef sunToPlanet(vec3 startPos, vec3 startDir) {
	vec3 endPos = vec3_subtract(planetPos, (vec3){0.0, 0.7, 2.0});
	vec3 endDir = normalize(vec3_subtract(startPos, endPos));
	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 10), startPos), vec3_add(vec3_scale(endDir, 30), endPos), endPos};
}

static CurveDef planetZoom(vec3 startPos, vec3 startDir) {
	vec3 endPos = vec3_subtract(planetPos, normalize((vec3){0.0, 0.6, 1.0}));
	vec3 endDir = vec3_subtract(endPos, planetPos);

	endPos = vec3_add(vec3_scale(endDir, 0.1), endPos);

	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 0.5), startPos), vec3_add(vec3_scale(endDir, 0.3), endPos), endPos};
}

static const vec3 surfacePos = {0.0, 2.0, 0.0};
static const vec3 sunDir = (vec3){0.0, 0.0, -4.86};
static CurveDef waterScene(vec3 startPos, vec3 startDir) {
	vec3 waterSceneStartPos = vec3_add(vec3_scale(normalize(sunDir), 50), surfacePos);
	waterSceneStartPos.y = 90.0;

	return (CurveDef){waterSceneStartPos, vec3_add((vec3){0.0, -25.0, 0.0}, waterSceneStartPos), vec3_add(vec3_scale(sunDir, 5), surfacePos), surfacePos};
}

static CurveDef waterDive(vec3 startPos, vec3 startDir) {
	vec3 endPos = vec3_add(vec3_scale(sunDir, -4), startPos);
	endPos.y = 0.0;

	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 2), startPos), (vec3){endPos.x, startPos.y, endPos.z}, endPos};
}

static CurveDef underwaterScene(vec3 startPos, vec3 startDir) {
	startPos = (vec3){17.0, 20.0, 30.0};
	vec3 endPos = vec3_scale(normalize((vec3){5.0, 1.0, 9.0}), 10.0);

	return (CurveDef){startPos, vec3_add((vec3){0.0, -1.0, 0.0}, startPos), vec3_scale(endPos, 3), endPos};
}

static const vec3 jellyfishPos = {0.0, -2.0, 0.0};
static CurveDef jellyfishZoom(vec3 startPos, vec3 startDir) {
	vec3 endDir = normalize(vec3_subtract(jellyfishPos, startPos));
	vec3 endPos = vec3_scale(endDir, -2.0);

	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 2.0), startPos), vec3_add(vec3_scale(endDir, 0.5), endPos), endPos};
}

static CurveDef cellScene(vec3 startPos, vec3 startDir) {
	return (CurveDef){(vec3){0.0, 0.0, 50.0}, (vec3){0.0, 0.0, 45.0}, (vec3){0.0, 0.0, 5.0}, (vec3){0.0, 0.0, 0.0}};
}

static const vec3 atomPos = {-1.0, 0.0, 0.0};
static CurveDef DNAScene(vec3 startPos, vec3 startDir) {
	vec3 endPos = vec3_add(atomPos, (vec3){0.0, 0.0, 0.6});
	vec3 endDir = normalize(vec3_subtract((vec3){0.0, 0.0, 50.0}, endPos));
	return (CurveDef){(vec3){0.0, 0.0, 50.0}, (vec3){0.0, 0.0, 25.0}, vec3_add(vec3_scale(endDir, 0.3), endPos), endPos};
}

static CurveDef AtomScene(vec3 startPos, vec3 startDir) {
	vec3 endPos = vec3_add(atomPos, (vec3){0.0, 0.0, 0.2});
	vec3 endDir = normalize(vec3_subtract(startPos, endPos));
	return (CurveDef){startPos, vec3_add(vec3_scale(startDir, 0.2), startPos), vec3_add(vec3_scale(endDir, 0.2), endPos), endPos};
}

// Curve computations

static const BezierParams curves[] = {
	{4.0, GALAXY_SCENE, straightGalaxy},
	{5.0, GALAXY_SCENE, galaxyZoom},
	{5.0, GALAXY_SCENE, hideGalaxy},
	{6.0, SUN_SCENE, sunScene},
	{5.0, PLANET_SCENE, sunToPlanet},
	{1.0, PLANET_SCENE, planetZoom},
	{5.0, WATER_SCENE, waterScene},
	{4.0, WATER_SCENE, waterDive},
	{4.0, UNDERWATER_SCENE, underwaterScene},
	{3.0, UNDERWATER_SCENE, jellyfishZoom},
	{7.0, CELL_SCENE, cellScene},
	{5.0, MOLECULE_SCENE, DNAScene},
	{7.0, MOLECULE_SCENE, AtomScene},
	{0.0, CREDIT_SCENE, NULL},
};
static const int steps = sizeof(curves) / sizeof(BezierParams);

static int curveIndex = 0;
static float lastTime = 0.0f;

mat4 getCameraMatrix(vec3 *camPos, float time) {
	BezierParams curve = curves[curveIndex];
	CurveDef c;
	if (curve.def != NULL) c = curve.def(currentStartPos, currentStartDir);
	
	if (lastTime + curve.length <= time && curveIndex < steps - 1) {
		curveIndex++;
		lastTime += curve.length;
		if (curve.def != NULL) {
			currentStartPos = c.P4;
			currentStartDir = normalize(vec3_subtract(c.P4, c.P3));
		}
		
		curve = curves[curveIndex];
		if (curve.def != NULL) c = curve.def(currentStartPos, currentStartDir);
	}
	
	currentScene = curve.scene;

	vec3 pos = currentStartPos, dir = currentStartDir;
	if (curve.def != NULL) {
		float t = (time - lastTime) / curve.length;
		pos = bezierUniform(t, c.P1, c.P2, c.P3, c.P4);
		dir = bezierDerivativeUniform(t, c.P1, c.P2, c.P3, c.P4);
	}
	
	*camPos = pos;
	return viewMatrix(pos, vec3_add(dir, pos), (vec3){0.0, 1.0, 0.0});
}

float getTime(int step) {
	if (step < steps) {
		float time = 0;
		for (int i = 0; i < step; i++) {
			time += curves[i].length;
		}
		return time;
	}
	return -1;
}