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

	// compute shader
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

// --------------------------- TEXT SHADERS ---------------------------

static const char textVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"out float id;"
"uniform mat4 model;"
"uniform float aspectRatio;"
"void main()"
"{"
	"id=pA.z;"
	"vec2 f=pA.xy;"
	"f.y*=aspectRatio;"
	"gl_Position=model*vec4(f.xy,-1,1);"
"}";

static const char textFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in float id;"
"uniform float time;"
"void main()"
"{"
	"float f=fract(sin(id*.1)*1e4)*.5+id/1e2;"
	"if(f>time*7.)"
		"discard;"
	"c=vec4(1.);"
"}";

// --------------------------- GALAXY SHADERS ---------------------------

static const char galaxyVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"layout(location=1) in float hA;"
"out float density;"
"out vec3 position;"
"flat out int star;"
"uniform mat4 projection,view;"
"uniform float screenWidth,r_max;"
"float s(vec2 f)"
"{"
	"float n=0.,a=1.,y=1.;"
	"for(int i=0;i<4;i++)"
		"n+=fract(sin(dot((f*y).xy,vec2(12.9898,78.233)))*43758.5453)*a,a*=.5,y*=2.;"
	"return n;"
"}"
"float s(vec2 f,float m,float p,float n,float x)"
"{"
	"float i=1./0.,a=f.x;"
	"if(f.x<=n)"
		"a+=2*acos(-1.);"
	"for(float y=a;y<=x;y+=2.*acos(-1.))"
	"{"
		"float e=m*exp(p*y);"
		"i=min(i,abs(f.y-e));"
	"}"
	"return i;"
"}"
"float f(vec3 f)"
"{"
	"return max(0.,abs(f.x)-.5)+max(0.,abs(f.y)-.05)+max(0.,abs(f.z)-.05);"
"}"
"void main()"
"{"
	"if(isnan(hA))"
		"density=-1.,gl_PointSize=60.*(screenWidth/8e2),gl_Position=projection*view*vec4(0.,0.,0.,1.);"
	"else"
	"{"
		"float p=pA.y*cos(pA.x),a=pA.z,y=pA.y*sin(pA.x);"
		"gl_Position=projection*view*vec4(vec3(p,a,y),1.);"
		"float n=log(r_max)/(4.*acos(-1.)),e=5.*acos(-1.),i=s(pA.xy,1,n,0.,e)*.5;"
		"i=min(i,s(pA.xy,1./exp(n*acos(-1.)),n,acos(-1.),e+acos(-1.))*.5);"
		"i=min(i,f(vec3(p,a,y)/2.));"
		"density=mix(0.,.5,i*2.);"
		"density*=max(hA,.3);"
		"position=vec3(p,a,y);"
		"star=0;"
		"if(pA.y<4.&&s(pA.xz)>1.5)"
			"star=1,gl_PointSize=2.;"
		"else"
		"{"
			"float m=density;"
			"if(pA.y>4.)"
				"m*=5.-pA.y;"
			"gl_PointSize=min(75.,mix(8.,75.,m*7./pA.y*1.5)*(screenWidth/8e2));"
		"}"
	"}"
"}";

const char galaxyFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in float density;"
"in vec3 position;"
"flat in int star;"
"float f(vec2 f)"
"{"
	"float r=0.,v=1.,y=1.;"
	"for(int i=0;i<4;i++)"
		"r+=fract(sin(dot((f*y).xy,vec2(12.9898,78.233)))*43758.5453)*v,v*=.5,y*=2.;"
	"return r;"
"}"
"void main()"
"{"
	"float v=length(gl_PointCoord-vec2(.5));"
	"if(v>.5)"
		"discard;"
	"if(density<0.)"
	"{"
		"float i=max(0.,1.-v*v*4.2+.05)*.9;"
		"i=min(i,1.);"
		"c=vec4(vec3(1.),i);"
	"}"
	"else if(star==1)"
		"c=vec4(1.);"
	"else"
	"{"
		"float i=length(position.xz),y=max(.01,mix(.3,.01,density*5./i*10.));"
		"if(i>4.)"
			"y*=5.-i;"
		"vec3 r=mix(vec3(.46,.54,.54),vec3(.7,.8,.85),density*10.);"
		"if(density<=.1)"
		"{"
			"r=mix(r,vec3(.35,.3,.3),(.1-density)*7.);"
			"if(f(position.xz)<.6)"
				"r=mix(r,vec3(.9,.2,.2),.3);"
		"}"
		"if(i<1.5)"
			"r=mix(vec3(.8,.7,.6),r,smoothstep(0.,2.,i));"
		"c=vec4(r,y);"
	"}"
"}";

// --------------------------- SPHERE SHADERS ---------------------------

static const char sphereVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"void main()"
"{"
	"gl_Position=vec4(pA,1.);"
"}";

static const char sphereGemoSrc[] = R"glsl(#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 100) out;

out vec3 fragPosition;
out vec3 fragNormal;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
uniform int subdivisions;
uniform float radius;

const int MAX_SUBDIVISIONS = 5;

vec3 calculateNormal(vec3 point) {
	return normalize(point);
}

void emitVertex(vec3 position) {
	fragPosition = position * radius;
	fragNormal = calculateNormal(position);
	gl_Position = projection * view * model * vec4(normalize(position) * radius, 1.);
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
static const char starFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragPosition,fragNormal;"
"uniform sampler2D noiseTexture;"
"void main()"
"{"
	"vec3 v=pow(abs(fragNormal),vec3(1.));"
	"v/=v.x+v.y+v.z;"
	"vec4 n=texture(noiseTexture,fragPosition.xy/2.+vec2(.5))*v.z+texture(noiseTexture,fragPosition.yz/2.+vec2(.5))*v.x+texture(noiseTexture,fragPosition.zx/2.+vec2(.5))*v.y;"
	"float f=abs(n.x*2.-1.);"
	"c=vec4(mix(mix(vec3(1.,.86,.73),vec3(1.,.45,0.),min(1.,f*10.+.5)),vec3(.82,.2,.01),f*2.),1.);"
"}";

// --------------------------- PLANET SHADERS ---------------------------

static const char planetFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragPosition,fragNormal;"
"uniform vec3 lightDir,camPos;"
"uniform float camDist;"
"uniform sampler2D noiseTexture;"
"const vec3 n=vec3(3.0117648,1.945098,.8784314),f=vec3(.4,.437,.443);"
"void main()"
"{"
	"vec3 v=pow(abs(fragNormal),vec3(1.));"
	"v/=v.x+v.y+v.z;"
	"vec4 b=texture(noiseTexture,fragPosition.xy*2.+vec2(.5))*v.z+texture(noiseTexture,fragPosition.yz*2.+vec2(.5))*v.x+texture(noiseTexture,fragPosition.zx*2.+vec2(.5))*v.y;"
	"vec3 u=f*(vec3(b)*.01+.5);"
	"u+=(normalize(n)+.3)*pow(max(dot(normalize(camPos-fragPosition-b.xyz),reflect(lightDir,fragNormal)),0.),15.)/5.;"
	"float l=min(camDist,.5);"
	"l*=1./.5;"
	"c=vec4(u*mix(1.,0.,l)+u*dot(fragNormal,lightDir),1.);"
"}";

// --------------------------- NOISE SHADERS ---------------------------

static const char postVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"out vec2 fragPos;"
"void main()"
"{"
	"gl_Position=vec4(pA,1.);"
	"fragPos=pA.xy;"
"}";

// Algorithm by patriciogonzalezvivo (https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83)
static const char snoiseFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"uniform float time;"
"uniform vec2 resolution;"
"vec3 v(vec3 v)"
"{"
	"return mod((v*34.+1.)*v,289.);"
"}"
"float t(vec2 y)"
"{"
	"const vec4 d=vec4(.211324865405187,.366025403784439,-.577350269189626,.024390243902439);"
	"vec2 f=floor(y+dot(y,d.yy)),m=y-f+dot(f,d.xx),r;"
	"r=m.x>m.y?"
		"vec2(1.,0.):"
		"vec2(0.,1.);"
	"vec4 c=m.xyxy+d.xxzz;"
	"c.xy-=r;"
	"f=mod(f,289.);"
	"vec3 x=v(v(f.y+vec3(0.,r.y,1.))+f.x+vec3(0.,r.x,1.)),z=max(.5-vec3(dot(m,m),dot(c.xy,c.xy),dot(c.zw,c.zw)),0.);"
	"z*=z;"
	"z*=z;"
	"vec3 t=2.*fract(x*d.www)-1.,e=abs(t)-.5,a=t-floor(t+.5);"
	"z*=1.79284291400159-.85373472095314*(a*a+e*e);"
	"vec3 w;"
	"w.x=a.x*m.x+e.x*m.y;"
	"w.yz=a.yz*c.xz+e.yz*c.yw;"
	"return 130.*dot(z,w);"
"}"
"float m(vec2 v)"
"{"
	"float f=0.,m=1,r=0.,e=1.;"
	"for(int z=0;z<4;z++)"
		"f+=t(v*e)*m,r+=m,m*=.5,e*=2.;"
	"return f/r;"
"}"
"void main()"
"{"
	"vec2 v=gl_FragCoord.xy/resolution.xy,e=v*10.;"
	"float f=m(e);"
	"c=vec4(vec3((f+1.)/2.),1.);"
"}";

// --------------------------- BLOOM SHADERS ---------------------------

static const char bloomVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"out vec2 texCoords;"
"uniform mat4 model,projection,view;"
"uniform float bloomRadius;"
"void main()"
"{"
	"gl_Position=projection*view*model*vec4(pA.x*bloomRadius*normalize(vec3(view[0][0],view[1][0],view[2][0]))+pA.y*bloomRadius*normalize(vec3(view[0][1],view[1][1],view[2][1])),1.);"
	"texCoords=pA.xy;"
"}";

static const char bloomFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec2 texCoords;"
"uniform vec3 bloomColor;"
"void main()"
"{"
	"if(length(texCoords)>1.)"
		"discard;"
	"c=vec4(bloomColor,(1.-length(texCoords))*1.5);"
"}";

// --------------------------- WATER SHADERS ---------------------------

static const char initialSpectrumFragSrc[] = "#version 330 core\n"
"#define M_PI 3.1415926535897932384626433832795\n"
"layout(location=0) out vec4 o0;"
"layout(location=1) out vec4 o1;"
"layout(location=2) out vec4 o2;"
"layout(location=3) out vec4 o3;"
"in vec2 fragPos;"
"struct SpectrumParameters{float scale;float angle;float spreadBlend;float swell;float alpha;float peakOmega;float gamma;float shortWavesFade;};"
"uniform uint seed,_N;"
"uniform float lengthScale0,lengthScale1,lengthScale2,lengthScale3,lowCutoff,highCutoff,gravity,depth;"
"uniform SpectrumParameters spectrums[8];"
"float _(uint f)"
"{"
	"f=f<<13U^f;"
	"f=f*(f*f*15731U+7901729U)+326512933U;"
	"return float(f&uint(2147483647U))/float(2147483647);"
"}"
"vec2 _(float f,float U)"
"{"
	"float _=sqrt(-2.*log(f)),a=2.*M_PI*U;"
	"return vec2(_*cos(a),_*sin(a));"
"}"
"float s(float f)"
"{"
	"return sqrt(gravity*f*tanh(min(f*depth,20.)));"
"}"
"float t(float f)"
"{"
	"float _=cosh(f*depth);"
	"return gravity*(depth*f/(_*_)+tanh(min(f*depth,20.)))/s(f)/2.;"
"}"
"float f(float f)"
"{"
	"float _=f*sqrt(depth/gravity);"
	"return _<=1.?"
		".5*_*_:"
		"_<2.?"
			"1.-.5*(2.-_)*(2.-_):"
			"1.;"
"}"
"float f(float _,SpectrumParameters a)"
"{"
	"float o=_<=a.peakOmega?"
		".07:"
		".09,e=1./_,U=a.peakOmega/_;"
	"return a.scale*f(_)*a.alpha*gravity*gravity*e*e*e*e*e*exp(-1.25*U*U*U*U)*pow(abs(a.gamma),exp((-_+a.peakOmega)*(_-a.peakOmega)/2./o/o/a.peakOmega/a.peakOmega));"
"}"
"float o(float f)"
"{"
	"float _=f*f,e=_*f,U=e*f;"
	"return f<5.?"
		"-.000564*U+.00776*e-.044*_+.192*f+.163:"
		"-4.8e-8*U+1.07e-5*e-.000953*_+.059*f+.393;"
"}"
"float o(float f,float _)"
"{"
	"return o(_)*pow(abs(cos(.5*f)),2.*_);"
"}"
"float s(float f,float _)"
"{"
	"return f>_?"
		"9.77*pow(abs(f/_),-2.5):"
		"6.97*pow(abs(f/_),5.);"
"}"
"float _(float f,float _,SpectrumParameters a)"
"{"
	"float U=s(_,a.peakOmega)+16.*tanh(min(_/a.peakOmega,20.))*a.swell*a.swell;"
	"return mix(2./M_PI*cos(f)*cos(f),o(f-a.angle,U),a.spreadBlend);"
"}"
"float t(float f,SpectrumParameters _)"
"{"
	"return exp(-_.shortWavesFade*_.shortWavesFade*f*f);"
"}"
"void main()"
"{"
	"vec2 a=(fragPos.xy+vec2(1.))*_N;"
	"uint U=uint(a.x+_N*a.y+_N);"
	"U+=seed;"
	"float o[4]=float[4](lengthScale0,lengthScale1,lengthScale2,lengthScale3);"
	"vec4 v=vec4(0.,0.,0.,1.);"
	"for(uint e=0U;e<4U;e++)"
	"{"
		"float y=2.*M_PI/o[e];"
		"vec2 S=(a-vec2(_N))*y;"
		"float l=length(S);"
		"U+=e+uint(_(U)*10.);"
		"if(lowCutoff<=l&&l<=highCutoff)"
		"{"
			"vec4 m=vec4(_(U),_(U*2U),_(U*3U),_(U*4U));"
			"vec2 n=_(m.x,m.y),u=_(m.z,m.w);"
			"float i=atan(S.y,S.x),h=s(l),N=t(l),r=f(h,spectrums[e*2U])*_(i,h,spectrums[e*2U])*t(l,spectrums[e*2U]);"
			"r+=f(h,spectrums[e*2U+1U])*_(i,h,spectrums[e*2U+1U])*t(l,spectrums[e*2U+1U]);"
			"v=vec4(vec2(u.x,n.y)*sqrt(2.*r*abs(N)/l*y*y),0.,1.);"
		"}"
		"if(e==0U)"
			"o0=v;"
		"else if(e==1U)"
			"o1=v;"
		"else if(e==2U)"
			"o2=v;"
		"else if(e==3U)"
			"o3=v;"
	"}"
"}";

static const char spectrumUpdateFragSrc[] = "#version 330 core\n"
"#define M_PI 3.1415926535897932384626433832795\n"
"layout(location=0) out vec4 o0;"
"layout(location=1) out vec4 o1;"
"layout(location=2) out vec4 o2;"
"layout(location=3) out vec4 o3;"
"layout(location=4) out vec4 o4;"
"layout(location=5) out vec4 o5;"
"layout(location=6) out vec4 o6;"
"layout(location=7) out vec4 o7;"
"in vec2 fragPos;"
"uniform uint _N;"
"uniform float lengthScale0,lengthScale1,lengthScale2,lengthScale3,repeatTime,frameTime,gravity;"
"uniform sampler2DArray initialSpectrum;"
"void o(vec4 o,int v)"
"{"
	"if(v==0)"
		"o0=o;"
	"else if(v==1)"
		"o1=o;"
	"else if(v==2)"
		"o2=o;"
	"else if(v==3)"
		"o3=o;"
	"else if(v==4)"
		"o4=o;"
	"else if(v==5)"
		"o5=o;"
	"else if(v==6)"
		"o6=o;"
	"else if(v==7)"
		"o7=o;"
"}"
"vec2 o(float v)"
"{"
	"return vec2(cos(v),sin(v));"
"}"
"vec2 v(vec2 v,vec2 o)"
"{"
	"return vec2(v.x*o.x-v.y*o.y,v.x*o.y+v.y*o.x);"
"}"
"void main()"
"{"
	"vec2 y=(fragPos+vec2(1.))/2.,_=y*_N;"
	"float i[4]=float[](lengthScale0,lengthScale1,lengthScale2,lengthScale3);"
	"for(int x=0;x<4;++x)"
	"{"
		"vec2 l=texture(initialSpectrum,vec3(y,x)).xy,n=texture(initialSpectrum,vec3(mod(1.-y,1.),x)).xy*vec2(1.,-1.),f=(_-float(_N)/2.)*2.*M_PI/i[x];"
		"float M=length(f),e=1./M;"
		"if(M<1e-4)"
			"e=1.;"
		"float c=2.*M_PI/repeatTime;"
		"vec2 s=o(floor(sqrt(gravity*M)/c)*c*frameTime),m=v(l,s)+v(n,vec2(s.x,-s.y)),t=vec2(-m.y,m),A=t*f.x*e,C=m,D=t*f.y*e,F=-m*f.x*f.x*e,G=t*f.x,H=-m*f.x*f.y*e,E=t*f.y,B=-m*f.y*f.y*e;"
		"o(vec4(vec2(A.x-D.y,A.y+D.x),vec2(C.x-H.y,C.y+H.x)),x*2);"
		"o(vec4(vec2(G.x-E.y,G.y+E.x),vec2(F.x-B.y,F.y+B.x)),x*2+1);"
	"}"
"}";

static const char waterVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"uniform mat4 model,view,projection;"
"uniform sampler2DArray displacementTextures;"
"out vec3 fragPos;"
"out vec2 UV;"
"out float depth;"
"void main()"
"{"
	"UV=pA.xz/50.+.5;"
	"vec3 v=vec3(0.);"
	"for(int f=0;f<4;f++)"
		"v+=texture(displacementTextures,vec3(UV,f)).xyz;"
	"fragPos=vec3(model*vec4(pA,1.))+v;"
	"vec4 f=projection*view*model*vec4(fragPos,1.);"
	"depth=f.z/f.w;"
	"v=mix(vec3(0.),v,pow(clamp(depth,0.,1.),1.));"
	"gl_Position=projection*view*model*vec4(pA+v,1.);"
"}";

static const char waterFragSrc[] = "#version 330 core\n"
"#define M_PI 3.1415926535897932384626433832795\n"
"out vec4 c;"
"in vec3 fragPos;"
"in vec2 UV;"
"in float depth;"
"uniform sampler2DArray displacementTextures,slopeTextures;"
"uniform mat4 model;"
"uniform vec3 camPos;"
"const vec3 v=vec3(0.,-1.,-4.86),f=vec3(1.,.694,.32),m=vec3(.016,.0736,.16),b=vec3(0,.02,.0159),s=vec3(.6,.5568,.492);"
"float e(vec3 v,vec3 f)"
"{"
	"return clamp(dot(v,f),0.,1.);"
"}"
"float e(vec3 v,vec3 f,float _)"
"{"
	"float m=max(.001,e(v,f)),c=m/(_*sqrt(1.-m*m)),p=c*c;"
	"return c<1.6?"
		"(1.-1.259*c+.396*p)/(3.535*c+2.181*p):"
		"0.;"
"}"
"float p(float f,float v)"
"{"
	"return exp((f*f-1.)/(v*v*f*f))/(M_PI*v*v*f*f*f*f);"
"}"
"void main()"
"{"
	"vec3 _=-normalize(v),d=normalize(camPos-fragPos),y=normalize(_+d);"
	"vec4 U=vec4(0.);"
	"vec2 u=vec2(0.);"
	"for(int i=0;i<4;i++)"
		"u+=texture(slopeTextures,vec3(UV,i)).xy,U+=texture(displacementTextures,vec3(UV,i));"
	"float i=mix(0.,clamp(U.w,0.,1.),pow(depth,1.));"
	"mat3 M=mat3(model);"
	"M=inverse(transpose(M));"
	"vec3 V=normalize(vec3(-u.x,1.,-u.y));"
	"V=normalize(M*normalize(V));"
	"float I=e(V,_),h=.075,r=max(1e-4,dot(V,y)),A=e(y,d,h),l=e(y,_,h),n=(1.33-1.)*(1.33-1.)/((1.33+1.)*(1.33+1.)),C=pow(1.-dot(V,d),5.*exp(-2.69*h)),t=n+(1.-n)*C/(1.+22.7*pow(h,1.5));"
	"t=clamp(t,0.,1.);"
	"vec3 D=f*t*(1./(1.+A+l))*p(r,h);"
	"D/=4*max(.001,e(vec3(0.,1.,0.),_));"
	"D*=e(V,_);"
	"vec3 B=vec3(.56,.8,1);"
	"B*=.5;"
	"float E=max(0.,U.y);"
	"vec3 F=m;"
	"float G=E*pow(e(_,-d),4.)*pow(.5-.5*dot(_,V),3.),H=pow(e(d,V),2.);"
	"vec3 J=(G+H)*F*f/(1.+l);"
	"J+=.5*I*F*f+b*f;"
	"vec3 w=(1.-t)*J+D+t*B;"
	"w=max(vec3(0.),w);"
	"w=mix(w,s,clamp(i,0.,1.));"
	"c=vec4(w,1.);"
"}";

static const char assembleMapsCompSrc[] = R"glsl(#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, rgba16f) uniform readonly image2DArray spectrumTextures;
layout(binding = 1, rgba16f) uniform image2DArray displacementTextures;
layout(binding = 2, rg16f) uniform writeonly image2DArray slopeTextures;

uniform vec2 lambda;
uniform float foamDecayRate, foamBias, foamThreshold, foamAdd;

vec4 Permute(vec4 data, vec3 id) {
	return data * (1.0 - 2.0 * mod(id.x + id.y, 2.0));
}

void main() {
	uvec3 id = gl_GlobalInvocationID;

	for (int i = 0; i < 4; i++) {
		vec4 htildeDisplacement = Permute(imageLoad(spectrumTextures, ivec3(id.xy, i * 2)), vec3(id));
		vec4 htildeSlope = Permute(imageLoad(spectrumTextures, ivec3(id.xy, (i * 2) + 1)), vec3(id));
		
		vec2 dxdz = htildeDisplacement.xy;
		vec2 dydxz = htildeDisplacement.zw;
		vec2 dyxdyz = htildeSlope.xy;
		vec2 dxxdzz = htildeSlope.zw;
		
		float jacobian = ((1.0 + (lambda.x * dxxdzz.x)) * (1.0 + (lambda.y * dxxdzz.y))) - (((lambda.x * lambda.y) * dydxz.y) * dydxz.y);
		vec3 displacement = vec3(lambda.x * dxdz.x, dydxz.x, lambda.y * dxdz.y);
		
		vec2 slopes = dyxdyz / (vec2(1.0) + abs(dxxdzz * lambda));
		float covariance = slopes.x * slopes.y;
		
		float foam = imageLoad(displacementTextures, ivec3(id.xy, i)).w;
		foam *= exp(-foamDecayRate);
		foam = clamp(foam, 0.0, 1.0);
		
		float biasedJacobian = max(0.0, -(jacobian - foamBias));
		
		if (biasedJacobian > foamThreshold)
			foam += foamAdd * biasedJacobian;
		
		imageStore(displacementTextures, ivec3(id.xy, i), vec4(displacement, foam / 5.0));
		imageStore(slopeTextures, ivec3(id.xy, i), vec4(slopes.xy, 0.0, 0.0));
	}
}
)glsl";

// --------------------------- FFT SHADERS ---------------------------

static const char horizontalFFTSrc[] = R"glsl(#version 430 core
#define SIZE 1024
#define LOG_SIZE 10

layout(local_size_x = SIZE, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, rgba16f) uniform image2DArray _FourierTarget;

shared vec4 fftGroupBuffer[2][SIZE];

vec2 ComplexMult(vec2 a, vec2 b) {
	return vec2((a.x * b.x) - (a.y * b.y), (a.x * b.y) + (a.y * b.x));
}

void ButterflyValues(uint step, uint index, out uvec2 indices, inout vec2 twiddle) {
	float twoPi = 2 * acos(-1.);
	uint b = uint(SIZE) >> ((step + 1u) & 31u);
	uint w = b * (index / b);
	uint i = (w + index) % uint(SIZE);
	
	twiddle.y = sin(((-twoPi) / float(SIZE)) * float(w));
	twiddle.x = cos(((-twoPi) / float(SIZE)) * float(w));
	
	twiddle.y = -twiddle.y;
	indices = uvec2(i, i + b);
}

vec4 FFT(uint threadIndex, vec4 inputValue) {
	fftGroupBuffer[0][threadIndex] = inputValue;
	barrier();

	bool flag = false;
	uvec2 inputsIndices;
	vec2 twiddle;
	
	for (uint step = 0u; step < uint(LOG_SIZE); step++) {
		ButterflyValues(step, threadIndex, inputsIndices, twiddle);
		
		vec4 v = fftGroupBuffer[uint(flag)][inputsIndices.y];
		fftGroupBuffer[uint(!flag)][threadIndex] = fftGroupBuffer[uint(flag)][inputsIndices.x] + vec4(ComplexMult(twiddle, v.xy), ComplexMult(twiddle, v.zw));
		
		flag = !flag;
		barrier();
	}
	
	return fftGroupBuffer[uint(flag)][threadIndex];
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
#define SIZE 1024
#define LOG_SIZE 10

layout(local_size_x = SIZE, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, rgba16f) uniform image2DArray _FourierTarget;

shared vec4 fftGroupBuffer[2][SIZE];

vec2 ComplexMult(vec2 a, vec2 b) {
	return vec2((a.x * b.x) - (a.y * b.y), (a.x * b.y) + (a.y * b.x));
}

void ButterflyValues(uint step, uint index, out uvec2 indices, inout vec2 twiddle) {
	float twoPi = 2 * acos(-1.);
	uint b = uint(SIZE) >> ((step + 1u) & 31u);
	uint w = b * (index / b);
	uint i = (w + index) % uint(SIZE);
	
	twiddle.y = sin(((-twoPi) / float(SIZE)) * float(w));
	twiddle.x = cos(((-twoPi) / float(SIZE)) * float(w));
	
	twiddle.y = -twiddle.y;
	indices = uvec2(i, i + b);
}

vec4 FFT(uint threadIndex, vec4 inputValue) {
	fftGroupBuffer[0][threadIndex] = inputValue;
	barrier();

	bool flag = false;
	uvec2 inputsIndices;
	vec2 twiddle;
	
	for (uint step = 0u; step < uint(LOG_SIZE); step++) {
		ButterflyValues(step, threadIndex, inputsIndices, twiddle);
		
		vec4 v = fftGroupBuffer[uint(flag)][inputsIndices.y];
		fftGroupBuffer[uint(!flag)][threadIndex] = fftGroupBuffer[uint(flag)][inputsIndices.x] + vec4(ComplexMult(twiddle, v.xy), ComplexMult(twiddle, v.zw));
		
		flag = !flag;
		barrier();
	}
	
	return fftGroupBuffer[uint(flag)][threadIndex];
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

static const char atmospherePostFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec2 fragPos;"
"uniform mat4 view,projection;"
"uniform vec3 cameraPos;"
"uniform sampler2DMS renderDepthTexture,renderColorTexture;"
"mat4 v=mat4(1.);"
"vec3 m(float f,vec2 c)"
"{"
	"vec4 m=v*vec4(c*2.-1.,f,1);"
	"return m.xyz/m.w;"
"}"
"const vec3 b=vec3(0.,-1.,-4.86),i=vec3(3.0117648,1.945098,.8784314);"
"const vec3 f=vec3(.83,.85,.87),r=vec3(.56,.8,1.);"
"void main()"
"{"
	"v=inverse(projection*view);"
	"float e=0.;"
	"for(int u=0;u<4;u++)"
		"e+=texelFetch(renderDepthTexture,ivec2(gl_FragCoord.xy),u).x;"
	"e/=4.;"
	"vec3 u=vec3(0);"
	"for(int y=0;y<4;y++)"
		"u+=texelFetch(renderColorTexture,ivec2(gl_FragCoord.xy),y).xyz;"
	"u/=4.;"
	"if(e>=1.)"
		"u=r;"
	"vec3 y=m(e,fragPos/2.+.5);"
	"float g=min(218.,y.y+150.)/218.;"
	"g=pow(clamp(g,0.,1.),1./1.63);"
	"float s=e*1e3f,n=.01/sqrt(log(2.))*max(0.,s-583.);"
	"n=exp2(-n*n);"
	"vec3 d=mix(f,u,clamp(g+n,0.,1.));"
	"c=vec4(d+max(vec3(0),i*pow(clamp(dot(normalize(cameraPos-y),normalize(b)),0.,1.),3500.)),1);"
"}";

// --------------------------- UNDERWATER SHADERS ---------------------------

static const char particleVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"uniform mat4 projection,view;"
"uniform vec3 camPos,camDir;"
"uniform float time,deltaTime,radius;"
"out vec2 stretchFactor;"
"out float pointSize;"
"void main()"
"{"
	"vec3 p=mod(pA-vec3(0.,time/40.,0.)-camPos*.1,radius)-vec3(radius/2.),r=radius/2.-abs(p),c=normalize(projection*view*vec4(camDir,0.)).xyz;"
	"stretchFactor=vec4(length(c)>0.?"
		"c:"
		"vec3(0.,deltaTime*10.,0.),0.).xy;"
	"pointSize=max(0.,2.*min(min(r.x,r.y),r.z)/radius);"
	"gl_Position=projection*view*vec4(p+camPos,1.);"
	"gl_PointSize=40.;"
"}";

static const char particleFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec2 stretchFactor;"
"in float pointSize;"
"float p(vec2 v,vec2 p,vec2 c)"
"{"
	"vec2 e=c-p;"
	"float r=length(e);"
	"vec2 f=e/r;"
	"float i=dot(v-p,f);"
	"i=clamp(i,0.,r);"
	"vec2 l=p+f*i;"
	"return length(v-l);"
"}"
"void main()"
"{"
	"vec2 v=stretchFactor*(.5-pointSize);"
	"float f=p(gl_PointCoord-vec2(.5),v,-v),e=1.-exp(-.5*pow((f-pointSize/2.)/.1,2.));"
	"if(e<=pointSize/2.||f>pointSize/2.)"
		"discard;"
	"c=vec4(1.,1.,1.,e);"
"}";

static const char underwaterPostFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec2 fragPos;"
"uniform mat4 view,projection;"
"uniform vec3 camPos;"
"uniform sampler2DMS underwaterDepthTexture,underwaterColorTexture;"
"mat4 v=mat4(1.);"
"vec3 m(float e,vec2 f)"
"{"
	"vec4 m=v*vec4(f*2.-1.,e,1.);"
	"return m.xyz/m.w;"
"}"
"void main()"
"{"
	"v=inverse(projection*view);"
	"float f=0.;"
	"for(int u=0;u<4;u++)"
		"f+=texelFetch(underwaterDepthTexture,ivec2(gl_FragCoord.xy),u).x;"
	"f/=4.;"
	"vec3 u=vec3(0.);"
	"for(int i=0;i<4;i++)"
		"u+=texelFetch(underwaterColorTexture,ivec2(gl_FragCoord.xy),i).xyz;"
	"u/=4;"
	"float i=exp(-f);"
	"i=clamp(i,0.,1.);"
	"vec3 b=m(f,fragPos/2.+.5);"
	"float e=smoothstep(-.5,1,dot(normalize(b-camPos),vec3(0.,1.,0.)));"
	"vec3 d=vec3(.66,.78,.98);"
	"if(f>=1.)"
		"u=mix(u,d,e);"
	"u=mix(vec3(0.,.1,.2),u,i);"
	"c=vec4(u,1.);"
"}";

// --------------------------- JELLYFISH SHADERS ---------------------------

// https://math.stackexchange.com/questions/2341764/asymmetric-periods-in-a-sine-curve
static const char jellyfishVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"out vec3 fragPos;"
"uniform mat4 projection,view;"
"uniform float time;"
"void main()"
"{"
	"float p=pA.y,n=time+p;"
	"vec3 v=pA*(.7+.4*((sin(n-.5*sin(n))+1.)/2.))*min(.7,exp(1.-p/1.5));"
	"v.y=p;"
	"fragPos=pA;"
	"gl_Position=projection*view*vec4(v,1.);"
"}";

static const char jellyfishFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragPos;"
"uniform vec3 camPos;"
"uniform float camDist;"
"float m(vec2 f,vec2 c)"
"{"
	"return max(0.,mix(1.,0.,abs(length(f+c)*2.-1.)*2.));"
"}"
"void main()"
"{"
	"float v=max(0.,length(fragPos.xz/2.)-.35),f=m(fragPos.xz,vec2(.75));"
	"f+=m(fragPos.xz,vec2(-.75));"
	"f+=m(fragPos.xz,vec2(-.75,.75));"
	"f+=m(fragPos.xz,vec2(.75,-.75));"
	"f*=v;"
	"vec3 b=vec3(.9,.12,.49)*f;"
	"float u=max(0.,1.-fragPos.y*2.-.5);"
	"u+=max(0.,fragPos.y/4.);"
	"u*=v;"
	"float z=.5+u+clamp(0.,1.,dot(normalize(fragPos-camPos),normalize(fragPos))*2.+.5)/4.+f;"
	"z=mix(1,z,clamp(0.,1.,camDist));"
	"c=vec4(vec3(.89,.69,.7)+b,z);"
"}";

// --------------------------- CELL SHADERS ---------------------------

static const char cellVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"out vec3 fragPos;"
"uniform mat4 model,projection,view;"
"void main()"
"{"
	"fragPos=pA;"
	"gl_PointSize=5.;"
	"gl_Position=projection*view*model*vec4(pA,1.);"
"}";

// The Book of Shaders by Patricio Gonzalez Vivo & Jen Lowe (https://thebookofshaders.com/12/)
static const char cellFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragPos;"
"uniform float scale,camDist;"
"const vec3 f=vec3(.329,.318,.384);"
"vec2 t(vec2 f)"
"{"
	"return fract(sin(vec2(dot(f,vec2(127.1,311.7)),dot(f,vec2(269.5,183.3))))*43758.5453);"
"}"
"void main()"
"{"
	"vec2 v=fragPos.xy*3./scale,b=fract(v);"
	"float i=1.;"
	"for(int s=-1;s<=1;s++)"
		"for(int d=-1;d<=1;d++)"
		"{"
			"vec2 m=vec2(float(d),float(s)),l=t(floor(v)+m);"
			"i=min(i,length(m+l-b));"
		"}"
	"i+=pow(1.-i,4.)*.8;"
	"i*=min(camDist,1);"
	"vec3 s=mix(i*f,f,clamp((camDist-35.)/10.,0.,1.));"
	"c=vec4(s,1.);"
"}";

// --------------------------- DNA SHADERS ---------------------------

static const char dnaVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"layout(location=3) in vec3 ipA;"
"out vec3 fragNormal,color;"
"uniform mat4 model,view,projection;"
"uniform float scale;"
"const vec3 v[]=vec3[](vec3(.569,.741,1.),vec3(.569,1.,.569),vec3(1.,.914,.569),vec3(1.,.741,.569));"
"int i(int v)"
"{"
	"int i=v;"
	"i^=i>>16;"
	"i*=2246822507;"
	"i^=i>>13;"
	"i*=3266489909;"
	"i^=i>>16;"
	"return(i+v)%4;"
"}"
"void main()"
"{"
	"int n=gl_InstanceID%2,l=i(gl_InstanceID-n);"
	"l=(l+n*2)%4;"
	"color=v[l];"
	"fragNormal=normalize(pA);"
	"gl_Position=projection*view*vec4(vec3(model*vec4(pA*-.3+ipA,1.))*scale,1.);"
"}";

static const char dnaFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragNormal,color;"
"uniform float camDist,scale;"
"void main()"
"{"
	"c=vec4(color*mix(0.,1.,(scale-.3)*2.)*((dot(vec3(0.,-1.,0.),fragNormal)+1.)/2.),camDist*camDist/5.);"
"}";

// --------------------------- ATOM SHADERS ---------------------------

static const char atomVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"layout(location=3) in vec3 ipA;"
"out vec3 fragPos,fragNormal,color;"
"uniform mat4 model,view,projection;"
"int e(int A)"
"{"
	"int i=A;"
	"i^=i>>16;"
	"i*=2246822507;"
	"i^=i>>13;"
	"i*=3266489909;"
	"i^=i>>16;"
	"return(i+A)%2;"
"}"
"void main()"
"{"
	"color=e(gl_InstanceID)==0?"
		"normalize(vec3(1.,.2,.1)):"
		"normalize(vec3(.1,.2,1.));"
	"fragNormal=normalize(pA);"
	"fragPos=pA*-.55+ipA;"
	"gl_Position=projection*view*model*vec4(fragPos,1.);"
"}";

static const char atomFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragPos,fragNormal,color;"
"void main()"
"{"
	"c=vec4(vec3(color*min(1.,length(fragPos)*0.6)*((dot(vec3(0.,-1.,0.),fragNormal)+1.)/2.)),1.);"
"}";

static const char electronVertSrc[] = "#version 330 core\n"
"layout(location=0) in vec3 pA;"
"layout(location=3) in vec3 ipA;"
"uniform mat4 model,view,projection;"
"uniform float time;"
"out vec3 fragNormal;"
"vec3 t(vec3 i)"
"{"
	"return normalize(i+fract(sin(dot(i,vec3(12.9898,78.233,54.53)))*43758.5453));"
"}"
"void main()"
"{"
	"fragNormal=normalize(pA);"
	"vec3 i=normalize(ipA+t(ipA))*30.,n=cross(normalize(ipA-t(ipA)),normalize(i));"
	"float v=time;"
	"gl_Position=projection*view*model*vec4(cos(v)*i+sin(v)*cross(n,i)+(1.-cos(v))*dot(n,i)*n-pA*.1,1.);"
"}";

static const char electronFragSrc[] = "#version 330 core\n"
"out vec4 c;"
"in vec3 fragNormal;"
"void main()"
"{"
	"c=vec4(0.,(dot(vec3(0.,-1.,0.),fragNormal)+1.)/2.,0.,1.);"
"}";

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

GLuint cellShader = 0;
GLuint dnaShader = 0;
GLuint atomShader = 0;
GLuint electronShader = 0;

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

	cellShader = compileShader(cellVertSrc, NULL, cellFragSrc);
	dnaShader = compileShader(dnaVertSrc, NULL, dnaFragSrc);
	atomShader = compileShader(atomVertSrc, NULL, atomFragSrc);
	electronShader = compileShader(electronVertSrc, NULL, electronFragSrc);
}