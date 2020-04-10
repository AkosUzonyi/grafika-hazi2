//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

GPUProgram gpuProgram; // vertex and fragment shaders

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1)) / 2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330
	precision highp float;

	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;

	void main() { fragmentColor = texture(textureUnit, texcoord); }
)";

class FullScreenTextQuad {
	unsigned int vao = 0, textureId = 0;
public:
	void genTexture() {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1,-1,1,-1,1,1,-1,1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}


	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTextQuad fullScreenTextQuad;

mat4 identityMat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);

mat4 transpose(const mat4& M) {
	mat4 Mt;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			Mt[i][j] = M[j][i];
	return Mt;
}

vec3 toDescartes(vec4 v) {
	return vec3(v.x / v.w, v.z / v.w, v.z / v.w);
}

vec4 toHomogeneousPoint(vec3 v) {
	return vec4(v.x, v.y, v.z, 1);
}

vec4 toHomogeneousVector(vec3 v) {
	return vec4(v.x, v.y, v.z, 1);
}

struct Light {
	vec3 pos;
	vec3 color;

	Light(vec3 pos, vec3 color) : pos(pos), color(color) {}
};

struct Hit {
	vec3 point, normal;
	float t;
};

float firstIntersect(float t1, float t2) {
	float min = t1 < t2 ? t1 : t2;
	if (min > 0)
		return min;

	float max = t1 > t2 ? t1 : t2;
	if (max > 0)
		return max;

	return -1;
}

struct Ray {
	vec4 origin, dir;

	Ray(vec3 o, vec3 d) : origin(toHomogeneousPoint(o)), dir(toHomogeneousVector(normalize(d))) {}
};

class Material {
public:
	virtual vec3 reflectLight(vec3 ambLight, vec3 inLight, vec3 normal, vec3 lightDir, vec3 eyeDir) const = 0;
};

class DiffuseMaterial : public Material {
	vec3 ka, ks, kd;
	float shine;
public:
	DiffuseMaterial(vec3 ka, vec3 ks, vec3 kd, float shine) : ka(ka), ks(ks), kd(kd), shine(shine) {}

	vec3 reflectLight(vec3 ambLight, vec3 inLight, vec3 normal, vec3 lightDir, vec3 eyeDir) const {
		return ka * ambLight + kd * dot(normal, lightDir) + ks * pow(dot(normalize((lightDir + eyeDir) / 2), normal), shine);
	}
};

class Camera {
	vec3 pos, look, up, right;
public:
	Camera(vec3 pos, vec3 look, vec3 up, vec3 right) : pos(pos), look(look), up(up), right(right) {}

	Ray getRay(float x, float y) const {
		return Ray(pos, look + right * x + up * y);
	}
};

class Shape {
	const Material& material;
public:
	Shape(const Material& material) : material(material) {}

	virtual Hit intersect(Ray ray) const = 0;

	const Material& getMaterial() const {
		return material;
	}
};

class QuadraticShape : public Shape {
	mat4 Q;

public:
	QuadraticShape(const Material& material, const mat4& Q) : Shape(material), Q(Q) {}

	void transform(mat4 M) {
		Q = M * Q * transpose(M);
	}

	Hit intersect(Ray ray) const {
		Hit hit;
		float a = dot(ray.dir * Q, ray.dir);
		float b = dot(ray.dir * Q, ray.origin) + dot(ray.origin * Q, ray.dir);
		float c = dot(ray.origin * Q, ray.origin);
		float disc = b * b - 4 * a * c;
		if (disc < 0) {
			hit.t = -1;
			return hit;
		}
		//printf("disc: %f\n", disc);
		float discSqrt = sqrt(disc);
		float t1 = (-b + discSqrt) / (2 * a);
		float t2 = (-b - discSqrt) / (2 * a);
		hit.t = firstIntersect(t1, t2);
		hit.point = toDescartes(ray.origin + ray.dir * hit.t);
		hit.normal = toDescartes(toHomogeneousPoint(hit.point) * Q * 2);
		return hit;
	}
};

class World {
	std::vector<Light> lights;
	std::vector<Shape*> shapes;
	Camera cam;
	DiffuseMaterial redDiffuseMaterial;
	vec3 ambLight;

public:
	World() : cam(vec3(0, 0, -2), vec3(0, 0, -1), vec3(0, 1, 0), vec3(1, 0, 0)), redDiffuseMaterial(vec3(1, 1, 1), vec3(1, 1, 1), vec3(1, 0 ,0), 2), ambLight(0.2, 0.2, 0.2) {
		shapes.push_back(new QuadraticShape(redDiffuseMaterial, mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,-1)));
		lights.push_back(Light(vec3(2, 0, 0), vec3(1, 1, 1)));
	}

	void render(std::vector<vec4>& image) {
		for (int i = 0; i < windowWidth; i++) {
			for (int j = 0; j < windowHeight; j++) {
				float x = (float)i / windowWidth * 2 - 1;
				float y = (float)j / windowHeight * 2 - 1;

				image.push_back(toHomogeneousVector(trace(cam.getRay(x, y))));
			}
		}
	}

	vec3 trace(Ray ray) const {
		//printf("%f %f %f\n", ray.dir.x, ray.dir.y, ray.dir.z);
		Hit firstHit;
		Shape* hitShape = nullptr;
		firstHit.t = INFINITY;
		for (auto shape : shapes) {
			Hit hit = shape->intersect(ray);
			if (hit.t < 0 || hit.t >= firstHit.t)
				continue;
			firstHit = hit;
			hitShape = shape;
		}

		if (hitShape == nullptr)
			return ambLight;

		vec3 reflectedLight;
		for (auto& light : lights) {
			reflectedLight = reflectedLight + hitShape->getMaterial().reflectLight(ambLight, light.color, firstHit.normal, light.pos - firstHit.point, toDescartes(vec4() - ray.dir));
		}

		return reflectedLight;
	}
};

World world;
std::vector<vec4> image;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
	fullScreenTextQuad.genTexture();
	world.render(image);
}

// Window has become invalid: Redraw
void onDisplay() {
	//std::vector<vec4> image(windowHeight * windowWidth, vec4(0.5, 0.5, 0, 0));
	fullScreenTextQuad.LoadTexture(image);
	fullScreenTextQuad.Draw();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	const char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
