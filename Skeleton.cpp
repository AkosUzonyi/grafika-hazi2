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
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
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


mat4 transpose(const mat4& M) {
	mat4 Mt;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			Mt[i][j] = M[j][i];
	return Mt;
}

vec3 cutToVec3(vec4 v) {
	return vec3(v.x, v.y, v.z);
}

vec4 toHomogeneousPoint(vec3 v) {
	return vec4(v.x, v.y, v.z, 1);
}

vec4 toHomogeneousVector(vec3 v) {
	return vec4(v.x, v.y, v.z, 0);
}

class Shape;
struct Hit {
	const Shape* shape = nullptr;
	vec3 point, normal;
	float t = INFINITY;
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

#define PV(v) printf("%s: %f %f %f\n", #v, v.x, v.y, v.z);

struct Ray {
	vec4 origin, dir;

	Ray(vec3 o, vec3 d) : origin(toHomogeneousPoint(o)), dir(toHomogeneousVector(normalize(d))) {}
};

class Camera {
	vec3 pos, look, up, right;
public:
	Camera(vec3 pos, vec3 look, vec3 up, vec3 right) : pos(pos), look(look), up(up), right(right) {}

	Ray getRay(float x, float y) const {
		return Ray(pos, look + right * x + up * y);
	}
};



class World;
class Material {
public:
	virtual vec3 trace(const World& world, vec3 point, vec3 normal, vec3 eyeDir, int depth) const = 0;
};

class DiffuseMaterial : public Material {
	vec3 ka, ks, kd;
	float shine;
public:
	DiffuseMaterial(vec3 ka, vec3 ks, vec3 kd, float shine) : ka(ka), ks(ks), kd(kd), shine(shine) {}
	vec3 trace(const World& world, vec3 point, vec3 normal, vec3 eyeDir, int depth) const;
};

class ReflectiveMaterial : public Material {
	vec3 n, k;
public:
	ReflectiveMaterial(vec3 n, vec3 k) : n(n), k(k) {}
	vec3 trace(const World& world, vec3 point, vec3 normal, vec3 eyeDir, int depth) const;
};

class Shape {
	const Material& material;
public:
	Shape(const Material& material) : material(material) {}

	virtual Hit intersect(Ray ray) const = 0;
	virtual void transform(mat4 M) = 0;

	void scale(float x, float y, float z) {
		transform(ScaleMatrix(vec3(1 / x, 1 / y, 1 / z)));
	}

	void rotate(float a, float x, float y, float z) {
		transform(RotationMatrix(-a, vec3(x, y, z)));
	}

	void translate(float x, float y, float z) {
		transform(TranslateMatrix(vec3(-x, -y, -z)));
	}

	const Material& getMaterial() const {
		return material;
	}
};

class YBoundedShape : public Shape {
	const Shape& shape;
	float ymin, ymax;
public:
	YBoundedShape(const Shape& shape, float ymin, float ymax) : Shape(shape.getMaterial()), shape(shape), ymin(ymin), ymax(ymax) {}

	Hit intersect(Ray ray) const {
		Hit hit = shape.intersect(ray);

		if (hit.point.y < ymin || hit.point.y > ymax)
			hit.shape = nullptr;

		return hit;
	}

	void transform(mat4 M) {
		printf("YBoundedShape::transform unsupported");
		exit(1);
	}
};

class QuadraticShape : public Shape {
public:
	mat4 Q;

	QuadraticShape(const Material& material, const mat4& Q = mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,-1)) : Shape(material), Q(Q) {}

	void transform(mat4 M);
	Hit intersect(Ray ray) const;
};

class World {
public:
	float holeHeight, holeRadius;
	std::vector<Shape*> shapes;
	std::vector<vec3> holePoints;
	vec3 ambLight;
	vec3 sky;
	vec3 sun;
	vec3 sunDir;

	Hit intersect(Ray ray) const;
	vec3 trace(Ray ray, int depth) const;
};



void QuadraticShape::transform(mat4 M) {
	Q = M * Q * transpose(M);
}

Hit QuadraticShape::intersect(Ray ray) const {
	Hit hit;
	float a = dot(ray.dir * Q, ray.dir);
	float b = dot(ray.dir * Q, ray.origin) + dot(ray.origin * Q, ray.dir);
	float c = dot(ray.origin * Q, ray.origin);
	float disc = b * b - 4 * a * c;
	if (disc < 0)
		return hit;

	float discSqrt = sqrt(disc);
	float t1 = (-b + discSqrt) / (2 * a);
	float t2 = (-b - discSqrt) / (2 * a);

	if (t1 < 0 && t2 < 0)
		return hit;

	hit.shape = this;
	hit.t = firstIntersect(t1, t2);
	hit.point = cutToVec3(ray.origin + ray.dir * hit.t);
	hit.normal = normalize(cutToVec3(toHomogeneousPoint(hit.point) * Q));
	return hit;
}

const int n = 3;
const float eps = 0.001;

vec3 DiffuseMaterial::trace(const World& world, vec3 point, vec3 normal, vec3 eyeDir, int depth) const {
	vec3 color;
	color = color + ka * world.ambLight;

	for (vec3 lightPoint : world.holePoints) {
		vec3 lightDir = normalize(lightPoint - point);
		float lightDist = length(lightPoint - point);
		float holeArea = world.holeRadius * world.holeRadius * M_PI;
		float lightAngle = std::fabs(lightDir.y);

		Ray rayToLight(point + normal * eps, lightDir);

		Hit hitToLight = world.intersect(rayToLight);
		if (hitToLight.shape && hitToLight.t < lightDist)
			continue;

		vec3 lightColor = world.trace(rayToLight, depth + 1);

		lightColor = lightColor * holeArea / n * lightAngle / (lightDist * lightDist);

		color = color + lightColor * (kd * std::fmax(dot(normal, lightDir), 0));
		color = color + lightColor * (ks * pow(std::fmax(dot(normalize((lightDir + eyeDir) / 2), normal), 0), shine));
	}

	return color;
}

vec3 operator/(vec3 a, vec3 b) {
	return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

vec3 ReflectiveMaterial::trace(const World& world, vec3 point, vec3 normal, vec3 eyeDir, int depth) const {
	float cosAngle = dot(normal, eyeDir);
	vec3 reflectDir = cosAngle * normal * 2 - eyeDir;

	vec3 one(1, 1, 1);
	vec3 f0 = ((n - one) * (n - one) + k * k) / ((n + one) * (n + one) + k * k);
	vec3 f = f0 + (one - f0) * pow(1 - cosAngle, 5);

	return world.trace(Ray(point + normal * eps, reflectDir), depth + 1) * f;
}

Hit World::intersect(Ray ray) const {
	Hit firstHit;
	for (auto shape : shapes) {
		Hit hit = shape->intersect(ray);
		if (!hit.shape || hit.t >= firstHit.t)
			continue;

		if (hit.point.y > holeHeight && length(vec2(hit.point.x, hit.point.z)) < holeRadius)
			continue;

		firstHit = hit;
	}
	return firstHit;
}

vec3 World::trace(Ray ray, int depth) const {
	if (depth > 4)
		return vec3(0, 0, 0);

	Hit hit = intersect(ray);

	if (hit.shape)
	{
		vec3 eyeDir = -cutToVec3(ray.dir);
		if (dot(eyeDir, hit.normal) < 0)
			hit.normal = -hit.normal;

		return hit.shape->getMaterial().trace(*this, hit.point, hit.normal, eyeDir, depth + 1);
	}
	else
	{
		return sky + sun * pow(std::fmax(dot(cutToVec3(ray.dir), sunDir), 0), 10);
	}
}

World world;
Camera camera(vec3(0, 0, 1.8), vec3(0, 0, -1), vec3(0, 1, 0), vec3(1, 0, 0));
std::vector<vec4> image;

// Initialization, create an OpenGL context
void onInitialization() {
	srand(time(NULL));
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
	fullScreenTextQuad.genTexture();

	DiffuseMaterial redDiffuseMaterial(vec3(1, 0.8, 0.8), vec3(1, 1, 1), vec3(1, 0.8, 0.8), 6);
	DiffuseMaterial greenDiffuseMaterial(vec3(0, 1, 0), vec3(1, 1, 1), vec3(0, 1 ,0), 30);
	ReflectiveMaterial gold(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
	ReflectiveMaterial silver(vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1));

	world.ambLight = vec3(0.2, 0.2, 0.2);
	world.sky = vec3(0.2, 0.2, 0.6);
	world.sun = vec3(5, 5, 2);
	world.sunDir = normalize(vec3(0, 1, 1));
	world.holeRadius = 0.6;

	QuadraticShape room(redDiffuseMaterial);
	room.scale(2, 1, 2);

	QuadraticShape ball(greenDiffuseMaterial);
	ball.scale(0.1, 0.1, 0.1);
	ball.translate(-0.2, -0.4, 1);

	QuadraticShape mirror(gold, mat4(1,0,0,0, 0,0,0,1, 0,0,1,0, 0,1,0,0));
	mirror.scale(0.5, 1, 0.5);
	mirror.translate(0.5, 0, -0.5);

	QuadraticShape tube(silver, mat4(1,0,0,0, 0,-1,0,0, 0,0,1,0, 0,0,0,-1));
	tube.scale(1, 4, 1);

	Hit holeHit = room.intersect(Ray(vec3(world.holeRadius, 100, 0), vec3(0, -1, 0)));
	world.holeHeight = holeHit.point.y;

	Hit tubeHit = tube.intersect(Ray(vec3(100, world.holeHeight, 0), vec3(-1, 0, 0)));
	float tubeScale = world.holeRadius / tubeHit.point.x;
	tube.scale(tubeScale, 1, tubeScale);

	YBoundedShape boundedTube(tube, world.holeHeight, world.holeHeight + 2);

	world.shapes.push_back(&mirror);
	world.shapes.push_back(&ball);
	world.shapes.push_back(&room);
	world.shapes.push_back(&boundedTube);

	for (int i = 0; i < n; i++) {
		float x, z;
		do {
			x = (float)rand() / RAND_MAX * 2 - 1;
			z = (float)rand() / RAND_MAX * 2 - 1;
		}
		while (x * x + z * z > 1);
		world.holePoints.push_back(vec3(x * world.holeRadius, world.holeHeight, z * world.holeRadius));
	}


	for (int j = 0; j < windowHeight; j++) {
		for (int i = 0; i < windowWidth; i++) {
			float x = (float)i / windowWidth * 2 - 1;
			float y = (float)j / windowHeight * 2 - 1;

			image.push_back(toHomogeneousPoint(world.trace(camera.getRay(x, y), 0)));
		}
	}
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
