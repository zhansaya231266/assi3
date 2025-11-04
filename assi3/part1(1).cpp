// Controls:
//  A / D : decrease / increase angle (theta)
//  W / S : increase / decrease camera height (H)
//  Q / E : decrease / increase orbit radius (R)
//  P     : toggle projection (perspective / orthographic)
//  R     : reset camera params
//  ESC   : exit

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <array>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
};
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 operator*(const Vec3& a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float len(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline Vec3 normalize(const Vec3& v) { float L = len(v); if (L == 0) return Vec3(0, 0, 0); return Vec3(v.x / L, v.y / L, v.z / L); }

std::vector<Vec3> vertices;
std::vector<std::array<int, 3>> faces;
std::vector<Vec3> faceNormals;
Vec3 modelCentroid(0, 0, 0);
float modelScale = 1.0f;

// camera (cylindrical)
float camTheta = 0.0f;
float camRadius = 3.0f;
float camHeight = 0.0f;
bool usePerspective = true;

int winW = 1024, winH = 768;

const float PI = 3.14159265358979323846f;
static float viewFactor = 4.0f;

// steps
const float DTH = 4.0f * PI / 180.0f; 
const float DR = 0.1f;
const float DH = 0.1f;

// matrix helpers (column-major) 
static std::array<float, 16> createPerspectiveMatrix(float fovy_radians, float aspect, float znear, float zfar) {
    float f = 1.0f / tanf(fovy_radians * 0.5f);
    std::array<float, 16> m{};
    // column-major: m[col*4 + row]
    m[0] = f / aspect; m[4] = 0.0f; m[8] = 0.0f;                      m[12] = 0.0f;
    m[1] = 0.0f;       m[5] = f;    m[9] = 0.0f;                      m[13] = 0.0f;
    m[2] = 0.0f;       m[6] = 0.0f; m[10] = (zfar + znear) / (znear - zfar); m[14] = (2.0f * zfar * znear) / (znear - zfar);
    m[3] = 0.0f;       m[7] = 0.0f; m[11] = -1.0f;                     m[15] = 0.0f;
    return m;
}

static std::array<float, 16> createLookAtMatrix(float eyeX, float eyeY, float eyeZ,
    float centerX, float centerY, float centerZ,
    float upX, float upY, float upZ)
{
    // compute forward vector f = normalize(center - eye)
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    float flen = sqrtf(fx * fx + fy * fy + fz * fz);
    if (flen == 0.0f) flen = 1.0f;
    fx /= flen; fy /= flen; fz /= flen;

    // normalize up
    float ux = upX, uy = upY, uz = upZ;
    float ulen = sqrtf(ux * ux + uy * uy + uz * uz);
    if (ulen == 0.0f) ulen = 1.0f;
    ux /= ulen; uy /= ulen; uz /= ulen;

    // s = f x up
    float sx = fy * uz - fz * uy;
    
    float sy = fz * ux - fx * uz;
    float sz = fx * uy - fy * ux;
    float slen = sqrtf(sx * sx + sy * sy + sz * sz);
    if (slen == 0.0f) slen = 1.0f;
    sx /= slen; sy /= slen; sz /= slen;

    // u = s x f
    float ux2 = sy * fz - sz * fy;
    float uy2 = sz * fx - sx * fz;
    float uz2 = sx * fy - sy * fx;

    // translation t = -R * eye (where R = [s; u; -f] rows)
    float tx = -(sx * eyeX + sy * eyeY + sz * eyeZ);
    float ty = -(ux2 * eyeX + uy2 * eyeY + uz2 * eyeZ);
    float tz = (fx * eyeX + fy * eyeY + fz * eyeZ); 

    std::array<float, 16> m{};
    m[0] = sx;   m[4] = ux2;  m[8] = -fx;  m[12] = 0.0f;
    m[1] = sy;   m[5] = uy2;  m[9] = -fy;  m[13] = 0.0f;
    m[2] = sz;   m[6] = uz2;  m[10] = -fz;  m[14] = 0.0f;
    m[3] = tx;   m[7] = ty;   m[11] = tz;   m[15] = 1.0f;
    return m;
}

// SMF loader
bool loadSMF(const std::string& fname) {
    std::ifstream in(fname);
    if (!in.is_open()) return false;
    vertices.clear(); faces.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string t; iss >> t;
        if (t == "v") {
            float x, y, z; iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        }
        else if (t == "f") {
            int a, b, c; if (!(iss >> a >> b >> c)) continue;
            faces.push_back({ a - 1,b - 1,c - 1 });
        }
    }
    in.close();
    return (!vertices.empty() && !faces.empty());
}

// UV sphere if no model provided 
void makeUVSphere(int stacks = 20, int sectors = 40, float radius = 1.0f) {
    vertices.clear(); faces.clear();
    for (int i = 0;i <= stacks;i++) {
        float V = (float)i / (float)stacks;
        float phi = V * PI;
        for (int j = 0;j <= sectors;j++) {
            float U = (float)j / (float)sectors;
            float theta = U * 2.0f * PI;
            float x = std::cos(theta) * std::sin(phi);
            float y = std::sin(theta) * std::sin(phi);
            float z = std::cos(phi);
            vertices.emplace_back(x * radius, y * radius, z * radius);
        }
    }
    int cols = sectors + 1;
    for (int i = 0;i < stacks;i++) {
        for (int j = 0;j < sectors;j++) {
            int v1 = i * cols + j;
            int v2 = v1 + cols;
            int v3 = v2 + 1;
            int v4 = v1 + 1;
            // two triangles v1,v2,v3 and v1,v3,v4
            faces.push_back({ v1,v2,v3 });
            faces.push_back({ v1,v3,v4 });
        }
    }
}

// -----------------------------------------------------
void computeModelInfo() {
    if (vertices.empty()) return;
    float sx = 0, sy = 0, sz = 0;
    for (auto& v : vertices) { sx += v.x; sy += v.y; sz += v.z; }
    float n = (float)vertices.size();
    modelCentroid = Vec3(sx / n, sy / n, sz / n);
    float maxd = 0.0f;
    for (auto& v : vertices) {
        Vec3 d = v - modelCentroid;
        float dist = len(d);
        if (dist > maxd) maxd = dist;
    }
    modelScale = (maxd > 0.0f) ? maxd : 1.0f;
    // set sensible camera defaults
    camRadius = modelScale * viewFactor;
    camHeight = 0.0f;
    camTheta = 0.0f;
}

// -----------------------------------------------------
void computeFaceNormals() {
    faceNormals.clear();
    faceNormals.reserve(faces.size());
    for (auto& f : faces) {
        Vec3 v1 = vertices[f[0]];
        Vec3 v2 = vertices[f[1]];
        Vec3 v3 = vertices[f[2]];
        Vec3 e1 = v2 - v1;
        Vec3 e2 = v3 - v1;
        Vec3 n = cross(e1, e2);
        n = normalize(n);
        faceNormals.push_back(n);
    }
}

// -----------------------------------------------------
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camTheta -= DTH;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camTheta += DTH;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camHeight += DH;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camHeight -= DH;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) camRadius = std::max(0.01f, camRadius - DR);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) camRadius += DR;
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) camRadius += 0.05f * modelScale; 
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) camRadius = std::max(0.01f, camRadius - 0.05f * modelScale); 
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
        static double last = 0;
        double t = glfwGetTime();
        if (t - last > 0.25) { usePerspective = !usePerspective; last = t; }
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        static double last = 0;
        double t = glfwGetTime();
        if (t - last > 0.25) {
            camRadius = modelScale * 3.0f;
            camHeight = 0.0f; camTheta = 0.0f;
            last = t;
        }
    }
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
}

// -----------------------------------------------------
void render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_FLAT);
    glDisable(GL_LIGHTING);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (winH == 0) ? 1.0f : (float)winW / (float)winH;
    if (usePerspective) {
        auto proj = createPerspectiveMatrix(45.0f * PI / 180.0f, aspect, 0.01f * modelScale, 100.0f * modelScale);
        glLoadMatrixf(proj.data());
    }
    else {
        float s = modelScale * 1.5f;
        if (aspect >= 1.0f) {
            glOrtho(-s * aspect, s * aspect, -s, s, -1000.0f * modelScale, 1000.0f * modelScale);
        }
        else {
            glOrtho(-s, s, -s / aspect, s / aspect, -1000.0f * modelScale, 1000.0f * modelScale);
        }
    }

    // view
    float camX = camRadius * std::cos(camTheta);
    float camY = camRadius * std::sin(camTheta);
    float camZ = camHeight;
    auto view = createLookAtMatrix(camX, camY, camZ,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(view.data());

    // draw axes (optional)
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(modelScale * 0.5f, 0, 0);
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, modelScale * 0.5f, 0);
    glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, modelScale * 0.5f);
    glEnd();

    // draw model centered at origin and scaled to unit radius: v' = (v - centroid) / modelScale
    glBegin(GL_TRIANGLES);
    for (size_t i = 0;i < faces.size();++i) {
        Vec3 n = faceNormals[i];
        // color = abs(normal)
        glColor3f(std::abs(n.x), std::abs(n.y), std::abs(n.z));
        // set normal (for completeness)
        glNormal3f(n.x, n.y, n.z);
        auto& f = faces[i];
        for (int k = 0;k < 3;k++) {
            Vec3 v = vertices[f[k]];
            Vec3 vt = (v - modelCentroid) * (1.0f / modelScale);
            glVertex3f(vt.x, vt.y, vt.z);
        }
    }
    glEnd();
}

// -----------------------------------------------------
void framebuffer_size_cb(GLFWwindow* window, int w, int h) {
    (void)window;
    winW = w; winH = h;
    glViewport(0, 0, w, h);
}

// -----------------------------------------------------
int main(int argc, char** argv) {
    std::string fname;
    if (argc >= 2) fname = argv[1];

    // init GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n"; return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    // create window
    GLFWwindow* window = glfwCreateWindow(winW, winH, "Part1: SMF Viewer", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create GLFW window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n"; return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_cb);

    // load smf
    bool loaded = false;
    if (!fname.empty()) {
        loaded = loadSMF(fname);
        if (!loaded) std::cerr << "Warning: failed to load SMF '" << fname << "'. Generating sphere.\n";
    }
    if (!loaded) {
        makeUVSphere(30, 40, 1.0f);
    }

    computeModelInfo();
    computeFaceNormals();

    std::cout << "Vertices: " << vertices.size() << ", Faces: " << faces.size() << "\n";
    std::cout << "Controls: A/D angle | W/S height | Q/E radius | P toggle proj | R reset | ESC exit | Z/X zoom out/in \n";

    // main loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        render();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
