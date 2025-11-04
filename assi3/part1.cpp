#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// constants
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float PI = 3.1415926535f;

// camera parameters 
float camera_radius = 5.0f; 
float camera_angle = PI / 4.0f; 
float camera_height = 0.5f; 

bool is_perspective = true; 
float last_key_press_time = 0.0f;
float key_debounce_time = 0.2f;

GLuint shaderProgram;
GLuint VBO, VAO;

struct Vec3 {
    float x, y, z;
};

struct Mat4 {
    float m[16];
};


// vector operations
Vec3 operator-(const Vec3& a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Vec3 operator+(const Vec3& a, const Vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Vec3 operator*(const Vec3& v, float s) { return { v.x * s, v.y * s, v.z * s }; }

float magnitude(const Vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

Vec3 normalize(const Vec3& v) {
    float len = magnitude(v);
    if (len > 0.00001f) {
        return { v.x / len, v.y / len, v.z / len };
    }
    return { 0.0f, 0.0f, 0.0f };
}

// vector products
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// scalar product
float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// matrix operations
Mat4 Mat4_Identity() {
    Mat4 res = { 0 };
    res.m[0] = res.m[5] = res.m[10] = res.m[15] = 1.0f; 
    return res;
}

// multiply two 4x4 matrices 
Mat4 Mat4_Multiply(const Mat4& A, const Mat4& B) {
    Mat4 C = { 0 };
    for (int i = 0; i < 4; ++i) { 
        for (int j = 0; j < 4; ++j) { 
            for (int k = 0; k < 4; ++k) {
                C.m[4 * j + i] += A.m[4 * k + i] * B.m[4 * j + k];
            }
        }
    }
    return C;
}

// view matrix
Mat4 Mat4_LookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye); 
    Vec3 s = normalize(cross(f, normalize(up))); 
    Vec3 u = cross(s, f); 

    Mat4 R = Mat4_Identity();
    R.m[0] = s.x; R.m[4] = s.y; R.m[8] = s.z;
    R.m[1] = u.x; R.m[5] = u.y; R.m[9] = u.z;
    R.m[2] = -f.x; R.m[6] = -f.y; R.m[10] = -f.z;

    Mat4 T = Mat4_Identity();
    T.m[12] = -eye.x;
    T.m[13] = -eye.y;
    T.m[14] = -eye.z;

    return Mat4_Multiply(R, T);
}

// projection matrices
Mat4 Mat4_Perspective(float fov, float aspect, float near, float far) {
    Mat4 res = { 0 };
    float tanHalfFov = std::tan(fov / 2.0f);

    res.m[0] = 1.0f / (aspect * tanHalfFov);
    res.m[5] = 1.0f / tanHalfFov;
    res.m[10] = -(far + near) / (far - near);
    res.m[11] = -1.0f;
    res.m[14] = -(2.0f * far * near) / (far - near);

    return res;
}

// orthographic projection
Mat4 Mat4_Orthographic(float left, float right, float bottom, float top, float near, float far) {
    Mat4 res = Mat4_Identity();

    res.m[0] = 2.0f / (right - left);
    res.m[5] = 2.0f / (top - bottom);
    res.m[10] = -2.0f / (far - near);
    res.m[12] = -(right + left) / (right - left);
    res.m[13] = -(top + bottom) / (top - bottom);
    res.m[14] = -(far + near) / (far - near);

    return res;
}

struct Vertex {
    Vec3 position;
};

struct FaceData {
    unsigned int v1, v2, v3;
    Vec3 color; 
};

std::vector<Vertex> vertices;
std::vector<FaceData> faces;
std::vector<float> vertexBufferData; 

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void checkShaderCompileErrors(unsigned int shader, std::string type);
void setupShaders();

// SMF loader
void loadSMF(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR::SMF_LOADER::File not successfully read: " << filename << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            vertices.push_back({ {x, y, z} });
        }
        else if (type == "f") {
            unsigned int v1, v2, v3;
            ss >> v1 >> v2 >> v3;
            faces.push_back({ v1 - 1, v2 - 1, v3 - 1, {0.0f, 0.0f, 0.0f} });
        }
    }

    file.close();
    std::cout << "Model loaded: " << vertices.size() << " vertices, " << faces.size() << " triangles." << std::endl;
}

// model processing
void calculateFaceNormals() {
    for (auto& face : faces) {
        Vec3 p1 = vertices[face.v1].position;
        Vec3 p2 = vertices[face.v2].position;
        Vec3 p3 = vertices[face.v3].position;

        Vec3 edge1 = p2 - p1;
        Vec3 edge2 = p3 - p1;

        Vec3 normal = cross(edge1, edge2);

        Vec3 normal_normalized = normalize(normal);

        face.color.x = std::abs(normal_normalized.x);
        face.color.y = std::abs(normal_normalized.y);
        face.color.z = std::abs(normal_normalized.z);
    }
}


// buffer setup
void setupBuffers() {
    vertexBufferData.reserve(faces.size() * 3 * 6);
    for (const auto& face : faces) {
        for (int i = 0; i < 3; ++i) {
            unsigned int v_index = (i == 0) ? face.v1 : ((i == 1) ? face.v2 : face.v3);
            const Vec3& pos = vertices[v_index].position;

            vertexBufferData.push_back(pos.x);
            vertexBufferData.push_back(pos.y);
            vertexBufferData.push_back(pos.z);
            vertexBufferData.push_back(face.color.x);
            vertexBufferData.push_back(face.color.y);
            vertexBufferData.push_back(face.color.z);
        }
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertexBufferData.size() * sizeof(float), vertexBufferData.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// matrix updates
Mat4 updateViewMatrix() {
    const Vec3 lookAtPoint = { 0.0f, 0.0f, 0.0f };
    const Vec3 upVector = { 0.0f, 1.0f, 0.0f };

	// calculate camera position in Cartesian coordinates
    // X = R * cos(θ)
    // Z = R * sin(θ)
    // Y = H
    float x = camera_radius * std::cos(camera_angle);
    float z = camera_radius * std::sin(camera_angle);
    float y = camera_height;

    Vec3 cameraPos = { x, y, z };

    return Mat4_LookAt(cameraPos, lookAtPoint, upVector);
}

// projection matrix update
Mat4 updateProjectionMatrix(float aspect) {
    if (is_perspective) {
        return Mat4_Perspective(PI / 4.0f, aspect, 0.1f, 100.0f);
    }
    else {
        float size = camera_radius / 3.0f;
        return Mat4_Orthographic(-size * aspect, size * aspect, -size, size, 0.1f, 100.0f);
    }
}

// shader setup
void setupShaders() {
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;

        out vec3 ourColor;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            ourColor = aColor; 
        }
    )";

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkShaderCompileErrors(vertexShader, "VERTEX");

    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        in vec3 ourColor;

        void main() {
            FragColor = vec4(ourColor, 1.0f); 
        }
    )";
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderCompileErrors(fragmentShader, "FRAGMENT");

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkShaderCompileErrors(shaderProgram, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void checkShaderCompileErrors(unsigned int shader, std::string type) {
    int success;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n" << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n" << std::endl;
        }
    }
}

// process input
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float current_time = (float)glfwGetTime();
    float camera_speed = 0.05f;
    float angle_speed = 0.02f;

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera_angle += angle_speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera_angle -= angle_speed;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera_radius = std::max(0.5f, camera_radius - camera_speed);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera_radius += camera_speed;

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera_height += camera_speed;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera_height -= camera_speed;

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && current_time - last_key_press_time > key_debounce_time) {
        is_perspective = !is_perspective;
        last_key_press_time = current_time;
        std::cout << "Projection switched to: " << (is_perspective ? "Perspective" : "Parallel (Orthographic)") << std::endl;
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <smf_filename>" << std::endl;
        return -1;
    }
    std::string smf_filename = argv[1];

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Part 1", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    loadSMF(smf_filename);
    calculateFaceNormals();

    setupShaders();
    setupBuffers();

	std::cout << "Controls: A/D angle | W/S zoom in/zoom out | Up/Down camera up/down | P toogle proj\n";

    // rendering
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        Mat4 model = Mat4_Identity();
        Mat4 view = updateViewMatrix();
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (float)width / (float)height;
        Mat4 projection = updateProjectionMatrix(aspect);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, model.m);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, view.m);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, projection.m);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertexBufferData.size() / 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}