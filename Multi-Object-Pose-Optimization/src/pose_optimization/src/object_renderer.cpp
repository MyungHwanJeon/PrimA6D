#include "object_renderer.h"

object_renderer::object_renderer()
{
    
}

bool object_renderer::init(int width, int height)
{
    return m_renderer.init(width, height);
}

void object_renderer::setLight(std::array<float, 3> lightCamPos,
                               std::array<float, 3> lightColor,
                               float lightAmbientWeight, float lightDiffuseWeight,
                               float lightSpecularWeight, float lightSpecularShininess)
{
    glm::vec3 _lightCamPos(lightCamPos[0], lightCamPos[1], lightCamPos[2]);
    glm::vec3 _lightColor(lightColor[0], lightColor[1], lightColor[2]);
    m_renderer.setLight(_lightCamPos, _lightColor, lightAmbientWeight,
                        lightDiffuseWeight, lightSpecularWeight,
                        lightSpecularShininess);
}

bool object_renderer::addObject(unsigned int objId, const std::string &filename)
{
    return m_renderer.addObject(objId, filename);
}

void object_renderer::removeObject(unsigned int objId)
{
    m_renderer.removeObject(objId);
}

void object_renderer::renderObject(unsigned int objId,
                    std::array<float, 9> R,
                    std::array<float, 3> t,
                    float fx, float fy, float cx, float cy)
{
    glm::mat3 _R;
    for (unsigned int x = 0; x < 3; x++)
    {
        for (unsigned int y = 0; y < 3; y++)
        {
            _R[x][y] = R[y * 3 + x];
        }
    }

    glm::vec3 _t;
    for (unsigned int x = 0; x < 3; x++)
    {
        _t[x] = t[x];
    }

    Pose pose(_R, _t);
    m_renderer.renderObject(objId, pose, fx, fy, cx, cy);
}

void object_renderer::renderObject(unsigned int objId, Eigen::Matrix3d R, Eigen::Vector3d t, Eigen::Matrix3d K)
{
    glm::mat3 _R;
    for (unsigned int x = 0; x < 3; x++)
    {
        for (unsigned int y = 0; y < 3; y++)
        {
            _R[x][y] = R(y, x);
        }
    }

    glm::vec3 _t;
    for (unsigned int x = 0; x < 3; x++)
    {
        _t[x] = t[x];
    }

    Pose pose(_R, _t);

    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    m_renderer.renderObject(objId, pose, fx, fy, cx, cy);
}

cv::Mat object_renderer::getDepthImage(unsigned int objId)
{
    float *buffData = m_renderer.getBuffer(objId, OA_DEPTH);

    int height = m_renderer.getHeight();
    int width = m_renderer.getWidth();
    int size = height * width;
    auto *arrData = new uint16_t[size];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int i = x + width * y;
            if (y < 0 || x < 0 || y >= height || x >= width)
            {
                arrData[i] = 0;
            }
            else
            {
                arrData[i] = static_cast<uint16_t>(round(buffData[i]));
            }
        }
    }

    cv::Mat depth_img = cv::Mat(height, width, CV_16UC1, arrData).clone();

    delete[] arrData;
    delete[] buffData;

    return depth_img;
}

cv::Mat object_renderer::getColorImage(unsigned int objId)
{
    float *buffData = m_renderer.getBuffer(objId, OA_COLORS);
    //  float *buffData = m_renderer.getBuffer(handle, OA_TEXTURED);

    int height = m_renderer.getHeight();
    int width = m_renderer.getWidth();
    int size = height * width * 3;
    auto *arrData = new uint8_t[size];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int i = (x + width * y) * 3;
            if (y < 0 || x < 0 || y >= height || x >= width)
            {
                arrData[i] = 0;
                arrData[i + 1] = 0;
                arrData[i + 2] = 0;
            }
            else
            {
                arrData[i] = static_cast<uint8_t>(buffData[i] * 255.0f);
                arrData[i + 1] = static_cast<uint8_t>(buffData[i + 1] * 255.0f);
                arrData[i + 2] = static_cast<uint8_t>(buffData[i + 2] * 255.0f);
            }
        }
    }

    cv::Mat color_img = cv::Mat(height, width, CV_8UC3, arrData).clone();

    delete[] arrData;
    delete[] buffData;
    

    return color_img;
}