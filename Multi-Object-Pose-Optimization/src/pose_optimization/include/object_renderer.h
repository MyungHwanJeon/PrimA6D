#ifndef _OBJECT_RENDERER_H_
#define _OBJECT_RENDERER_H_

#include "usrdef.h"
#include <Renderer.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

class object_renderer
{
public:
    object_renderer();


    bool init(int width, int height);

    void setLight(std::array<float, 3> lightCamPos,
                  std::array<float, 3> lightColor,
                  float lightAmbientWeight, float lightDiffuseWeight,
                  float lightSpecularWeight, float lightSpecularShininess);

    bool addObject(unsigned int objId, const std::string &filename);

    void removeObject(unsigned int objId);

    void renderObject(unsigned int objId,
                      std::array<float, 9> R,
                      std::array<float, 3> t,
                      float fx, float fy, float cx, float cy);

    void renderObject(unsigned int objId, Eigen::Matrix3d R, Eigen::Vector3d t, Eigen::Matrix3d K);

    cv::Mat getDepthImage(unsigned int objId);

    cv::Mat getColorImage(unsigned int objId);

private:
    Renderer m_renderer;
};

#endif
