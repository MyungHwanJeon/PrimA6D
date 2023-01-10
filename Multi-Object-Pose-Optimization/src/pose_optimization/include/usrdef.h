#ifndef _USRDEF_H_
#define _USRDEF_H_

#include <iostream>
#include <sstream>
#include <fstream>

// using namespace std;

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <termios.h>
#include <random>
char getch();

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <std_msgs/Time.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <vision_msgs/BoundingBox2D.h>
#include <realsense2_camera/Extrinsics.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "franka_rpm_msgs/FrankaState.h"



#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.h>




// using namespace cv;

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <pcl/io/ply_io.h>
// #include <pcl/point_types.h>
// #include <pcl/registration/icp.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/filters/filter_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void addNormal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals);

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include <Eigen/StdVector>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <opencv2/core/eigen.hpp>


// using namespace Eigen;

#define PI 3.1415926535

typedef std::tuple<double, double, double, double, double, double, double> model_info;
model_info get_model_param(std::string dataset, int idx);

#endif

