#ifndef _HAND_EYE_CALIBRATION_WITH_DEPTH_H_
#define _HAND_EYE_CALIBRATION_WITH_DEPTH_H_

#include "usrdef.h"

class hand_eye_calibration
{
public:

    hand_eye_calibration(ros::NodeHandle nh);

    void robot_manipulator_pose_callback(const franka_rpm_msgs::FrankaState::ConstPtr& msg);

    void camera_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg);
    void camera_color_callback(const sensor_msgs::Image::ConstPtr& msg);
    void camera_depth_callback(const sensor_msgs::Image::ConstPtr& msg);

    void publish_sensored_pointcloud();
    void publish_object_pose_marker_from_world();
    void publish_object_pose_image_from_cam();

    void run();

private:
    ros::NodeHandle m_nh;

    ros::Publisher m_pub_camera_color;
    ros::Publisher m_pub_sensored_pts;    
    ros::Publisher m_pub_aruco_pts;    
    ros::Publisher m_pub_refined_pts; 
    ros::Publisher m_pub_refined_pcl;    

    ros::Subscriber m_sub_camera_info;
    ros::Subscriber m_sub_camera_color;
    ros::Subscriber m_sub_camera_depth;
    ros::Subscriber m_sub_robot_manipulator_pose;   

    Eigen::Matrix4d m_T_Cam2Gripper;
    Eigen::Matrix4d m_T_axis_align;   // coordinate axis align btw robot manupulator and camera
    
    Eigen::Matrix4d m_T_robot_manipulator_pose;       
    Eigen::Matrix4d m_T_robot_manipulator_ee_to_cam;
    Eigen::Matrix4d m_T_robot_manipulator_ee_to_gripper; 
    double m_time_robot_manipulator_pose;    
    bool m_new_robot_manipulator_pose;    

    Eigen::Matrix3d m_camera_instricsic;
    Eigen::VectorXd m_camera_distortion;
    bool m_new_camera_info;

    cv::Mat m_camera_color;
    double m_time_camera_color;
    bool m_new_camera_color;

    cv::Mat  m_camera_depth;
    double m_time_camera_depth;
    bool m_new_camera_depth;

    std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> m_vec_T_robot_manipulator_pose;
    std::vector<std::array<double, 3>> m_vec_sensored_pts;
    std::vector<std::array<double, 3>> m_vec_aruco_pts;

    std::vector<std::array<double, 3>> m_aruco_marker_center;
    void setup_aruco_marker_center();

    void hand_eye_calibration_with_depth();
    Eigen::Matrix4d m_T_hand_eye;

public:
    bool m_save = false;
    int m_data_index = 0;
};


#endif