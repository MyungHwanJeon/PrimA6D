#ifndef _VISUALIZATION_H_
#define _VISUALIZATION_H_

#include "usrdef.h"

class visualization
{
public:

    visualization(ros::NodeHandle nh);

    void object_pose_from_world_callback(const vision_msgs::Detection2DArray::ConstPtr &msg);
    void object_pose_from_cam_callback(const vision_msgs::Detection2DArray::ConstPtr &msg);    
    void robot_manipulator_pose_callback(const franka_rpm_msgs::FrankaState::ConstPtr& msg);

    // void object_pose_PrimA6D_callback(const vision_msgs::Detection2DArray::ConstPtr &msg);


    void camera_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg);
    void camera_color_callback(const sensor_msgs::Image::ConstPtr& msg);
    void camera_depth_callback(const sensor_msgs::Image::ConstPtr& msg);

    void publish_sensored_pointcloud();
    void publish_robot_manipulator_path();
    void publish_object_pose_marker_from_world();
    void publish_object_pose_image_from_cam();
    void init_object_mesh_in_rviz();

    void setup_drawing_param(std::string dataset);

    void timer_callback(const ros::TimerEvent& event);    

private:
    ros::NodeHandle m_nh;
    ros::Publisher m_pub_object_pose_marker;
    ros::Publisher m_pub_object_pose_image;
    ros::Publisher m_pub_sensored_pcl;    
    ros::Publisher m_pub_robot_manipulator_path;

    ros::Subscriber m_sub_camera_info;
    ros::Subscriber m_sub_camera_color;
    ros::Subscriber m_sub_camera_depth;
    ros::Subscriber m_sub_object_pose_from_world;
    ros::Subscriber m_sub_object_pose_from_cam;
    ros::Subscriber m_sub_robot_manipulator_pose;   

    int m_n_object = 30;

    // std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> m_T_object_pose_from_world;
    // std::vector<Eigen::MatrixXd,Eigen::aligned_allocator<Eigen::MatrixXd>> m_object_pose_uncertainty_from_world;
    // std::vector<double> m_object_pose_uncertainty_norm_from_world;
    std::map<int, Eigen::Matrix4d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix4d>>> m_T_object_pose_from_world;
    std::map<int, Eigen::MatrixXd, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::MatrixXd>>> m_object_pose_uncertainty_from_world;
    std::map<int, double> m_object_pose_uncertainty_norm_from_world;
    double m_time_object_pose_from_world;  
    bool m_new_object_pose_from_world;

    // std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> m_T_object_pose_from_cam;
    // std::vector<Eigen::MatrixXd,Eigen::aligned_allocator<Eigen::MatrixXd>> m_object_pose_uncertainty_from_cam;
    // std::vector<double> m_object_pose_uncertainty_norm_from_cam;
    std::map<int, Eigen::Matrix4d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix4d>>> m_T_object_pose_from_cam;
    std::map<int, Eigen::MatrixXd, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::MatrixXd>>> m_object_pose_uncertainty_from_cam;
    std::map<int, double> m_object_pose_uncertainty_norm_from_cam;
    double m_time_object_pose_from_cam;
    bool m_new_object_pose_from_cam;

    Eigen::Matrix4d m_T_Cam2Gripper;
    Eigen::Matrix4d m_T_axis_align;   // coordinate axis align btw robot manupulator and camera
    
    // Eigen::Matrix4d m_T_robot_manipulator_pose;       
    Eigen::Matrix4d m_T_robot_manipulator_ee_to_cam;
    Eigen::Matrix4d m_T_robot_manipulator_ee_to_gripper; 
    // double m_time_robot_manipulator_pose;    
    bool m_new_robot_manipulator_pose;    

    std::vector<double> m_time_robot_manipulator_pose;
    std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> m_T_robot_manipulator_pose;
    std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> m_vec_T_robot_manipulator_pose;


    Eigen::Matrix3d m_camera_instricsic;
    Eigen::VectorXd m_camera_distortion;
    bool m_new_camera_info;

    cv::Mat m_camera_color;
    double m_time_camera_color;
    bool m_new_camera_color;

    cv::Mat  m_camera_depth;
    double m_time_camera_depth;
    bool m_new_camera_depth;
    Eigen::Matrix4d m_T_robot_manipulator_pose_at_camera_depth;    

    std::map<int, model_info> m_model_info;
    std::map<int, cv::Scalar> m_colors;

    ros::Timer m_timer;

    bool m_init_mesh;
    
    bool m_use_robot_franka;

    std::map<int, std::string> m_vis_model;

};


#endif