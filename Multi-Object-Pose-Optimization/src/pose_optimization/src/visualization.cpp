#include "visualization.h"

visualization::visualization(ros::NodeHandle nh): m_nh(nh)
{    
    

    std::string dataset;
    m_nh.getParam("/dataset", dataset);         
    setup_drawing_param(dataset);

    XmlRpc::XmlRpcValue dataset_object;
    m_nh.getParam("/dataset_object", dataset_object);    
    for (int32_t i = 0; i < dataset_object.size(); i++)
    {                        
        m_vis_model[static_cast<int>(dataset_object[i]["index"])] = static_cast<std::string>(dataset_object[i]["vis_model_path"]);
    }

    for (auto iter = m_vis_model.begin(); iter != m_vis_model.end(); iter++)
    {
        m_T_object_pose_from_world[iter->first] = Eigen::Matrix4d::Identity();
        m_object_pose_uncertainty_from_world[iter->first] = Eigen::MatrixXd::Identity(6, 6);
        m_object_pose_uncertainty_norm_from_world[iter->first] = 1e+10;

        m_T_object_pose_from_cam[iter->first] = Eigen::Matrix4d::Identity();
        m_object_pose_uncertainty_from_cam[iter->first] = Eigen::MatrixXd::Identity(6, 6);
        m_object_pose_uncertainty_norm_from_cam[iter->first] = 1e+10;
    }    
    m_new_object_pose_from_world = false;
    m_time_object_pose_from_world = 0.;
    m_new_object_pose_from_cam = false;  
    m_time_object_pose_from_cam = 0.;

    m_nh.getParam("use_robot_franka", m_use_robot_franka); 
    m_T_axis_align = Eigen::Matrix4d::Identity();
    if (m_use_robot_franka)
    {
        m_T_axis_align = Eigen::Matrix4d::Zero();
        m_T_axis_align(0, 1) = -1;
        m_T_axis_align(1, 0) = 1;
        m_T_axis_align(2, 2) = 1;
        m_T_axis_align(3, 3) = 1;
    }

    m_new_robot_manipulator_pose = false;

    m_camera_instricsic = Eigen::Matrix3d::Identity();
    m_camera_distortion = Eigen::VectorXd::Zero(5);
    m_new_camera_info = false;

    m_camera_color = cv::Mat(480, 640, CV_8UC3);
    m_time_camera_color = 0.;
    m_new_camera_color = false;

    m_camera_depth = cv::Mat(480, 640, CV_16UC1);
    m_time_camera_depth = 0.;
    m_new_camera_depth = false;       

    m_init_mesh = false;   

             
    
    m_pub_sensored_pcl = m_nh.advertise<sensor_msgs::PointCloud2>("/pose_estimation/graph_opt/sensored_pcl", 1);
    m_pub_object_pose_marker = m_nh.advertise<visualization_msgs::MarkerArray>("/pose_estimation/graph_opt/object_marker", 1);    
    m_pub_object_pose_image = m_nh.advertise<sensor_msgs::Image>("/pose_estimation/PrimA6D/object_image", 1);      
    m_pub_robot_manipulator_path = m_nh.advertise<nav_msgs::Path>("/pose_estimation/robot_arm/ee/path", 1);
    
    m_sub_object_pose_from_world = m_nh.subscribe<vision_msgs::Detection2DArray>("/pose_estimation/graph_opt/object/pose_from_world", 1, &visualization::object_pose_from_world_callback, this);
    m_sub_object_pose_from_cam = m_nh.subscribe<vision_msgs::Detection2DArray>("/pose_estimation/graph_opt/object/pose_from_cam", 1, &visualization::object_pose_from_cam_callback, this);
    // m_sub_object_pose_from_cam = m_nh.subscribe<vision_msgs::Detection2DArray>("/pose_estimation/PrimA6D/detection2D_array", 1000, &visualization::object_pose_from_cam_callback, this);
    // m_sub_robot_manipulator_pose = m_nh.subscribe<geometry_msgs::PoseStamped>("/pose_estimation/graph_opt/robot_arm/ee/pose", 1, &visualization::robot_manipulator_pose_callback, this);
    m_sub_robot_manipulator_pose = m_nh.subscribe<franka_rpm_msgs::FrankaState>("/franka_rpm/franka_states", 1000, &visualization::robot_manipulator_pose_callback, this);
        
    m_sub_camera_info = m_nh.subscribe<sensor_msgs::CameraInfo>("/pose_estimation/PrimA6D/camera_info", 1000, &visualization::camera_info_callback, this);    
    m_sub_camera_color = m_nh.subscribe<sensor_msgs::Image>("/pose_estimation/PrimA6D/color_raw", 1000, &visualization::camera_color_callback, this);
    m_sub_camera_depth = m_nh.subscribe<sensor_msgs::Image>("/pose_estimation/PrimA6D/depth_raw", 1000, &visualization::camera_depth_callback, this);

    m_timer = nh.createTimer(ros::Duration(0.033), boost::bind(&visualization::timer_callback, this, _1));
}

void visualization::camera_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    m_camera_instricsic(0, 0) = msg->K[0];
    m_camera_instricsic(0, 2) = msg->K[2];
    m_camera_instricsic(1, 1) = msg->K[4];
    m_camera_instricsic(1, 2) = msg->K[5];

    m_camera_distortion[0] = msg->D[0];
    m_camera_distortion[1] = msg->D[1];
    m_camera_distortion[2] = msg->D[2];
    m_camera_distortion[3] = msg->D[3];
    m_camera_distortion[4] = msg->D[4];

    m_new_camera_info = true;
}

void visualization::camera_color_callback(const sensor_msgs::Image::ConstPtr& msg)
{
    m_time_camera_color = msg->header.stamp.toSec();

    m_camera_color = cv::Mat(msg->height, msg->width, CV_8UC3, const_cast<unsigned char *>(msg->data.data()), msg->step).clone();
    
    m_new_camera_color = true;
}

void visualization::camera_depth_callback(const sensor_msgs::Image::ConstPtr& msg)
{
    m_time_camera_depth = msg->header.stamp.toSec();

    m_camera_depth = cv::Mat(msg->height, msg->width, CV_16UC1, const_cast<unsigned char *>(msg->data.data()), msg->step).clone();       

    m_new_camera_depth = true;
}

void visualization::object_pose_from_world_callback(const vision_msgs::Detection2DArray::ConstPtr& msg)
{
    m_time_object_pose_from_world = msg->header.stamp.toSec(); //msg->header.stamp.toSec();

    // uncertainty init //
    for (auto iter = m_vis_model.begin(); iter != m_vis_model.end(); iter++)
    {
        // object pose uncertainty //
        Eigen::VectorXd object_pose_uncertainty = Eigen::VectorXd::Zero(3);
        object_pose_uncertainty[0] = 1e+10;
        object_pose_uncertainty[1] = 1e+10;
        object_pose_uncertainty[2] = 1e+10;

        m_object_pose_uncertainty_from_world[iter->first] = object_pose_uncertainty;

        // object_pose_uncertainty_norm //
        m_object_pose_uncertainty_norm_from_world[iter->first] = 1e+10;
    }

    for (int i = 0; i < msg->detections.size(); i++)
    {
        // object pose //  
        Eigen::VectorXd object_pose_postion = Eigen::VectorXd::Zero(3);
        object_pose_postion[0] = msg->detections[i].results[0].pose.pose.position.x;
        object_pose_postion[1] = msg->detections[i].results[0].pose.pose.position.y;
        object_pose_postion[2] = msg->detections[i].results[0].pose.pose.position.z;

        Eigen::Quaterniond object_pose_orientation = Eigen::Quaterniond();;
        object_pose_orientation.x() = msg->detections[i].results[0].pose.pose.orientation.x;
        object_pose_orientation.y() = msg->detections[i].results[0].pose.pose.orientation.y;
        object_pose_orientation.z() = msg->detections[i].results[0].pose.pose.orientation.z;
        object_pose_orientation.w() = msg->detections[i].results[0].pose.pose.orientation.w;

        Eigen::Matrix4d object_pose = Eigen::Matrix4d::Identity();
        object_pose.block(0, 0, 3, 3) = object_pose_orientation.normalized().toRotationMatrix();
        object_pose.block(0, 3, 3, 1) = object_pose_postion;

        m_T_object_pose_from_world[msg->detections[i].results[0].id] = object_pose;

        // object pose uncertainty //
        Eigen::MatrixXd object_pose_uncertainty = Eigen::MatrixXd::Identity(6, 6);
        for (int ii=0; ii<6; ii++)
        {
            for (int jj=0; jj<6; jj++)
            {
                object_pose_uncertainty(ii, jj) = msg->detections[i].results[0].pose.covariance[ii + jj*6];
            }
        }        

        m_object_pose_uncertainty_from_world[msg->detections[i].results[0].id] = object_pose_uncertainty;  

        // object_pose_uncertainty_norm //
        m_object_pose_uncertainty_norm_from_world[msg->detections[i].results[0].id] = object_pose_uncertainty.norm();
    }

    m_new_object_pose_from_world = true;
}

void visualization::object_pose_from_cam_callback(const vision_msgs::Detection2DArray::ConstPtr& msg)
{
    m_time_object_pose_from_cam = msg->header.stamp.toSec();

    // uncertainty init //
    for (auto iter = m_vis_model.begin(); iter != m_vis_model.end(); iter++)
    {
        // object pose uncertainty //
        Eigen::VectorXd object_pose_uncertainty = Eigen::VectorXd::Zero(3);
        object_pose_uncertainty[0] = 1e+10;
        object_pose_uncertainty[1] = 1e+10;
        object_pose_uncertainty[2] = 1e+10;

        m_object_pose_uncertainty_from_cam[iter->first] = object_pose_uncertainty;

        // object_pose_uncertainty_norm //
        m_object_pose_uncertainty_norm_from_cam[iter->first] = 1e+10;
    }

    for (int i = 0; i < msg->detections.size(); i++)
    {
        // object pose //  
        Eigen::VectorXd object_pose_postion = Eigen::VectorXd::Zero(3);
        object_pose_postion[0] = msg->detections[i].results[0].pose.pose.position.x;
        object_pose_postion[1] = msg->detections[i].results[0].pose.pose.position.y;
        object_pose_postion[2] = msg->detections[i].results[0].pose.pose.position.z;

        Eigen::Quaterniond object_pose_orientation = Eigen::Quaterniond();;
        object_pose_orientation.x() = msg->detections[i].results[0].pose.pose.orientation.x;
        object_pose_orientation.y() = msg->detections[i].results[0].pose.pose.orientation.y;
        object_pose_orientation.z() = msg->detections[i].results[0].pose.pose.orientation.z;
        object_pose_orientation.w() = msg->detections[i].results[0].pose.pose.orientation.w;

        Eigen::Matrix4d object_pose = Eigen::Matrix4d::Identity();
        object_pose.block(0, 0, 3, 3) = object_pose_orientation.normalized().toRotationMatrix();
        object_pose.block(0, 3, 3, 1) = object_pose_postion;

        m_T_object_pose_from_cam[msg->detections[i].results[0].id] = object_pose;

        // object pose uncertainty //
        Eigen::MatrixXd object_pose_uncertainty = Eigen::MatrixXd::Identity(6, 6);
        for (int ii=0; ii<6; ii++)
        {
            for (int jj=0; jj<6; jj++)
            {
                object_pose_uncertainty(ii, jj) = msg->detections[i].results[0].pose.covariance[ii + jj*6];
            }
        }    

        m_object_pose_uncertainty_from_cam[msg->detections[i].results[0].id] = object_pose_uncertainty;

        // object_pose_uncertainty_norm //
        m_object_pose_uncertainty_norm_from_cam[msg->detections[i].results[0].id] = object_pose_uncertainty.norm();
    }

    m_new_object_pose_from_cam = true;
}

void visualization::robot_manipulator_pose_callback(const franka_rpm_msgs::FrankaState::ConstPtr& msg)
{    
    m_time_robot_manipulator_pose.push_back(msg->header.stamp.toSec());

    Eigen::Matrix4d T_robot_manipulator_pose = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            T_robot_manipulator_pose(j, i) = msg->O_T_EE[i * 4 + j];
        }
    }

    m_T_robot_manipulator_pose.push_back(T_robot_manipulator_pose);
    // m_T_robot_manipulator_pose.block(0, 3, 3, 1) *= 1000.;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m_T_robot_manipulator_ee_to_cam(j, i) = msg->EE_T_CAM[i * 4 + j];
        }
    }
    // m_T_robot_manipulator_ee_to_cam.block(0, 3, 3, 1) *= 1000.;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m_T_robot_manipulator_ee_to_gripper(j, i) = msg->EE_T_GRIPPER[i * 4 + j];
        }
    }
    // m_T_robot_manipulator_ee_to_gripper.block(0, 3, 3, 1) *= 1000.;

    m_vec_T_robot_manipulator_pose.push_back(T_robot_manipulator_pose * m_T_robot_manipulator_ee_to_gripper);

    m_new_robot_manipulator_pose = true;
}

void visualization::publish_sensored_pointcloud()
{
    double min_dist = 1e+15;
    long min_idx = 0;
    for (int idx = m_time_robot_manipulator_pose.size() - 1; idx > 0; idx--)
    {        
        double dist = abs(m_time_robot_manipulator_pose[idx] - (m_time_camera_color));        
        if (dist <= min_dist)
        {
            min_idx = idx;
            min_dist = dist;
        }            
        // else
        // {
        //     break;
        // }                                    
    }       
    if (min_dist > 0.001)    return;        

    Eigen::Matrix4d T_robot_manipulator_pose = m_T_robot_manipulator_pose[min_idx];
    m_T_robot_manipulator_pose.erase(m_T_robot_manipulator_pose.begin(), m_T_robot_manipulator_pose.begin() + min_idx);
    m_time_robot_manipulator_pose.erase(m_time_robot_manipulator_pose.begin(), m_time_robot_manipulator_pose.begin() + min_idx);


    PointCloud::Ptr sensored_cloud(new PointCloud);
    for (int m = 0; m < m_camera_depth.rows; m += 1)
    {
        for (int n = 0; n < m_camera_depth.cols; n += 1)
        {
            ushort d = m_camera_depth.ptr<ushort>(m)[n];
            if (d <= 0)
                continue;
            if (d >= 3000)
                continue;

            PointT p;

            p.z = double(d);
            p.x = ((n - m_camera_instricsic(0, 2)) * p.z / m_camera_instricsic(0, 0));
            p.y = ((m - m_camera_instricsic(1, 2)) * p.z / m_camera_instricsic(1, 1));            
            p.x /= 1000.;
            p.y /= 1000.;
            p.z /= 1000.;

            p.r = m_camera_color.ptr<uchar>(m)[n * 3];
            p.g = m_camera_color.ptr<uchar>(m)[n * 3 + 1];
            p.b = m_camera_color.ptr<uchar>(m)[n * 3 + 2];

            sensored_cloud->points.push_back(p);
        }
    }
    
    Eigen::Matrix4d T_Base2Cam = T_robot_manipulator_pose * m_T_robot_manipulator_ee_to_cam * m_T_axis_align;     
    pcl::transformPointCloud(*sensored_cloud, *sensored_cloud, T_Base2Cam);

    sensor_msgs::PointCloud2 cloudmsg;
    pcl::toROSMsg(*sensored_cloud, cloudmsg);
    cloudmsg.header.frame_id = "panda_link0";
    m_pub_sensored_pcl.publish(cloudmsg);
}

void visualization::publish_robot_manipulator_path()
{        
    nav_msgs::Path robot_manipulator_path_msgs;

    robot_manipulator_path_msgs.header.frame_id = "panda_link0";
    robot_manipulator_path_msgs.header.stamp = ros::Time::now();

    int token = int(m_vec_T_robot_manipulator_pose.size() / 500) + 1;
    for (int i=0; i<m_vec_T_robot_manipulator_pose.size(); i+=token)
    {        
        geometry_msgs::PoseStamped pose_msgs;
        pose_msgs.header.frame_id = "panda_link0";
        pose_msgs.header.seq = i;        
             
        Eigen::Matrix4d T_base2cam = m_vec_T_robot_manipulator_pose[i];

        Eigen::Quaterniond q(Eigen::Matrix3d(T_base2cam.block(0, 0, 3, 3)));
        pose_msgs.pose.orientation.x = q.x();
        pose_msgs.pose.orientation.y = q.y();
        pose_msgs.pose.orientation.z = q.z();
        pose_msgs.pose.orientation.w = q.w();
        pose_msgs.pose.position.x = T_base2cam(0, 3);
        pose_msgs.pose.position.y = T_base2cam(1, 3);
        pose_msgs.pose.position.z = T_base2cam(2, 3);

        robot_manipulator_path_msgs.poses.push_back(pose_msgs);
    }

    m_pub_robot_manipulator_path.publish(robot_manipulator_path_msgs);
}

void visualization::publish_object_pose_marker_from_world()
{
    visualization_msgs::MarkerArray marker_array;
    
    for (auto iter = m_vis_model.begin(); iter != m_vis_model.end(); iter++)
    {
        if (m_object_pose_uncertainty_norm_from_world[iter->first] != 1e+10)
        {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "panda_link0";
            marker.header.stamp = ros::Time();
            marker.ns = std::string("object") + std::to_string(iter->first);
            marker.id = iter->first;
            marker.type = visualization_msgs::Marker::MESH_RESOURCE;
            marker.action = visualization_msgs::Marker::ADD;

            Eigen::Matrix4d T_EE2Obj = m_T_object_pose_from_world[iter->first];
            T_EE2Obj.block(0, 3, 3, 1) /= 1000.;

            marker.pose.position.x = T_EE2Obj(0, 3);
            marker.pose.position.y = T_EE2Obj(1, 3);
            marker.pose.position.z = T_EE2Obj(2, 3);
            Eigen::Quaterniond quat(Eigen::Matrix3d(T_EE2Obj.block(0, 0, 3, 3)));
            marker.pose.orientation.x = quat.x();
            marker.pose.orientation.y = quat.y();
            marker.pose.orientation.z = quat.z();
            marker.pose.orientation.w = quat.w();
            marker.scale.x = 1;
            marker.scale.y = 1;
            marker.scale.z = 1;
            marker.mesh_resource = m_vis_model[iter->first];
            marker.mesh_use_embedded_materials = true;

            marker_array.markers.push_back(marker);
        }
    }

    m_pub_object_pose_marker.publish(marker_array);
}

void visualization::init_object_mesh_in_rviz()
{
    visualization_msgs::MarkerArray marker_array;
    // for (int i = 0; i < m_n_object; i++)
    for (auto iter = m_vis_model.begin(); iter != m_vis_model.end(); iter++)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "panda_link0";
        marker.header.stamp = ros::Time();
        marker.ns = std::string("object") + std::to_string(iter->first);
        marker.id = iter->first;
        marker.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::Marker::ADD;

        Eigen::Matrix4d T_EE2Obj = Eigen::Matrix4d::Identity();
        T_EE2Obj(2, 3) = -5000;
        T_EE2Obj.block(0, 3, 3, 1) /= 1000.;

        marker.pose.position.x = T_EE2Obj(0, 3);
        marker.pose.position.y = T_EE2Obj(1, 3);
        marker.pose.position.z = T_EE2Obj(2, 3);
        Eigen::Quaterniond quat(Eigen::Matrix3d(T_EE2Obj.block(0, 0, 3, 3)));
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
        marker.scale.x = 1;
        marker.scale.y = 1;
        marker.scale.z = 1;
        marker.mesh_resource = m_vis_model[iter->first];
        marker.mesh_use_embedded_materials = true;

        marker_array.markers.push_back(marker);
    }

    m_pub_object_pose_marker.publish(marker_array);
}

void visualization::publish_object_pose_image_from_cam()
{
    // for (int i = 0; i < m_n_object; i++)
    cv::Mat vis_camera_color = m_camera_color.clone();
    for (auto iter = m_vis_model.begin(); iter != m_vis_model.end(); iter++)
    {
        if (m_object_pose_uncertainty_norm_from_cam[iter->first] != 1e+10)
        {
            const int diameter = 0;
            const int min_x = 1;
            const int min_y = 2;
            const int min_z = 3;
            const int size_x = 4;
            const int size_y = 5;
            const int size_z = 6;

            model_info model_info_ = m_model_info[iter->first % 100];
            Eigen::MatrixXd corner_pts(4, 8);
            corner_pts.block(0, 0, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_), std::get<min_y>(model_info_), std::get<min_z>(model_info_), 1.);
            corner_pts.block(0, 1, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_), std::get<min_y>(model_info_), std::get<min_z>(model_info_) + std::get<size_z>(model_info_), 1.);
            corner_pts.block(0, 2, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_), std::get<min_y>(model_info_) + std::get<size_y>(model_info_), std::get<min_z>(model_info_), 1.);
            corner_pts.block(0, 3, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_), std::get<min_y>(model_info_) + std::get<size_y>(model_info_), std::get<min_z>(model_info_) + std::get<size_z>(model_info_), 1.);
            corner_pts.block(0, 4, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_) + std::get<size_x>(model_info_), std::get<min_y>(model_info_), std::get<min_z>(model_info_), 1.);
            corner_pts.block(0, 5, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_) + std::get<size_x>(model_info_), std::get<min_y>(model_info_), std::get<min_z>(model_info_) + std::get<size_z>(model_info_), 1.);
            corner_pts.block(0, 6, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_) + std::get<size_x>(model_info_), std::get<min_y>(model_info_) + std::get<size_y>(model_info_), std::get<min_z>(model_info_), 1.);
            corner_pts.block(0, 7, 4, 1) = Eigen::Vector4d(std::get<min_x>(model_info_) + std::get<size_x>(model_info_), std::get<min_y>(model_info_) + std::get<size_y>(model_info_), std::get<min_z>(model_info_) + std::get<size_z>(model_info_), 1.);                                                             

            Eigen::MatrixXd projected_corner_pts = m_camera_instricsic * m_T_object_pose_from_cam[iter->first].block(0, 0, 3, 4) * corner_pts;
            projected_corner_pts.block(0, 0, 1, 8).array() /= projected_corner_pts.block(2, 0, 1, 8).array();
            projected_corner_pts.block(1, 0, 1, 8).array() /= projected_corner_pts.block(2, 0, 1, 8).array();
            projected_corner_pts.block(2, 0, 1, 8).array() /= projected_corner_pts.block(2, 0, 1, 8).array();

            std::vector<cv::Point> polyline_pts;
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 0), projected_corner_pts(1, 0)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 1), projected_corner_pts(1, 1)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 3), projected_corner_pts(1, 3)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 2), projected_corner_pts(1, 2)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 0), projected_corner_pts(1, 0)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 4), projected_corner_pts(1, 4)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 6), projected_corner_pts(1, 6)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 2), projected_corner_pts(1, 2)));
            cv::polylines(vis_camera_color, polyline_pts, true, m_colors[iter->first % 100], 2);

            polyline_pts.clear();
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 5), projected_corner_pts(1, 5)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 4), projected_corner_pts(1, 4)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 6), projected_corner_pts(1, 6)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 7), projected_corner_pts(1, 7)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 5), projected_corner_pts(1, 5)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 1), projected_corner_pts(1, 1)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 3), projected_corner_pts(1, 3)));
            polyline_pts.push_back(cv::Point(projected_corner_pts(0, 7), projected_corner_pts(1, 7)));

            cv::polylines(vis_camera_color, polyline_pts, true, m_colors[iter->first % 100], 2);        
        }
    }

    sensor_msgs::Image img_msg;
    cv_bridge::CvImage img_bridge;
    std_msgs::Header header;
    header.stamp = ros::Time().fromSec(m_time_camera_color);
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, vis_camera_color);
    img_bridge.toImageMsg(img_msg);

    m_pub_object_pose_image.publish(img_msg);
}

void visualization::setup_drawing_param(std::string dataset)
{
    if (dataset.compare("ycbv") == 0)
    {
        for (int i=1; i<=21; i++)
        {
            m_model_info[i] = get_model_param(dataset, i);
        }
    }

    if (dataset.compare("tless") == 0)
    {        
        for (int i=1; i<=30; i++)
        {
            m_model_info[i] = get_model_param(dataset, i);
        }
    }
            
    m_colors[1] = cv::Scalar(0, 0, 0);
    m_colors[2] = cv::Scalar(255, 0, 0);
    m_colors[3] = cv::Scalar(0, 255, 0);
    m_colors[4] = cv::Scalar(0, 0, 255);
    m_colors[5] = cv::Scalar(255, 255, 0);
    m_colors[6] = cv::Scalar(255, 0, 255);
    m_colors[7] = cv::Scalar(0, 255, 255);
    m_colors[8] = cv::Scalar(255, 255, 255);
    m_colors[9] = cv::Scalar(128, 0, 0);
    m_colors[10] = cv::Scalar(0, 128, 0);
    m_colors[11] = cv::Scalar(0, 0, 128);
    m_colors[12] = cv::Scalar(128, 128, 0);
    m_colors[13] = cv::Scalar(128, 0, 128);
    m_colors[14] = cv::Scalar(0, 128, 128);
    m_colors[15] = cv::Scalar(128, 128, 128);
    m_colors[16] = cv::Scalar(255, 128, 0);
    m_colors[17] = cv::Scalar(255, 0, 128);
    m_colors[18] = cv::Scalar(128, 255, 0);
    m_colors[19] = cv::Scalar(0, 255, 128);
    m_colors[20] = cv::Scalar(128, 0, 255);
    m_colors[21] = cv::Scalar(0, 128, 255);        
    m_colors[22] = cv::Scalar(64, 0, 0);
    m_colors[23] = cv::Scalar(0, 64, 0);
    m_colors[24] = cv::Scalar(0, 0, 64);
    m_colors[25] = cv::Scalar(64, 64, 0);
    m_colors[26] = cv::Scalar(64, 0, 64);
    m_colors[27] = cv::Scalar(0, 64, 64);
    m_colors[28] = cv::Scalar(64, 255, 255);
    m_colors[29] = cv::Scalar(255, 64, 255);
    m_colors[30] = cv::Scalar(255, 128, 64);
}

void visualization::timer_callback(const ros::TimerEvent& event)
{    
    if (!m_init_mesh)
    {
        init_object_mesh_in_rviz();
    }    

    if (m_new_robot_manipulator_pose && m_new_camera_depth && m_new_camera_color)
    {                
        publish_sensored_pointcloud();             
        publish_robot_manipulator_path();          

        m_new_robot_manipulator_pose = false;
        m_new_camera_color = false;
        m_new_camera_depth = false;
    }

    if (m_new_object_pose_from_world)
    {        
        publish_object_pose_marker_from_world();        

        m_init_mesh = true;
        m_new_object_pose_from_world = false;
    }    
    
    if (m_new_object_pose_from_cam)
    {
        publish_object_pose_image_from_cam(); 

        m_new_object_pose_from_cam = false;
    }    
}

int main(int argc, char **argv)
{    
    ros::init(argc, argv, "visualization");
    
    ROS_INFO("[visualization] ROS Package Start");

    ros::NodeHandle nh;        
    visualization vis(nh);    
    
    ros::spin();
    // ros::Rate rate(30);
    // while (ros::ok())
    // {
    //     ros::spinOnce();

    //     vis.run();        

    //     rate.sleep();
    // }

  return 0;
}