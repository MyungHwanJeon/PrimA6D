#include "hand_eye_calibration_with_depth.h"

hand_eye_calibration::hand_eye_calibration(ros::NodeHandle nh): m_nh(nh)
{    
    m_T_axis_align = Eigen::Matrix4d::Zero();
    m_T_axis_align(0, 1) = -1;
    m_T_axis_align(1, 0) = 1;
    m_T_axis_align(2, 2) = 1;
    m_T_axis_align(3, 3) = 1;

    m_T_robot_manipulator_pose = Eigen::Matrix4d::Identity();        
    m_time_robot_manipulator_pose = 0.;
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

    setup_aruco_marker_center();

    m_T_hand_eye = Eigen::Matrix4d::Identity();

    m_data_index = 0;
    
    m_pub_sensored_pts = m_nh.advertise<sensor_msgs::PointCloud2>("/hand_eye_calibration/sensored_pts", 1);
    m_pub_aruco_pts = m_nh.advertise<sensor_msgs::PointCloud2>("/hand_eye_calibration/aruco_pts", 1);
    m_pub_refined_pts = m_nh.advertise<sensor_msgs::PointCloud2>("/hand_eye_calibration/refined_pts", 1);
    m_pub_refined_pcl = m_nh.advertise<sensor_msgs::PointCloud2>("/hand_eye_calibration/refined_pcl", 1);
    m_pub_camera_color = m_nh.advertise<sensor_msgs::Image>("/hand_eye_calibration/image_raw", 1);    
    
    m_sub_robot_manipulator_pose = m_nh.subscribe<franka_rpm_msgs::FrankaState>("/franka_rpm/franka_states", 1, &hand_eye_calibration::robot_manipulator_pose_callback, this);
        
    m_sub_camera_info = m_nh.subscribe<sensor_msgs::CameraInfo>("/camera/color/camera_info", 1, &hand_eye_calibration::camera_info_callback, this);    
    m_sub_camera_color = m_nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw", 1, &hand_eye_calibration::camera_color_callback, this);
    m_sub_camera_depth = m_nh.subscribe<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 1, &hand_eye_calibration::camera_depth_callback, this);      
}

void hand_eye_calibration::camera_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg)
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

void hand_eye_calibration::camera_color_callback(const sensor_msgs::Image::ConstPtr& msg)
{
    m_time_camera_color = msg->header.stamp.toSec();

    m_camera_color = cv::Mat(msg->height, msg->width, CV_8UC3, const_cast<unsigned char *>(msg->data.data()), msg->step).clone();

    m_new_camera_color = true;
}

void hand_eye_calibration::camera_depth_callback(const sensor_msgs::Image::ConstPtr& msg)
{
    m_time_camera_depth = msg->header.stamp.toSec();

    m_camera_depth = cv::Mat(msg->height, msg->width, CV_16UC1, const_cast<unsigned char *>(msg->data.data()), msg->step).clone();   

    m_new_camera_depth = true;
}

void hand_eye_calibration::robot_manipulator_pose_callback(const franka_rpm_msgs::FrankaState::ConstPtr& msg)
{
    m_time_robot_manipulator_pose = double(msg->header.stamp.sec) + double(msg->header.stamp.nsec) * 1e-9; //msg->stamp.toSec();

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m_T_robot_manipulator_pose(j, i) = msg->O_T_EE[i * 4 + j];
        }
    }
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

    m_new_robot_manipulator_pose = true;
}

void hand_eye_calibration::hand_eye_calibration_with_depth()
{        
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    cv::aruco::detectMarkers(m_camera_color, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    

    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(10, 7, 0.1, 0.02, dictionary);
    cv::aruco::refineDetectedMarkers(m_camera_color, board, markerCorners, markerIds, rejectedCandidates);
    cv::aruco::drawDetectedMarkers(m_camera_color, markerCorners, markerIds);

    sensor_msgs::Image img_msg;
    cv_bridge::CvImage img_bridge;
    std_msgs::Header header;
    header.stamp = ros::Time().fromSec(m_time_camera_color);
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, m_camera_color);
    img_bridge.toImageMsg(img_msg);
    m_pub_camera_color.publish(img_msg);



    double color[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {128, 128, 128}};
    double corner_pos[4][2] = {{-50, -50}, {-50, 50}, {50, 50}, {50, -50}};

    PointCloud::Ptr sensored_pts(new PointCloud);
    PointCloud::Ptr aruco_pts(new PointCloud);    
    PointCloud::Ptr sensored_pts_vis(new PointCloud);
    PointCloud::Ptr aruco_pts_vis(new PointCloud);  
    for (int i = 0; i < markerCorners.size(); i++)    
    {        
        PointT corner_p[4];
        for (int j = 0; j < 4; j++)
        {
            int m = markerCorners[i][j].y;
            int n = markerCorners[i][j].x;
            ushort d = m_camera_depth.ptr<ushort>(m)[n];
            
            corner_p[j].z = double(d);
            corner_p[j].x = ((n - m_camera_instricsic(0, 2)) * corner_p[j].z / m_camera_instricsic(0, 0));
            corner_p[j].y = ((m - m_camera_instricsic(1, 2)) * corner_p[j].z / m_camera_instricsic(1, 1));
            corner_p[j].x /= 1000.;
            corner_p[j].y /= 1000.;
            corner_p[j].z /= 1000.;
            corner_p[j].r = 255;//color[j][0];
            corner_p[j].g = 0;//color[j][1];
            corner_p[j].b = 0;//color[j][2];            
        }

        double corner_distance[3] = {0.,};
        for (int j = 0; j < 3; j++)
        {
            double diff_x = corner_p[j].x - corner_p[j+1].x;
            double diff_y = corner_p[j].y - corner_p[j+1].y;
            double diff_z = corner_p[j].z - corner_p[j+1].z;
            corner_distance[j] = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
        }
        double threshold = 0.097;
        if (corner_distance[0] > threshold && corner_distance[1] > threshold && corner_distance[2] > threshold)
        {
            m_vec_T_robot_manipulator_pose.push_back(m_T_robot_manipulator_pose);

            for (int j = 0; j < 4; j++)
            {
                sensored_pts->points.push_back(corner_p[j]);
                sensored_pts_vis->points.push_back(corner_p[j]);
            }

            std::array<double, 3> center = m_aruco_marker_center[markerIds[i]];
            for (int j = 0; j < 4; j++)
            {
                PointT p;
                p.x = center[0] + corner_pos[j][0];
                p.y = center[1] + corner_pos[j][1];
                p.z = center[2];
                p.x /= 1000.;
                p.y /= 1000.;
                p.z /= 1000.;
                p.r = 0;//color[j][0];
                p.g = 0;//color[j][1];
                p.b = 255;//color[j][2];
                aruco_pts->points.push_back(p);
                aruco_pts_vis->points.push_back(p);
            }
        }
    }
    
    pcl::transformPointCloud(*sensored_pts, *sensored_pts, m_T_axis_align);      

    Eigen::Matrix4d T_robot_manipulator_pose_inv = m_T_robot_manipulator_pose.inverse();
    pcl::transformPointCloud(*aruco_pts, *aruco_pts, T_robot_manipulator_pose_inv);

    if (m_save)
    {
        for (int i = 0; i < sensored_pts->points.size(); i++)
        {
            std::array<double, 3> sensored_pts_ = {sensored_pts->points[i].x, sensored_pts->points[i].y, sensored_pts->points[i].z};
            m_vec_sensored_pts.push_back(sensored_pts_);

            std::array<double, 3> aruco_pts_ = {aruco_pts->points[i].x, aruco_pts->points[i].y, aruco_pts->points[i].z};
            m_vec_aruco_pts.push_back(aruco_pts_);
        }

        {
            std::ofstream file;
            file.open(std::string("/home/oem/Desktop/Multi_Object_pose_Optimization/src/pose_optimization/hand_eye_calibration_data/") + std::to_string(m_data_index) + std::string(".csv"));
            file << std::setprecision(20);
            for (int i = 0; i < m_vec_sensored_pts.size(); i++)
            {
                file << m_vec_sensored_pts[i][0] << ","
                     << m_vec_sensored_pts[i][1] << ","
                     << m_vec_sensored_pts[i][2] << ","
                     << m_vec_aruco_pts[i][0] << ","
                     << m_vec_aruco_pts[i][1] << ","
                     << m_vec_aruco_pts[i][2] << "\n";
            }
            file.close();
            m_save = false;
            m_data_index++;
        }

        Eigen::VectorXd sensored_pts_mean = Eigen::VectorXd::Zero(3);
        Eigen::VectorXd aruco_pts_mean = Eigen::VectorXd::Zero(3);
        for (int i = 0; i < 3; i++)
        {
            double sensored_sum = 0;
            double aruco_sum = 0;
            for (int j = 0; j < m_vec_sensored_pts.size(); j++)
            {
                sensored_sum += m_vec_sensored_pts[j][i];
                aruco_sum += m_vec_aruco_pts[j][i];
            }
            sensored_pts_mean[i] = sensored_sum / m_vec_sensored_pts.size();
            aruco_pts_mean[i] = aruco_sum / m_vec_aruco_pts.size();
        }

        Eigen::MatrixXd sensored_mat(3, m_vec_sensored_pts.size());
        Eigen::MatrixXd aruco_mat(3, m_vec_sensored_pts.size());
        for (int i = 0; i < m_vec_sensored_pts.size(); i++)
        {
            sensored_mat(0, i) = m_vec_sensored_pts[i][0] - sensored_pts_mean[0];
            sensored_mat(1, i) = m_vec_sensored_pts[i][1] - sensored_pts_mean[1];
            sensored_mat(2, i) = m_vec_sensored_pts[i][2] - sensored_pts_mean[2];

            aruco_mat(0, i) = m_vec_aruco_pts[i][0] - aruco_pts_mean[0];
            aruco_mat(1, i) = m_vec_aruco_pts[i][1] - aruco_pts_mean[1];
            aruco_mat(2, i) = m_vec_aruco_pts[i][2] - aruco_pts_mean[2];
        }

        Eigen::MatrixXd H = sensored_mat * aruco_mat.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullV | Eigen::ComputeFullU);

        Eigen::MatrixXd R = svd.matrixV() * svd.matrixU().transpose();

        if (R.determinant() < 0)
        {
            Eigen::Matrix3d V = svd.matrixV();
            V(2, 0) *= -1;
            V(2, 1) *= -1;
            V(2, 2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        Eigen::VectorXd t = -R * sensored_pts_mean + aruco_pts_mean;
        m_T_hand_eye.block(0, 0, 3, 3) = R;
        m_T_hand_eye.block(0, 3, 3, 1) = t;

        double rmse = 0.;
        for (int i=0; i<m_vec_sensored_pts.size(); i++)
        {
            int robot_manipulator_idx = int(i / 4);
            Eigen::VectorXd transformed_sensored_pt = Eigen::VectorXd::Zero(4);
            transformed_sensored_pt << m_vec_sensored_pts[i][0], m_vec_sensored_pts[i][1], m_vec_sensored_pts[i][2], 1;
            transformed_sensored_pt =  m_vec_T_robot_manipulator_pose[robot_manipulator_idx] * m_T_hand_eye * transformed_sensored_pt;            
            Eigen::VectorXd aruco_pt = Eigen::VectorXd::Zero(4);
            aruco_pt << m_vec_aruco_pts[i][0], m_vec_aruco_pts[i][1], m_vec_aruco_pts[i][2], 1;
            aruco_pt = m_vec_T_robot_manipulator_pose[robot_manipulator_idx] * aruco_pt;
 
            Eigen::VectorXd residual = transformed_sensored_pt - aruco_pt;            
            rmse += residual.norm();
        }
        rmse = rmse / m_vec_sensored_pts.size();
        std::cout << m_T_hand_eye << std::endl;
        std::cout << rmse << std::endl;
    }

  
// 0.999961 -0.00655526  0.00593217    0.092329
//    0.006662    0.999813  -0.0181578  0.00308126
// -0.00581203   0.0181966    0.999818   0.0622155
//           0           0           0           1

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

    Eigen::Matrix4d T_Base2Cam = m_T_robot_manipulator_pose * m_T_hand_eye * m_T_axis_align;      
    pcl::transformPointCloud(*sensored_cloud, *sensored_cloud, T_Base2Cam);    
    pcl::transformPointCloud(*sensored_pts_vis, *sensored_pts_vis, T_Base2Cam);    
    // pcl::transformPointCloud(*aruco_pts_vis, *aruco_pts_vis, T_Base2Cam);    
 
    sensor_msgs::PointCloud2 cloudmsg;    
    pcl::toROSMsg(*sensored_cloud, cloudmsg);
    cloudmsg.header.frame_id = "panda_link0";
    m_pub_refined_pcl.publish(cloudmsg);
    
    pcl::toROSMsg(*sensored_pts_vis, cloudmsg);
    cloudmsg.header.frame_id = "panda_link0";
    m_pub_refined_pts.publish(cloudmsg);
   
    pcl::toROSMsg(*aruco_pts_vis, cloudmsg);
    cloudmsg.header.frame_id = "panda_link0";
    m_pub_aruco_pts.publish(cloudmsg);
}

void hand_eye_calibration::setup_aruco_marker_center()
{
    double offset_x = 109.;
    double offset_y = 0.;    

    Eigen::Matrix4d offset_mat = Eigen::Matrix4d::Identity();
    offset_mat << 1.00124200793653, 0.000688612697346, 0., 0.698299960716213, 
                0., 1.0, 0, 0.,
                0., 0., 1.0, 0,
                0., 0., 0., 1.0;

    for (int i = 0; i < 7; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            Eigen::VectorXd cp = Eigen::VectorXd::Zero(4);
            cp << offset_x + 15 + 50 + 120 * i, -540 + 120 * j, 10, 1;
            cp = offset_mat * cp;
        
            std::array<double, 3> cp_ = {cp[0], cp[1], cp[2]};            
            m_aruco_marker_center.push_back(cp_);
        }
    }
}


void hand_eye_calibration::run()
{
    if (m_new_robot_manipulator_pose && m_new_camera_depth && m_new_camera_color)
    {                
        hand_eye_calibration_with_depth();     

        m_new_robot_manipulator_pose = false;
        m_new_camera_color = false;
        m_new_camera_depth = false;                  
    }
}




int main(int argc, char **argv)
{    
    ros::init(argc, argv, "hand_eye_calibration");
    
    ROS_INFO("[hand_eye_calibration] ROS Package Start");

    ros::NodeHandle nh;        
    hand_eye_calibration hec(nh);    

    // ros::spin();
    ros::Rate rate(30);
    while (ros::ok())
    {
        if ('s' == getch()) hec.m_save = true;            
        
        ros::spinOnce();
        hec.run();        
        rate.sleep();
    }

  return 0;
}