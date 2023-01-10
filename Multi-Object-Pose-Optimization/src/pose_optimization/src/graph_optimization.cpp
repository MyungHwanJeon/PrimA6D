#include "graph_optimization.h"

graph_optimization::graph_optimization(ros::NodeHandle nh): m_nh(nh)
{
    global_mutex = new std::mutex();

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 10;
    parameters.relinearizeSkip = 1;
    m_isam = new ISAM2(parameters);         

    m_T_axis_align = Eigen::Matrix4d::Identity();

    m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
    m_T_sym_obj_constraint(2, 3) = 10;    

    m_camera_instricsic = Eigen::Matrix3d::Identity();
    m_camera_distortion = Eigen::VectorXd::Zero(5);
    m_new_camera_info = false;

    std::string dataset;
    m_nh.getParam("/dataset", dataset);         

    XmlRpc::XmlRpcValue dataset_object;
    m_nh.getParam("/dataset_object", dataset_object);    
    for (int32_t i = 0; i < dataset_object.size(); i++)
    {                
        ObjectInfo object_info;
        object_info.index = static_cast<int>(dataset_object[i]["index"]);
        object_info.name = static_cast<std::string>(dataset_object[i]["name"]);
        object_info.threshold = static_cast<double>(dataset_object[i]["threshold"]);
        // object_info.is_symmetry = static_cast<bool>(dataset_object[i]["symmetry"]);
        object_info.model_path = ros::package::getPath("pose_optimization") + static_cast<std::string>(dataset_object[i]["model_path"]);
        object_info.n_instance = static_cast<int>(dataset_object[i]["n_instance"]);
        object_info.model_spec = get_model_param(dataset, object_info.index);

        m_object_info[static_cast<int>(dataset_object[i]["index"])] = object_info;
    }    
    
    m_nh.getParam("robot_base_weight", m_weight.robot_base);
    m_nh.getParam("robot_manipulator_weight", m_weight.robot_manipulator);
    m_nh.getParam("object_weight", m_weight.object);
    m_nh.getParam("object_icp_weight", m_weight.object_icp);

    m_nh.getParam("seq_index", m_seq_index);
    m_nh.getParam("save_data", m_save_data);   
    m_save_path = ros::package::getPath("pose_optimization");    
    m_nh.getParam("seq_name", m_seq_name);     
    
    m_nh.getParam("img_width", m_img_size[0]);  
    m_nh.getParam("img_height", m_img_size[1]);  

    m_nh.getParam("add_noise", m_add_noise);     
    m_nh.getParam("tranalation_noise", m_tranalation_noise);     
    m_nh.getParam("rotation_noise", m_rotation_noise);

    m_nh.getParam("use_robot_franka", m_use_robot_franka);
    if (m_use_robot_franka)
    {
        m_T_axis_align = Eigen::Matrix4d::Zero();
        m_T_axis_align(0, 1) = -1;
        m_T_axis_align(1, 0) = 1;
        m_T_axis_align(2, 2) = 1;
        m_T_axis_align(3, 3) = 1;
    }    

    mkdir((m_save_path + std::string("/data/")).c_str(), 0777);

    for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
    {
        m_icp_thread[iter->first].object_idx = iter->first;
        m_icp_thread[iter->first].renderer.init(m_img_size[0], m_img_size[1]);
        m_icp_thread[iter->first].renderer.addObject(iter->first, m_object_info[iter->first].model_path);

        m_icp_thread[iter->first].active = true; 
        m_icp_thread[iter->first].thd = std::thread(&graph_optimization::Thread_ICP, this, iter->first);
    }

    m_object_pose_data.active = true; 
    m_object_pose_data.thd = std::thread(&graph_optimization::add_factor, this);

    m_first_call = true;

    m_pub_refined_object_pose_from_world = m_nh.advertise<vision_msgs::Detection2DArray>("/pose_estimation/graph_opt/object/pose_from_world", 1);
    m_pub_refined_object_pose_from_cam = m_nh.advertise<vision_msgs::Detection2DArray>("/pose_estimation/graph_opt/object/pose_from_cam", 1);
    m_pub_refined_robot_manipulator_path = m_nh.advertise<nav_msgs::Path>("/pose_estimation/graph_opt/robot_arm/ee/path", 1);
    m_pub_refined_robot_manipulator_pose = m_nh.advertise<geometry_msgs::PoseStamped>("/pose_estimation/graph_opt/robot_arm/ee/pose", 1);        

    m_sub_object_pose = m_nh.subscribe<vision_msgs::Detection2DArray>("/pose_estimation/PrimA6D/detection2D_array", 1000, boost::bind(&graph_optimization::object_pose_callback, this, _1));
    m_sub_robot_manipulator_pose = m_nh.subscribe<franka_rpm_msgs::FrankaState>("/franka_rpm/franka_states", 1000, boost::bind(&graph_optimization::robot_manipulator_pose_callback, this, _1));
        
    m_sub_camera_info = m_nh.subscribe<sensor_msgs::CameraInfo>("/pose_estimation/PrimA6D/camera_info", 1000, boost::bind(&graph_optimization::camera_info_callback, this, _1));
    m_sub_camera_color = m_nh.subscribe<sensor_msgs::Image>("/pose_estimation/PrimA6D/color_raw", 1000, boost::bind(&graph_optimization::camera_color_callback, this, _1));
    m_sub_camera_depth = m_nh.subscribe<sensor_msgs::Image>("/pose_estimation/PrimA6D/depth_raw", 1000, boost::bind(&graph_optimization::camera_depth_callback, this, _1));    

    m_timer = nh.createTimer(ros::Duration(0.1), boost::bind(&graph_optimization::optimization, this, _1));

    std::cout << "[System] init " << std::endl;    
}

graph_optimization::~graph_optimization()
{
    for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
    {
        m_icp_thread[iter->first].active = false;
        m_icp_thread[iter->first].cv.notify_all();        
        if(m_icp_thread[iter->first].thd.joinable()) m_icp_thread[iter->first].thd.detach();        
        while(m_icp_thread[iter->first].thd.joinable()) m_icp_thread[iter->first].thd.join();   
    }

    m_object_pose_data.active = false;
    m_object_pose_data.cv.notify_all();    
    if(m_object_pose_data.thd.joinable()) m_object_pose_data.thd.detach();    
    while(m_object_pose_data.thd.joinable()) m_object_pose_data.thd.join();          
}

void graph_optimization::camera_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    // m_camera_instricsic(0, 0) = msg->K[0];
    // m_camera_instricsic(0, 2) = msg->K[2];
    // m_camera_instricsic(1, 1) = msg->K[4];
    // m_camera_instricsic(1, 2) = msg->K[5];

    // m_camera_distortion[0] = msg->D[0];
    // m_camera_distortion[1] = msg->D[1];
    // m_camera_distortion[2] = msg->D[2];
    // m_camera_distortion[3] = msg->D[3];
    // m_camera_distortion[4] = msg->D[4];

    // m_new_camera_info = true;    
    
    m_camera_info_data.push_back(make_pair(msg->header.stamp.toSec(), *msg));
    m_camera_info_data.sort();
    m_camera_info_data.unique();
}

void graph_optimization::camera_color_callback(const sensor_msgs::Image::ConstPtr& msg)
{            
    m_camera_color_data.push_back(make_pair(msg->header.stamp.toSec(), *msg));
    m_camera_color_data.sort();
    m_camera_color_data.unique();
}

void graph_optimization::camera_depth_callback(const sensor_msgs::Image::ConstPtr& msg)
{        
    m_camera_depth_data.push_back(make_pair(msg->header.stamp.toSec(), *msg));     
    m_camera_depth_data.sort();     
    m_camera_depth_data.unique();    
}

void graph_optimization::object_pose_callback(const vision_msgs::Detection2DArray::ConstPtr& msg)
{    
    m_object_pose_data.push_back(make_pair(msg->header.stamp.toSec(), *msg));         
    m_object_pose_data.sort(); 
    m_object_pose_data.unique();    
    m_object_pose_data.cv.notify_all();    
}

void graph_optimization::robot_manipulator_pose_callback(const franka_rpm_msgs::FrankaState::ConstPtr& msg)
{        
    m_robot_manipulator_data.push_back(make_pair(msg->header.stamp.toSec(), *msg));          
    m_robot_manipulator_data.sort();     
    m_robot_manipulator_data.unique();   

    if (m_first_call)
    {
        Eigen::Matrix4d T_base2ee = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d T_ee2cam = Eigen::Matrix4d::Identity();

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m_T_cam_to_gripper(j, i) = msg->CAM_T_GRIPPER[i * 4 + j];     
                T_base2ee(j, i) = msg->O_T_EE[i * 4 + j];      
                T_ee2cam(j, i) = msg->EE_T_CAM[i * 4 + j];               
            }
        }
        m_T_cam_to_gripper.block(0, 3, 3, 1) *= 1000.;
        T_base2ee.block(0, 3, 3, 1) *= 1000.;
        T_ee2cam.block(0, 3, 3, 1) *= 1000.;
        
        m_first_call = false;
    }    
}

void graph_optimization::publish_refined_object_pose_from_world()
{
    vision_msgs::Detection2DArray detection_array_msg;
    detection_array_msg.header.frame_id = "panda_link0";
    detection_array_msg.header.stamp = ros::Time::now();
    
    for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
    {        
        auto search_l = m_factor_counter.find(std::string("l") + std::to_string(iter->first));
        if (search_l != m_factor_counter.end())        
        {
            auto search_c = m_factor_counter.find(std::string("c") + std::to_string(iter->first));   
            auto search_icp = m_factor_counter.find(std::string("l_icp") + std::to_string(iter->first));             
            bool is_symmetry = false;             
            if (search_c != m_factor_counter.end())        
            {                                
                if ((float(search_c->second) / search_l->second) > 0.6)    is_symmetry = true;                
            }            

            // if (is_symmetry)
            // {
            //     std::cout << "id: " << iter->first << ", sym: 1, ratio: " << float(search_c->second) / search_l->second << " " <<
            //     search_l->second << " " << search_c->second << " " << search_icp->second <<std::endl; 
            // }
            // else
            // {
            //     std::cout << "id: " << iter->first << ", sym: 0, ratio: " << float(search_c->second) / search_l->second << " " <<
            //     search_l->second << " " << search_c->second << " " << search_icp->second <<std::endl; 
            // }
            

            if (!m_optimized_estimate.exists(symbol('l', iter->first)))   continue;   
            gtsam::Pose3 T_Base2Obj = m_optimized_estimate.at<Pose3>(symbol('l', iter->first));            
            
            if (is_symmetry)
            {
                gtsam::Pose3 T_obj_center = m_optimized_estimate.at<Pose3>(symbol('l', iter->first));
                gtsam::Pose3 T_obj_rot_axis = m_optimized_estimate.at<Pose3>(symbol('c', iter->first));

                gtsam::Point3 rot_axis = (T_obj_rot_axis.translation() - T_obj_center.translation());
                rot_axis = normalize(rot_axis);

                gtsam::Matrix33 axis_rotation = Eigen::Matrix3d::Identity();
                // if (rot_axis[2] >= 0)
                // std::cout << "axis : " << rot_axis << std::endl;
                // if (rot_axis[0] < 0)
                // {
                    // std::cout << "aaaaaaaaaaaaaaaaaaa" << std::endl;
                    if (iter->first == 20)
                    {
                        axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(-1, 0, 0), rot_axis);
                    }
                    else if (iter->first == 17)
                    {
                        axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(1, 0, 0), rot_axis);
                    }
                    else
                    {
                        axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, 1), rot_axis);
                    }
                    // axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, 1), rot_axis);
                // }   
                // else
                // {
                    // std::cout << "bbbbbbbbbbbbbbbb" << std::endl;
                    // if (iter->first == 20)
                    // {
                    //     axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(1, 0, 0), rot_axis);
                    // }
                    // else
                    // {
                    //     axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, -1), rot_axis);
                    // }                    
                    // // axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, -1), rot_axis);
                    // Eigen::Matrix3d reverse;
                    // reverse = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
                    // axis_rotation = axis_rotation * reverse;
                // }                             

                T_Base2Obj = gtsam::Pose3(gtsam::Rot3(axis_rotation), gtsam::Point3(T_obj_center.translation()));
            }                        

            vision_msgs::Detection2D detection_msg;

            detection_msg.header.frame_id = "panda_link0";
            detection_msg.header.stamp = ros::Time::now();;
            
            vision_msgs::ObjectHypothesisWithPose pose_msg;
            pose_msg.id = iter->first;
            pose_msg.pose.pose.position.x = T_Base2Obj.translation().x();
            pose_msg.pose.pose.position.y = T_Base2Obj.translation().y();
            pose_msg.pose.pose.position.z = T_Base2Obj.translation().z();

            gtsam::Quaternion q = T_Base2Obj.rotation().toQuaternion();   
            pose_msg.pose.pose.orientation.x = q.x();
            pose_msg.pose.pose.orientation.y = q.y();
            pose_msg.pose.pose.orientation.z = q.z();
            pose_msg.pose.pose.orientation.w = q.w();

            Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
            if (m_isam->valueExists(symbol('l', iter->first)) && m_object_info[iter->first].is_factor)    cov = m_isam->marginalCovariance(symbol('l', iter->first));
            for (int ii=0; ii<6; ii++)
            {
                for (int jj=0; jj<6; jj++)
                {
                    pose_msg.pose.covariance[ii + jj*6] = cov(ii, jj);
                }
            }

            detection_msg.results.push_back(pose_msg);
            detection_array_msg.detections.push_back(detection_msg);
        }
    }
    m_pub_refined_object_pose_from_world.publish(detection_array_msg);
}

void graph_optimization::publish_refined_object_pose_from_cam()
{
    vision_msgs::Detection2DArray detection_array_msg;
    detection_array_msg.header.frame_id = "panda_link0";
    detection_array_msg.header.stamp = ros::Time::now();

    auto search = m_factor_counter.find(std::string("x"));
    if (search == m_factor_counter.end())   return;    

    if (!m_optimized_estimate.exists(symbol('x', search->second-1)))   return;       
    gtsam::Pose3 T_Base2Cam = m_optimized_estimate.at<Pose3>(symbol('x', search->second-1));

    for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
    {              
        auto search_l = m_factor_counter.find(std::string("l") + std::to_string(iter->first));
        if (search_l != m_factor_counter.end())
        {
            auto search_c = m_factor_counter.find(std::string("c") + std::to_string(iter->first));                    
            bool is_symmetry = false;             
            if (search_c != m_factor_counter.end())        
            {                
                if ((float(search_c->second) / search_l->second) > 0.7)    is_symmetry = true;                
            }            

            if (!m_optimized_estimate.exists(symbol('l', iter->first)))   continue;   
            gtsam::Pose3 T_Base2Obj = m_optimized_estimate.at<Pose3>(symbol('l', iter->first));

            if (is_symmetry)
            {
                gtsam::Pose3 T_obj_center = m_optimized_estimate.at<Pose3>(symbol('l', iter->first));
                gtsam::Pose3 T_obj_rot_axis = m_optimized_estimate.at<Pose3>(symbol('c', iter->first));

                gtsam::Point3 rot_axis = (T_obj_rot_axis.translation() - T_obj_center.translation());
                rot_axis = normalize(rot_axis);
                                
                gtsam::Matrix33 axis_rotation = Eigen::Matrix3d::Identity();
                if (rot_axis[2] >= 0)
                {
                    axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, 1), rot_axis);
                }   
                else
                {
                    axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, -1), rot_axis);
                    Eigen::Matrix3d reverse;
                    reverse = Eigen::AngleAxisd(M_PI,Eigen:: Vector3d::UnitX()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
                    axis_rotation = axis_rotation * reverse;
                }              

                T_Base2Obj = gtsam::Pose3(gtsam::Rot3(axis_rotation), gtsam::Point3(T_obj_center.translation()));
            }
            gtsam::Pose3 T_Cam2Obj = gtsam::Pose3(m_T_axis_align.inverse()) * (T_Base2Cam.inverse() * T_Base2Obj);

            vision_msgs::Detection2D detection_msg;

            detection_msg.header.frame_id = "panda_link0";
            detection_msg.header.stamp = ros::Time::now();
            
            vision_msgs::ObjectHypothesisWithPose pose_msg;
            pose_msg.id = iter->first;
            pose_msg.pose.pose.position.x = T_Cam2Obj.translation().x();
            pose_msg.pose.pose.position.y = T_Cam2Obj.translation().y();
            pose_msg.pose.pose.position.z = T_Cam2Obj.translation().z();

            gtsam::Quaternion q = T_Cam2Obj.rotation().toQuaternion();   
            pose_msg.pose.pose.orientation.x = q.x();
            pose_msg.pose.pose.orientation.y = q.y();
            pose_msg.pose.pose.orientation.z = q.z();
            pose_msg.pose.pose.orientation.w = q.w();

            Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
            if (m_isam->valueExists(symbol('l', iter->first)) && m_object_info[iter->first].is_factor)    cov = m_isam->marginalCovariance(symbol('l', iter->first));
            for (int ii=0; ii<6; ii++)
            {
                for (int jj=0; jj<6; jj++)
                {
                    pose_msg.pose.covariance[ii + jj*6] = cov(ii, jj);
                }
            }
            
            detection_msg.results.push_back(pose_msg);
            detection_array_msg.detections.push_back(detection_msg);
        }
    }
    m_pub_refined_object_pose_from_cam.publish(detection_array_msg);
}

void graph_optimization::publish_refined_robot_manipulator_pose()
{
    auto search = m_factor_counter.find(std::string("x"));
    if (search == m_factor_counter.end())   return;    

    if (!m_optimized_estimate.exists(symbol('x', search->second-1)))   return; 
    gtsam::Pose3 T_base2cam = m_optimized_estimate.at<Pose3>(symbol('x', search->second - 1));    

    geometry_msgs::PoseStamped refined_robot_manipulator_pose_msgs;

    refined_robot_manipulator_pose_msgs.header.frame_id = "panda_link0";
    refined_robot_manipulator_pose_msgs.header.stamp = ros::Time::now();

    gtsam::Quaternion q = T_base2cam.rotation().toQuaternion();
    refined_robot_manipulator_pose_msgs.pose.orientation.x = q.x();
    refined_robot_manipulator_pose_msgs.pose.orientation.y = q.y();
    refined_robot_manipulator_pose_msgs.pose.orientation.z = q.z();
    refined_robot_manipulator_pose_msgs.pose.orientation.w = q.w();
    refined_robot_manipulator_pose_msgs.pose.position.x = T_base2cam.translation().x() / 1000.;
    refined_robot_manipulator_pose_msgs.pose.position.y = T_base2cam.translation().y() / 1000.;
    refined_robot_manipulator_pose_msgs.pose.position.z = T_base2cam.translation().z() / 1000.;

    m_pub_refined_robot_manipulator_pose.publish(refined_robot_manipulator_pose_msgs);
}

void graph_optimization::publish_refined_robot_manipulator_path()
{        
    nav_msgs::Path refined_robot_manipulator_path_msgs;

    refined_robot_manipulator_path_msgs.header.frame_id = "panda_link0";
    refined_robot_manipulator_path_msgs.header.stamp = ros::Time::now();

    auto search = m_factor_counter.find(std::string("x"));
    if (search == m_factor_counter.end())   return; 

    int token = int(search->second / 500) + 1;
    for (int i=0; i<search->second; i+=token)
    {
        if (!m_optimized_estimate.exists(symbol('x', i)))   continue;   

        geometry_msgs::PoseStamped pose_msgs;
        pose_msgs.header.frame_id = "panda_link0";
        pose_msgs.header.seq = i;        
             
        gtsam::Pose3 T_base2cam = m_optimized_estimate.at<Pose3>(symbol('x', i)) * gtsam::Pose3(m_T_cam_to_gripper);

        gtsam::Quaternion q = T_base2cam.rotation().toQuaternion();
        pose_msgs.pose.orientation.x = q.x();
        pose_msgs.pose.orientation.y = q.y();
        pose_msgs.pose.orientation.z = q.z();
        pose_msgs.pose.orientation.w = q.w();
        pose_msgs.pose.position.x = T_base2cam.translation().x() / 1000.;
        pose_msgs.pose.position.y = T_base2cam.translation().y() / 1000.;
        pose_msgs.pose.position.z = T_base2cam.translation().z() / 1000.;

        refined_robot_manipulator_path_msgs.poses.push_back(pose_msgs);
    }

    m_pub_refined_robot_manipulator_path.publish(refined_robot_manipulator_path_msgs);
}

Eigen::Matrix3d graph_optimization::find_rot_btw_two_vectors(Eigen::Vector3d v1, Eigen::Vector3d v2)
{
    Eigen::Vector3d v_c = v1.cross(v2);
    double v_d = v1.dot(v2);
    Eigen::Matrix3d v_c_mat = Eigen::Matrix3d::Identity();
    v_c_mat << 0, -v_c[2], v_c[1], v_c[2], 0, -v_c[0], -v_c[1], v_c[0], 0;

    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity() + v_c_mat + v_c_mat * v_c_mat * (1 / 1 + v_d);

    return rot;
}

double graph_optimization::rotation_error(Eigen::Matrix3d r1, Eigen::Matrix3d r2)
{
    double error_cos = ((r1 * r2.inverse()).trace() - 1.0) * 0.5;
    error_cos = std::min(1.0, std::max(-1.0, error_cos));

    double rotation_error = acos(error_cos) * (180. / PI);

    return rotation_error;
}

void graph_optimization::add_factor()
{    
    std::map<int, Eigen::Matrix3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix3d>>> pre_R_robot_manipulator_pose;
    std::map<int, Eigen::Matrix3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix3d>>> pre_R_object_pose;

    std::map<int, Eigen::Matrix4d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Matrix4d>>> T_object_pose;
    std::map<int, Eigen::VectorXd, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::VectorXd>>> object_pose_uncertainty;
    std::map<int, double> object_pose_uncertainty_norm;    
    std::map<int, Eigen::VectorXd, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::VectorXd>>> object_bbox;    
    std::map<int, cv::Mat> object_mask;    

    std::map<int, int> n_invalid_object; 

    for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
    { 
        pre_R_robot_manipulator_pose[iter->first] = Eigen::Matrix3d::Identity();
        pre_R_object_pose[iter->first] = Eigen::Matrix3d::Identity();

        T_object_pose[iter->first] = Eigen::Matrix4d::Identity();
        object_pose_uncertainty[iter->first] = Eigen::VectorXd::Zero(3);
        object_pose_uncertainty_norm[iter->first] = 0.;
        object_bbox[iter->first] = Eigen::VectorXd::Zero(4);
        object_mask[iter->first] = cv::Mat(m_img_size[0], m_img_size[1], CV_8UC1);

        n_invalid_object[iter->first] = 0;
    }

    std::cout << "[add factor thread] start" << std::endl;

    while (1)
    {
        std::unique_lock<std::mutex> ul(m_object_pose_data.mtx);
        m_object_pose_data.cv.wait(ul);
        if (m_object_pose_data.active == false)
        {
            global_mutex->lock();
            std::cout << "add factor thread finish" << std::endl;
            global_mutex->unlock();
            return;
        }
        ul.unlock();

        int maximum_size = 10;
        if (m_object_pose_data.size() > maximum_size)
        {            
            int delete_range = 0;
            for (int i = m_robot_manipulator_data.size() - 1; i >= 0; i--)
            {
                if (m_object_pose_data[m_object_pose_data.size() - 1].first - m_robot_manipulator_data[i].first > 0)
                {
                    delete_range = i;
                    break;
                }
            }
            if (m_robot_manipulator_data.size() > 0)
                m_robot_manipulator_data.erase(0, delete_range);

            for (int i = m_camera_color_data.size() - 1; i >= 0; i--)
            {
                if (m_object_pose_data[m_object_pose_data.size() - 1].first - m_camera_color_data[i].first > 0)
                {
                    delete_range = i;
                    break;
                }
            }
            if (m_camera_color_data.size() > 0)
                m_camera_color_data.erase(0, delete_range);

            for (int i = m_camera_depth_data.size() - 1; i >= 0; i--)
            {
                if (m_object_pose_data[m_object_pose_data.size() - 1].first - m_camera_depth_data[i].first > 0)
                {
                    delete_range = i;
                    break;
                }
            }
            if (m_camera_depth_data.size() > 0)
                m_camera_depth_data.erase(0, delete_range);

            for (int i = m_camera_info_data.size() - 1; i >= 0; i--)
            {
                if (m_object_pose_data[m_object_pose_data.size() - 1].first - m_camera_info_data[i].first > 0)
                {
                    delete_range = i;
                    break;
                }
            }
            if (m_camera_info_data.size() > 0)
                m_camera_info_data.erase(0, delete_range);

            m_object_pose_data.clear();

            std::cout << "[add factor thread] Clear Sensor Data" << std::endl;
        }

        // std::cout << "::::::::::::::" << m_object_pose_data.size() << " " 
        //     << m_robot_manipulator_data.size() << " " 
        //     << m_camera_color_data.size() << " " 
        //     << m_camera_depth_data.size() << " " 
        //     << m_camera_info_data.size() << std::endl;

        while (!m_object_pose_data.empty() && m_object_pose_data.size() <= maximum_size)
        {            
            if (m_robot_manipulator_data.size() < 1)  break;            
 
            auto selected_data_object_pose = m_object_pose_data.front();   
            int index_object_pose = 0;                            
            auto end_data_robot_manipulator = m_robot_manipulator_data.back(); 
            // int size_data_robot_manipulator = m_robot_manipulator_data.size();                             
            
            int color_idx = -1;
            for (int i = 0; i < m_camera_color_data.size(); i++)
            {
                if (selected_data_object_pose.first == m_camera_color_data[i].first)
                {
                    color_idx = i;                    
                    break;
                }
            }            
            
            int depth_idx = -1;
            for (int i = 0; i < m_camera_depth_data.size(); i++)
            {
                if (selected_data_object_pose.first == m_camera_depth_data[i].first)
                {
                    depth_idx = i;                    
                    break;
                }
            }

            int camera_info_idx = -1;
            for (int i = 0; i < m_camera_info_data.size(); i++)
            {
                if (selected_data_object_pose.first == m_camera_info_data[i].first)
                {
                    camera_info_idx = i;                    
                    break;
                }
            }              
                                    
            if (end_data_robot_manipulator.first > selected_data_object_pose.first && color_idx != -1 && depth_idx != -1 && camera_info_idx != -1)
            {                                             
                /* find sync sensor data */
                m_object_pose_data.erase(0, index_object_pose + 1);

                auto selected_data_camera_color = m_camera_color_data[color_idx];                
                m_camera_color_data.erase(0, color_idx + 1);                
                auto selected_data_camera_depth = m_camera_depth_data[depth_idx];
                m_camera_depth_data.erase(0, depth_idx + 1);
                auto selected_data_camera_info = m_camera_info_data[camera_info_idx];
                m_camera_info_data.erase(0, camera_info_idx + 1);

                double min_dist = 1e+15;
                int min_idx = 0;                             
                for (int i=m_robot_manipulator_data.size() - 1; i>=0; i--)                
                {                    
                    double dist = abs(m_robot_manipulator_data[i].first - selected_data_object_pose.first);                                        
                    if (dist <= min_dist)
                    {                        
                        min_dist = dist;
                        min_idx = i;
                    }                    
                    // else
                    // {
                    //     break;
                    // }
                }

                if (min_dist > 0.001)
                {
                    std::cout << "[add factor thread] time sync error : " << min_dist << " " << m_robot_manipulator_data.size() << " " << min_idx << std::endl;  
                    // std::cout << "idx : " << min_idx << " " << color_idx << " " << depth_idx << " " << camera_info_idx << std::endl;                    
                    // if (m_robot_manipulator_data.size()) m_robot_manipulator_data.erase(0, min_idx + 1);

                    break;
                } 

                m_save_time.push_back(selected_data_object_pose.first);

                auto selected_data_robot_manipulator = m_robot_manipulator_data[min_idx];
                m_robot_manipulator_data.erase(0, min_idx + 1);

                // std::cout.precision(20);
                // std::cout << selected_data_object_pose.first << " " << abs(selected_data_object_pose.first - selected_data_robot_manipulator.first) << std::endl;
                // std::cout << m_robot_manipulator_data.size() << " "
                //      << m_object_pose_data.size() << " "
                //      << m_camera_color_data.size() << " "
                //      << m_camera_depth_data.size() << std::endl
                //      << std::endl;

                /* add robot manipulator factor */
                Eigen::Matrix4d T_robot_manipulator_pose = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d T_robot_manipulator_ee_to_cam = Eigen::Matrix4d::Identity();
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        T_robot_manipulator_pose(j, i) = selected_data_robot_manipulator.second.O_T_EE[i * 4 + j];
                        T_robot_manipulator_ee_to_cam(j, i) = selected_data_robot_manipulator.second.EE_T_CAM[i * 4 + j];
                    }
                }
                T_robot_manipulator_pose.block(0, 3, 3, 1) *= 1000.;
                T_robot_manipulator_ee_to_cam.block(0, 3, 3, 1) *= 1000.;

                Eigen::Matrix4d T_Base2Cam = T_robot_manipulator_pose * T_robot_manipulator_ee_to_cam;
                gtsam::noiseModel::Diagonal::shared_ptr robot_manipulator_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * m_weight.robot_manipulator);

                if(m_add_noise)
                {                    
                    Eigen::Matrix4d T_noise = Eigen::Matrix4d::Identity();

                    std::random_device rd; 
                    std::mt19937 gen(rd()); 

                    std::normal_distribution<double> d1(0, m_tranalation_noise);
                    T_noise(0, 3) = d1(gen);
                    T_noise(1, 3) = d1(gen);
                    T_noise(2, 3) = d1(gen);

                    std::normal_distribution<double> d2(0, m_rotation_noise);
                    Eigen::Matrix3d m;
                    m = Eigen::AngleAxisd(d2(gen) * (M_PI / 180.), Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(d2(gen) * (M_PI / 180.), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(d2(gen) * (M_PI / 180.), Eigen::Vector3d::UnitZ());
                    T_noise.block(0, 0, 3, 3) = m;

                    T_Base2Cam = T_Base2Cam * T_noise;
                }

                global_mutex->lock();                
                auto search = m_factor_counter.find(std::string("x"));                
                if (search == m_factor_counter.end())
                {
                    // if (m_use_robot_franka)
                    // {                        
                        m_GTSAM_graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::symbol('b', 0), gtsam::Pose3::identity(), gtsam::noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * m_weight.robot_base)));
                        m_initial_estimate.insert(gtsam::symbol('b', 0), gtsam::Pose3::identity());
                    // }
                    // else
                    // {
                        // m_GTSAM_graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::symbol('b', 0), gtsam::Pose3::identity(), robot_manipulator_noise));
                        // m_initial_estimate.insert(gtsam::symbol('b', 0), gtsam::Pose3::identity());
                    // }

                    m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('b', 0), gtsam::symbol('x', 0), gtsam::Pose3(T_Base2Cam), robot_manipulator_noise));
                    m_initial_estimate.insert(gtsam::symbol('x', 0), gtsam::Pose3(T_Base2Cam));

                    m_factor_counter.insert(std::multimap<std::string, long>::value_type(std::string("x"), 0));
                }
                else
                {
                    m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('b', 0), gtsam::symbol('x', search->second), gtsam::Pose3(T_Base2Cam), robot_manipulator_noise));
                    m_initial_estimate.insert(gtsam::symbol('x', search->second), gtsam::Pose3(T_Base2Cam));
                }
                
                search = m_factor_counter.find(std::string("x"));                
                int index_robot_manipulator_pose = search->second;                
                search->second++;                
                
                global_mutex->unlock();

                /* add object pose factor */
                for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
                {
                    // object pose uncertainty //
                    object_pose_uncertainty[iter->first][0] = 1e+10;
                    object_pose_uncertainty[iter->first][1] = 1e+10;
                    object_pose_uncertainty[iter->first][2] = 1e+10;

                    // object_pose_uncertainty_norm //
                    object_pose_uncertainty_norm[iter->first] = 1e+10;
                }

                for (int i = 0; i < selected_data_object_pose.second.detections.size(); i++)
                {                    
                    // object pose //
                    Eigen::VectorXd object_pose_postion = Eigen::VectorXd::Zero(3);
                    object_pose_postion[0] = selected_data_object_pose.second.detections[i].results[0].pose.pose.position.x;
                    object_pose_postion[1] = selected_data_object_pose.second.detections[i].results[0].pose.pose.position.y;
                    object_pose_postion[2] = selected_data_object_pose.second.detections[i].results[0].pose.pose.position.z;

                    if (object_pose_postion[2] < 0) continue;

                    Eigen::Quaterniond object_pose_orientation = Eigen::Quaterniond();
                    object_pose_orientation.x() = selected_data_object_pose.second.detections[i].results[0].pose.pose.orientation.x;
                    object_pose_orientation.y() = selected_data_object_pose.second.detections[i].results[0].pose.pose.orientation.y;
                    object_pose_orientation.z() = selected_data_object_pose.second.detections[i].results[0].pose.pose.orientation.z;
                    object_pose_orientation.w() = selected_data_object_pose.second.detections[i].results[0].pose.pose.orientation.w;

                    Eigen::Matrix4d object_pose = Eigen::Matrix4d::Identity();
                    object_pose.block(0, 0, 3, 3) = object_pose_orientation.normalized().toRotationMatrix();
                    object_pose.block(0, 3, 3, 1) = object_pose_postion;                    

                    // discern the instances of same object
                    int instance_id = selected_data_object_pose.second.detections[i].results[0].id;                    

                    if (m_object_info[instance_id].n_instance > 1)
                    {                       
                        bool is_assigned = false;
                        for (int iii=0; iii < m_object_info[instance_id].n_instance; iii++)
                        {
                            if (!m_optimized_estimate.exists(symbol('l', int(instance_id + 100*iii))))
                            {
                                instance_id = selected_data_object_pose.second.detections[i].results[0].id + 100*iii;
                                is_assigned = true;
                                break;
                            }
                            else
                            {
                                gtsam::Pose3 T_Base2Obj_ = m_optimized_estimate.at<Pose3>(symbol('l', int(instance_id + 100*iii)));
                                Eigen::Matrix4d T_Base2Obj_l = T_Base2Obj_.matrix();                                
                                Eigen::Matrix4d T_Base2Obj_new = T_Base2Cam * object_pose;                                
                                double rot_diff = rotation_error(T_Base2Obj_l.block(0, 0, 3, 3), T_Base2Obj_new.block(0, 0, 3, 3));
                                double tra_diff = (T_Base2Obj_l.block(0, 3, 3, 1) - T_Base2Obj_new.block(0, 3, 3, 1)).norm();
                                double radius = std::get<0>(m_object_info[instance_id].model_spec) / 2.;
                                if (tra_diff < radius && rot_diff < 20)
                                {
                                    instance_id = selected_data_object_pose.second.detections[i].results[0].id + 100 * iii;
                                    is_assigned = true;
                                    break;
                                }
                            }
                        }
                        if (!is_assigned)
                        {
                            std::cout << "no assign : " << instance_id << std::endl;
                            continue;
                        }   
                    }
                    // std::cout << instance_id << std::endl;

                    // if (instance_id != 20)   continue;

                    float amb_threshold = 0.4;
                    m_object_info[instance_id].is_symmetry = false;
                    if (selected_data_object_pose.second.detections[i].results[0].pose.covariance[0] > amb_threshold)    m_object_info[instance_id].is_symmetry = true;
                    if (selected_data_object_pose.second.detections[i].results[0].pose.covariance[7] > amb_threshold)    m_object_info[instance_id].is_symmetry = true;
                    if (selected_data_object_pose.second.detections[i].results[0].pose.covariance[14] > amb_threshold)    m_object_info[instance_id].is_symmetry = true;

                    if (pre_R_object_pose[instance_id] != Eigen::Matrix3d::Identity())
                    {
                        if (!m_object_info[instance_id].is_symmetry)
                        {
                            double R_robot_manipulator_pose_diff = rotation_error(T_robot_manipulator_pose.block(0, 0, 3, 3), pre_R_robot_manipulator_pose[instance_id]);
                            double R_object_pose_diff = rotation_error(object_pose.block(0, 0, 3, 3), pre_R_object_pose[instance_id]);
                            // std::cout << instance_id << " " << R_robot_manipulator_pose_diff << " " << R_object_pose_diff << std::endl;
                            if (R_object_pose_diff > R_robot_manipulator_pose_diff + 20)
                            {
                                n_invalid_object[instance_id]++;
                                if (n_invalid_object[instance_id] > 5)
                                {
                                    pre_R_object_pose[instance_id] = Eigen::Matrix3d::Identity();
                                    n_invalid_object[instance_id] = 0;
                                }
                                continue;
                            }
                        }
                        else
                        {
                            Eigen::Vector3d rot_vector = object_pose.block(0, 0, 3, 3) * Eigen::Vector3d(0, 0, 1);
                            Eigen::Vector3d pre_rot_vector = pre_R_object_pose[instance_id] * Eigen::Vector3d(0, 0, 1);
                            double vector_diff = acos((rot_vector.dot(pre_rot_vector)) / (rot_vector.norm() * pre_rot_vector.norm())) * (180. / PI);
                            // std::cout << instance_id << " " << vector_diff << std::endl;
                            if (vector_diff > 20)
                            {
                                n_invalid_object[instance_id]++;
                                if (n_invalid_object[instance_id] > 5)
                                {
                                    pre_R_object_pose[instance_id] = Eigen::Matrix3d::Identity();
                                    n_invalid_object[instance_id] = 0;
                                }
                                continue;
                            }
                        }
                    }
                    pre_R_robot_manipulator_pose[instance_id] = T_robot_manipulator_pose.block(0, 0, 3, 3);
                    pre_R_object_pose[instance_id] = object_pose.block(0, 0, 3, 3);

                    T_object_pose[instance_id] = object_pose;

                    // object pose uncertainty //
                    object_pose_uncertainty[instance_id][0] = selected_data_object_pose.second.detections[i].results[0].pose.covariance[0];
                    object_pose_uncertainty[instance_id][1] = selected_data_object_pose.second.detections[i].results[0].pose.covariance[7];
                    object_pose_uncertainty[instance_id][2] = selected_data_object_pose.second.detections[i].results[0].pose.covariance[14];

                    object_pose_uncertainty_norm[instance_id] = selected_data_object_pose.second.detections[i].results[0].score;                    

                    // object pose bbox //
                    object_bbox[instance_id][0] = selected_data_object_pose.second.detections[i].bbox.center.x;
                    object_bbox[instance_id][1] = selected_data_object_pose.second.detections[i].bbox.center.y;
                    object_bbox[instance_id][2] = selected_data_object_pose.second.detections[i].bbox.size_x;
                    object_bbox[instance_id][3] = selected_data_object_pose.second.detections[i].bbox.size_y;

                    // object mask //
                    object_mask[instance_id] = cv::Mat(selected_data_object_pose.second.detections[i].source_img.height, selected_data_object_pose.second.detections[i].source_img.width, CV_8UC1, const_cast<unsigned char *>(selected_data_object_pose.second.detections[i].source_img.data.data()), selected_data_object_pose.second.detections[i].source_img.step).clone();
                }            

                global_mutex->lock();
                for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
                {                    
                    if (object_pose_uncertainty_norm[iter->first] != 1e+10)
                    {
                        Eigen::Matrix4d T_Cam2Obj = m_T_axis_align * T_object_pose[iter->first];
                        Eigen::Matrix4d T_Base2Obj = T_Base2Cam * T_Cam2Obj;

                        if (T_Base2Obj(2, 3) < 0)
                            continue;

                        if (iter->first == 20)
                        {
                            m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
                            m_T_sym_obj_constraint(0, 3) = -10;
                        }
                        else if (iter->first == 17)
                        {
                            m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
                            m_T_sym_obj_constraint(0, 3) = 10;
                        }
                        else
                        {
                            m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
                            m_T_sym_obj_constraint(2, 3) = 10;    
                        }

                        Eigen::Matrix4d T_Cam2Obj_sym = T_Cam2Obj;
                        Eigen::Matrix4d T_Cam2Obj_sym_constraint = T_Cam2Obj * m_T_sym_obj_constraint;
                        Eigen::Matrix4d T_Base2Obj_sym = T_Base2Cam * T_Cam2Obj;
                        Eigen::Matrix4d T_Base2Obj_sym_constraint = T_Base2Cam * T_Cam2Obj * m_T_sym_obj_constraint;
                        // T_Cam2Obj_sym.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();                        
                        T_Cam2Obj_sym_constraint.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();                        
                        // T_Base2Obj_sym.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();                        
                        T_Base2Obj_sym_constraint.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

                        auto search = m_factor_counter.find(std::string("l") + std::to_string(iter->first));
                        if (search == m_factor_counter.end())
                        {
                            if (m_object_info[iter->first].is_symmetry)
                            {
                                m_initial_estimate.insert(gtsam::symbol('l', iter->first), gtsam::Pose3(T_Base2Obj_sym));   //
                                m_initial_estimate.insert(gtsam::symbol('c', iter->first), gtsam::Pose3(T_Base2Obj_sym_constraint));
                            }
                            else
                            {
                                m_initial_estimate.insert(gtsam::symbol('l', iter->first), gtsam::Pose3(T_Base2Obj));
                                m_initial_estimate.insert(gtsam::symbol('c', iter->first), gtsam::Pose3(T_Base2Obj_sym_constraint));
                            }

                            m_factor_counter.insert(std::multimap<std::string, long>::value_type(std::string("l") + std::to_string(iter->first), 0));
                            m_factor_counter.insert(std::multimap<std::string, long>::value_type(std::string("c") + std::to_string(iter->first), 0));
                            m_factor_counter.insert(std::multimap<std::string, long>::value_type(std::string("l_icp") + std::to_string(iter->first), 0));
                        }

                        // std::cout << iter->first << " ";
                        // std::cout << object_pose_uncertainty[iter->first][0] << " ";
                        // std::cout << object_pose_uncertainty[iter->first][1] << " ";
                        // std::cout << object_pose_uncertainty[iter->first][2] << std::endl;

                        if (object_pose_uncertainty_norm[iter->first] < m_object_info[iter->first].threshold)
                        {
                            cv::Mat camera_color = cv::Mat(selected_data_camera_color.second.height, selected_data_camera_color.second.width, CV_8UC3, const_cast<unsigned char *>(selected_data_camera_color.second.data.data()), selected_data_camera_color.second.step);
                            cv::cvtColor(camera_color, camera_color, cv::COLOR_BGR2RGB);
                                
                            cv::Mat camera_depth = cv::Mat(selected_data_camera_depth.second.height, selected_data_camera_depth.second.width, CV_16UC1, const_cast<unsigned char *>(selected_data_camera_depth.second.data.data()), selected_data_camera_depth.second.step);

                            IcpData icp_data;
                            icp_data.camera_depth = camera_depth.clone();
                            icp_data.object_color = camera_color.clone();
                            icp_data.object_mask = object_mask[iter->first].clone();
                            icp_data.object_idx = iter->first;
                            icp_data.robot_manipulator_idx = index_robot_manipulator_pose;
                            icp_data.uncertainty = object_pose_uncertainty_norm[iter->first];
                            icp_data.is_symmetry = m_object_info[iter->first].is_symmetry;
                            for (int ii = 0; ii < 4; ii++)
                            {
                                for (int jj = 0; jj < 4; jj++)
                                {
                                    icp_data.T_Cam2Obj[ii * 4 + jj] = T_object_pose[iter->first](jj, ii);
                                    icp_data.T_Base2Cam[ii * 4 + jj] = T_Base2Cam(jj, ii);
                                }
                            }
                            for (int ii=0; ii<9; ii++)  icp_data.K[ii] = selected_data_camera_info.second.K[ii];
                            m_icp_thread[iter->first].push(icp_data);
                            m_icp_thread[iter->first].cv.notify_all();
                        }
                        else
                        {
                            if (m_object_info[iter->first].is_symmetry)
                            {
                                gtsam::Vector noise = gtsam::Vector::Ones(6);                                
                                noise(0) = 1e+5;
                                noise(1) = 1e+5;
                                noise(2) = 1e+5;
                                noise(3) = object_pose_uncertainty_norm[iter->first] * m_weight.object;
                                noise(4) = object_pose_uncertainty_norm[iter->first] * m_weight.object;
                                noise(5) = object_pose_uncertainty_norm[iter->first] * m_weight.object;
                                gtsam::noiseModel::Diagonal::shared_ptr object_pose_noise = gtsam::noiseModel::Diagonal::Variances(noise);
                                gtsam::noiseModel::Diagonal::shared_ptr object_pose_noise1 = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * object_pose_uncertainty_norm[iter->first] * m_weight.object);

                                m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('x', index_robot_manipulator_pose), gtsam::symbol('l', iter->first), gtsam::Pose3(T_Cam2Obj_sym), object_pose_noise1));
                                m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('x', index_robot_manipulator_pose), gtsam::symbol('c', iter->first), gtsam::Pose3(T_Cam2Obj_sym_constraint), object_pose_noise1));

                                search = m_factor_counter.find(std::string("c") + std::to_string(iter->first));
                                search->second++;
                            }
                            else
                            {
                                gtsam::noiseModel::Diagonal::shared_ptr object_pose_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * object_pose_uncertainty_norm[iter->first] * m_weight.object);

                                m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('x', index_robot_manipulator_pose), gtsam::symbol('l', iter->first), gtsam::Pose3(T_Cam2Obj), object_pose_noise));
                            }
                            m_object_info[iter->first].is_factor = true;

                            search = m_factor_counter.find(std::string("l") + std::to_string(iter->first));
                            search->second++;   
                        }

                                            
                    }
                }
                global_mutex->unlock();
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }
}

void graph_optimization::optimization(const ros::TimerEvent& event)
{      
    if (m_GTSAM_graph.size() > 0)
    {                            
        global_mutex->lock();

        // m_optimized_estimate = LevenbergMarquardtOptimizer(m_GTSAM_graph, m_initial_estimate).optimize(); 

        m_isam->update(m_GTSAM_graph, m_initial_estimate);
        m_isam->update();
        m_optimized_estimate = m_isam->calculateEstimate();   
        // m_optimized_estimate = m_isam->calculateBestEstimate();   
        m_GTSAM_graph.resize(0);
        m_initial_estimate.clear();  

          
        
        // // std::cout << m_isam->valueExists(symbol('l', 4)) << std::endl;
        // auto search = m_factor_counter.find(std::string("l") + std::to_string(20));
        // if (m_isam->valueExists(symbol('l', 20)) && search->second > 2)
        // {            
        //     std::cout << m_isam->marginalCovariance(symbol('l', 20))(0, 0) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 20))(1, 1) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 20))(2, 2) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 20))(3, 3) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 20))(4, 4) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 20))(5, 5) << std::endl;

        //     std::cout << std::endl;

        //     auto search = m_factor_counter.find(std::string("c") + std::to_string(20));
        //     if (m_isam->valueExists(symbol('c', 20)) && search->second > 2)
        //     {
        //         std::cout << m_isam->marginalCovariance(symbol('c', 20))(0, 0) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 20))(1, 1) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 20))(2, 2) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 20))(3, 3) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 20))(4, 4) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 20))(5, 5) << std::endl;
        //     }
        //     std::cout << "----------------------" << std::endl;
        // }

        // std::cout << "4444444444444444444444444444" << std::endl;

        // search = m_factor_counter.find(std::string("l") + std::to_string(4));
        // if (m_isam->valueExists(symbol('l', 4)) && search->second > 2)
        // {            
        //     std::cout << m_isam->marginalCovariance(symbol('l', 4))(0, 0) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 4))(1, 1) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 4))(2, 2) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 4))(3, 3) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 4))(4, 4) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 4))(5, 5) << std::endl;

        //     std::cout << std::endl;

        //     auto search = m_factor_counter.find(std::string("c") + std::to_string(4));
        //     if (m_isam->valueExists(symbol('c', 4)) && search->second > 2)
        //     {
        //         std::cout << m_isam->marginalCovariance(symbol('c', 4))(0, 0) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 4))(1, 1) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 4))(2, 2) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 4))(3, 3) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 4))(4, 4) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 4))(5, 5) << std::endl;
        //     }
        //     std::cout << "----------------------" << std::endl;
        // }        

        // std::cout << "212121212121212121212121" << std::endl;

        // search = m_factor_counter.find(std::string("l") + std::to_string(21));
        // if (m_isam->valueExists(symbol('l', 21)) && search->second > 2)
        // {            
        //     std::cout << m_isam->marginalCovariance(symbol('l', 21))(0, 0) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 21))(1, 1) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 21))(2, 2) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 21))(3, 3) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 21))(4, 4) << std::endl;
        //     std::cout << m_isam->marginalCovariance(symbol('l', 21))(5, 5) << std::endl;

        //     std::cout << std::endl;

        //     auto search = m_factor_counter.find(std::string("c") + std::to_string(21));
        //     if (m_isam->valueExists(symbol('c', 21)) && search->second > 2)
        //     {
        //         std::cout << m_isam->marginalCovariance(symbol('c', 21))(0, 0) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 21))(1, 1) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 21))(2, 2) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 21))(3, 3) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 21))(4, 4) << std::endl;
        //         std::cout << m_isam->marginalCovariance(symbol('c', 21))(5, 5) << std::endl;
        //     }
        //     std::cout << "----------------------" << std::endl;
        // }           

        global_mutex->unlock();  
        
        publish_refined_robot_manipulator_pose();                            
        publish_refined_robot_manipulator_path();                            
        
        publish_refined_object_pose_from_world();                
        publish_refined_object_pose_from_cam();
                
        if (m_save_data)    save_data();        
        
    }    
}

void graph_optimization::save_data()
{    
    std::ofstream file;
    file.open(m_save_path + std::string("/data/") + m_seq_name + std::string("/seq_") + std::to_string(m_seq_index) + std::string(".csv"));
    
    file << std::setprecision(20);
    for (auto iter = m_object_info.begin(); iter != m_object_info.end(); iter++)
    {
        if(m_object_info[iter->first].is_factor)
        {
            if (!m_optimized_estimate.exists(symbol('l', iter->first)))   continue;   
            gtsam::Pose3 T_Base2Obj_ = m_optimized_estimate.at<Pose3>(symbol('l', iter->first));
            if (m_object_info[iter->first].is_symmetry)
            {
                gtsam::Pose3 T_obj_center = m_optimized_estimate.at<Pose3>(symbol('l', iter->first));
                gtsam::Pose3 T_obj_rot_axis = m_optimized_estimate.at<Pose3>(symbol('c', iter->first));                

                gtsam::Point3 rot_axis = (T_obj_rot_axis.translation() - T_obj_center.translation());
                rot_axis = normalize(rot_axis);
                                
                gtsam::Matrix33 axis_rotation = Eigen::Matrix3d::Identity();
                if (rot_axis[2] >= 0)
                {
                    axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, 1), rot_axis);
                }   
                else
                {
                    axis_rotation = find_rot_btw_two_vectors(gtsam::Point3(0, 0, -1), rot_axis);
                    Eigen::Matrix3d reverse;
                    reverse = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
                    axis_rotation = axis_rotation * reverse;
                }              

                T_Base2Obj_ = gtsam::Pose3(gtsam::Rot3(axis_rotation), gtsam::Point3(T_obj_center.translation()));
            }

            Eigen::Matrix4d T_Base2Obj = T_Base2Obj_.matrix();            
            file << "object," << std::to_string(iter->first%100);
            for (int ii = 0; ii < 4; ii++)
            {
                for (int jj = 0; jj < 4; jj++)
                {
                    file << "," << T_Base2Obj(ii, jj);
                }
            }
            file << "\n";            
        }
    }

    int x_idx = 0;
    while(true)
    {                
        if (!m_optimized_estimate.exists(symbol('x', x_idx)))   break;          
        gtsam::Pose3 T_Base2Cam_ = m_optimized_estimate.at<Pose3>(symbol('x', x_idx));

        Eigen::Matrix4d T_Base2Cam = T_Base2Cam_.matrix();
        file << "cam," << x_idx;
        file << "," << (ros::Time().fromSec(m_save_time[x_idx])).toNSec();
        for (int ii = 0; ii < 4; ii++)
        {
            for (int jj = 0; jj < 4; jj++)
            {
                file << "," << T_Base2Cam(ii, jj);
            }
        }
        file << "\n";

        x_idx++;        
    }
    file.close();
}

void graph_optimization::Thread_ICP(int object_idx)
{
    ros::Publisher icp_pub = m_nh.advertise<sensor_msgs::PointCloud2>("icp_pcl" + std::to_string(object_idx), 1);
    ros::Publisher tar_pub = m_nh.advertise<sensor_msgs::PointCloud2>("tar_pcl" + std::to_string(object_idx), 1);    

    std::cout << "[obj " << object_idx << " ICP thread] start" << std::endl;

    while (1)
    {
        std::unique_lock<std::mutex> ul(m_icp_thread[object_idx].mtx);
        m_icp_thread[object_idx].cv.wait(ul);
        if (m_icp_thread[object_idx].active == false)
        {            
            global_mutex->lock();
            std::cout << object_idx << " ICP thread finish" << std::endl;    
            global_mutex->unlock();        
            return;
        }
        ul.unlock();

        while (!m_icp_thread[object_idx].data_queue.empty())
        {            
            auto data = m_icp_thread[object_idx].pop();

            Eigen::Matrix4d T_Cam2Obj = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d T_Base2Cam = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    T_Cam2Obj(j, i) = data.T_Cam2Obj[i * 4 + j];
                    T_Base2Cam(j, i) = data.T_Base2Cam[i * 4 + j];
                }
            }            
            Eigen::Matrix3d camera_instricsic = Eigen::Matrix3d::Identity();
            camera_instricsic(0, 0) = data.K[0];
            camera_instricsic(0, 2) = data.K[2];
            camera_instricsic(1, 1) = data.K[4];
            camera_instricsic(1, 2) = data.K[5];
            

            // image rendering using estimate //            
            m_icp_thread[object_idx].renderer.renderObject(object_idx, T_Cam2Obj.block(0, 0, 3, 3), T_Cam2Obj.block(0, 3, 3, 1), camera_instricsic);   

            cv::Mat rendered_color = m_icp_thread[object_idx].renderer.getColorImage(object_idx);
            cv::Mat rendered_depth = m_icp_thread[object_idx].renderer.getDepthImage(object_idx);  

            cv::Mat rendered_gray;
            cv::Mat rendered_mask;
            cv::cvtColor(rendered_color, rendered_color, CV_BGR2RGB);
            cv::cvtColor(rendered_color, rendered_gray, CV_RGB2GRAY);
            cv::threshold(rendered_gray, rendered_mask, 10, 255, cv::THRESH_BINARY);  

            Eigen::Matrix4d T_Cloud2Origin = Eigen::Matrix4d::Identity();
            T_Cloud2Origin.block(0, 3, 3, 1) = -T_Cam2Obj.block(0, 3, 3, 1);

            int interval = 2;
            pcl::PointCloud<PointT>::Ptr rendered_cloud(new pcl::PointCloud<PointT>);
            for (int m = 0; m < rendered_depth.rows; m+=interval)
            {
                for (int n = 0; n < rendered_depth.cols; n+=interval)
                {
                    if (rendered_mask.ptr<uchar>(m)[n] < 0.5)
                        continue;
                    if (data.object_mask.ptr<uchar>(m)[n] < 0.5 )
                        continue;    

                    ushort d = rendered_depth.ptr<ushort>(m)[n];
                    if (d <= 0)
                        continue;

                    PointT p;
                    p.z = double(d);
                    p.x = (n - camera_instricsic(0, 2)) * p.z / camera_instricsic(0, 0);
                    p.y = (m - camera_instricsic(1, 2)) * p.z / camera_instricsic(1, 1);

                    p.b = rendered_color.ptr<uchar>(m)[n * 3];
                    p.g = rendered_color.ptr<uchar>(m)[n * 3 + 1];
                    p.r = rendered_color.ptr<uchar>(m)[n * 3 + 2];

                    rendered_cloud->points.push_back(p);
                }
            }

            // sensored depth //
            PointCloud::Ptr sensored_cloud(new PointCloud);
            for (int m = 0; m < data.camera_depth.rows; m+=interval)
            {
                for (int n = 0; n < data.camera_depth.cols; n+=interval)
                {                                                            
                    if (data.object_mask.ptr<uchar>(m)[n] < 0.5 )
                        continue;

                    ushort d = data.camera_depth.ptr<ushort>(m)[n];
                    if (d <= 0)
                        continue;
                    if (d >= 2000)
                        continue;

                    PointT p;

                    p.z = double(d);
                    p.x = (n - camera_instricsic(0, 2)) * p.z / camera_instricsic(0, 0);
                    p.y = (m - camera_instricsic(1, 2)) * p.z / camera_instricsic(1, 1);

                    p.b = data.object_color.ptr<uchar>(m)[n * 3];
                    p.g = data.object_color.ptr<uchar>(m)[n * 3 + 1];
                    p.r = data.object_color.ptr<uchar>(m)[n * 3 + 2];

                    sensored_cloud->points.push_back(p);
                }
            }       

            if (rendered_cloud->points.size() > 50 && sensored_cloud->points.size() > 50)
            {
                // pcl::transformPointCloud(*rendered_cloud, *rendered_cloud, T_Cloud2Origin);
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr rendered_cloud_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                addNormal(rendered_cloud, rendered_cloud_normals);

                // pcl::transformPointCloud(*sensored_cloud, *sensored_cloud, T_Cloud2Origin);
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sensored_cloud_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
                addNormal(sensored_cloud, sensored_cloud_normals);

                pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
                icp.setInputSource(rendered_cloud_normals);
                icp.setInputTarget(sensored_cloud_normals);
                // icp.setMaxCorrespondenceDistance(100);
                // icp.setMaximumOptimizerIterations(100);
                icp.setMaximumIterations(50);
                icp.setRANSACIterations(50);
                icp.setEuclideanFitnessEpsilon(0.1);

                pcl::PointCloud<pcl::PointXYZRGBNormal> output;
                icp.align(output);             
                Eigen::Matrix4f T_source2target = icp.getFinalTransformation();                            

                // std::cout << icp.getFitnessScore() << " " << object_idx << " " << m_icp_thread[object_idx].data_queue.size() << " " << data.uncertainty << std::endl;
                
                Eigen::Matrix4d icp_T_Cam2Obj = m_T_axis_align * T_Cam2Obj;
                noiseModel::Diagonal::shared_ptr object_pose_noise = noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * data.uncertainty * m_weight.object);
                if (icp.hasConverged() == true && icp.getFitnessScore() < 15)
                {
                    // global_mutex->lock();  
                    // std::cout << "[obj " << object_idx << " ICP thread] " << icp.getFitnessScore() << " " << m_icp_thread[object_idx].data_queue.size() << std::endl;
                    // global_mutex->unlock();  

                    icp_T_Cam2Obj = m_T_axis_align * T_source2target.cast<double>() * T_Cam2Obj;                    
                    object_pose_noise = noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * icp.getFitnessScore() * m_weight.object_icp);
                }

                if (icp.hasConverged() == true)
                {                    
                    if (data.is_symmetry)
                    {
                        if (object_idx == 20)
                        {
                            m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
                            m_T_sym_obj_constraint(0, 3) = -10;    
                        }
                        else if (object_idx == 17)
                        {                            
                            m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
                            m_T_sym_obj_constraint(0, 3) = 10;
                        }
                        else
                        {
                            m_T_sym_obj_constraint = Eigen::Matrix4d::Identity();
                            m_T_sym_obj_constraint(2, 3) = 10;    
                        }


                        Eigen::Matrix4d T_Base2Obj = T_Base2Cam * icp_T_Cam2Obj;
                        Eigen::Matrix4d T_Cam2Obj_sym = icp_T_Cam2Obj;
                        Eigen::Matrix4d T_Cam2Obj_sym_constraint = icp_T_Cam2Obj * m_T_sym_obj_constraint;
                        Eigen::Matrix4d T_Base2Obj_sym = T_Base2Cam * T_Cam2Obj_sym;
                        Eigen::Matrix4d T_Base2Obj_sym_constraint = T_Base2Cam * T_Cam2Obj_sym_constraint;
                        // T_Cam2Obj_sym.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
                        T_Cam2Obj_sym_constraint.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
                        // T_Base2Obj_sym.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
                        T_Base2Obj_sym_constraint.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();

                        global_mutex->lock();

                        // noiseModel::Diagonal::shared_ptr object_pose_noise = noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * data.uncertainty * m_weight.object);

                        gtsam::Vector noise = gtsam::Vector::Ones(6);
                        noise(0) = 1e+5;
                        noise(1) = 1e+5;
                        noise(2) = 1e+5;
                        noise(3) = icp.getFitnessScore() * m_weight.object_icp;
                        noise(4) = icp.getFitnessScore() * m_weight.object_icp;
                        noise(5) = icp.getFitnessScore() * m_weight.object_icp;
                        gtsam::noiseModel::Diagonal::shared_ptr object_pose_noise = gtsam::noiseModel::Diagonal::Variances(noise);
                        noiseModel::Diagonal::shared_ptr object_pose_noise1 = noiseModel::Diagonal::Variances(gtsam::Vector::Ones(6) * icp.getFitnessScore() * m_weight.object_icp);

                        m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('x', data.robot_manipulator_idx), gtsam::symbol('l', object_idx), gtsam::Pose3(T_Cam2Obj_sym), object_pose_noise1));
                        m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('x', data.robot_manipulator_idx), gtsam::symbol('c', object_idx), gtsam::Pose3(T_Cam2Obj_sym_constraint), object_pose_noise1));

                        if (m_initial_estimate.exists(gtsam::symbol('l', object_idx)))
                        {
                            m_initial_estimate.update(gtsam::symbol('l', object_idx), gtsam::Pose3(T_Base2Obj_sym));
                        }
                        if (m_initial_estimate.exists(gtsam::symbol('c', object_idx)))
                        {
                            m_initial_estimate.update(gtsam::symbol('c', object_idx), gtsam::Pose3(T_Base2Obj_sym_constraint));
                        }

                        global_mutex->unlock();

                        auto search = m_factor_counter.find(std::string("c") + std::to_string(object_idx));
                        search->second++; 
                    }
                    else
                    {
                        Eigen::Matrix4d T_Base2Obj = T_Base2Cam * icp_T_Cam2Obj;

                        global_mutex->lock();
                        m_GTSAM_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::symbol('x', data.robot_manipulator_idx), gtsam::symbol('l', object_idx), gtsam::Pose3(icp_T_Cam2Obj), object_pose_noise));

                        if (m_initial_estimate.exists(gtsam::symbol('l', object_idx)))
                        {
                            m_initial_estimate.update(gtsam::symbol('l', object_idx), gtsam::Pose3(T_Base2Obj));
                        }

                        global_mutex->unlock();
                    }

                    m_object_info[data.object_idx].is_factor = true;

                    auto search = m_factor_counter.find(std::string("l") + std::to_string(object_idx));                    
                    search->second++;    
                    search = m_factor_counter.find(std::string("l_icp") + std::to_string(object_idx));
                    search->second++;  
                }                

                if (icp.hasConverged() == true && icp.getFitnessScore() < 15)
                {                    
                    Eigen::Matrix4d tt = T_Base2Cam * m_T_axis_align * T_source2target.cast<double>();
                    pcl::transformPointCloud(*rendered_cloud_normals, *rendered_cloud_normals, tt);
                    for (int kk = 0; kk < rendered_cloud_normals->points.size(); kk++)
                    {
                        rendered_cloud_normals->points[kk].x /= 1000.;
                        rendered_cloud_normals->points[kk].y /= 1000.;
                        rendered_cloud_normals->points[kk].z /= 1000.;
                    }
                    sensor_msgs::PointCloud2 icp_output;
                    pcl::toROSMsg(*rendered_cloud_normals.get(), icp_output);
                    icp_output.header.frame_id = "panda_link0";
                    icp_pub.publish(icp_output);
                    
                    tt = T_Base2Cam * m_T_axis_align;
                    pcl::transformPointCloud(*sensored_cloud_normals, *sensored_cloud_normals, tt);
                    for (int kk = 0; kk < sensored_cloud_normals->points.size(); kk++)
                    {
                        sensored_cloud_normals->points[kk].x /= 1000.;
                        sensored_cloud_normals->points[kk].y /= 1000.;
                        sensored_cloud_normals->points[kk].z /= 1000.;
                    }
                    sensor_msgs::PointCloud2 tar_output;
                    pcl::toROSMsg(*sensored_cloud_normals.get(), tar_output);
                    tar_output.header.frame_id = "panda_link0";
                    tar_pub.publish(tar_output);
                }
            }
            else
            {
                std::cout << "pcl : input_ is empty!" << std::endl;
            }
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "graph_optimization");

    ROS_INFO("[graph_optimization] ROS Package Start");

    ros::NodeHandle nh;
    graph_optimization go(nh);

    ros::spin();
    // ros::MultiThreadedSpinner spinner(4);
    // spinner.spin();

    // ros::AsyncSpinner spinner(0);
    // spinner.start();
    // ros::waitForShutdown();

    // ros::Rate rate(30);
    // while (ros::ok())
    // {
    //     ros::spinOnce();

    //     go.add_factor();

    //     rate.sleep();
    // }

    return 0;
}