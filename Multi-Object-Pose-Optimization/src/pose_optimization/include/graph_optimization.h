#ifndef _GRAPH_OPTIMIZATION_H_
#define _GRAPH_OPTIMIZATION_H_

#include "usrdef.h"

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam/slam/RotateFactor.h>
#include <gtsam/sfm/TranslationFactor.h>

#include "object_renderer.h"


using namespace gtsam;

struct ObjectInfo
{
    int index;
    std::string name;
    double threshold;
    bool is_symmetry;    
    bool is_factor;
    std::string model_path;
    int n_instance;
    model_info model_spec;

    ObjectInfo()
    {
        index = 0;
        name = std::string("");
        threshold = 0.;
        is_symmetry = false;        
        is_factor = false;
        model_path = std::string("");
        n_instance = 1;
    }
};

struct Weight
{
    double robot_base;
    double robot_manipulator;
    double object;
    double object_icp;

    Weight()
    {
        robot_base = 0.;
        robot_manipulator = 0.;
        object = 0.;
        object_icp = 0.;
    }
};


struct IcpData
{
    cv::Mat camera_depth;
    cv::Mat object_mask;
    cv::Mat object_color;
    double T_Cam2Obj[16];
    double T_Base2Cam[16];
    double K[9];
    double uncertainty;
    int object_idx;
    long robot_manipulator_idx;
    bool is_symmetry;

     IcpData()
     {
         camera_depth = cv::Mat(480, 640, CV_16UC1);
         object_mask = cv::Mat(480, 640, CV_8UC1);    
         object_color = cv::Mat(480, 640, CV_8UC3);        
         object_idx = 0;
         robot_manipulator_idx = 0;  
         uncertainty = 0.;    
         is_symmetry = false;   
     }
};

struct IcpThread
{
    std::mutex mtx;
    std::deque<IcpData> data_queue;    
    std::condition_variable cv;
    std::thread thd;
    object_renderer renderer;
    int object_idx;    
    int max_queue_size;
    bool active;    

    IcpThread() : active(false), max_queue_size(100)    {}

    void push(IcpData data)
    {
        mtx.lock();

        if (data_queue.size() < max_queue_size)
        {
            if (data_queue.size() == 0)
            {
                // cout << "[ICP Queue " << object_idx << "] New data is inserted" << endl;
            }

            data_queue.push_back(data);                    
        }
        else
        {
            for (int i=data_queue.size()-1; i >= 0 ; i--)
            {
                if (data_queue[i].uncertainty > data.uncertainty)
                {
                    data_queue.erase(data_queue.begin() + i);
                    data_queue.push_back(data);
                }

                break;
            }
        }            
        mtx.unlock();
    }

    IcpData pop()
    {
        IcpData result;
        mtx.lock();        
        result = data_queue.front();                                
        data_queue.pop_front();    

        if (data_queue.size() == 0)
        {
            // cout << "[ICP Queue " << object_idx << "] Empty" << endl;
        }

        mtx.unlock();        
        return result;
    }
};

template <typename T>
struct SensorData
{
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::pair<double, T>> data_deque;
    long cnt;    
    std::thread thd;
    bool active; 

    SensorData() : active(false), cnt(0){}

    void push_front(std::pair<double, T> data)
    {
        mtx.lock();
        data_deque.push_front(data);
        cnt++;
        mtx.unlock();
    }

    void push_back(std::pair<double, T> data)
    {
        mtx.lock();
        data_deque.push_back(data);
        cnt++;
        mtx.unlock();
    }    

    std::pair<double, T> pop_front()
    {
        std::pair<double, T> result;
        mtx.lock();
        result = data_deque.front();
        data_deque.pop_front();
        data_deque.shrink_to_fit();
        cnt--;
        mtx.unlock();
        return result;
    }

    std::pair<double, T> pop_back()
    {
        std::pair<double, T> result;
        mtx.lock();
        result = data_deque.back();
        data_deque.pop_back();
        data_deque.shrink_to_fit();
        cnt--;
        mtx.unlock();
        return result;
    }    

    std::pair<double, T> front()
    {
        std::pair<double, T> result;
        mtx.lock();
        result = data_deque.front();
        mtx.unlock();
        return result;
    }

    std::pair<double, T> back()
    {
        std::pair<double, T> result;
        mtx.lock();
        result = data_deque.back();
        mtx.unlock();
        return result;
    }

    void del_front()
    {        
        mtx.lock();
        data_deque.pop_front();
        data_deque.shrink_to_fit();
        cnt--;
        mtx.unlock();        
    }

    void del_back()
    {        
        mtx.lock();
        data_deque.pop_back();
        data_deque.shrink_to_fit();
        cnt--;
        mtx.unlock();        
    }

    void erase(int start, int end)
    {        
        mtx.lock();
        if (data_deque.size() > 0)
        {
            data_deque.erase(data_deque.begin() + start, data_deque.begin() + end);
        }        
        data_deque.shrink_to_fit();        
        cnt = data_deque.size();
        mtx.unlock();        
    }

    void clear()
    {
        mtx.lock();
        data_deque.clear();
        mtx.unlock();   
    }

    bool empty()
    {
        return data_deque.empty();
    }

    int size()
    {
        return data_deque.size();
    }

    std::pair<double, T> operator[](std::size_t idx)
    {
        return data_deque[idx];
    }

    void unique()
    {
        mtx.lock();
        if (data_deque.size() > 1)
        {            
            std::unique(data_deque.begin(), data_deque.end());
        }
        mtx.unlock();
    }

    void sort()
    {
        mtx.lock();
        if (data_deque.size() > 1)
        {            
            std::sort(data_deque.begin(), data_deque.end(), [](std::pair<double, T> const &a, std::pair<double, T> const &b)
                      { return a.first < b.first; });
        }
        mtx.unlock();
    }
};

class graph_optimization
{
public:

    graph_optimization(ros::NodeHandle nh);
    ~graph_optimization();

    void object_pose_callback(const vision_msgs::Detection2DArray::ConstPtr& msg);
    void robot_manipulator_pose_callback(const franka_rpm_msgs::FrankaState::ConstPtr& msg);

    void camera_info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg);
    void camera_color_callback(const sensor_msgs::Image::ConstPtr& msg);
    void camera_depth_callback(const sensor_msgs::Image::ConstPtr& msg);

    void publish_refined_object_pose_from_world();
    void publish_refined_object_pose_from_cam();

    void publish_refined_robot_manipulator_pose();
    void publish_refined_robot_manipulator_path();    

    void optimization(const ros::TimerEvent& event);
    void add_factor();    

    Eigen::Matrix3d find_rot_btw_two_vectors(Eigen::Vector3d v1, Eigen::Vector3d v2);
    double rotation_error(Eigen::Matrix3d r1, Eigen::Matrix3d r2);

    void save_data();

private:    

    NonlinearFactorGraph m_GTSAM_graph;
    Values m_initial_estimate;
    Values m_optimized_estimate;
    ISAM2 *m_isam;
    Values m_isam_current_estimate;
    std::vector<bool> m_object_key_exist;
    Marginals m_marginals;
 

    ros::NodeHandle m_nh;
    ros::Publisher m_pub_refined_object_pose_from_world;
    ros::Publisher m_pub_refined_object_pose_from_cam;
    ros::Publisher m_pub_refined_robot_manipulator_path;
    ros::Publisher m_pub_refined_robot_manipulator_pose;        

    ros::Subscriber m_sub_camera_info;
    ros::Subscriber m_sub_camera_color;
    ros::Subscriber m_sub_camera_depth;
    ros::Subscriber m_sub_object_pose;
    ros::Subscriber m_sub_robot_manipulator_pose;        

    int m_n_object = 21;        

    Eigen::Matrix4d m_T_axis_align;   // coordinate axis align btw robot manupulator and camera
    Eigen::Matrix4d m_T_sym_obj_constraint;    

    Eigen::Matrix3d m_camera_instricsic;
    Eigen::VectorXd m_camera_distortion;
    bool m_new_camera_info;

    std::multimap<std::string, long> m_factor_counter;

    ros::Timer m_timer;
    
    std::map<int, ObjectInfo> m_object_info;
    Weight m_weight;
            
    // IcpThread m_icp_thread[30];       
    std::map<int, IcpThread> m_icp_thread;

    std::mutex *global_mutex;
    void Thread_ICP(int object_idx);

    int m_seq_index;
    std::string m_seq_name;
    bool m_save_data;
    std::string m_save_path;
    std::vector<double> m_save_time;

    SensorData<vision_msgs::Detection2DArray> m_object_pose_data;
    SensorData<franka_rpm_msgs::FrankaState> m_robot_manipulator_data;    
    SensorData<sensor_msgs::CameraInfo> m_camera_info_data;
    SensorData<sensor_msgs::Image> m_camera_color_data;
    SensorData<sensor_msgs::Image> m_camera_depth_data;        

    bool m_first_call;   
    Eigen::Matrix4d m_T_cam_to_gripper;    

    bool m_add_noise;
    double m_tranalation_noise;
    double m_rotation_noise;

    bool m_use_robot_franka;  

    int m_img_size[2];        

    std::map<int, model_info> m_model_info;
};

#endif