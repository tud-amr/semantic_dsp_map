/**
 * @file mapping.cpp
 * @author Clarence Chen (g-ch@github.com)
 * @brief An exampe of using the SemanticDSPMap in a ROS node
 * @version 0.1
 * @date 2023-12-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <std_msgs/Int32.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <settings/external_settings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/PoseStamped.h>

#include <mask_kpts_msgs/MaskGroup.h>
#include <mask_kpts_msgs/MaskKpts.h>
#include <mask_kpts_msgs/Keypoint.h>

#include <geometry_msgs/PoseStamped.h>
#include <yaml-cpp/yaml.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "semantic_dsp_map.h"

class MappingNode
{
public:
    MappingNode(std::string yaml_file, std::string object_info_csv_file = "") :
    yaml_file_(yaml_file),
    object_info_csv_file_(object_info_csv_file)
    {
        initialize();
    };
    ~MappingNode(){};

    
private:
    cv::Mat depth_image_;
    Eigen::Vector3d current_camera_position_, last_camera_position_;
    Eigen::Quaterniond current_camera_orientation_, last_camera_orientation_;

    double time_stamp_double_;

    uint32_t depth_seq_;
    uint32_t masks_seq_;

    std::string yaml_file_;
    std::string object_info_csv_file_;

    std::string frame_id_;

    bool visualize_with_zero_center_;

    TrackingResultHandler tracking_result_handler_;

    SemanticDSPMap dsp_map_;

    ros::NodeHandle nh_;

    ros::Publisher occupied_point_pub_;
    ros::Publisher map_pose_pub_;
    ros::Publisher freespace_point_pub_;

    bool if_output_freespace_;

    /// @brief Initialize the node
    void initialize()
    {
        // Read the package path
        std::string package_path = ros::package::getPath("semantic_dsp_map");
        std::cout << "Package Path: " << package_path << std::endl;

        // Read csv file to set the object information if the file is provided
        if(object_info_csv_file_ != ""){
            std::string object_info_csv_file_path = package_path + "/cfg/" + object_info_csv_file_;
            ObjectInfoHandler object_info_handler;
            object_info_handler.readObjectInfo(object_info_csv_file_path);
        }

        // Read yaml file to set the parameters
        YAML::Node config = YAML::LoadFile(package_path + "/cfg/" + yaml_file_);
        bool if_consider_depth_noise = config["if_consider_depth_noise"].as<bool>();
        bool if_use_independent_filter = config["if_use_independent_filter"].as<bool>();
        bool if_out_evaluation_format = config["if_out_evaluation_format"].as<bool>();
        if_output_freespace_ = config["if_output_freespace"].as<bool>();

        std::string depth_image_topic = config["depth_image_topic"].as<std::string>();
        std::string camera_pose_topic = config["camera_pose_topic"].as<std::string>();
        std::string mask_group_topic = config["mask_group_topic"].as<std::string>();

        frame_id_ = config["frame_id"].as<std::string>();
        visualize_with_zero_center_ = config["visualize_with_zero_center"].as<bool>();

        float detection_probability = 1.0f, noise_number = 0.001f, occupancy_threshold = 0.1f;
        int nb_ptc_num_per_point = 3, max_obersevation_lost_time = 10;
        // If depth noise is considered, use parameters in the YAML. Otherwise use default parameters.
        if(if_consider_depth_noise){
            detection_probability = config["detection_probability"].as<float>();
            noise_number = config["noise_number"].as<float>();
            nb_ptc_num_per_point = config["nb_ptc_num_per_point"].as<int>();
            occupancy_threshold = config["occupancy_threshold"].as<float>();
            max_obersevation_lost_time = config["max_obersevation_lost_time"].as<int>();
        }
        
        float forgetting_rate = config["forgetting_rate"].as<float>();
        int max_forget_count = config["max_forget_count"].as<int>();
        
        float id_transition_probability = config["id_transition_probability"].as<float>();
        float match_score_threshold = config["match_score_threshold"].as<float>();

        float beyesian_movement_distance_threshold = config["beyesian_movement_distance_threshold"].as<float>();
        float beyesian_movement_probability_threshold = config["beyesian_movement_probability_threshold"].as<float>();
        float beyesian_movement_increment = config["beyesian_movement_increment"].as<float>();
        float beyesian_movement_decrement = config["beyesian_movement_decrement"].as<float>();

        float depth_noise_model_first_order = config["depth_noise_model_first_order"].as<float>();
        float depth_noise_model_zero_order = config["depth_noise_model_zero_order"].as<float>();
    
        // Set parameters for the SemanticDSPMap
        dsp_map_.setMapParameters(detection_probability, noise_number, nb_ptc_num_per_point, occupancy_threshold, max_obersevation_lost_time, forgetting_rate, max_forget_count, match_score_threshold, id_transition_probability);
        dsp_map_.setMapOptions(if_consider_depth_noise, if_use_independent_filter);
        dsp_map_.setVisualizeOptions(visualize_with_zero_center_, if_out_evaluation_format);

        dsp_map_.setBeyesianMovementParameters(beyesian_movement_distance_threshold, beyesian_movement_probability_threshold, beyesian_movement_increment, beyesian_movement_decrement);
        dsp_map_.setDepthNoiseModelParameters(depth_noise_model_first_order, depth_noise_model_zero_order);

        occupied_point_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("occupied_point", 1);
        freespace_point_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("freespace_point", 1);

        map_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("map_pose", 1);
        
        // Create a synchronizer to subscribe the depth image, camera pose, and mask group
        message_filters::Subscriber<sensor_msgs::Image> depth_image_sub(nh_, depth_image_topic, 1);
        message_filters::Subscriber<geometry_msgs::PoseStamped> camera_pose_sub(nh_, camera_pose_topic, 1);
        message_filters::Subscriber<mask_kpts_msgs::MaskGroup> mask_group_sub(nh_, mask_group_topic, 1);

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped, mask_kpts_msgs::MaskGroup> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_image_sub, camera_pose_sub, mask_group_sub);
        sync.registerCallback(boost::bind(&MappingNode::syncCallback, this, _1, _2, _3));

        ros::spin();
    }



    /// @brief The callback function for the depth image, camera pose, and mask group
    /// @param depth_image_msg 
    /// @param camera_pose_msg 
    /// @param mask_group_msg
    void syncCallback(const sensor_msgs::ImageConstPtr& depth_image_msg, const geometry_msgs::PoseStampedConstPtr& camera_pose_msg, const mask_kpts_msgs::MaskGroup::ConstPtr& mask_group_msg)
    {   
        static int count = 0;
        count++;
        time_stamp_double_ = camera_pose_msg->header.stamp.toSec();

        // Ignore the first two depth images
        if(depth_image_msg->header.seq < 2){
            return;
        }

        // Convert the depth image to cv::Mat
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            depth_image_ = cv_ptr->image;

            depth_seq_ = depth_image_msg->header.seq;

            current_camera_position_ = Eigen::Vector3d(camera_pose_msg->pose.position.x, camera_pose_msg->pose.position.y, camera_pose_msg->pose.position.z);
            current_camera_orientation_ = Eigen::Quaterniond(camera_pose_msg->pose.orientation.w, camera_pose_msg->pose.orientation.x, camera_pose_msg->pose.orientation.y, camera_pose_msg->pose.orientation.z);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Check if the sequence number of depth image and mask group are the same
        masks_seq_ = mask_group_msg->header.seq;
        if(depth_seq_ != masks_seq_){
            std::cout << "Error!!!! The sequence number of depth image and mask group are not the same!" << std::endl;
            std::cout << "depth_seq_: " << depth_seq_ << ", masks_seq_: " << masks_seq_ << std::endl;
            return;
        }

        // Set new tracking result
        tracking_result_handler_.tracking_result.clear();

        for(int i = 0; i < mask_group_msg->objects.size(); ++i)
        {
            // Only consider the object that has a valid track id. Static objects will also be considered.
            if(mask_group_msg->objects[i].track_id < 0 || mask_group_msg->objects[i].track_id > 65535){
                continue;
            }

            MaskKpts mask_kpts;
            mask_kpts.track_id = mask_group_msg->objects[i].track_id;
            mask_kpts.label = mask_group_msg->objects[i].label;

            mask_kpts.mask = cv_bridge::toCvCopy(mask_group_msg->objects[i].mask, sensor_msgs::image_encodings::MONO8)->image;
            
            if(mask_kpts.label != "static") // If the object is not static, add the bounding box and keypoints
            {
                mask_kpts.bbox.x1 = mask_group_msg->objects[i].bbox_tl.x;
                mask_kpts.bbox.y1 = mask_group_msg->objects[i].bbox_tl.y;
                mask_kpts.bbox.x2 = mask_group_msg->objects[i].bbox_br.x;
                mask_kpts.bbox.y2 = mask_group_msg->objects[i].bbox_br.y;

                for(int j = 0; j < mask_group_msg->objects[i].kpts_curr.size(); ++j)
                {
                    Eigen::Vector3d kpt;
                    kpt[0] = mask_group_msg->objects[i].kpts_curr[j].x;
                    kpt[1] = mask_group_msg->objects[i].kpts_curr[j].y;
                    kpt[2] = mask_group_msg->objects[i].kpts_curr[j].z;
                    mask_kpts.kpts_current.push_back(kpt);
                }

                for(int j = 0; j < mask_group_msg->objects[i].kpts_last.size(); ++j)
                {
                    Eigen::Vector3d kpt;
                    kpt[0] = mask_group_msg->objects[i].kpts_last[j].x;
                    kpt[1] = mask_group_msg->objects[i].kpts_last[j].y;
                    kpt[2] = mask_group_msg->objects[i].kpts_last[j].z;
                    mask_kpts.kpts_previous.push_back(kpt);
                }

                // Check if kpts_curr.size()!= kpts_last.size()
                if(mask_kpts.kpts_current.size() != mask_kpts.kpts_previous.size()){
                    std::cout << "Error!!!! kpts_curr.size()!= kpts_last.size()" << std::endl;
                    continue;
                }
            }

            tracking_result_handler_.tracking_result.push_back(mask_kpts);
        }

        
        // Visualization variables update
        last_camera_position_ = current_camera_position_;
        last_camera_orientation_ = current_camera_orientation_;
    
        // Update the map
        updateMap();
    }


    void updateMap()
    {
        // Do mapping
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr occupied_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr freespace_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        double time = ros::Time::now().toSec();
        static int count = 0;
        static double total_time_cost = 0.0;

        dsp_map_.update(depth_image_, tracking_result_handler_.tracking_result, current_camera_position_, current_camera_orientation_, occupied_point_cloud, freespace_point_cloud, if_output_freespace_, time_stamp_double_);
        double time_cost = ros::Time::now().toSec() - time;
        count++;
        total_time_cost += time_cost;

        ROS_INFO_THROTTLE(1, "Updating and mapping result getting time cost: %f s", time_cost);
        ROS_INFO_THROTTLE(1, "Average time cost: %f s", total_time_cost / count);

        // Transform Occupied point cloud to a ros message and publish
        sensor_msgs::PointCloud2 occupied_point_cloud_msg;
        pcl::toROSMsg(*occupied_point_cloud, occupied_point_cloud_msg);

        occupied_point_cloud_msg.header.frame_id = frame_id_;
        occupied_point_cloud_msg.header.stamp = ros::Time::now();
        occupied_point_pub_.publish(occupied_point_cloud_msg);

        geometry_msgs::PoseStamped map_pose_msg;
        map_pose_msg.header.frame_id = frame_id_;
        map_pose_msg.header.stamp = occupied_point_cloud_msg.header.stamp;

        if(visualize_with_zero_center_){
            map_pose_msg.pose.position.x = 0;
            map_pose_msg.pose.position.y = 0;
            map_pose_msg.pose.position.z = 0;
        }else{
            map_pose_msg.pose.position.x = current_camera_position_.x();
            map_pose_msg.pose.position.y = current_camera_position_.y();
            map_pose_msg.pose.position.z = current_camera_position_.z();
        }

        map_pose_msg.pose.orientation.w = current_camera_orientation_.w();
        map_pose_msg.pose.orientation.x = current_camera_orientation_.x();
        map_pose_msg.pose.orientation.y = current_camera_orientation_.y();
        map_pose_msg.pose.orientation.z = current_camera_orientation_.z();
        map_pose_pub_.publish(map_pose_msg);

        if(if_output_freespace_){
            // Transform Freespace point cloud to a ros message and publish
            sensor_msgs::PointCloud2 freespace_point_cloud_msg;
            pcl::toROSMsg(*freespace_point_cloud, freespace_point_cloud_msg);
            freespace_point_cloud_msg.header.frame_id = frame_id_;
            freespace_point_cloud_msg.header.stamp = ros::Time::now();
            freespace_point_pub_.publish(freespace_point_cloud_msg);
        }
    }
};


/// @brief Main function
/// @param argc 
/// @param argv Up to two arguments are allowed. The first argument is the yaml file name. The second argument is the object information csv file name.
/// @return 
int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapping_with_external_data");

    // Read the yaml file name from the command line
    std::string yaml_file = "options.yaml";
    std::string object_info_csv_file = "";

    if(argc == 2){
        yaml_file = argv[1];
        std::cout << "yaml_file: " << yaml_file << std::endl;
    }else if(argc == 3){
        yaml_file = argv[1];
        std::cout << "yaml_file: " << yaml_file << std::endl;

        object_info_csv_file = argv[2];
        std::cout << "object_info_csv_file: " << object_info_csv_file << std::endl;

    }else if(argc == 1){
        std::cout << "No yaml file is provided. Will use the default yaml file: options.yaml" << std::endl;
    }else{
        std::cout << "Error!!!! Too many arguments! The command should be: rosrun semantic_dsp_map mapping_with_external_data [yaml_file]" << std::endl;
        return -1;
    }

    if(yaml_file.find(".yaml") == std::string::npos){
        std::cout << "Error!!!! The yaml file name should end with .yaml" << std::endl;
        return -1;
    }

    if(object_info_csv_file == ""){
        MappingNode mapping_node(yaml_file);
    }else{
        MappingNode mapping_node(yaml_file, object_info_csv_file);
    }
    
    return 0;
}
