/**
 * @file external_evaluation.cpp
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
#include "dataset_process/file_retriever.h"
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "semantic_dsp_map.h"

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


class MappingNode
{
public:
    MappingNode():
    depth_pose_seq_(1000000),
    masks_seq_(1000000),
    if_output_freespace_(false)
    {
        initialize();
    };
    ~MappingNode(){};

    
private:
    cv::Mat depth_image_;
    Eigen::Vector3d current_camera_position_, last_camera_position_;
    Eigen::Quaterniond current_camera_orientation_, last_camera_orientation_;

    uint32_t depth_pose_seq_;
    uint32_t masks_seq_;

    TrackingResultHandler tracking_result_handler_;

    SemanticDSPMap dsp_map_;

    ros::NodeHandle nh_;

    ros::Publisher occupied_point_pub_;
    ros::Publisher occupied_point_vtk_color_pub_;
    ros::Publisher freespace_point_pub_;

    bool if_output_freespace_;

    /// @brief Initialize the node
    void initialize()
    {
        // Read the package path
        std::string package_path = ros::package::getPath("semantic_dsp_map");
        std::cout << "Package Path: " << package_path << std::endl;

        // Read yaml file to set the parameters
        YAML::Node config = YAML::LoadFile(package_path + "/cfg/options.yaml");
        bool if_consider_depth_noise = config["if_consider_depth_noise"].as<bool>();
        bool if_consider_tracking_noise = config["if_consider_tracking_noise"].as<bool>();
        bool if_use_pignistic_probability = config["if_use_pignistic_probability"].as<bool>();
        bool if_use_independent_filter = config["if_use_independent_filter"].as<bool>();
        bool if_out_evaluation_format = config["if_out_evaluation_format"].as<bool>();
        bool if_use_template_matching = config["if_use_template_matching"].as<bool>();
        if_output_freespace_ = config["if_output_freespace"].as<bool>();


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
        float match_score_threshold = config["match_score_threshold"].as<float>();


        float beyesian_movement_distance_threshold = config["beyesian_movement_distance_threshold"].as<float>();
        float beyesian_movement_probability_threshold = config["beyesian_movement_probability_threshold"].as<float>();
        float beyesian_movement_increment = config["beyesian_movement_increment"].as<float>();
        float beyesian_movement_decrement = config["beyesian_movement_decrement"].as<float>();
        
    
        // Set parameters for the SemanticDSPMap
        dsp_map_.setMapParameters(detection_probability, noise_number, nb_ptc_num_per_point, occupancy_threshold, max_obersevation_lost_time, forgetting_rate, match_score_threshold);
        dsp_map_.setTemplatePath(package_path + "/data/VirtualKitti2/template");
        dsp_map_.setMapOptions(if_consider_depth_noise, if_consider_tracking_noise, if_use_pignistic_probability, if_use_independent_filter, if_use_template_matching);
        dsp_map_.useEvaluationFormat(if_out_evaluation_format);
        dsp_map_.setBeyesianMovementParameters(beyesian_movement_distance_threshold, beyesian_movement_probability_threshold, beyesian_movement_increment, beyesian_movement_decrement);

        occupied_point_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("occupied_point", 1);
        occupied_point_vtk_color_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("occupied_point_vtk_color", 1);
        freespace_point_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("freespace_point", 1);
        
        // Create a synchronizer to subscribe the depth image and camera pose. Use syncDepthPoseCallback as the callback function 
        message_filters::Subscriber<sensor_msgs::Image> depth_image_sub(nh_, "/camera_depth_image", 1);
        message_filters::Subscriber<geometry_msgs::PoseStamped> camera_pose_sub(nh_, "/camera_pose", 1);
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_image_sub, camera_pose_sub);
        sync.registerCallback(boost::bind(&MappingNode::syncDepthPoseCallback, this, _1, _2));

        ros::Subscriber mask_sub = nh_.subscribe("/mask_group_super_glued", 1, &MappingNode::maskGroupSuperGluedCallback, this);

        ros::spin();
    }


    /// @brief The callback function for the depth image and camera pose subscriber
    /// @param depth_image_msg 
    /// @param camera_pose_msg 
    void syncDepthPoseCallback(const sensor_msgs::ImageConstPtr& depth_image_msg, const geometry_msgs::PoseStampedConstPtr& camera_pose_msg)
    {
        // Convert the depth image to cv::Mat
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(depth_image_msg, sensor_msgs::image_encodings::TYPE_32FC1);
            depth_image_ = cv_ptr->image;

            depth_pose_seq_ = depth_image_msg->header.seq;

            std::cout << "depth_pose_seq_: " << depth_pose_seq_  << std::endl;

            current_camera_position_ = Eigen::Vector3d(camera_pose_msg->pose.position.x, camera_pose_msg->pose.position.y, camera_pose_msg->pose.position.z);
            current_camera_orientation_ = Eigen::Quaterniond(camera_pose_msg->pose.orientation.w, camera_pose_msg->pose.orientation.x, camera_pose_msg->pose.orientation.y, camera_pose_msg->pose.orientation.z);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }


    /// @brief The callback function for the mask group subscriber
    /// @param msg
    void maskGroupSuperGluedCallback(const mask_kpts_msgs::MaskGroup::ConstPtr& msg)
    {
        // Check if the sequence number of depth image and mask group are the same
        masks_seq_ = msg->header.seq;
        if(depth_pose_seq_ != masks_seq_){
            std::cout << "Error!!!! The sequence number of depth image and mask group are not the same!" << std::endl;
            std::cout << "depth_pose_seq_: " << depth_pose_seq_ << ", masks_seq_: " << masks_seq_ << std::endl;
            return;
        }

        // Set new tracking result
        tracking_result_handler_.tracking_result.clear();

        float camera_fx = g_camera_fx;
        float camera_fy = g_camera_fy;
        float camera_cx = g_camera_cx;
        float camera_cy = g_camera_cy;

        for(int i = 0; i < msg->objects.size(); ++i)
        {
            // Only consider the object that has a valid track id. Static objects will also be considered.
            if(msg->objects[i].track_id < 0 || msg->objects[i].track_id > 65535){
                continue;
            }

            MaskKpts mask_kpts;
            mask_kpts.track_id = msg->objects[i].track_id;
            mask_kpts.label = msg->objects[i].label;

            mask_kpts.mask = cv_bridge::toCvCopy(msg->objects[i].mask, sensor_msgs::image_encodings::MONO8)->image;
            
            if(mask_kpts.label != "static") // If the object is not static, add the bounding box and keypoints
            {
                mask_kpts.bbox.x1 = msg->objects[i].bbox_tl.x;
                mask_kpts.bbox.y1 = msg->objects[i].bbox_tl.y;
                mask_kpts.bbox.x2 = msg->objects[i].bbox_br.x;
                mask_kpts.bbox.y2 = msg->objects[i].bbox_br.y;

                for(int j = 0; j < msg->objects[i].kpts_curr.size(); ++j)
                {
                    Eigen::Vector3d kpt;
                    kpt[0] = msg->objects[i].kpts_curr[j].x;
                    kpt[1] = msg->objects[i].kpts_curr[j].y;
                    kpt[2] = msg->objects[i].kpts_curr[j].z;
                    mask_kpts.kpts_current.push_back(kpt);
                }

                for(int j = 0; j < msg->objects[i].kpts_last.size(); ++j)
                {
                    Eigen::Vector3d kpt;
                    kpt[0] = msg->objects[i].kpts_last[j].x;
                    kpt[1] = msg->objects[i].kpts_last[j].y;
                    kpt[2] = msg->objects[i].kpts_last[j].z;
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

        // Draw the tracking result in one image for visualization
        if(tracking_result_handler_.tracking_result.size() > 0){
            int img_width = tracking_result_handler_.tracking_result[0].mask.cols;
            int img_height = tracking_result_handler_.tracking_result[0].mask.rows;

            // Set the matrix that turn global frame to camera frame
            Eigen::Matrix4d global_to_camera_current = Eigen::Matrix4d::Identity();
            global_to_camera_current.block<3, 3>(0, 0) = current_camera_orientation_.toRotationMatrix();
            global_to_camera_current.block<3, 1>(0, 3) = current_camera_position_;
            global_to_camera_current = global_to_camera_current.inverse().eval();

            Eigen::Matrix4d global_to_camera_last = Eigen::Matrix4d::Identity();
            global_to_camera_last.block<3, 3>(0, 0) = last_camera_orientation_.toRotationMatrix();
            global_to_camera_last.block<3, 1>(0, 3) = last_camera_position_;
            global_to_camera_last = global_to_camera_last.inverse().eval();

            cv::Mat tracking_result_img(img_height, img_width, CV_8UC3, cv::Scalar(0, 0, 0));
            for(int i = 0; i < tracking_result_handler_.tracking_result.size(); ++i)
            {
                // Only draw the bounding box and keypoints of the dynamic objects
                if(tracking_result_handler_.tracking_result[i].label == "static"){
                    continue;
                }

                cv::Mat mask = tracking_result_handler_.tracking_result[i].mask;
                for(int j = 0; j < mask.rows; ++j)
                {
                    for(int k = 0; k < mask.cols; ++k)
                    {
                        if(mask.at<uint8_t>(j, k) > 0){
                            tracking_result_img.at<cv::Vec3b>(j, k)[0] = 255;
                            tracking_result_img.at<cv::Vec3b>(j, k)[1] = 255;
                            tracking_result_img.at<cv::Vec3b>(j, k)[2] = 255;
                        }
                    }
                }
                
                // Draw the bounding box
                cv::rectangle(tracking_result_img, cv::Point(tracking_result_handler_.tracking_result[i].bbox.x1, tracking_result_handler_.tracking_result[i].bbox.y1), cv::Point(tracking_result_handler_.tracking_result[i].bbox.x2, tracking_result_handler_.tracking_result[i].bbox.y2), cv::Scalar(0, 0, 255), 1);
                
                // Draw the current keypoints
                for(int j = 0; j < tracking_result_handler_.tracking_result[i].kpts_current.size(); ++j)
                {
                    // Project the 3D point to the image plane using camera intrinsic parameters
                    Eigen::Vector4d global_frame_point, camera_frame_point;
                    global_frame_point.x() = tracking_result_handler_.tracking_result[i].kpts_current[j].x();
                    global_frame_point.y() = tracking_result_handler_.tracking_result[i].kpts_current[j].y();
                    global_frame_point.z() = tracking_result_handler_.tracking_result[i].kpts_current[j].z();
                    global_frame_point.w() = 1.0;

                    camera_frame_point = global_to_camera_current * global_frame_point;

                    double image_frame_x = camera_frame_point.x() / camera_frame_point.z() * camera_fx + camera_cx;
                    double image_frame_y = camera_frame_point.y() / camera_frame_point.z() * camera_fy + camera_cy;

                    // std::cout << "camera_frame_point" << camera_frame_point.transpose() << std::endl;
                    // std::cout << "image_frame_x: " << image_frame_x << ", image_frame_y: " << image_frame_y << std::endl;

                    cv::circle(tracking_result_img, cv::Point(image_frame_x, image_frame_y), 1, cv::Scalar(0, 255, 0), 1);

                    // Project keypoints in last frame to the current frame
                    Eigen::Vector4d global_frame_point_last, camera_frame_point_last;
                    global_frame_point_last.x() = tracking_result_handler_.tracking_result[i].kpts_previous[j].x();
                    global_frame_point_last.y() = tracking_result_handler_.tracking_result[i].kpts_previous[j].y();
                    global_frame_point_last.z() = tracking_result_handler_.tracking_result[i].kpts_previous[j].z();
                    global_frame_point_last.w() = 1.0;

                    camera_frame_point_last = global_to_camera_last * global_frame_point_last;

                    double image_frame_x_last = camera_frame_point_last.x() / camera_frame_point_last.z() * camera_fx + camera_cx;
                    double image_frame_y_last = camera_frame_point_last.y() / camera_frame_point_last.z() * camera_fy + camera_cy;

                    cv::circle(tracking_result_img, cv::Point(image_frame_x_last, image_frame_y_last), 1, cv::Scalar(0, 0, 255), 1);
                    
                    // cv::circle(tracking_result_img, tracking_result_handler_.tracking_result[i].kpts_current[j].pt, 1, cv::Scalar(0, 255, 0), 1);
                }
                // Draw the Track ID in the middle of the bounding box
                cv::putText(tracking_result_img, std::to_string(tracking_result_handler_.tracking_result[i].track_id), cv::Point((tracking_result_handler_.tracking_result[i].bbox.x1 + tracking_result_handler_.tracking_result[i].bbox.x2) / 2, (tracking_result_handler_.tracking_result[i].bbox.y1 + tracking_result_handler_.tracking_result[i].bbox.y2) / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }

            cv::imshow("tracking_result", tracking_result_img);
            cv::waitKey(1);
        }

        // For visualization variables update
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
        dsp_map_.update(depth_image_, tracking_result_handler_.tracking_result, current_camera_position_, current_camera_orientation_, occupied_point_cloud, freespace_point_cloud, if_output_freespace_);
    
        // Transform Occupied point cloud to a ros message and publish
        sensor_msgs::PointCloud2 occupied_point_cloud_msg;
        pcl::toROSMsg(*occupied_point_cloud, occupied_point_cloud_msg);

        occupied_point_cloud_msg.header.frame_id = "map";
        occupied_point_cloud_msg.header.stamp = ros::Time::now();
        occupied_point_pub_.publish(occupied_point_cloud_msg);

        // Publish a point cloud with vitrual kitti 2 color
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr occupied_point_cloud_vtk_color(new pcl::PointCloud<pcl::PointXYZRGB>);
        colorPointCloudWithVirtualKitti2Color(occupied_point_cloud, occupied_point_cloud_vtk_color);

        sensor_msgs::PointCloud2 occupied_point_cloud_vtk_color_msg;
        pcl::toROSMsg(*occupied_point_cloud_vtk_color, occupied_point_cloud_vtk_color_msg);
        occupied_point_cloud_vtk_color_msg.header.frame_id = "map";
        occupied_point_cloud_vtk_color_msg.header.stamp = ros::Time::now();
        occupied_point_vtk_color_pub_.publish(occupied_point_cloud_vtk_color_msg);

        if(if_output_freespace_){
            // Transform Freespace point cloud to a ros message and publish
            sensor_msgs::PointCloud2 freespace_point_cloud_msg;
            pcl::toROSMsg(*freespace_point_cloud, freespace_point_cloud_msg);
            freespace_point_cloud_msg.header.frame_id = "map";
            freespace_point_cloud_msg.header.stamp = ros::Time::now();
            freespace_point_pub_.publish(freespace_point_cloud_msg);

            // Save the freespace point cloud to a pcd file
            /*std::string file_path = "/home/cc/chg_ws/ros_ws/semantic_map_ws/src/data_reader_evaluator/data/freespace_folder/temp";
            std::string file_name = "freespace_" + std::to_string(depth_pose_seq_) + ".pcd";
            freespace_point_cloud->width = freespace_point_cloud->points.size();
            freespace_point_cloud->height = 1;
            pcl::io::savePCDFileASCII(file_path + "/" + file_name, *freespace_point_cloud);*/
        }
    }

};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapping_with_external_data");

    MappingNode mapping_node;

    return 0;
}
