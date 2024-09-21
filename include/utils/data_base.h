/**
 * @file data_base.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This file contains the input data structure and object level data structure used in the map.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>

// A global variable to store the global time stamp
uint32_t global_time_stamp = 0;

/// @brief A struct to store the data of 2d bounding box
struct BBox2D
{
    int x1;
    int y1;
    int x2;
    int y2;
};

/// @brief A struct to store the data of 3d bounding box
struct BBox3D
{
    Eigen::Vector3d center;
    Eigen::Vector3d size;
    // Eigen::Vector3d orientation;
};

/// @brief A struct to store the data of a frame
struct InputFrameDataWithGT
{   
    // Dataset_Type
    int dataset_type;

    // Frame id
    int frame_id;

    // Camera pose in the world coordinate system.
    Eigen::Vector3d camera_position;
    Eigen::Quaterniond camera_orientation;
    
    // RGB image. CV_8UC3
    cv::Mat rgb_image;  

    // Colored depth image for visualization. CV_8UC3
    cv::Mat depth_image; 
    // Depth value matrix. CV_32FC1. meters.
    cv::Mat depth_value_mat; 

    // Colored segmentation image for visualization. CV_8UC3
    cv::Mat instance_image; 
    // Segmentation value mask matrix. CV_16UC1. Upper digits: label id. Lower digits: instance id.
    cv::Mat instance_value_mat;  

    // Ground truth depth image
    cv::Mat gt_depth_value_mat; 

    // Ground truth tracking data
    std::vector<Eigen::Vector3d> gt_tracked_object_positions;
    std::vector<Eigen::Quaterniond> gt_tracked_object_orientations;
    std::vector<int> gt_tracked_object_ids;
    std::vector<int> gt_tracked_object_labels;
};

/// @brief A struct to define a point with semantic id and track id
struct LabeledPoint
{
    ///< Position. Global coordinate system.
    Eigen::Vector3f position; //12 bytes
    ///< Standard deviation of the position
    float sigma; //4 bytes

    ///< Track id in Multi-Object Tracking (MOT)
    uint16_t track_id; //2 bytes
    ///< Semantic id (label)
    uint8_t label_id; //1 byte
    ///< If the point is valid
    bool is_valid;  //1 byte

};


/// @brief A struct to store the sensor odometry data for localization
struct OdomData
{
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};


/*************************************************************************************************************************************
 * The following part is the default data for the object labels. These data will be changed if a new object information csv file is read.
 * **********************************************************************************************************************************/

/// Define the label id for the object
std::unordered_map<std::string, int> g_label_id_map_default = {
    {"Background", 0},
    {"Terrain", 2},
    {"Sky", 3},
    {"Tree", 4},
    {"Vegetation", 5},
    {"Building", 6},
    {"Road", 7},
    {"GuardRail", 8},
    {"TrafficSign", 9},
    {"TrafficLight", 10},
    {"Pole", 11},
    {"Misc", 12},
    {"Truck", 13},
    {"Car", 14},
    {"Person", 15}
};


std::unordered_map<int, std::string> g_label_id_map_reversed = {
    {0, "Background"},
    {2, "Terrain"},
    {3, "Sky"},
    {4, "Tree"},
    {5, "Vegetation"},
    {6, "Building"},
    {7, "Road"},
    {8, "GuardRail"},
    {9, "TrafficSign"},
    {10, "TrafficLight"},
    {11, "Pole"},
    {12, "Misc"},
    {13, "Truck"},
    {14, "Car"},
    {15, "Person"}
};


std::unordered_set<int> g_movable_object_label_ids_set = {13, 14, 15};

/// Define the label id map for the static objects
std::unordered_map<std::string, int> g_label_id_map_static = {
    {"Background", 0},
    {"Terrain", 2},
    {"Sky", 3},
    {"Tree", 4},
    {"Vegetation", 5},
    {"Building", 6},
    {"Road", 7},
    {"GuardRail", 8},
    {"TrafficSign", 9},
    {"TrafficLight", 10},
    {"Pole", 11},
    {"Misc", 12}
};

/// Define the reversed label id map for the static objects
std::unordered_map<int, std::string> g_label_id_map_static_reversed = {
    {0, "Background"},
    {2, "Terrain"},
    {3, "Sky"},
    {4, "Tree"},
    {5, "Vegetation"},
    {6, "Building"},
    {7, "Road"},
    {8, "GuardRail"},
    {9, "TrafficSign"},
    {10, "TrafficLight"},
    {11, "Pole"},
    {12, "Misc"}
};

/// Define the instance id to label id map
std::unordered_map<std::string, int> g_label_to_instance_id_map_default = {
    {"Background", 65535},
    {"Terrain", 65534},
    {"Sky", 65533},
    {"Tree", 65532},
    {"Vegetation", 65531},
    {"Building", 65530},
    {"Road", 65529},
    {"GuardRail", 65528},
    {"TrafficSign", 65527},
    {"TrafficLight", 65526},
    {"Pole", 65525},
    {"Misc", 65524}
};

int g_max_movable_object_instance_id = 65523;

/// Define the label id to instance id map
std::unordered_map<int, std::string> g_instance_id_to_label_map_default = {
    {65535, "Background"},
    {65534, "Terrain"},
    {65533, "Sky"},
    {65532, "Tree"},
    {65531, "Vegetation"},
    {65530, "Building"},
    {65529, "Road"},
    {65528, "GuardRail"},
    {65527, "TrafficSign"},
    {65526, "TrafficLight"},
    {65525, "Pole"},
    {65524, "Misc"}
};


/// Define the color for each label. BGR
std::unordered_map<int, cv::Vec3b> g_label_color_map_default = {
    {0, cv::Vec3b(0, 0, 0)},
    {2, cv::Vec3b(200, 0, 210)},
    {3, cv::Vec3b(255, 200, 90)},
    {4, cv::Vec3b(0, 199, 0)},
    {5, cv::Vec3b(0, 240, 90)},
    {6, cv::Vec3b(140, 140, 140)},
    {7, cv::Vec3b(100, 60, 100)},
    {8, cv::Vec3b(255, 100, 250)},
    {9, cv::Vec3b(0, 255, 255)},
    {10, cv::Vec3b(0, 200, 200)},
    {11, cv::Vec3b(0, 130, 255)},
    {12, cv::Vec3b(80, 80, 80)},
    {13, cv::Vec3b(60, 60, 160)},
    {14, cv::Vec3b(80, 127, 255)},
    {15, cv::Vec3b(139, 139, 0)}
};




