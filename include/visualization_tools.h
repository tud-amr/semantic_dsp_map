/**
 * @file visualization_tools.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This file contains some simple visualization tools
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <random>
#include "data_base.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>


///@brief Define a function to map a value to a color
cv::Scalar mapValueToColor(double value, double minVal, double maxVal) {
    if(value < minVal) value = minVal;
    if(value > maxVal) value = maxVal;

    int value_int = int(value);
    
    static std::vector<cv::Scalar> colors(256);

    static bool initialized = false;
    if(!initialized) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        for(int i = 0; i < 256; i++) {
            colors[i] = cv::Scalar(dis(gen), dis(gen), dis(gen));
        }

        initialized = true;
    }

    return colors[value_int];
}


/// @brief Recolor the point cloud with the label color map defined using the Virtual KITTI 2 color map
/// @param source_cloud Original point cloud. The label id is stored in the r channel.
/// @param virtual_kitti_cloud Output point cloud with the color of the label color map.
/// @return Number of points in the output point cloud
int colorPointCloudWithVirtualKitti2Color(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &virtual_kitti_cloud)
{
    // Check if the source_cloud is empty
    if(source_cloud->points.size() == 0){
        std::cerr << "Error: source_cloud is empty." << std::endl;
        return -1;
    }

    // Resize the virtual_kitti_cloud with the same size as source_cloud
    virtual_kitti_cloud->points.resize(source_cloud->points.size());

    // Iterate through the source_cloud and color the virtual_kitti_cloud
    for(int i=0; i<source_cloud->points.size(); ++i)
    {
        pcl::PointXYZRGB point;
        uint8_t label_id = source_cloud->points[i].r;

        // Check if the label_id exists in label_color_map_default. If so, use the color in label_color_map_default. 
        if(label_color_map_default.find(label_id) != label_color_map_default.end()){
            point.r = label_color_map_default[label_id][2];
            point.g = label_color_map_default[label_id][1];
            point.b = label_color_map_default[label_id][0];
        }else{
            point.r = 255;
            point.g = 255;
            point.b = 255;
        }

        point.x = source_cloud->points[i].x;
        point.y = source_cloud->points[i].y;
        point.z = source_cloud->points[i].z;

        virtual_kitti_cloud->points[i] = point;
    }

    return virtual_kitti_cloud->points.size();
}

