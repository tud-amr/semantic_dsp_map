/**
 * @file data_base.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This file is the header file that handles tracking results given by the external tracking algorithm we use.
 * @version 0.1
 * @date 2023-10-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include "data_base.h"

struct MaskKpts
{
    int track_id;
    std::string label;

    std::vector<Eigen::Vector3d> kpts_current;
    std::vector<Eigen::Vector3d> kpts_previous;

    /// TODO: Merge the masks into a single image
    cv::Mat mask;
    BBox2D bbox;
};

class TrackingResultHandler
{
public:
    TrackingResultHandler(){};
    ~TrackingResultHandler(){};

    std::vector<MaskKpts> tracking_result;
};