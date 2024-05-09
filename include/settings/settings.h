
/**
 * @file settings.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This head file includes the parameters that can be changed by the user.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

/// NOTE: C_VOXEL_NUM_AXIS_N*3 + C_MAX_PARTICLE_NUM_PER_VOXEL_N should be no more than 31
/// The particle index works as: | bits for z | bits for y | bits for x | bits for particle in a voxel |

#define STORAGE_TYPE 0 ///< 0: row major, 1: column major, 2: morton code

/// TODO: Make ring buffer size of Z axis configurable

const uint8_t C_VOXEL_NUM_AXIS_N = 8;  ///< Number of bits used to store the voxel index along an axis. The voxel number on one axis is then 2^C_VOXEL_NUM_AXIS_N
const uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 2;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
const float C_VOXEL_SIZE = 0.2f; ///< Voxel size. Unit: meter

/// TODO: Make the following configurable. 0.3 is just for debugging
const float C_PARTICLE_OCC_INIT_WEIGHT = 0.001; ///< Initial weight of a newborn particle
const float C_PARTICLE_FREE_INIT_WEIGHT = 0.001; ///< Initial weight of a newborn particle

const float c_min_rightly_updated_pdf = 0.1f;

const float g_camera_fx = 725.0087f; ///< Focal length in x direction. Unit: pixel
const float g_camera_fy = 725.0087f; ///< Focal length in y direction. Unit: pixel
const float g_camera_cx = 620.5f; ///< Principal point in x direction. Unit: pixel
const float g_camera_cy = 187.f; ///< Principal point in y direction. Unit: pixel

const int g_image_width = 1242; ///< Image width. Unit: pixel
const int g_image_height = 375; ///< Image height. Unit: pixel

const float g_depth_error_stddev_at_one_meter = 0.05f; ///< Depth error standard deviation at one meter. Unit: meter 0.01

const float g_depth_range_min = 0.3f; ///< Depth range min. Unit: meter
const float g_depth_range_max = 40.f; ///< Depth range max. Unit: meter. Usually we suppose the depth range is big than the max visiable depth of the map, i.e., every point in the map is visiable unless there is occlusion. Make the map smaller if you want to use a smaller depth range.

