
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
/// For example: The particle index works as: "| bits for z | bits for y | bits for x | bits for particle in a voxel |" for row major

#define STORAGE_TYPE 0 ///< 0: row major, 1: column major, 2: morton code. Morton code is a bit more efficient but needs C_VOXEL_NUM_AXIS_X_N==C_VOXEL_NUM_AXIS_Y_N==C_VOXEL_NUM_AXIS_Z_N (or C_VOXEL_NUM_AXIS_Z_N=C_VOXEL_NUM_AXIS_X_N-1). The row major and column major are more flexible.

#define VERBOSE_MODE 0 ///< 0: no verbose, 1: verbose: show more output information

#define SETTING 3 ///<0: KITTI_360, 1:CODA, 2:VIRTUAL_KITTI2, 3:ZED2


#if SETTING == 3 ///< ZED2 MODE. USE BOOST MODE
    #define BOOST_MODE 1 ///< 0: no boost, 1: boost
#else
    #define BOOST_MODE 0 ///< 0: no boost, 1: boost
#endif


#if SETTING == 0 ///< KITTI_360
    constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 8;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 8;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 8;

    constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 3;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
    constexpr float C_VOXEL_SIZE = 0.15f; ///< Voxel size. Unit: meter

    // Kitti-360 settings
    constexpr float g_camera_fx_set = 552.554261;
    constexpr float g_camera_fy_set = 552.554261;
    constexpr float g_camera_cx_set = 682.049453;
    constexpr float g_camera_cy_set = 238.769549;

    constexpr int g_image_width_set = 1408;
    constexpr int g_image_height_set = 376;

    constexpr bool g_consider_instance = false; ///< Whether to consider instance segmentation. If true, the instance id will be used to distinguish different objects. If false, the object id will be used to distinguish different objects.

    constexpr float g_depth_range_min = 0.3f; ///< Depth range min. Unit: meter
    constexpr float g_depth_range_max = 30.f; ///< Depth range max. Unit: meter. Usually we suppose the depth range is big than the max visiable depth of the map, i.e., every point in the map is visiable unless there is occlusion. Make the map smaller if you want to use a smaller depth range.


#elif SETTING == 1 ///< CODA
    constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 8;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 8;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 7;

    constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 2;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
    constexpr float C_VOXEL_SIZE = 0.15f; ///< Voxel size. Unit: meter

    // Coda settings
    constexpr float g_camera_fx_set = 569.8286f; ///< Focal length in x direction. Unit: pixel
    constexpr float g_camera_fy_set = 565.4818f; ///< Focal length in y direction. Unit: pixel
    constexpr float g_camera_cx_set = 439.2660f; ///< Principal point in x direction. Unit: pixel
    constexpr float g_camera_cy_set = 360.5810f; ///< Principal point in y direction. Unit: pixel

    constexpr int g_image_width_set = 960; ///< Image width. Unit: pixel
    constexpr int g_image_height_set = 540; ///< Image height. Unit: pixel

    constexpr bool g_consider_instance = true;

    constexpr float g_depth_error_stddev_at_one_meter = 0.1f; ///< Depth error standard deviation at one meter. Unit: meter 0.01

    constexpr float g_depth_range_min = 0.3f; ///< Depth range min. Unit: meter
    constexpr float g_depth_range_max = 10.f; ///< Depth range max. Unit: meter. Usually we suppose the depth range is big than the max visiable depth of the map, i.e., every point in the map is visiable unless there is occlusion. Make the map smaller if you want to use a smaller depth range.

#elif SETTING == 2 ///< VIRTUAL_KITTI2
    constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 8;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 7;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 8;

    constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 3;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
    constexpr float C_VOXEL_SIZE = 0.2f; ///< Voxel size. Unit: meter

    constexpr float g_camera_fx_set = 725.0087f; ///< Focal length in x direction. Unit: pixel
    constexpr float g_camera_fy_set = 725.0087f; ///< Focal length in y direction. Unit: pixel
    constexpr float g_camera_cx_set = 620.5f; ///< Principal point in x direction. Unit: pixel
    constexpr float g_camera_cy_set = 187.f; ///< Principal point in y direction. Unit: pixel

    constexpr int g_image_width_set = 1242; ///< Image width. Unit: pixel
    constexpr int g_image_height_set = 375; ///< Image height. Unit: pixel

    constexpr bool g_consider_instance = true;

    constexpr float g_depth_range_min = 0.3f; ///< Depth range min. Unit: meter
    constexpr float g_depth_range_max = 30.f; ///< Depth range max. Unit: meter. Usually we suppose the depth range is big than the max visiable depth of the map, i.e., every point in the map is visiable unless there is occlusion. Make the map smaller if you want to use a smaller depth range.

#elif SETTING == 3 ///< ZED2
    constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 7;
    constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 5;  // Y is the height. Use smaller range to reduce the computation.
    constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 7;

    constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 2;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
    constexpr float C_VOXEL_SIZE = 0.15f; ///< Voxel size. Unit: meter

    constexpr float g_camera_fx_set = 527.8191528320312;
    constexpr float g_camera_fy_set = 527.8191528320312;
    constexpr float g_camera_cx_set = 633.9357299804688;
    constexpr float g_camera_cy_set = 366.3338623046875;

    constexpr int g_image_width_set = 1280;
    constexpr int g_image_height_set = 720;

    constexpr bool g_consider_instance = true; ///< Whether to consider instance segmentation. If true, the instance id will be used to distinguish different objects. If false, the object id will be used to distinguish different objects.

    constexpr float g_depth_range_min = 0.3f; ///< Depth range min. Unit: meter
    constexpr float g_depth_range_max = 15.f; ///< Depth range max. Unit: meter. Usually we suppose the depth range is big than the max visiable depth of the map, i.e., every point in the map is visiable unless there is occlusion. Make the map smaller if you want to use a smaller depth range.

#else
    #error "Please define the SETTING macro to one of the supported parameters or add your own parameters."

#endif



#if BOOST_MODE == 0
    constexpr float g_camera_fx = g_camera_fx_set;
    constexpr float g_camera_fy = g_camera_fy_set;
    constexpr float g_camera_cx = g_camera_cx_set;
    constexpr float g_camera_cy = g_camera_cy_set;
    constexpr int g_image_width = g_image_width_set;
    constexpr int g_image_height = g_image_height_set;
#else
    /********* For Boost Mode. We make the image smaller  **********/
    constexpr float g_image_rescale = 0.5f;
    constexpr float g_camera_fx = g_image_rescale * g_camera_fx_set;
    constexpr float g_camera_fy = g_image_rescale * g_camera_fy_set;
    constexpr float g_camera_cx = g_image_rescale * g_camera_cx_set;
    constexpr float g_camera_cy = g_image_rescale * g_camera_cy_set;
    constexpr int g_image_width = g_image_rescale * g_image_width_set;
    constexpr int g_image_height = g_image_rescale * g_image_height_set;
#endif


constexpr float C_PARTICLE_OCC_INIT_WEIGHT = 0.05; ///< Initial weight of a newborn particle
constexpr float C_PARTICLE_FREE_INIT_WEIGHT = 0.05; ///< Initial weight of a newborn particle for free space. Not used in the current version.
constexpr float c_min_rightly_updated_pdf = 0.1f; ///< The minimum value of the pdf to treat one particle is correctly updated with an observation. Used for foggeting function.
constexpr float g_depth_error_stddev_at_one_meter = 0.1f; ///< Depth error standard deviation at one meter. Unit: meter 0.01. Not important in the current version.
