/**
 * @file buffer.h
 * @brief This head file defines the ringbuffer for particle storage. The storage is shared by multiple threads.
 * 1. The size of the map is C_VOXEL_SIZE * 2^C_VOXEL_NUM_AXIS_N. e.g. 0.1m x 2^8 = 25.6m
 * 2. The index of a voxel in storage is encoded twice. One is encoded by ringbuffer index. The other is encoded by row-major/col-mojor/Morton-code.
 *    You can select the second encoding method by setting STORAGE_TYPE. The first encoding method is fixed.
 * 3. The particles are stored in voxels. The index of a particle is | voxel index bits | particle index bits |.
 * 
 * @author Clarence Chen (g-ch@github.com)
 * @version 0.1
 * @date 2023-06-28
 * @copyright Copyright (c) 2023
 * 
 */



#pragma once

#include <vector>
#include <array>
#include <Eigen/Dense>
#include "../settings/settings.h"

// Derived parameters
const uint32_t C_VOXEL_NUM_AXIS = 1 << C_VOXEL_NUM_AXIS_N; ///< The number of voxels on each axis. 
const uint32_t C_VOXEL_NUM_TOTAL = 1 << 3*C_VOXEL_NUM_AXIS_N; ///< The total number of voxels in the map.
const uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL = 1 << C_MAX_PARTICLE_NUM_PER_VOXEL_N; ///< Max particle number in a voxel.

/**
 * @brief The status of a particle
 * 
 */
enum class Particle_Status{
    INVALID,
    UPDATED,
    REGULAR_BORN,
    GUESSED_BORN,
    COPIED,
    TIMEPTC
}; 

/**
 * @brief The struct to store a particle's information
 * 
 */
struct Particle
{
    Particle_Status status; 
    uint32_t time_stamp;
    float occ_weight;
    float free_weight;

    uint8_t forget_count; ///< The number of time steps that the particle is not rightly updated. To forget the particles updated with wrong IDs.

    Eigen::Vector3f pos;
    uint8_t label_id;
    uint16_t track_id;  ///< TODO: Merge label_id and track_id.
    uint8_t color_h;  ///< TODO: Remove color_h.
};

///< Define the maximum number of particles in the map. The map is divided into voxels. Each voxel has a maximum number of C_MAX_PARTICLE_NUM_PER_VOXEL particles.
const uint32_t C_MAX_PARTICLE_NUM = C_VOXEL_NUM_AXIS * C_VOXEL_NUM_AXIS * C_VOXEL_NUM_AXIS * C_MAX_PARTICLE_NUM_PER_VOXEL;

///< Define a vector with pre-allocated size to store particles. Works like a array and will be indexed as a ringbuffer.
std::vector<Particle> PARTICLE_ARRAY(C_MAX_PARTICLE_NUM);

static const float c_max_pyramid_length = static_cast<float>(C_VOXEL_NUM_AXIS >> 1) * C_VOXEL_SIZE * 1.732f; ///< The maximum pyramid length. 
static const float c_max_pyramid_end_length = c_max_pyramid_length / g_camera_cx; ///< The maximum pyramid end length. 
static const float c_max_voxel_partition_in_pixel = (c_max_pyramid_end_length / C_VOXEL_SIZE) * (c_max_pyramid_end_length / C_VOXEL_SIZE); ///< The maximum voxel number in a pyramid projected to the image plane.
static const float c_max_particle_num_pyramid_scale = 0.3f; ///< Normally the space in a pyramid is not fully occupied by particles. This scale is used to estimate the maximum particle number in a pyramid.
const int C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID = static_cast<uint32_t>(c_max_particle_num_pyramid_scale * static_cast<float>(C_VOXEL_NUM_AXIS >> 1) * 1.732f * 0.3 * static_cast<float>(C_MAX_PARTICLE_NUM_PER_VOXEL-1) * c_max_voxel_partition_in_pixel); ///< The maximum particle number in a pyramid. 0.3 is a scale to use smaller memory. The overflowed particles (rare) are stored in a map.

///< Define an array to store the index of particles in each pyramid. The pyramids are projected to the image plane. Each pixel has a pyramid.
std::vector<std::vector<std::vector<uint32_t>>> particle_to_pixel_index_array(g_image_height, std::vector<std::vector<uint32_t>>(g_image_width, std::vector<uint32_t>(C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID, 0)));

///< Define an unordered map to store the index of particles that overflow the maximum particle number in a pyramid. The pyramids are projected to the image plane. Each pixel has a pyramid. The key is the pixel index given by row * g_image_width + col. The value is the overflowed particle indices in a pyramid.
std::unordered_map<uint32_t, std::vector<uint32_t>> particle_to_pixel_index_map;

///< Define an array to store the number of particles in each pyramid
std::vector<std::vector<uint32_t>> particle_to_pixel_num_array(g_image_height, std::vector<uint32_t>(g_image_width, 0));


/*********** Below are the variables frequently used. We define them here to share among threads. ***********/
float map_p_min_const; ///< Min map range on each axis (map frame). E.g. minimum -x or -y or -z
float map_p_max_const; ///< Max map range on each axis (map frame). E.g. maximum x or y or z
float voxel_size_recip; ///< Reciprocal of voxel size. 
Eigen::Vector3f map_center_pos; ///< The center position of the map. Used to calculate the voxel index in map frame.
Eigen::Vector3f ego_center_pos; ///< The center position of the ego vehicle. Updated every time step.

uint32_t low_index_move_bit_const; ///< The number of bits to move from index_x to the part of x in the complete voxel index 
uint32_t med_index_move_bit_const; // The number of bits to move from index_y to the part of y in the complete voxel index
uint32_t high_index_move_bit_const; ///< The number of bits to move from index_z to the part of z in the complete voxel index
uint32_t low_index_retrieve_mask_const; ///< The number used to mask other bits in an index except x's index
uint32_t med_index_retrieve_mask_const; ///< The number used to mask other bits in an index except y's index
uint32_t high_index_retrieve_mask_const; ///< The number used to mask other bits in an index except z's index

uint32_t voxel_time_stamps_x[C_VOXEL_NUM_AXIS]; ///< The time stamp of the voxels on x axis. Used to check if a particle is outdated. The time step is updated when the voxels with x index moves out of map range.
uint32_t voxel_time_stamps_y[C_VOXEL_NUM_AXIS]; ///< The time stamp of the voxels on y axis. Used to check if a particle is outdated. The time step is updated when the voxels with y index moves out of map range.
uint32_t voxel_time_stamps_z[C_VOXEL_NUM_AXIS]; ///< The time stamp of the voxels on z axis. Used to check if a particle is outdated. The time step is updated when the voxels with z index moves out of map range.

int buffer_moved_steps_x; ///< The number of voxels moved on x axis. Positive means moving to positive x direction. Negative means moving to negative x direction.
int buffer_moved_steps_y; ///< The number of voxels moved on y axis. Positive means moving to positive y direction. Negative means moving to negative y direction.
int buffer_moved_steps_z; ///< The number of voxels moved on z axis. Positive means moving to positive z direction. Negative means moving to negative z direction.

int buffer_moved_equivalent_steps_x; ///< The cycle length of the ringbuffer is C_VOXEL_NUM_AXIS. This variable is used to calculate the equivalent steps in the ringbuffer. 
int buffer_moved_equivalent_steps_y; ///< The cycle length of the ringbuffer is C_VOXEL_NUM_AXIS. This variable is used to calculate the equivalent steps in the ringbuffer.
int buffer_moved_equivalent_steps_z; ///< The cycle length of the ringbuffer is C_VOXEL_NUM_AXIS. This variable is used to calculate the equivalent steps in the ringbuffer.


