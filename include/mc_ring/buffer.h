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
// const uint32_t C_VOXEL_NUM_AXIS = 1 << C_VOXEL_NUM_AXIS_N; ///< The number of voxels on each axis. 

const uint32_t C_VOXEL_NUM_AXIS_X = 1 << C_VOXEL_NUM_AXIS_X_N; ///< The number of voxels on x axis.
const uint32_t C_VOXEL_NUM_AXIS_Y = 1 << C_VOXEL_NUM_AXIS_Y_N; ///< The number of voxels on y axis.
const uint32_t C_VOXEL_NUM_AXIS_Z = 1 << C_VOXEL_NUM_AXIS_Z_N; ///< The number of voxels on z axis.

const uint32_t C_VOXEL_NUM_AXIS_N_XY_BIGGEST = C_VOXEL_NUM_AXIS_X_N > C_VOXEL_NUM_AXIS_Y_N ? C_VOXEL_NUM_AXIS_X_N : C_VOXEL_NUM_AXIS_Y_N;
const uint32_t C_VOXEL_NUM_AXIS_N_BIGGEST = C_VOXEL_NUM_AXIS_N_XY_BIGGEST > C_VOXEL_NUM_AXIS_Z_N ? C_VOXEL_NUM_AXIS_N_XY_BIGGEST : C_VOXEL_NUM_AXIS_Z_N;

// const uint32_t C_VOXEL_NUM_TOTAL = 1 << 3*C_VOXEL_NUM_AXIS_N; ///< The total number of voxels in the map.

const uint32_t C_VOXEL_NUM_TOTAL = C_VOXEL_NUM_AXIS_X * C_VOXEL_NUM_AXIS_Y * C_VOXEL_NUM_AXIS_Z; ///< The total number of voxels in the map.
const uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL = 1 << C_MAX_PARTICLE_NUM_PER_VOXEL_N; ///< Max particle number in a voxel.

/**
 * @brief The status of a particle
 * 
 */
enum class Particle_Status : uint8_t{
    INVALID,
    UPDATED,
    REGULAR_BORN,
    GUESSED_BORN,
    COPIED,
    TIMEPTC
}; 


/**
 * @brief The struct to store the basic state of a particle
 * 
 */
struct ParticleBasicState{
    float x;
    float y;
    float z;
    float weight;
};

/**
 * @brief The struct to store a particle's information with more states
 * 
 */
struct Particle
{
    // Group the largest data types together and align them on natural boundaries
    ParticleBasicState pos;          // 16 bytes 

    // Following types are smaller but grouped together for better packing
    uint16_t time_stamp;          // 2 bytes
    uint16_t track_id;            // 2 bytes
    uint16_t label_id;             // 1 byte
    Particle_Status status;       // 1 byte
    uint8_t forget_count;         // 1 byte
};

///< Define the maximum number of particles in the map. The map is divided into voxels. Each voxel has a maximum number of C_MAX_PARTICLE_NUM_PER_VOXEL particles.
// const uint32_t C_MAX_PARTICLE_NUM = C_VOXEL_NUM_AXIS * C_VOXEL_NUM_AXIS * C_VOXEL_NUM_AXIS * C_MAX_PARTICLE_NUM_PER_VOXEL;
const uint32_t C_MAX_PARTICLE_NUM = C_VOXEL_NUM_TOTAL * C_MAX_PARTICLE_NUM_PER_VOXEL; ///< The maximum number of particles in the map.

///< Define the particle array to store all the particles in the map.
std::array<Particle, C_MAX_PARTICLE_NUM> PARTICLE_ARRAY;


///< Define an unordered map to store the index of particles in a pyramids. The pyramids are projected to the image plane. Each pixel has a pyramid. The key is the pixel index given by row * g_image_width + col. The value is the overflowed particle indices in a pyramid.
std::unordered_map<uint32_t, std::vector<uint32_t>> particle_to_pixel_index_map;

///< Define an array to store the number of particles in each pyramid
std::vector<std::vector<uint32_t>> particle_to_pixel_num_array(g_image_height, std::vector<uint32_t>(g_image_width, 0));


/*********** Below are the variables frequently used. We define them here to share among threads. ***********/
float map_p_min_const[3]; ///< Min map range on each axis (map frame). E.g. minimum -x or -y or -z
float map_p_max_const[3]; ///< Max map range on each axis (map frame). E.g. maximum x or y or z
float voxel_size_recip; ///< Reciprocal of voxel size. 
Eigen::Vector3f map_center_pos; ///< The center position of the map. Used to calculate the voxel index in map frame.
Eigen::Vector3f ego_center_pos; ///< The center position of the ego vehicle. Updated every time step.

uint32_t low_index_move_bit_const; ///< The number of bits to move from index_x to the part of x in the complete voxel index in row major case or z's index in column major case
uint32_t med_index_move_bit_const; // The number of bits to move from index_y to the part of y in the complete voxel index
uint32_t high_index_move_bit_const; ///< The number of bits to move from index_z to the part of z in the complete voxel index in row major case or x's index in column major case
uint32_t low_index_retrieve_mask_const; ///< The number used to mask other bits in an index except x's index in row major case or z's index in column major case
uint32_t med_index_retrieve_mask_const; ///< The number used to mask other bits in an index except y's index
uint32_t high_index_retrieve_mask_const; ///< The number used to mask other bits in an index except z's index in row major case or x's index in column major case

uint32_t voxel_time_stamps_x[C_VOXEL_NUM_AXIS_X]; ///< The time stamp of the voxels on x axis. Used to check if a particle is outdated. The time step is updated when the voxels with x index moves out of map range.
uint32_t voxel_time_stamps_y[C_VOXEL_NUM_AXIS_Y]; ///< The time stamp of the voxels on y axis. Used to check if a particle is outdated. The time step is updated when the voxels with y index moves out of map range.
uint32_t voxel_time_stamps_z[C_VOXEL_NUM_AXIS_Z]; ///< The time stamp of the voxels on z axis. Used to check if a particle is outdated. The time step is updated when the voxels with z index moves out of map range.

int buffer_moved_steps_x; ///< The number of voxels moved on x axis. Positive means moving to positive x direction. Negative means moving to negative x direction.
int buffer_moved_steps_y; ///< The number of voxels moved on y axis. Positive means moving to positive y direction. Negative means moving to negative y direction.
int buffer_moved_steps_z; ///< The number of voxels moved on z axis. Positive means moving to positive z direction. Negative means moving to negative z direction.

int buffer_moved_equivalent_steps_x; ///< The cycle length of the ringbuffer is C_VOXEL_NUM_AXIS_X. This variable is used to calculate the equivalent steps in the ringbuffer. 
int buffer_moved_equivalent_steps_y; ///< The cycle length of the ringbuffer is C_VOXEL_NUM_AXIS_Y. This variable is used to calculate the equivalent steps in the ringbuffer.
int buffer_moved_equivalent_steps_z; ///< The cycle length of the ringbuffer is C_VOXEL_NUM_AXIS_Z. This variable is used to calculate the equivalent steps in the ringbuffer.


