/**
 * @file mt_basic.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This file contains the basic operations for multi-threading.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "buffer.h"
#include <mutex>
#include "operations.h"

static std::mutex mt_basic_mtx; ///< Mutex for multi-threading

/**
 * @brief The operation to find the visible particles for a buffer. The visible particles are added to the pyramid.
 * 
 */
class MTVisibleParitlcesIdxCheckOperation : public RingBufferOperations{
public:
    MTVisibleParitlcesIdxCheckOperation(uint32_t id, Eigen::Vector3i start_point_idx, int main_axis, int main_axis_min_idx, int main_axis_max_idx, const Eigen::Matrix4f &extrinsic_matrix, const cv::Mat &depth_img)  
        : thread_id_(id), 
          start_point_idx_(start_point_idx),
          main_axis_(main_axis),
          main_axis_min_idx_(main_axis_min_idx),
          main_axis_max_idx_(main_axis_max_idx),
          extrinsic_matrix_(extrinsic_matrix),
          depth_img_(depth_img)
    {};

    ~MTVisibleParitlcesIdxCheckOperation(){};

    void operator()()
    {
        // Check if main_axis_max_idx is bigger than main_axis_min_idx
        if(main_axis_max_idx_ <= main_axis_min_idx_){
            std::cout << "main_axis_max_idx_ <= main_axis_min_idx_ in MTVisibleParitlcesIdxCheckOperation" << std::endl;
            return;
        }

        // Define variables and containers for getIdxOfVisibleParitlces        
        /// TODO: Use better way to calculate the size of the container and use a smaller container. using bool vector takes only one bit for each element.
        std::vector<std::vector<std::vector<bool>>> visited_points(C_VOXEL_NUM_AXIS_X+1, std::vector<std::vector<bool>>(C_VOXEL_NUM_AXIS_Y+1, std::vector<bool>(C_VOXEL_NUM_AXIS_Z+1, false)));
        std::vector<std::vector<std::vector<bool>>> added_voxels_idx(C_VOXEL_NUM_AXIS_X, std::vector<std::vector<bool>>(C_VOXEL_NUM_AXIS_Y, std::vector<bool>(C_VOXEL_NUM_AXIS_Z, false)));
        std::queue<Eigen::Vector3i> point_queue;

        // Calculate the offset variable from a voxel index to global frame position
        Eigen::Vector3f voxel_to_global_offset = map_center_pos + Eigen::Vector3f(map_p_min_const[0], map_p_min_const[1], map_p_min_const[2]);

        // Get the intrinsic matrix
        Eigen::Matrix3f intrinsic_matrix;
        intrinsic_matrix << g_camera_fx, 0, g_camera_cx, 0, g_camera_fy, g_camera_cy, 0, 0, 1;
        
        // Start the BFS from the point that is definately in the FOV
        point_queue.push(start_point_idx_);
        while(!point_queue.empty())
        {
            Eigen::Vector3i current_point = point_queue.front();
            point_queue.pop();

            if(visited_points[current_point.x()][current_point.y()][current_point.z()]) {continue;}

            // Check if the main axis is out of range
            if(current_point(main_axis_) < main_axis_min_idx_ || current_point(main_axis_) >= main_axis_max_idx_) {continue;}

            // Mark this point as visited
            visited_points[current_point.x()][current_point.y()][current_point.z()] = true;

            // Check if this point is in the FOV
            Eigen::Vector3f point_global_frame = current_point.cast<float>() * C_VOXEL_SIZE + voxel_to_global_offset;
            
            if(isPointInFrustum(point_global_frame, extrinsic_matrix_, intrinsic_matrix, g_image_width, g_image_height)){
                // Get the index of voxels that share this vertex
                for (int dx = -1; dx <= 0; dx++) {
                    for (int dy = -1; dy <= 0; dy++) {
                        for (int dz = -1; dz <= 0; dz++) {
                            Eigen::Vector3i adjacent_voxel_idx;
                            adjacent_voxel_idx.x() = current_point.x() + dx;
                            adjacent_voxel_idx.y() = current_point.y() + dy;
                            adjacent_voxel_idx.z() = current_point.z() + dz;

                            // Check for main axis. Skip if the voxel indices are out of bounds. 
                            if(adjacent_voxel_idx(main_axis_) < main_axis_min_idx_ || adjacent_voxel_idx(main_axis_) >= main_axis_max_idx_) {continue;}

                            // Skip if the voxel indices are out of bounds
                            if (adjacent_voxel_idx.x() < 0 || adjacent_voxel_idx.x() >= C_VOXEL_NUM_AXIS_X || adjacent_voxel_idx.y() < 0 || adjacent_voxel_idx.y() >= C_VOXEL_NUM_AXIS_Y || adjacent_voxel_idx.z() < 0 || adjacent_voxel_idx.z() >= C_VOXEL_NUM_AXIS_Z) {continue;}

                            // Check if voxel has been added before
                            if (added_voxels_idx[adjacent_voxel_idx.x()][adjacent_voxel_idx.y()][adjacent_voxel_idx.z()]){continue;}
                            
                            // Calculate the voxel index in storage
                            uint32_t voxel_idx, buffer_x_index, buffer_y_index, buffer_z_index;
                            mapXYZIdxToRingbufferXYZIdx(adjacent_voxel_idx.x(), adjacent_voxel_idx.y(), adjacent_voxel_idx.z(), buffer_x_index, buffer_y_index, buffer_z_index);
                            ringbufferXYZIdxToVoxelIdx(buffer_x_index, buffer_y_index, buffer_z_index, voxel_idx);
                            uint32_t particle_start_index = voxel_idx << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

                            // Update the time particle. The time particle is always the first particle in the voxel.
                            PARTICLE_ARRAY[particle_start_index].time_stamp = global_time_stamp;

                            // Add the particle index to the pyramid if the particle is in depth range. Ignore the time particle.
                            for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i){
                                Particle *ptc = &PARTICLE_ARRAY[particle_start_index+i];

                                // Check if the particle is valid and not outdated
                                if(ptc->status != Particle_Status::INVALID && ptc->time_stamp >= voxel_time_stamps_x[buffer_x_index] && ptc->time_stamp >= voxel_time_stamps_y[buffer_y_index] && ptc->time_stamp >= voxel_time_stamps_z[buffer_z_index]){
                                    // Calculate the particle position the image frame
                                    int row, col;
                                    float camera_frame_z;
                                    if(calculateParticleBasicStateInImage(*ptc, extrinsic_matrix_, intrinsic_matrix, row, col, camera_frame_z)){
                                        // Check if the particle is in depth range
                                        /// TODO: Use faster way to determine the visibility of a particle.
                                        static const float one_sigma_error_coeff = g_depth_error_stddev_at_one_meter + 1.f;
                                        if(camera_frame_z > depth_img_.at<float>(row, col) * one_sigma_error_coeff){
                                            // Skip if the particle is out of depth range. The particle will not be updated.
                                            continue;
                                        }
                                        
                                        int id = row * g_image_width + col;
                                        mt_basic_mtx.lock();
                                        particle_to_pixel_index_map[id].push_back(particle_start_index+i);
                                        particle_to_pixel_num_array[row][col] ++;
                                        mt_basic_mtx.unlock();
                                    }
                                }
                            }

                            // Mark the voxel as added
                            added_voxels_idx[adjacent_voxel_idx.x()][adjacent_voxel_idx.y()][adjacent_voxel_idx.z()] = true;
                        }
                    }
                }

                static const int dir[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

                // Add the 6 neighboring points to the queue
                for(int i=0; i<6; ++i){
                    Eigen::Vector3i neighbor_point;
                    neighbor_point.x() = current_point.x() + dir[i][0];
                    neighbor_point.y() = current_point.y() + dir[i][1];
                    neighbor_point.z() = current_point.z() + dir[i][2];

                    // Check main axis. Skip if the point indices are out of bounds.
                    if(neighbor_point(main_axis_) < main_axis_min_idx_ || neighbor_point(main_axis_) >= main_axis_max_idx_) {continue;}

                    // Skip if the point indices are out of bounds or the point has been visited
                    if (neighbor_point.x() < 0 || neighbor_point.x() > C_VOXEL_NUM_AXIS_X || neighbor_point.y() < 0 || neighbor_point.y() > C_VOXEL_NUM_AXIS_Y || neighbor_point.z() < 0 || neighbor_point.z() > C_VOXEL_NUM_AXIS_Z || visited_points[neighbor_point.x()][neighbor_point.y()][neighbor_point.z()]){continue;}
                    // Add the point to the queue
                    point_queue.push(neighbor_point);
                }
            }
        }
    }


private:
    int thread_id_;
    int main_axis_;
    int main_axis_min_idx_;
    int main_axis_max_idx_;
    Eigen::Vector3i start_point_idx_;

    Eigen::Matrix4f extrinsic_matrix_;
    cv::Mat depth_img_;
};


/**
 * @brief The operation to clear the buffer for a thread.
 *  
 */
class MTBufferClearOperation{
public:
    MTBufferClearOperation(uint32_t id, uint32_t start_vxl_idx, uint32_t end_vxl_idx)  
        : thread_id(id), 
          start_voxel_idx(start_vxl_idx),
          end_voxel_idx(end_vxl_idx)
    {};

    ~MTBufferClearOperation(){};

    void operator()()
    {
        // Clear voxel by voxel. Retain the first time particle.
        for(uint32_t i=start_voxel_idx; i<end_voxel_idx; ++i)
        {
            uint32_t start_ptc_seq = i << C_MAX_PARTICLE_NUM_PER_VOXEL_N;
            // Add time particle. Always the first particle in a voxel.
            PARTICLE_ARRAY[start_ptc_seq].status = Particle_Status::TIMEPTC;
            PARTICLE_ARRAY[start_ptc_seq].pos.x = 0.f;
            PARTICLE_ARRAY[start_ptc_seq].pos.y = 0.f;
            PARTICLE_ARRAY[start_ptc_seq].pos.z = 0.f;
            PARTICLE_ARRAY[start_ptc_seq].pos.weight = 0.f;
            PARTICLE_ARRAY[start_ptc_seq].time_stamp = 0;

            for(int j=1; j<C_MAX_PARTICLE_NUM_PER_VOXEL; ++j)
            {
                PARTICLE_ARRAY[start_ptc_seq+j].status = Particle_Status::INVALID;
                PARTICLE_ARRAY[start_ptc_seq+j].pos.x = 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].pos.y = 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].pos.z = 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].pos.weight = 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].time_stamp = 0;
            }

        }
    }

private:
    uint32_t thread_id;
    uint32_t start_voxel_idx;
    uint32_t end_voxel_idx;
};
