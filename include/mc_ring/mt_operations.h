/**
 * @file mt_operations.h
 * @author your name (you@domain.com)
 * @brief This file contains the multi-threaded operations used in the map. The operations are used to speed up the map update.
 * 
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#pragma once


#include "mt_basic.h"
#include <thread>
#include <memory>
#include <math.h>



class MTRingBufferOperations : public RingBufferOperations
{
public:
    MTRingBufferOperations(){
        thread_num_ = 2;
    };

    ~MTRingBufferOperations(){};

    // Test the running time of multi-thread filling PARTICLE_ARRAY
    void mtTest()
    {
        mtClearBuffer();
    }

    /// @brief  Set the number of threads
    /// @param thread_num 
    void setThreadNum(uint32_t thread_num){
        // Check if the thread number is no less than 2
        if(thread_num < 2){
            std::cout << "Thread number should be no less than 2. Set thread number to 2." << std::endl;
            thread_num = 2;
        }else{
            thread_num_ = thread_num;
        }
    }

    
    /// @brief The Function use multiple threads to find the particles in the FOV with Breadth-first search. We search the vertexes and then check voxels to realize searching in continous space.
    /// @param extrinsic_matrix The extrinsic matrix of the camera
    /// @param depth_img The depth image. Used to check if a particle is occluded.
    void mtUpdateVisibleParitlcesWithBFS(const Eigen::Matrix4f& extrinsic_matrix, const cv::Mat &depth_img){
        // Set elements in particle_to_pixel_num_array to zero for a fresh start
        for(uint32_t i=0; i<g_image_height; ++i){
            for(uint32_t j=0; j<g_image_width; ++j){
                particle_to_pixel_num_array[i][j] = 0;
            }
        }
        // Clear particle_to_pixel_index_map
        particle_to_pixel_index_map.clear();

        mtGetIdxOfVisibleParitlcesWithPartition(extrinsic_matrix, depth_img);
    }


private:
    u_int32_t thread_num_; // The number of threads

    /// @brief Clear the PARTICLE_ARRAY
    void mtClearBuffer()
    {
        // Test the running time of multi-thread filling PARTICLE_ARRAY
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Define a vector to store the threads
        std::vector<std::thread> threads(thread_num_);

        // Divide the voxel into thread_num_ parts
        uint32_t voxel_num_per_thread = C_VOXEL_NUM_TOTAL / thread_num_;
        uint32_t voxel_num_remain = C_VOXEL_NUM_TOTAL % thread_num_;

        for(uint32_t i=0; i<thread_num_; ++i)
        {
            uint32_t start_voxel_idx = i * voxel_num_per_thread;
            uint32_t end_voxel_idx = (i+1) * voxel_num_per_thread;
            if(i == thread_num_-1){end_voxel_idx += voxel_num_remain;}
            
            // Create a thread
            std::shared_ptr<MTBufferClearOperation> op = std::make_shared<MTBufferClearOperation>(i, start_voxel_idx, end_voxel_idx);
            threads[i] = std::thread(*op);
        }

        for(uint32_t i=0; i<thread_num_; ++i)
        {
            threads[i].join();
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Time used for multi-thread filling PARTICLE_ARRAY: " << time_used.count() << " s" << std::endl;
    }


    /// @brief The Function uses multiple threads to find the particles in the FOV with Breadth-first search. We search the vertexes and then check voxels to realize searching in continous space.
    /// @param extrinsic_matrix The extrinsic matrix of the camera
    /// @param depth_img The depth image. Used to check if a particle is occluded.
    void mtGetIdxOfVisibleParitlcesWithPartition(const Eigen::Matrix4f& extrinsic_matrix, const cv::Mat &depth_img)
    {        
        // Calculate the global position of the start points. Make sure each thread has a start point.
        std::vector<Eigen::Vector3i> mt_bfs_start_point_idx_vec;
        Eigen::Matrix4f extrinsic_matrix_inv = extrinsic_matrix.inverse();
        for(uint32_t i=0; i<thread_num_; ++i){
            Eigen::Vector3f start_point_camera_frame;
            float start_z = g_depth_range_max / pow(1.26f, static_cast<float>(thread_num_ -1 - i)); // 2^(1/3) = 1.26. Divide the fov pyramid uniformly by its volume. 
            start_point_camera_frame << 0.f, 0.f, start_z;

            Eigen::Vector3f start_point_global_frame = extrinsic_matrix_inv.block<3,3>(0,0) * start_point_camera_frame + extrinsic_matrix_inv.block<3,1>(0,3);
            
            Eigen::Vector3f start_point_map_frame = start_point_global_frame - map_center_pos;
            Eigen::Vector3i start_point_idx;
            start_point_idx << static_cast<int>((start_point_map_frame.x()+map_p_max_const[0])*voxel_size_recip), static_cast<int>((start_point_map_frame.y()+map_p_max_const[1])*voxel_size_recip), static_cast<int>((start_point_map_frame.z()+map_p_max_const[2])*voxel_size_recip);

            mt_bfs_start_point_idx_vec.push_back(start_point_idx);
        }

        // Calculate the main axis of the camera center in map frame
        int main_axis = 0, max_value = 0;
        for(int i=0; i<3; ++i){
            int delt_axis = abs(mt_bfs_start_point_idx_vec[0](i) - mt_bfs_start_point_idx_vec[thread_num_-1](i));
            if(delt_axis > max_value){
                max_value = delt_axis;
                main_axis = i;
            }
        }

        // Calculate space partitioning along the main axis. The space is divided into thread_num_ cubic spaces along the main axis.
        std::vector<uint32_t> min_main_axis_idx(thread_num_);
        std::vector<uint32_t> max_main_axis_idx(thread_num_);
        min_main_axis_idx[0] = 0;
        max_main_axis_idx[0] = mt_bfs_start_point_idx_vec[0](main_axis);
        for(int i=1; i<thread_num_-1; ++i){
            min_main_axis_idx[i] = mt_bfs_start_point_idx_vec[i-1](main_axis);
            max_main_axis_idx[i] = mt_bfs_start_point_idx_vec[i](main_axis);
        }
        min_main_axis_idx[thread_num_-1] = mt_bfs_start_point_idx_vec[thread_num_-2](main_axis);
        // max_main_axis_idx[thread_num_-1] = C_VOXEL_NUM_AXIS;
        max_main_axis_idx[thread_num_-1] = 1 << C_VOXEL_NUM_AXIS_N_BIGGEST;

        // Reduce the main axis in mt_bfs_start_point_idx_vec by 1 to avoid the voxel in the boundary
        for(int i=0; i<thread_num_-1; ++i){
            mt_bfs_start_point_idx_vec[i](main_axis) -= 1;
            if(mt_bfs_start_point_idx_vec[i](main_axis) < 0) {mt_bfs_start_point_idx_vec[i](main_axis) = 0;}
        }
        // The last start point should be increased by 2 to the last-1 start point. A different search direction is used for the last thread.
        mt_bfs_start_point_idx_vec[thread_num_-1](main_axis) = mt_bfs_start_point_idx_vec[thread_num_-2](main_axis) + 2;

        // Start the threads
        std::vector<std::thread> threads(thread_num_);
        for(int i=0; i<thread_num_; ++i){
            std::shared_ptr<MTVisibleParitlcesIdxCheckOperation> op 
                = std::make_shared<MTVisibleParitlcesIdxCheckOperation>(i, mt_bfs_start_point_idx_vec[i], main_axis, min_main_axis_idx[i], max_main_axis_idx[i], extrinsic_matrix, depth_img);
            threads[i] = std::thread(*op);
            // Delay 3ms to avoid the threads start at the same time
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }

        for(int i=0; i<thread_num_; ++i){
            threads[i].join();
        }

    }
    
};


