/********************************************************************************************************************************************
 * @file operations.h
 * @author Clarence Chen (g-ch@github.com)
 * 
 * @brief This head file defines some basic operations for particles in the ringbuffer. 
 * 
 * Usage:
 * 1. updateEgoCenterPos
 * 2. Add a particle
 * 3. Remove a particle
 * 4. Calculate weight summation in a voxel
 * 5. Rendering to get occupancy status and semantics
 * 
 * Below is some information that developers should know.
 * 1. Note the position displacement between two update steps should be less than the map size. (Don't move like a lightning.)
 * 2. The interface uses global frame for ego position and particles' positions. 
 * 3. Run runSystemChecking() and runInitialization() only once.
 * 
 * @version 0.1
 * @date 2023-06-28
 * @copyright Copyright (c) 2023
 * 
 */


#pragma once

#include <random>
#include "libmorton/include/libmorton/morton.h"
#include "buffer.h"
#include <iostream>
#include <unordered_set>
#include <pcl/point_types.h>

#define INVALID_PARTICLE_INDEX 0xffffffff

/**
 * @brief The class to define the basic operations for particles in the ringbuffer. Single thread.
 * 
 */
class RingBufferOperations
{
public:
    RingBufferOperations(){

        runSystemChecking(); // Run system checking

        initialize(); // Initialize the ring buffer
    }

    ~RingBufferOperations(){};

    /// @brief  Run system checking
    void runSystemChecking(){
        if(C_VOXEL_NUM_AXIS_N*3 + C_MAX_PARTICLE_NUM_PER_VOXEL_N > 31){
            std::cout << "ERROR: C_VOXEL_NUM_AXIS_N*3 + C_MAX_PARTICLE_NUM_PER_VOXEL_N should be no more than 31" << std::endl;
            exit(-1);
        }

        if(sizeof(uint32_t) != 4){
            std::cout << "ERROR: uint32_t is not 4 bytes in this system! System not supported!" << std::endl;
            exit(-1);
        }
    }

    /// @brief Update the ego center position. Necessary for each step.
    /// @param pos 
    void updateEgoCenterPos(const Eigen::Vector3f &pos){
        // Update the ego center position and the time stamp of the map
        ego_center_pos = pos; // Update ego center position.

        updateRingbufferIndexParams(); // Update the ring buffer indexes
    }

    /// @brief Get the position of a particle by its index
    /// @param particle_index 
    /// @param pos 
    inline void getParticlePosByIndex(const uint32_t &particle_index, Eigen::Vector3f &pos){
        pos = PARTICLE_ARRAY[particle_index].pos;
    }

    /// @brief Get the weight and position of a particle by its index\
    /// @param particle_index
    /// @param pos
    /// @param occ_weight
    /// @param free_weight
    inline void getParticlePosWeightByIndex(const uint32_t &particle_index, Eigen::Vector3f &pos, float &occ_weight, float &free_weight){
        pos = PARTICLE_ARRAY[particle_index].pos;
        occ_weight = PARTICLE_ARRAY[particle_index].occ_weight;
        free_weight = PARTICLE_ARRAY[particle_index].free_weight;
    }


    /// @brief Add a set of particles given in point cloud form to the ring buffer
    /// @param ptc_cloud 
    /// @param label_id 
    /// @param track_id 
    /// @param ptc_indices 
    /// @return 
    inline uint32_t addMatchedParticles(const pcl::PointCloud<pcl::PointXYZ>::Ptr &ptc_cloud, const uint8_t &label_id, const uint16_t &track_id, std::unordered_set<uint32_t> &ptc_indices){
        uint32_t voxel_index, particle_index;
        for(int i=0; i<ptc_cloud->size(); ++i){
            Particle particle;
            particle.status = Particle_Status::GUESSED_BORN;
            particle.time_stamp = global_time_stamp;
            particle.forget_count = 0;
            particle.pos.x() = ptc_cloud->points[i].x;
            particle.pos.y() = ptc_cloud->points[i].y;
            particle.pos.z() = ptc_cloud->points[i].z;
            /// TODO: Use real weight of the particle
            particle.occ_weight = 0.2; 
            particle.free_weight = 0;
            particle.label_id = label_id;
            particle.track_id = track_id;
            particle.color_h = 0;

            particle_index = addParticleByGlobalPos(particle, voxel_index);
            if(particle_index != INVALID_PARTICLE_INDEX){
                ptc_indices.insert(particle_index);
            }
        }
        return ptc_indices.size();
    }


    /// @brief Add a new particle to a voxel with a global position
    /// @param particle_pos global position of the particle. The status of the particle is set to REGULAR_BORN.
    /// @return if success, return the particle index; if the voxel is full or the voxel is outside the map, return 0xffffffff.
    inline uint32_t addNewParticle(const Eigen::Vector3f &particle_pos){
        Particle particle;
        particle.status = Particle_Status::REGULAR_BORN;
        particle.time_stamp = global_time_stamp;
        particle.forget_count = 0;
        particle.pos = particle_pos;
        particle.occ_weight = C_PARTICLE_OCC_INIT_WEIGHT;
        particle.free_weight = C_PARTICLE_FREE_INIT_WEIGHT;

        return addParticleByGlobalPos(particle);
    }

    /// @brief Add a new particle to a voxel with a global position and semantics information
    /// @param particle_pos 
    /// @param label_id 
    /// @param track_id 
    /// @param color_h 
    /// @param voxel_index Return the voxel index of the particle. 0xffffffff if failed.
    /// @param particle_index Return the particle index of the particle. 0xffffffff if failed.
    inline void addNewParticleWithSemantics(const Eigen::Vector3f &particle_pos, const uint8_t &label_id, const uint16_t &track_id, const uint8_t &color_h, uint32_t &voxel_index, uint32_t &particle_index){
        Particle particle;
        particle.status = Particle_Status::REGULAR_BORN;
        particle.time_stamp = global_time_stamp;
        particle.forget_count = 0;
        particle.pos = particle_pos;
        particle.occ_weight = C_PARTICLE_OCC_INIT_WEIGHT;
        particle.free_weight = C_PARTICLE_FREE_INIT_WEIGHT;
        particle.label_id = label_id;
        particle.track_id = track_id;
        particle.color_h = color_h;

        particle_index = addParticleByGlobalPos(particle, voxel_index);
    }

    /// @brief Add a new Guessed particle to a voxel with a global position and semantics information
    /// @param particle_pos 
    /// @param label_id 
    /// @param track_id 
    /// @param color_h 
    /// @param voxel_index Return the voxel index of the particle. 0xffffffff if failed.
    /// @param particle_index Return the particle index of the particle. 0xffffffff if failed.
    inline void addGuessedParticles(const Eigen::Vector3f &particle_pos, const uint8_t &label_id, const uint16_t &track_id, const uint8_t &color_h, uint32_t &voxel_index, uint32_t &particle_index){
        Particle particle;
        particle.status = Particle_Status::GUESSED_BORN;
        particle.time_stamp = global_time_stamp;
        particle.forget_count = 0;
        particle.pos = particle_pos;
        particle.occ_weight = C_PARTICLE_OCC_INIT_WEIGHT;
        particle.free_weight = C_PARTICLE_FREE_INIT_WEIGHT;
        particle.label_id = label_id;
        particle.track_id = track_id;
        particle.color_h = color_h;

        particle_index = addParticleByGlobalPos(particle, voxel_index);
    }

    /// @brief Add a particle to a voxel by its a global position
    /// @param particle A particle. 
    /// @return if success, return the particle index; if the voxel is full, return -2; if the voxel is outside the map, return -1.
    inline uint32_t addExistingParticleByGlobalPos(const Particle &particle){
        return addParticleByGlobalPos(particle);
    }

    /// @brief Delete particles in a set composed of their indices
    /// @param ptc_indices 
    inline void deleteParticlesInSet(const std::unordered_set<uint32_t> &ptc_indices)
    {
        for(auto it = ptc_indices.begin(); it != ptc_indices.end(); ++it){
            deleteParticleByIndex(*it);
        }
    }

    /// @brief Delete a particle by its index
    /// @param particle_index
    inline void deleteParticleByIndex(const uint32_t &particle_index){
        PARTICLE_ARRAY[particle_index].status = Particle_Status::INVALID;
    }

    /// @brief Delete particles in a voxel the index of the time particle, which is the first particle in the voxel
    /// @param time_particle_index
    inline void deleteParticlesInVoxelByTimeParticleIndex(const uint32_t &time_particle_index){
        PARTICLE_ARRAY[time_particle_index].time_stamp = 0; // Set the time stamp to 0 to indicate that the voxel is invalid.         
        for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i){
            PARTICLE_ARRAY[time_particle_index+i].status = Particle_Status::INVALID;
        }
    }

    /// @brief Retrieve the particles whose indices are in a specified set to a point cloud
    /// @param ptc_indices 
    /// @param ptc_cloud 
    /// @param weight_threshold 
    inline void getParticlesAsPointCloud(const std::unordered_set<uint32_t> &ptc_indices, pcl::PointCloud<pcl::PointXYZ>::Ptr &ptc_cloud, double weight_threshold = 0.1){
        ptc_cloud->clear();
        ptc_cloud->reserve(ptc_indices.size());
        for(auto it = ptc_indices.begin(); it != ptc_indices.end(); ++it){
            if(PARTICLE_ARRAY[*it].occ_weight > weight_threshold){
                pcl::PointXYZ ptc;
                ptc.x = PARTICLE_ARRAY[*it].pos.x();
                ptc.y = PARTICLE_ARRAY[*it].pos.y();
                ptc.z = PARTICLE_ARRAY[*it].pos.z();
                ptc_cloud->push_back(ptc);
            }
        }

    }


    /// @brief Move a particle to a new position
    /// @param particle_index 
    /// @param new_pos 
    /// @param particle_new_index Return the new index of the particle if success. Otherwise, return -1 or -2.
    inline uint32_t moveParticleToNewPosition(const uint32_t &particle_index, const Eigen::Vector3f &new_pos){
        // Update the particle position and add it to the new voxel
        PARTICLE_ARRAY[particle_index].pos = new_pos;
        uint32_t particle_index_new = addExistingParticleByGlobalPos(PARTICLE_ARRAY[particle_index]);

        // Delete the particle from the old voxel regardless if the particle is added to the new voxel successfully.
        deleteParticleByIndex(particle_index);

        return particle_index_new;
    }

    /// @brief Move particles in a set to a new position by a transformation matrix. Return the new indices of the particles.
    /// @param ptc_indices A set of particle indices to be moved
    /// @param t_matrix The transformation matrix
    /// @param new_ptc_indices The new indices of the particles
    inline void moveParticlesInSetByTransformation(const std::unordered_set<uint32_t> &ptc_indices, const Eigen::Matrix4f &t_matrix, std::unordered_set<uint32_t> &new_ptc_indices)
    {
        const int particle_vector_size = ptc_indices.size();
        std::vector<Particle> new_particles(particle_vector_size);

        // Make a copy of the particles to be moved
        int i = 0;
        for(auto it = ptc_indices.begin(); it != ptc_indices.end(); ++it){
            new_particles[i] = PARTICLE_ARRAY[*it];
            new_particles[i].pos = t_matrix.block<3, 3>(0, 0) * PARTICLE_ARRAY[*it].pos + t_matrix.block<3, 1>(0, 3);

            deleteParticleByIndex(*it);
            ++i;
        }

        // Add the copy of the particles if the voxel is not full
        for(int i=0; i<particle_vector_size; ++i){
            uint32_t voxel_index, particle_index;
            particle_index = addParticleByGlobalPos(new_particles[i], voxel_index);
            if(particle_index != INVALID_PARTICLE_INDEX){
                new_ptc_indices.insert(particle_index);
            }
        }   
    }



    /// @brief Move particles in a set to a new position by a transformation matrix. Return the new indices of the particles.
    /// @param ptc_indices A vector of sets of particle indices to be moved
    /// @param t_matrices A vector of transformation matrices
    /// @param new_ptc_indices A vector of new indices of the particles
    inline void moveParticlesInSetsByTransformations(const std::vector<std::unordered_set<uint32_t>> &ptc_indices, const std::vector<Eigen::Matrix4f> &t_matrices, std::vector<std::unordered_set<uint32_t>> &new_ptc_indices)
    {
        if(ptc_indices.empty() || t_matrices.empty()){
            return;
        }

        const int particle_vector_size = ptc_indices.size();
        std::vector<std::vector<Particle>> new_particles(particle_vector_size);

        // Make a copy of the particles to be moved
        for(int i=0; i<particle_vector_size; ++i){
            new_particles[i].resize(ptc_indices[i].size());
            int j = 0;
            for(auto it = ptc_indices[i].begin(); it != ptc_indices[i].end(); ++it){
                new_particles[i][j] = PARTICLE_ARRAY[*it];
                new_particles[i][j].pos = t_matrices[i].block<3, 3>(0, 0) * PARTICLE_ARRAY[*it].pos + t_matrices[i].block<3, 1>(0, 3);

                deleteParticleByIndex(*it);
                ++j;
            }
        }

        // Add the copy of the particles if the voxel is not full
        new_ptc_indices.resize(particle_vector_size);
        for(int i=0; i<particle_vector_size; ++i){
            for(int j=0; j<new_particles[i].size(); ++j){
                uint32_t voxel_index, particle_index;
                particle_index = addParticleByGlobalPos(new_particles[i][j], voxel_index);
                if(particle_index != INVALID_PARTICLE_INDEX){
                    new_ptc_indices[i].insert(particle_index);
                }
            }
        }
    }


    /// @brief Calculate the weight summation in a voxel
    /// @param voxel_index 
    /// @param weight_sum
    inline void calculateWeightSumInVoxel(const uint32_t &voxel_index, float &weight_sum)
    {
        uint32_t particle_start_index = voxel_index << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

        uint32_t x_index, y_index, z_index; // Voxel index in the ring buffer
        voxelIdxToRingbufferIdx(voxel_index, x_index, y_index, z_index);

        weight_sum = 0.f;
        for(uint32_t i=1; i < C_MAX_PARTICLE_NUM_PER_VOXEL; ++i) // The first particle (i=0) is the time particle. Skip.
        {
            if(!isParticleVacant(PARTICLE_ARRAY[particle_start_index+i], x_index, y_index, z_index)){
                weight_sum += PARTICLE_ARRAY[particle_start_index+i].occ_weight;
            }
        }
    }

    /// @brief Calculate the weight summation in a voxel and the track id and label_id of the voxel
    /// @param voxel_index 
    /// @param weight_sum 
    /// @param guessed_weight
    /// @param label_id 
    /// @param track_id 
    inline void calculateWeightAndSemanticsInVoxel(const uint32_t &voxel_index, float &weight_sum, float &guessed_weight, uint8_t &label_id, uint16_t &track_id)
    {
        uint32_t particle_start_index = voxel_index << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

        uint32_t x_index, y_index, z_index; // Voxel index in the ring buffer
        voxelIdxToRingbufferIdx(voxel_index, x_index, y_index, z_index);

        std::map<uint16_t, float> track_id_weight_map;
        std::map<uint16_t, uint8_t> track_id_label_map;

        // Check if the voxel is valid with the time particle. If not updated at all, return 0. 
        if(!isVoxelValid(voxel_index, x_index, y_index, z_index)){
            weight_sum = -1.f;
            guessed_weight = 0.f;
            label_id = 0;
            track_id = 0;
            return;
        }

        weight_sum = 0.f;
        guessed_weight = 0.f;
        for(uint32_t i=1; i < C_MAX_PARTICLE_NUM_PER_VOXEL; ++i) // The first particle (i=0) is the time particle. Skip.
        {
            if(!isParticleVacant(PARTICLE_ARRAY[particle_start_index+i], x_index, y_index, z_index)){
                weight_sum += PARTICLE_ARRAY[particle_start_index+i].occ_weight;

                /// TODO: Check if this limit for the weight can improve the rerecognition performance
                if(PARTICLE_ARRAY[particle_start_index+i].occ_weight > 1.f){
                    PARTICLE_ARRAY[particle_start_index+i].occ_weight = 1.f;
                }

                if(PARTICLE_ARRAY[particle_start_index+i].status == Particle_Status::GUESSED_BORN){
                    guessed_weight += PARTICLE_ARRAY[particle_start_index+i].occ_weight;
                }else if(PARTICLE_ARRAY[particle_start_index+i].status == Particle_Status::UPDATED && PARTICLE_ARRAY[particle_start_index+i].occ_weight < C_PARTICLE_OCC_INIT_WEIGHT){
                    // Ignore and remove very low-weight particle to make some space
                    PARTICLE_ARRAY[particle_start_index+i].status = Particle_Status::INVALID;
                    continue;
                }

                /// THIS LINE IS FOR DEBUGGING
                // guessed_weight += C_PARTICLE_OCC_INIT_WEIGHT;

                // Check if the track id exists in the map. Otherwise, set the weight to 0.
                if(track_id_weight_map.find(PARTICLE_ARRAY[particle_start_index+i].track_id) == track_id_weight_map.end()){
                    track_id_weight_map[PARTICLE_ARRAY[particle_start_index+i].track_id] = 0.f;
                }
                // Update the weight, semantic id
                track_id_weight_map[PARTICLE_ARRAY[particle_start_index+i].track_id] += PARTICLE_ARRAY[particle_start_index+i].occ_weight;
                track_id_label_map[PARTICLE_ARRAY[particle_start_index+i].track_id] = PARTICLE_ARRAY[particle_start_index+i].label_id;
            }
        }

        // Find the track id with the maximum weight
        float max_weight = 0.f;
        for(auto it = track_id_weight_map.begin(); it != track_id_weight_map.end(); ++it){
            if(it->second > max_weight){
                max_weight = it->second;
                track_id = it->first;
                label_id = track_id_label_map[it->first];
            }
        }
    }


    /// @brief Calculate the pignistic probability in a voxel and the track id, label_id of the voxel
    /// @param voxel_index 
    /// @param pignistic_probability 
    /// @param guessed_weight
    /// @param label_id 
    /// @param track_id 
    inline void calculatePignisticProbAndSemanticsInVoxel(const uint32_t &voxel_index, float &pignistic_probability, float &guessed_weight, uint8_t &label_id, uint16_t &track_id) //CHG
    {
        uint32_t particle_start_index = voxel_index << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

        uint32_t x_index, y_index, z_index; // Voxel index in the ring buffer
        voxelIdxToRingbufferIdx(voxel_index, x_index, y_index, z_index);

        std::map<uint16_t, float> track_id_weight_map;
        std::map<uint16_t, uint8_t> track_id_label_map;

        float weight_sum = 0.f;
        float weight_sum_free = 0.f;
        guessed_weight = 0.f;
        for(uint32_t i=1; i < C_MAX_PARTICLE_NUM_PER_VOXEL; ++i) // The first particle (i=0) is the time particle. Skip.
        {
            if(!isParticleVacant(PARTICLE_ARRAY[particle_start_index+i], x_index, y_index, z_index)){
                weight_sum += PARTICLE_ARRAY[particle_start_index+i].occ_weight;
                weight_sum_free += PARTICLE_ARRAY[particle_start_index+i].free_weight;

                if(PARTICLE_ARRAY[particle_start_index+i].status == Particle_Status::GUESSED_BORN){
                    guessed_weight += PARTICLE_ARRAY[particle_start_index+i].occ_weight;
                }

                // Check if the track id exists in the map. Otherwise, set the weight to 0.
                if(track_id_weight_map.find(PARTICLE_ARRAY[particle_start_index+i].track_id) == track_id_weight_map.end()){
                    track_id_weight_map[PARTICLE_ARRAY[particle_start_index+i].track_id] = 0.f;
                }
                // Update the weight, semantic id
                track_id_weight_map[PARTICLE_ARRAY[particle_start_index+i].track_id] += PARTICLE_ARRAY[particle_start_index+i].occ_weight;
                track_id_label_map[PARTICLE_ARRAY[particle_start_index+i].track_id] = PARTICLE_ARRAY[particle_start_index+i].label_id;
            }
        }

        pignistic_probability = weight_sum + 0.5f*(1.f-weight_sum-weight_sum_free);

        // std::cout << "weight_sum: " << weight_sum << std::endl;
        // std::cout << "weight_sum_free: " << weight_sum_free << std::endl;
        // std::cout << "pignistic_probability: " << pignistic_probability << std::endl;

        // Find the track id with the maximum weight
        float max_weight = 0.f;
        for(auto it = track_id_weight_map.begin(); it != track_id_weight_map.end(); ++it){
            if(it->second > max_weight){
                max_weight = it->second;
                track_id = it->first;
                label_id = track_id_label_map[it->first];
            }
        }
    }


    /// @brief Determine if a voxel is occupied given its index and a weight threshold
    /// @param voxel_index 
    /// @param weight_threshold 
    /// @return true if occupied
    inline bool determineIfVoxelOccupied(const uint32_t &voxel_index, float weight_threshold=0.f)
    {
        float weight_sum;
        calculateWeightSumInVoxel(voxel_index, weight_sum);
        if(weight_sum > weight_threshold){
            return true;
        }else{
            return false;
        }
    }
    
    /// @brief Determine if a voxel is occupied given its index and a weight threshold. Also return the track id, semantic id and color_h.
    /// @param voxel_index: Input the voxel index
    /// @param label_id: Return the semantic id of the voxel
    /// @param track_id: Return the track id of the voxel
    /// @param weight_threshold: (Optional) Input the weight threshold
    /// @return 
    inline int determineIfVoxelOccupied(const uint32_t &voxel_index, uint8_t &label_id, uint16_t &track_id, float weight_threshold=0.f)
    {
        float weight_sum, guessed_weight;
        calculateWeightAndSemanticsInVoxel(voxel_index, weight_sum, guessed_weight, label_id, track_id);
        if(weight_sum > weight_threshold){
            return 1; // Occupied
        }else{
            if(weight_sum < 0){
                return -1; // Unknow area
            }else if(guessed_weight >= C_PARTICLE_OCC_INIT_WEIGHT){
                return 2; // Guessed occupied
            }else{
                return 0; // Free
            }
        }
    }

    /// @brief Determine if a voxel is occupied given its index and a pignistic probability threshold
    /// @param voxel_index: Input the voxel index
    /// @param label_id: Return the semantic id of the voxel
    /// @param track_id: Return the track id of the voxel
    /// @param color_h: Return the color_h of the voxel
    /// @param pignistic_probability_threshold: (Optional) Input the pignistic probability threshold
    /// @return 
    inline int determineIfVoxelOccupiedConsiderFreePoint(const uint32_t &voxel_index, uint8_t &label_id, uint16_t &track_id, float pignistic_probability_threshold=0.f) //CHG
    {
        float pignistic_probability, guessed_weight;
        calculatePignisticProbAndSemanticsInVoxel(voxel_index, pignistic_probability, guessed_weight, label_id, track_id);
        if(pignistic_probability > pignistic_probability_threshold){
            return 1;
        }else{
            if(guessed_weight >= C_PARTICLE_OCC_INIT_WEIGHT){
                return 2;
            }else{
                return 0;
            }
        }
    }
    
    /// @brief Get the voxel global position by its index
    /// @param voxel_index
    /// @param voxel_pos
    inline void getVoxelGlobalPosition(const uint32_t &voxel_index, Eigen::Vector3f &voxel_pos)
    {
        voxelIdxToGlobalFramePos(voxel_index, voxel_pos);
    }

    /// @brief The Function is used to find the particles in the FOV with Breadth-first search. We search the vertexes and then check voxels to realize searching in continous space.
    /// @param extrinsic_matrix The extrinsic matrix of the camera
    /// @param depth_img The depth image. Used to check if a particle is occluded.
    void updateVisibleParitlcesWithBFS(const Eigen::Matrix4f& extrinsic_matrix, const cv::Mat &depth_img){
        // Set elements in particle_to_pixel_num_array to zero for a fresh start
        for(uint32_t i=0; i<g_image_height; ++i){
            for(uint32_t j=0; j<g_image_width; ++j){
                particle_to_pixel_num_array[i][j] = 0;
            }
        }
        // Clear particle_to_pixel_index_map
        particle_to_pixel_index_map.clear();
        
        std::cout << "Start BFS" << std::endl;

        // Find the visible particles
        getIdxOfVisibleParitlces(extrinsic_matrix, depth_img);
    }

    /// @brief The Function checks if a point is in the frustum of the camera
    /// @param point The point to be checked. Coordinate in the global frame.
    /// @param extrinsic_matrix The extrinsic matrix of the camera
    /// @param intrinsic_matrix The intrinsic matrix of the camera
    /// @param image_width The width of the image
    /// @param image_height The height of the image
    /// @return true if the point is in the frustum
    inline bool checkIfPointInFrustum(const Eigen::Vector3f& point, const Eigen::Matrix4f& extrinsic_matrix, const Eigen::Matrix3f& intrinsic_matrix, const int &image_width, const int &image_height)
    {
        return isPointInFrustum(point, extrinsic_matrix, intrinsic_matrix, image_width, image_height);
    }



    /// @brief Clear the ring buffer and time stamps. The movement of the ring buffer is retained.
    void clear()
    {
        // Initialize voxel_time_stamps_x, voxel_time_stamps_y, voxel_time_stamps_z
        global_time_stamp = 0;
        for(int i=0; i<C_VOXEL_NUM_AXIS; ++i){
            voxel_time_stamps_x[i] = 0;
            voxel_time_stamps_y[i] = 0;
            voxel_time_stamps_z[i] = 0;
        }

        // Fill the PARTICLE_ARRAY with invalid particles (take a vacant position) and time particles. Time consumption: 290 ms on a single thread of AMD 4800HS CPU        
        for(uint32_t i=0; i<C_VOXEL_NUM_TOTAL; ++i)
        {
            uint32_t start_ptc_seq = i << C_MAX_PARTICLE_NUM_PER_VOXEL_N;
            // Add time particle. Always the first particle in a voxel.
            PARTICLE_ARRAY[start_ptc_seq].status = Particle_Status::TIMEPTC;
            PARTICLE_ARRAY[start_ptc_seq].pos << 0.f, 0.f, 0.f;
            PARTICLE_ARRAY[start_ptc_seq].occ_weight = 0.f;
            PARTICLE_ARRAY[start_ptc_seq].free_weight = 0.f;
            PARTICLE_ARRAY[start_ptc_seq].time_stamp = 0;

            for(int j=1; j<C_MAX_PARTICLE_NUM_PER_VOXEL; ++j)
            {
                PARTICLE_ARRAY[start_ptc_seq+j].status = Particle_Status::INVALID;
                PARTICLE_ARRAY[start_ptc_seq+j].pos << 0.f, 0.f, 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].occ_weight = 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].free_weight = 0.f;
                PARTICLE_ARRAY[start_ptc_seq+j].time_stamp = 0;
            }

        }
    }

    /// @brief Initialize the ring buffer. A necessary step.
    void initialize(){
        // Clear the ring buffer
        clear();

        // Initialize buffer moved steps
        buffer_moved_steps_x = buffer_moved_steps_y = buffer_moved_steps_z = 0;
        buffer_moved_equivalent_steps_x = buffer_moved_equivalent_steps_y = buffer_moved_equivalent_steps_z = 0;

        // Calculate the private variables or parameters
        map_p_max_const = (C_VOXEL_NUM_AXIS >> 1) * C_VOXEL_SIZE;
        map_p_min_const = -map_p_max_const;
        voxel_size_recip = 1.f / C_VOXEL_SIZE;

        low_index_move_bit_const = 0;
        med_index_move_bit_const = low_index_move_bit_const + C_VOXEL_NUM_AXIS_N;
        high_index_move_bit_const = med_index_move_bit_const + C_VOXEL_NUM_AXIS_N;

        uint32_t naive_full_number = 0;  
        for(int i=0; i<C_VOXEL_NUM_AXIS_N; ++i){
            // Let naive_full_number be 00...0011...11. The number of 1 is determined by C_VOXEL_NUM_AXIS_N. 
            naive_full_number |= (1<<i);
        }
        low_index_retrieve_mask_const = naive_full_number << low_index_move_bit_const;
        med_index_retrieve_mask_const = naive_full_number << med_index_move_bit_const;
        high_index_retrieve_mask_const = naive_full_number << high_index_move_bit_const;

        map_center_pos << 0.f, 0.f, 0.f;
        // map_start_center_pos << 0.f, 0.f, 0.f;
        ego_center_pos << 0.f, 0.f, 0.f;
    }

protected:
    /// @brief Add a particle to the voxel with its global position
    /// @param particle 
    /// @return if success, return the particle index; otherwise, return 0xffffffff.
    inline uint32_t addParticleByGlobalPos(const Particle &particle){
        uint32_t voxel_index;
        return addParticleByGlobalPos(particle, voxel_index);
    }

    /// @brief Add a particle to the voxel with its global position
    /// @param particle
    /// @param voxel_index Return the voxel index of the particle
    /// @return if success, return the particle index; otherwise, return 0xffffffff.
    inline uint32_t addParticleByGlobalPos(const Particle &particle, uint32_t &voxel_index){
        // Add a particle to the voxel with the given global position. Return 0 if success. Return -1 if the voxel is outside the map.
        uint32_t x_index_buffer, y_index_buffer, z_index_buffer;
        globalFramePostoVoxelIdx(particle.pos, voxel_index, x_index_buffer, y_index_buffer, z_index_buffer);

        if(voxel_index != INVALID_PARTICLE_INDEX){
            uint32_t particle_start_index = voxel_index << C_MAX_PARTICLE_NUM_PER_VOXEL_N;
            // The first particle (i=0) is the time particle. Skip.
            for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i){
                Particle *p_ori = &PARTICLE_ARRAY[particle_start_index+i];
                if(isParticleVacant(*p_ori, x_index_buffer, y_index_buffer, z_index_buffer)){
                    *p_ori = particle;
                    return particle_start_index+i; // Particle new index
                }
            }
            // std::cout << "Voxel full. Abort." << std::endl;
            return INVALID_PARTICLE_INDEX; // Fail to add a particle. The voxel is full.
        }else{
            // std::cout << "Particle out of range." << std::endl;
            return INVALID_PARTICLE_INDEX; // Fail to add a particle. The voxel is outside the map.
        }
    }

    /// @brief Determine if a particle position in the buffer is vacant
    /// @param particle The particle to be checked
    /// @param x_index_buffer The x ring buffer index of the voxel that the particle is in.
    /// @param y_index_buffer The y ring buffer index of the voxel that the particle is in.
    /// @param z_index_buffer The z ring buffer index of the voxel that the particle is in.
    inline bool isParticleVacant(const Particle &particle, const uint32_t &x_index_buffer, const uint32_t &y_index_buffer, const uint32_t &z_index_buffer){
        if(particle.status == Particle_Status::INVALID || particle.time_stamp < voxel_time_stamps_x[x_index_buffer] || particle.time_stamp < voxel_time_stamps_y[y_index_buffer] || particle.time_stamp < voxel_time_stamps_z[z_index_buffer]){
            return true;
        }else{
            return false;
        }
    }

    /// @brief Determine if a voxel is valid. The voxel is invalid if the time particle is not updated or smaller than the voxel time stamp.
    /// @param voxel_index 
    /// @param x_index_buffer 
    /// @param y_index_buffer 
    /// @param z_index_buffer 
    /// @return 
    inline bool isVoxelValid(const uint32_t &voxel_index, const uint32_t &x_index_buffer, const uint32_t &y_index_buffer, const uint32_t &z_index_buffer){
        // Time particle
        uint32_t particle_start_index = voxel_index << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

        if(PARTICLE_ARRAY[particle_start_index].time_stamp == 0){
            return false;
        }

        if(PARTICLE_ARRAY[particle_start_index].time_stamp < voxel_time_stamps_x[x_index_buffer] || PARTICLE_ARRAY[particle_start_index].time_stamp < voxel_time_stamps_y[y_index_buffer] || PARTICLE_ARRAY[particle_start_index].time_stamp < voxel_time_stamps_z[z_index_buffer]){
            return false;
        }else{
            return true;
        }
    }


    /// @brief Calculate the voxel index from the position in the global frame
    inline void globalFramePostoVoxelIdx(const Eigen::Vector3f &pos, uint32_t &voxel_index)
    {
        Eigen::Vector3f map_frame_pos = pos - map_center_pos;
        uint32_t x_index, y_index, z_index;
        mapFramePostoVoxelIdx(map_frame_pos, voxel_index, x_index, y_index, z_index);
    }

    /// @brief  Calculate the voxel index from the position in the global frame
    inline void globalFramePostoVoxelIdx(const Eigen::Vector3f &pos, uint32_t &voxel_index, uint32_t &x_index, uint32_t &y_index, uint32_t &z_index)
    {
        Eigen::Vector3f map_frame_pos = pos - map_center_pos;
        mapFramePostoVoxelIdx(map_frame_pos, voxel_index, x_index, y_index, z_index);
    }

    /// @brief Calculate the voxel index from the position in the ego vehicle frame
    inline void egoFramePostoVoxelIdx(const Eigen::Vector3f &pos, uint32_t &voxel_index)
    {
        Eigen::Vector3f map_frame_pos = pos + ego_center_pos - map_center_pos;
        uint32_t x_index, y_index, z_index;
        mapFramePostoVoxelIdx(map_frame_pos, voxel_index, x_index, y_index, z_index);
    }

    /// @brief Calculate the voxel index from the position in the map frame
    inline void mapFramePostoVoxelIdx(const Eigen::Vector3f &pos, uint32_t &voxel_index, uint32_t &x_index, uint32_t &y_index, uint32_t &z_index)
    {
        // Calculate the voxel index in the map frame
        uint32_t map_x_index = static_cast<uint32_t>((pos.x() - map_p_min_const) * voxel_size_recip);
        uint32_t map_y_index = static_cast<uint32_t>((pos.y() - map_p_min_const) * voxel_size_recip);
        uint32_t map_z_index = static_cast<uint32_t>((pos.z() - map_p_min_const) * voxel_size_recip);
        
        // Check if the index is in the map space.
        if(map_x_index >= 0 && map_x_index < C_VOXEL_NUM_AXIS && map_y_index >= 0 && map_y_index < C_VOXEL_NUM_AXIS && map_z_index >= 0 && map_z_index < C_VOXEL_NUM_AXIS){
            // Calculate the voxel index in the ring buffer
            mapXYZIdxToRingbufferXYZIdx(map_x_index, map_y_index, map_z_index, x_index, y_index, z_index);
            // std::cout << "x_index="<<x_index<<" y_index="<<y_index << " z_index="<<z_index << std::endl;

            ringbufferXYZIdxToVoxelIdx(x_index, y_index, z_index, voxel_index);
        }
        else{ 
            // Use Maximum or -1 when the position is outside
            voxel_index = INVALID_PARTICLE_INDEX; // If set index to int, then -1.
        }
    }

    /// @brief Calculate the voxel index from the ring buffer index of each axis
    /// @param x_index Ring buffer index of x axis
    /// @param y_index Ring buffer index of y axis
    /// @param z_index Ring buffer index of z axis
    /// @param voxel_index Voxel index in storage
    inline void ringbufferXYZIdxToVoxelIdx(const uint32_t &x_index, const uint32_t &y_index, const uint32_t &z_index, uint32_t &voxel_index)
    {
#if (STORAGE_TYPE == 1)
        // Column-major order
        voxel_index = (((x_index << C_VOXEL_NUM_AXIS_N) | y_index) << C_VOXEL_NUM_AXIS_N) | z_index;
#elif (STORAGE_TYPE == 2)
        // Morton code
        voxel_index = libmorton::morton3D_32_encode(x_index, y_index, z_index);
#else
        // Row-major order
        voxel_index = (((z_index << C_VOXEL_NUM_AXIS_N) | y_index) << C_VOXEL_NUM_AXIS_N) | x_index;
#endif
        // std::cout << "voxel_index = " << voxel_index << std::endl;
    }

    /// @brief Calculate the position of a particle in the global frame from the particle index
    inline void particleIdxToGlobalFramePos(const uint32_t &particle_index, Eigen::Vector3f &pos)
    {
        particleIdxToMapFramePos(particle_index, pos);
        pos += map_center_pos;
    }

    /// @brief Calculate the position of a particle in the map frame from the particle index
    inline void particleIdxToMapFramePos(const uint32_t &particle_index, Eigen::Vector3f &pos)
    {
        uint32_t voxel_index = particle_index >> C_MAX_PARTICLE_NUM_PER_VOXEL_N;
        voxelIdxToMapFramePos(voxel_index, pos);
    }

    /// @brief Calculate the position of a voxel in the global frame from the voxel index
    inline void voxelIdxToGlobalFramePos(const uint32_t &voxel_index, Eigen::Vector3f &pos)
    {
        voxelIdxToMapFramePos(voxel_index, pos);
        pos += map_center_pos;
    }

    /// @brief Calculate the ringbuffer x,y,z index a voxel from the voxel index
    inline void voxelIdxToRingbufferIdx(const uint32_t &voxel_index, uint32_t &x_index, uint32_t &y_index, uint32_t &z_index)
    {
#if (STORAGE_TYPE == 1)
        // Column-major order
        z_index = (voxel_index & low_index_retrieve_mask_const) >> low_index_move_bit_const;
        y_index = (voxel_index & med_index_retrieve_mask_const) >> med_index_move_bit_const;
        x_index = (voxel_index & high_index_retrieve_mask_const) >> high_index_move_bit_const;
#elif (STORAGE_TYPE == 2)
        // Morton code
        libmorton::morton3D_32_decode(voxel_index, x_index, y_index, z_index);
#else
        // Row-major order
        x_index = (voxel_index & low_index_retrieve_mask_const) >> low_index_move_bit_const;
        y_index = (voxel_index & med_index_retrieve_mask_const) >> med_index_move_bit_const;
        z_index = (voxel_index & high_index_retrieve_mask_const) >> high_index_move_bit_const;
#endif
    }
    
    /// @brief Calculate the position of a voxel in the map frame from the voxel index
    inline void voxelIdxToMapFramePos(const uint32_t &voxel_index, Eigen::Vector3f &pos)
    {   
        uint32_t x_index, y_index, z_index;
        voxelIdxToRingbufferIdx(voxel_index, x_index, y_index, z_index);

        // Calculate the voxel index in the map frame
        uint32_t map_x_index, map_y_index, map_z_index;
        ringbufferXYZIdxToMapXYZIdx(x_index, y_index, z_index, map_x_index, map_y_index, map_z_index);
        
        // Calculate the position in the map frame
        pos.x() = map_x_index * C_VOXEL_SIZE + map_p_min_const;
        pos.y() = map_y_index * C_VOXEL_SIZE + map_p_min_const;
        pos.z() = map_z_index * C_VOXEL_SIZE + map_p_min_const;
    }

    /// @brief Calculate the position of a voxel in the global frame from the voxel index in the map frame
    inline void mapXYZIdxToGlobalPose(const uint32_t &map_x_index, const uint32_t &map_y_index, const uint32_t &map_z_index, Eigen::Vector3f &pos)
    {
        pos.x() = map_x_index * C_VOXEL_SIZE + map_p_min_const + map_center_pos.x();
        pos.y() = map_y_index * C_VOXEL_SIZE + map_p_min_const + map_center_pos.y();
        pos.z() = map_z_index * C_VOXEL_SIZE + map_p_min_const + map_center_pos.z();
    }



    /// @brief Calculate the ring buffer index of each axis from the map index
    inline void mapXYZIdxToRingbufferXYZIdx(const uint32_t &map_x_index, const uint32_t &map_y_index, const uint32_t &map_z_index, uint32_t &buffer_x_index, uint32_t &buffer_y_index, uint32_t &buffer_z_index)
    {
        // map_x_index and buffer_x_index should be in the range of [0, C_VOXEL_NUM_AXIS-1]
        int buffer_idx_x_temp = static_cast<int>(map_x_index) + buffer_moved_equivalent_steps_x;
        int buffer_idx_y_temp = static_cast<int>(map_y_index) + buffer_moved_equivalent_steps_y;
        int buffer_idx_z_temp = static_cast<int>(map_z_index) + buffer_moved_equivalent_steps_z;

        // Correct the buffer index if it is out of range
        oneAxisIdxCorrection(buffer_idx_x_temp, buffer_x_index);
        oneAxisIdxCorrection(buffer_idx_y_temp, buffer_y_index);
        oneAxisIdxCorrection(buffer_idx_z_temp, buffer_z_index);
    }

    /// @brief Calculate the ring buffer index of each axis from the map index
    inline void mapXYZIdxToRingbufferXYZIdx(const int &map_x_index, const int &map_y_index, const int &map_z_index, uint32_t &buffer_x_index, uint32_t &buffer_y_index, uint32_t &buffer_z_index)
    {
        // map_x_index and buffer_x_index should be in the range of [0, C_VOXEL_NUM_AXIS-1]
        int buffer_idx_x_temp = map_x_index + buffer_moved_equivalent_steps_x;
        int buffer_idx_y_temp = map_y_index + buffer_moved_equivalent_steps_y;
        int buffer_idx_z_temp = map_z_index + buffer_moved_equivalent_steps_z;

        // Correct the buffer index if it is out of range
        oneAxisIdxCorrection(buffer_idx_x_temp, buffer_x_index);
        oneAxisIdxCorrection(buffer_idx_y_temp, buffer_y_index);
        oneAxisIdxCorrection(buffer_idx_z_temp, buffer_z_index);
    }

    /// @brief Calculate the map index from the ring buffer index of each axis
    inline void ringbufferXYZIdxToMapXYZIdx(const uint32_t &buffer_x_index, const uint32_t &buffer_y_index, const uint32_t &buffer_z_index, uint32_t &map_x_index, uint32_t &map_y_index, uint32_t &map_z_index)
    {
        // buffer_x_index and map_x_index should be in the range of [0, C_VOXEL_NUM_AXIS-1]
        int map_idx_x_temp = static_cast<int>(buffer_x_index) - buffer_moved_equivalent_steps_x;
        int map_idx_y_temp = static_cast<int>(buffer_y_index) - buffer_moved_equivalent_steps_y;
        int map_idx_z_temp = static_cast<int>(buffer_z_index) - buffer_moved_equivalent_steps_z;

        // Correct the map index if it is out of range
        oneAxisIdxCorrection(map_idx_x_temp, map_x_index);
        oneAxisIdxCorrection(map_idx_y_temp, map_y_index);
        oneAxisIdxCorrection(map_idx_z_temp, map_z_index);
    }


    /// @brief Correct the index of an axis if it is out of range
    inline void oneAxisIdxCorrection(const int &index_ori, uint32_t &index_corrected){
        // Correct the index of an axis if it is out of range
        if(index_ori < 0){
            index_corrected = index_ori + C_VOXEL_NUM_AXIS;
        }else if(index_ori >= C_VOXEL_NUM_AXIS){
            index_corrected = index_ori - C_VOXEL_NUM_AXIS;
        }else{
            index_corrected = index_ori;
        }
    }


    /// @brief Correct the index of an axis if it is out of range
    inline void oneAxisIdxCorrection(int &index){
        // Correct the index of an axis if it is out of range
        if(index < 0){
            index = index + C_VOXEL_NUM_AXIS;
        }else if(index >= C_VOXEL_NUM_AXIS){
            index = index - C_VOXEL_NUM_AXIS;
        }else{
            // Do nothing
        }
    }

    /// @brief Update the ring buffer indexes and the time stamps of the voxels on each axis
    void updateRingbufferIndexParams()
    {
        // std::cout << "ego_center_pos=" << ego_center_pos << std::endl;
        // Calculate the moved indexes from the center of global frame
        int center_moved_steps_x = static_cast<int>(ego_center_pos.x() * voxel_size_recip);
        int center_moved_steps_y = static_cast<int>(ego_center_pos.y() * voxel_size_recip);
        int center_moved_steps_z = static_cast<int>(ego_center_pos.z() * voxel_size_recip);

        // Calculate the map center position in the global frame
        map_center_pos.x() = static_cast<float>(center_moved_steps_x) * C_VOXEL_SIZE;
        map_center_pos.y() = static_cast<float>(center_moved_steps_y) * C_VOXEL_SIZE;
        map_center_pos.z() = static_cast<float>(center_moved_steps_z) * C_VOXEL_SIZE;
        
        // Calculate the newly moved indexes
        int new_moved_index_x = center_moved_steps_x - buffer_moved_steps_x;
        int new_moved_index_y = center_moved_steps_y - buffer_moved_steps_y;
        int new_moved_index_z = center_moved_steps_z - buffer_moved_steps_z;

        // std::cout << "new_moved_index_x=" << new_moved_index_x << ", new_moved_index_y=" << new_moved_index_y << ", new_moved_index_z=" << new_moved_index_z << std::endl;

        // Update the time stamps of the voxels on each axis to mask the outdated particles caused by the movement of the ring buffer
        if(new_moved_index_x > 0){
            for(int i=0; i<new_moved_index_x; ++i){
                int buffer_idx_x_temp = i + buffer_moved_equivalent_steps_x;
                oneAxisIdxCorrection(buffer_idx_x_temp);
                voxel_time_stamps_x[buffer_idx_x_temp] = global_time_stamp;
            }
        }else if (new_moved_index_x < 0)
        {
            for(int i=0; i<-new_moved_index_x; ++i){
                int buffer_idx_x_temp = C_VOXEL_NUM_AXIS - 1 - i + buffer_moved_equivalent_steps_x;
                oneAxisIdxCorrection(buffer_idx_x_temp);
                voxel_time_stamps_x[buffer_idx_x_temp] = global_time_stamp;
            }
        }else{
            // Do nothing
        }

        if(new_moved_index_y > 0){
            for(int i=0; i<new_moved_index_y; ++i){
                int buffer_idx_y_temp = i + buffer_moved_equivalent_steps_y;
                oneAxisIdxCorrection(buffer_idx_y_temp);
                voxel_time_stamps_y[buffer_idx_y_temp] = global_time_stamp;
            }
        }else if (new_moved_index_y < 0)
        {
            for(int i=0; i<-new_moved_index_y; ++i){
                int buffer_idx_y_temp = C_VOXEL_NUM_AXIS - 1 - i + buffer_moved_equivalent_steps_y;
                oneAxisIdxCorrection(buffer_idx_y_temp);
                voxel_time_stamps_y[buffer_idx_y_temp] = global_time_stamp;
            }
        }else{
            // Do nothing
        }

        if(new_moved_index_z > 0){
            for(int i=0; i<new_moved_index_z; ++i){
                int buffer_idx_z_temp = i + buffer_moved_equivalent_steps_z;
                oneAxisIdxCorrection(buffer_idx_z_temp);
                voxel_time_stamps_z[buffer_idx_z_temp] = global_time_stamp;
            }
        }else if (new_moved_index_z < 0)
        {
            for(int i=0; i<-new_moved_index_z; ++i){
                int buffer_idx_z_temp = C_VOXEL_NUM_AXIS - 1 - i + buffer_moved_equivalent_steps_z;
                oneAxisIdxCorrection(buffer_idx_z_temp);
                voxel_time_stamps_z[buffer_idx_z_temp] = global_time_stamp;
            }
        }else{
            // Do nothing
        }

        // Update the moved steps on each axis
        buffer_moved_steps_x = center_moved_steps_x;
        buffer_moved_steps_y = center_moved_steps_y;
        buffer_moved_steps_z = center_moved_steps_z;

        // std::cout << "buffer_moved_steps_x=" << buffer_moved_steps_x << ", buffer_moved_steps_y=" << buffer_moved_steps_y << ", buffer_moved_steps_z=" << buffer_moved_steps_z << std::endl;

        getEquivalentSteps(buffer_moved_steps_x, buffer_moved_equivalent_steps_x);
        getEquivalentSteps(buffer_moved_steps_y, buffer_moved_equivalent_steps_y);
        getEquivalentSteps(buffer_moved_steps_z, buffer_moved_equivalent_steps_z);


        // std::cout << "buffer_moved_steps_x=" << buffer_moved_steps_x << ", buffer_moved_equivalent_steps_x=" << buffer_moved_equivalent_steps_x << std::endl;
    }

    /// @brief Calculate the equivalent steps of a given steps considering ringbuffer.
    /// @param ori_steps 
    /// @param equivalent_steps 
    inline void getEquivalentSteps(const int &ori_steps, int &equivalent_steps){
        if(ori_steps > 0){
            equivalent_steps = ori_steps % C_VOXEL_NUM_AXIS;
        }else if(ori_steps < 0){
            equivalent_steps = -(-ori_steps % C_VOXEL_NUM_AXIS);
        }else{
            equivalent_steps = ori_steps; //0
        }
    }

    /// @brief Check if a point is in the FOV of the camera regardless of occlusion
    /// @param point 
    /// @param extrinsic_matrix 
    /// @param intrinsic_matrix 
    /// @param image_width 
    /// @param image_height 
    /// @return Return true if the point is in the FOV of the camera and false otherwise.
    inline bool isPointInFrustum(const Eigen::Vector3f& point, const Eigen::Matrix4f& extrinsic_matrix, const Eigen::Matrix3f& intrinsic_matrix, const int &image_width, const int &image_height)
    {
        // Convert point to homogeneous coordinates
        Eigen::Vector4f homogeneous_point(point.x(), point.y(), point.z(), 1.0);

        // Apply extrinsic matrix to get point in camera coordinates
        Eigen::Vector4f point_camera_frame = extrinsic_matrix * homogeneous_point;

        // Calculate FOV parameters. Only calculate once.
        static const float tan_half_FOVx = tan(atan2(image_width / 2.0, intrinsic_matrix(0,0)));
        static const float tan_half_FOVy = tan(atan2(image_height / 2.0, intrinsic_matrix(1,1)));

        // Check if point is within the frustum
        if (point_camera_frame.z() < g_depth_range_min || point_camera_frame.z() > g_depth_range_max) return false;
        if (std::abs(point_camera_frame.x()) > point_camera_frame.z() * tan_half_FOVx) return false;
        if (std::abs(point_camera_frame.y()) > point_camera_frame.z() * tan_half_FOVy) return false;
        
        return true;
    }

    /// @brief Calculate the position of a particle in the image frame.
    /// @param ptc Particle
    /// @param extrinsic_matrix
    /// @param intrinsic_matrix
    /// @param row The calculated row index of the particle in the image.
    /// @param col The calculated column index of the particle in the image.
    /// @param camera_frame_z The calculated z value of the particle in the camera frame.
    inline bool calculateParticlePositionInImage(const Particle &ptc, const Eigen::Matrix4f& extrinsic_matrix, const Eigen::Matrix3f& intrinsic_matrix, int &row, int &col, float &camera_frame_z)
    {
        // Convert point to homogeneous coordinates
        Eigen::Vector4f homogeneous_point(ptc.pos.x(), ptc.pos.y(), ptc.pos.z(), 1.0);

        // Apply extrinsic matrix to get point in camera coordinates
        Eigen::Vector4f ptc_camera_frame = extrinsic_matrix * homogeneous_point;

        // Check if point is in the depth range
        if (ptc_camera_frame.z() < g_depth_range_min || ptc_camera_frame.z() > g_depth_range_max) return false;

        // Calculate the position in the image frame
        Eigen::Vector3f ptc_image_frame = intrinsic_matrix * ptc_camera_frame.head<3>() / ptc_camera_frame.z();
        row = static_cast<int>(ptc_image_frame.y());
        col = static_cast<int>(ptc_image_frame.x());

        if(row < 0 || row >= g_image_height || col < 0 || col >= g_image_width){
            return false;
        }
        
        camera_frame_z = ptc_camera_frame.z();

        return true;
    }

private:

    /// @brief The Function is used to find the particles in the FOV with Breadth-first search. We search the vertexes and then check voxels to realize searching in continous space. The result is written to particle_to_pixel_index_array and particle_to_pixel_num_array.
    /// @param extrinsic_matrix The extrinsic matrix of the camera
    /// @param depth_img The depth image. Used to check if a particle is occluded.
    void getIdxOfVisibleParitlces(const Eigen::Matrix4f& extrinsic_matrix, const cv::Mat &depth_img){        
        // Define variables and containers for getIdxOfVisibleParitlces
        std::vector<std::vector<std::vector<bool>>> visited_points(C_VOXEL_NUM_AXIS+1, std::vector<std::vector<bool>>(C_VOXEL_NUM_AXIS+1, std::vector<bool>(C_VOXEL_NUM_AXIS+1, false)));
        std::vector<std::vector<std::vector<bool>>> added_voxels_idx(C_VOXEL_NUM_AXIS, std::vector<std::vector<bool>>(C_VOXEL_NUM_AXIS, std::vector<bool>(C_VOXEL_NUM_AXIS, false)));
        std::queue<Eigen::Vector3i> point_queue;

        // Calculate the offset variable from a voxel index to global frame position
        Eigen::Vector3f voxel_to_global_offset = map_center_pos + Eigen::Vector3f(map_p_min_const, map_p_min_const, map_p_min_const);

        // std::cout << "map_center_pos = " << map_center_pos.transpose() << std::endl;

        // Get the intrinsic matrix
        Eigen::Matrix3f intrinsic_matrix;
        intrinsic_matrix << g_camera_fx, 0, g_camera_cx, 0, g_camera_fy, g_camera_cy, 0, 0, 1;
        
        // Calculate the global position of the point (0,0,1) in the camera frame and then search from this point
        Eigen::Vector3f start_point_camera_frame;
        start_point_camera_frame << 0.f, 0.f, 1.f;
        // Eigen::Vector3f start_point_global_frame = extrinsic_matrix.block<3,3>(0,0).inverse() * start_point_camera_frame - extrinsic_matrix.block<3,1>(0,3);
        
        Eigen::Matrix4f inverse_extrinsic_matrix = extrinsic_matrix.inverse().eval();
        Eigen::Vector3f start_point_global_frame = inverse_extrinsic_matrix.block<3,3>(0,0) * start_point_camera_frame + inverse_extrinsic_matrix.block<3,1>(0,3);

        // std::cout << "start_point_global_frame = " << start_point_global_frame.transpose() << std::endl;
        
        Eigen::Vector3f start_point_map_frame = start_point_global_frame - map_center_pos;
        Eigen::Vector3i start_point_idx;
        start_point_idx << static_cast<int>((start_point_map_frame.x()+map_p_max_const)*voxel_size_recip), static_cast<int>((start_point_map_frame.y()+map_p_max_const)*voxel_size_recip), static_cast<int>((start_point_map_frame.z()+map_p_max_const)*voxel_size_recip);

        // Start the BFS from the point that is definately in the FOV
        point_queue.push(start_point_idx);

        // std::cout << "start_point_idx = " << start_point_idx.transpose() << std::endl;

        int valid_particle_num = 0;
        while(!point_queue.empty())
        {
            Eigen::Vector3i current_point = point_queue.front(); //vertex.
            point_queue.pop();

            if(visited_points[current_point.x()][current_point.y()][current_point.z()]) continue;

            // Mark this point as visited
            visited_points[current_point.x()][current_point.y()][current_point.z()] = true;

            // Check if this point is in the FOV
            Eigen::Vector3f point_global_frame = current_point.cast<float>() * C_VOXEL_SIZE + voxel_to_global_offset;
            
            if(isPointInFrustum(point_global_frame, extrinsic_matrix, intrinsic_matrix, g_image_width, g_image_height)){
                // std::cout << "Point " << current_point.transpose() << " is in the FOV." << std::endl;

                // Get the index of voxels that share this vertex
                for (int dx = -1; dx <= 0; dx++) {
                    for (int dy = -1; dy <= 0; dy++) {
                        for (int dz = -1; dz <= 0; dz++) {
                            Eigen::Vector3i adjacent_voxel_idx;
                            adjacent_voxel_idx.x() = current_point.x() + dx;
                            adjacent_voxel_idx.y() = current_point.y() + dy;
                            adjacent_voxel_idx.z() = current_point.z() + dz;

                            // Skip if the voxel indices are out of bounds
                            if (adjacent_voxel_idx.x() < 0 || adjacent_voxel_idx.x() >= C_VOXEL_NUM_AXIS || adjacent_voxel_idx.y() < 0 || adjacent_voxel_idx.y() >= C_VOXEL_NUM_AXIS || adjacent_voxel_idx.z() < 0 || adjacent_voxel_idx.z() >= C_VOXEL_NUM_AXIS) {continue;}

                            // Check if voxel has been added before
                            if (added_voxels_idx[adjacent_voxel_idx.x()][adjacent_voxel_idx.y()][adjacent_voxel_idx.z()]){continue;}
                            
                            // Calculate the voxel index in storage
                            uint32_t voxel_idx, buffer_x_index, buffer_y_index, buffer_z_index;
                            mapXYZIdxToRingbufferXYZIdx(adjacent_voxel_idx.x(), adjacent_voxel_idx.y(), adjacent_voxel_idx.z(), buffer_x_index, buffer_y_index, buffer_z_index);
                            ringbufferXYZIdxToVoxelIdx(buffer_x_index, buffer_y_index, buffer_z_index, voxel_idx);
                            uint32_t particle_start_index = voxel_idx << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

                            bool voxel_observed = false;
                            int valid_particle_num_in_voxel = 0;

                            // Check if the voxel is valid
                            //if(PARTICLE_ARRAY[particle_start_index].time_stamp > 0 && (PARTICLE_ARRAY[particle_start_index].time_stamp < voxel_time_stamps_x[buffer_x_index] || PARTICLE_ARRAY[particle_start_index].time_stamp < voxel_time_stamps_y[buffer_y_index] || PARTICLE_ARRAY[particle_start_index].time_stamp < voxel_time_stamps_z[buffer_z_index])){
                                // Invalid voxel. Traverse a new voxel after ring buffer move. Delete the outdated particles in the voxel.
                            //    deleteParticlesInVoxelByTimeParticleIndex(particle_start_index);
                            //}else{
                                // Add the particle index to the pyramid if the particle is in depth range. Ignore the time particle.
                                for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i){
                                    Particle *ptc = &PARTICLE_ARRAY[particle_start_index+i];

                                    // Check if the particle is valid and not outdated
                                    // if(ptc->status != Particle_Status::INVALID && ptc->time_stamp >= voxel_time_stamps_x[buffer_x_index] && ptc->time_stamp >= voxel_time_stamps_y[buffer_y_index] && ptc->time_stamp >= voxel_time_stamps_z[buffer_z_index]){
                                    if(ptc->status != Particle_Status::INVALID){

                                        if(ptc->time_stamp < voxel_time_stamps_x[buffer_x_index] || ptc->time_stamp < voxel_time_stamps_y[buffer_y_index] || ptc->time_stamp < voxel_time_stamps_z[buffer_z_index]){
                                            // The particle is outdated. Delete the particle.
                                            ptc->status = Particle_Status::INVALID;
                                            continue;
                                        }

                                        // Calculate the particle position in the image frame
                                        valid_particle_num_in_voxel ++;

                                        int row, col;
                                        float camera_frame_z;
                                        if(calculateParticlePositionInImage(*ptc, extrinsic_matrix, intrinsic_matrix, row, col, camera_frame_z)){
                                            // Check if the particle is in depth range
                                            static const float one_sigma_error_coeff = g_depth_error_stddev_at_one_meter + 1.f;

                                            if(depth_img.at<float>(row, col) > g_depth_range_max){
                                                // Object too far to be measured in the depth image. The particle should be free.
                                                // ptc->status = Particle_Status::INVALID;
                                                ptc->occ_weight = C_PARTICLE_OCC_INIT_WEIGHT;
                                                voxel_observed = true; // In free space, the voxel is observed.
                                                continue;
                                            }
                                            
                                            if(camera_frame_z > depth_img.at<float>(row, col) * one_sigma_error_coeff){
                                                // Skip if depth is invalid or the particle is out of depth range. The particle will not be updated.
                                                continue;
                                            }

                                            voxel_observed = true; // The voxel is observed.

                                            if(particle_to_pixel_num_array[row][col] < C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID){
                                                // Add the particle index to the pyramid
                                                particle_to_pixel_index_array[row][col][particle_to_pixel_num_array[row][col]] = particle_start_index+i;
                                                particle_to_pixel_num_array[row][col] ++;
                                            }else{
                                                // If no enough space in the pyramid (particle_to_pixel_index_array). The particle will be stored in particle_to_pixel_index_map.
                                                int id = row * g_image_width + col;
                                                particle_to_pixel_index_map[id].push_back(particle_start_index+i);
                                                particle_to_pixel_num_array[row][col] ++;
                                                // std::cout << "No enough space in the pyramid. The particle will be stored in particle_to_pixel_index_map." << std::endl;
                                            }

                                            valid_particle_num ++;
                                        }
                                    }
                                }
                            //}

                            // Update the time particle. The time particle is always the first particle in the voxel.
                            if(voxel_observed){
                                PARTICLE_ARRAY[particle_start_index].time_stamp = global_time_stamp;
                            }else{
                                if(valid_particle_num_in_voxel == 0){
                                    // Imagine a particle in the central of the voxel. Check if the particle is the valid depth range.
                                    Particle imaginary_particle;
                                    int row, col;
                                    float camera_frame_z;
                                    mapXYZIdxToGlobalPose(adjacent_voxel_idx.x(), adjacent_voxel_idx.y(), adjacent_voxel_idx.z(), imaginary_particle.pos);
                                    if(calculateParticlePositionInImage(imaginary_particle, extrinsic_matrix, intrinsic_matrix, row, col, camera_frame_z)){
                                        // Check if the particle is in depth range. If the particle is in depth range, the voxel is observed.
                                        if(camera_frame_z <= depth_img.at<float>(row, col)){
                                            PARTICLE_ARRAY[particle_start_index].time_stamp = global_time_stamp;
                                        }
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

                    // Skip if the point indices are out of bounds or the point has been visited
                    if (neighbor_point.x() < 0 || neighbor_point.x() > C_VOXEL_NUM_AXIS || neighbor_point.y() < 0 || neighbor_point.y() > C_VOXEL_NUM_AXIS || neighbor_point.z() < 0 || neighbor_point.z() > C_VOXEL_NUM_AXIS || visited_points[neighbor_point.x()][neighbor_point.y()][neighbor_point.z()]){continue;}
                    // Add the point to the queue
                    point_queue.push(neighbor_point);
                }
            }
        }

        std::cout << "valid_particle_num = " << valid_particle_num << std::endl;
    }
};

