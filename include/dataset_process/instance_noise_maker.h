/**
 * @file instance_noise_maker.h
 * @author Clarence (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-19
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

class InstanceNoiseMaker
{
public:
    InstanceNoiseMaker(){};
    ~InstanceNoiseMaker(){};

    struct TrackingNoise
    {
        int start_frame_id;
        int end_frame_id;
        int instance_id;
    };


    /// @brief The function to get the noise id map for one frame after the noise map is generated
    /// @param frame_id 
    /// @param ori_instance_id_list 
    /// @param noise_id_map 
    void getNoiseIDMapOneFrame(const int frame_id, const std::vector<int> &ori_instance_id_list, std::unordered_map<int, int> &noise_id_map)
    {
        for (int i = 0; i < ori_instance_id_list.size(); i++)
        {
            int noised_id = getNoisedTrackingID(i, frame_id);
            noise_id_map.insert(std::make_pair(i, noised_id));
        }
    }


    /// @brief The function to get the noised instance id after the noise map is generated
    /// @param instance_id 
    /// @param frame_id 
    /// @return the noised instance id
    int getNoisedTrackingID(const int instance_id, const int frame_id)
    {
        int noised_instance_id = instance_id;
        auto iter = tracking_noise_map.find(instance_id);
        if (iter != tracking_noise_map.end())
        {
            TrackingNoise tracking_noise = iter->second;
            if (frame_id >= tracking_noise.start_frame_id && frame_id <= tracking_noise.end_frame_id)
            {
                noised_instance_id = tracking_noise.instance_id;
            }
        }
        return noised_instance_id;
    }


    /// @brief Generate the noise for the tracking result
    /// @param ori_instance_id_list the id list of the instances in the dataset
    /// @param start_frame_id the start frame id of the sequence
    /// @param end_frame_id the end frame id of the sequence
    /// @param alter_id_start_frame_id the start instance id of wrong ids that can be generated
    /// @param alter_id_end_frame_id the end instance id of wrong ids that can be generated
    void generateTrackingNoiseMap(const std::vector<int> &ori_instance_id_list, const int start_frame_id, const int end_frame_id, const int alter_id_start_frame_id = 100, const int alter_id_end_frame_id = 255, const double percentage = 1.0)
    {
        std::cout << "Generate the tracking noise map... ori_instance_id_list size = " << ori_instance_id_list.size() << std::endl; 

        int alter_id_this = alter_id_start_frame_id;

        for (int i = 0; i < ori_instance_id_list.size(); i++)
        {
            // Generate a random number to determine whether to add noise to the tracking result
            double random_number = (double)rand() / RAND_MAX;
            if (random_number < percentage) 
            {
                TrackingNoise tracking_noise;
                // Generate a random number between start_frame_id and end_frame_id
                int random_frame_id = rand() % (end_frame_id - start_frame_id + 1) + start_frame_id;

                if(random_number < percentage*0.5){
                    // Change the instance id for just one frame
                    tracking_noise.instance_id = alter_id_this;
                    tracking_noise.start_frame_id = random_frame_id;
                    tracking_noise.end_frame_id = random_frame_id;

                }else{
                    // Change the instance id for the frames from a random number to the end of the sequence
                    tracking_noise.instance_id = alter_id_this;
                    tracking_noise.start_frame_id = random_frame_id;
                    tracking_noise.end_frame_id = end_frame_id;
                }

                // Show the tracking noise
                std::cout << "Instance id: " << i << " -> " << tracking_noise.instance_id << " from " << tracking_noise.start_frame_id << " to " << tracking_noise.end_frame_id << std::endl;

                tracking_noise_map.insert(std::make_pair(i, tracking_noise));
                
                // Update the wrong id
                alter_id_this++;
                if(alter_id_this > alter_id_end_frame_id){
                    std::cout << "No usable wrong id to be chosen! Break." << std::endl;
                    break;
                }
            }
        }
    }

private:
    std::unordered_map<int, TrackingNoise> tracking_noise_map;

};
