/**
 * @file file_retriever.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This head file includes the class to get data from different datasets.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <dirent.h> 
#include <filesystem>
#include <fstream>
#include <sstream>
#include "data_base.h"
#include "object_layer.h"
#include "visualization_tools.h"
#include <iostream>
#include <png++/png.hpp>
#include "settings/external_settings.h"
#include "settings/settings.h"
#include "instance_noise_maker.h"

// #include <NumCpp.hpp>
// Install Numcpp: Follow https://dpilger26.github.io/NumCpp/doxygen/html/md__mnt_c__github__num_cpp_docs_markdown__installation.html


// Basic function to read image names from a folder
int findImages(std::string folder_path, std::vector<std::string> &file_names, std::string suffix, bool add_path = true, bool add_suffix = true, bool sort_names = true)
{
    DIR* directory_pointer;

    int image_num = 0;

    // Create a dirent structure pointer to store the information about the file
    struct dirent* directory_entry_pointer;

    // Open the directory using the path
    directory_pointer = opendir(folder_path.c_str());

    // Check if the directory was opened successfully
    if (directory_pointer == NULL){
        std::cout << "Could not open directory" << std::endl;
        return -1;
    }

    // Loop through all the files in the directory
    while ((directory_entry_pointer = readdir(directory_pointer)) != NULL)
    {
        std::string file_name = directory_entry_pointer->d_name;
        if (file_name.find(suffix) != std::string::npos){
            std::filesystem::path p(file_name);
            std::string file_name_no_suffix = p.stem().string();
            
            file_names.push_back(file_name_no_suffix);
            image_num ++;
        }
    }
    closedir(directory_pointer);

    // Sort the file names
    if(sort_names)
    {
        // Define a custom sort function to sort the file names by number order
        auto compareFileNames = [](const std::string& a, const std::string& b) {
            // Get the numeric part of the file names
            auto extractNumber = [](const std::string& s) {
                size_t i = 0;
                while (i < s.length() && !std::isdigit(s[i])) {
                    i++;
                }
                size_t j = i;
                while (j < s.length() && std::isdigit(s[j])) {
                    j++;
                }
                return std::stoi(s.substr(i, j - i));
            };

            return extractNumber(a) < extractNumber(b);
        };

        // Sort the file names using the custom sort function
        std::sort(file_names.begin(), file_names.end(), compareFileNames);
    }

    // Add the path to the file names
    if(add_path){
        for (int i = 0; i < file_names.size(); i++){
            file_names[i] = folder_path + file_names[i];
        }
    }

    // Add the suffix to the file names
    if(add_suffix){
        for (int i = 0; i < file_names.size(); i++){
            file_names[i] = file_names[i] + suffix;
        }
    }

    return image_num;
}



/** The class to get data from different datasets. 
 * Usage: Define a FileRetriever object to read data in a scene folder. Call getNextFrame() to get the next frame data.
 * NOTE: 1. When using TARTAN_AIR_SHIBUYA, the scene_folder_path should be "xxx/RoadCrossing0x/"
 *       2. When using UT_KINECT, the scene_folder_path should be "xxx/RGB/s0x_e0x/". The depth image and joint folder, "depth" and "joint", should has the same parent folder with RGB. 
 * **/
class FileRetriever
{
public:
    // Constructor. Initialize the data reading. If visualize is true, the data will be visualized.
    FileRetriever(const std::string scene_folder_path, const int dataset_type, const bool visualize): 
        dataset_type_(dataset_type), 
        dataset_folder_path_(scene_folder_path),
        current_frame_id_(0),
        visualize_(visualize)
    {
        // Check if the scene folder exists
        if(!std::filesystem::exists(dataset_folder_path_))
        {
            std::cout << "Error: The scene folder does not exist!" << std::endl;
            return;
        }

        // Add a '/' to the end of the scene folder path if it does not exist
        dataset_folder_path_fs_ = dataset_folder_path_;
        if (dataset_folder_path_fs_.string().back() != '/') {
            dataset_folder_path_fs_ += '/';
            dataset_folder_path_ = dataset_folder_path_fs_.string();
        }

        std::cout << "Reading data from " << dataset_folder_path_ << std::endl;

        // Initialize the data reading
        int flag = 0;
        if(dataset_type == Dataset_Type::VIRTUAL_KITTI2)
        {
            flag = initializeDataReadingVirtualKitti2(dataset_folder_path_, visualize_);
        }
        else
        {
            std::cout << "Error: Unknown dataset type!" << std::endl;
        }

        // Check if the data reading is initialized successfully
        if(flag == -1){
            std::cout << "Error: Failed to initialize data reading!" << std::endl;
        }
        if(flag == 0){
            std::cout << "Warning: No data found in the given folder! Failed to initialize data reading!" << std::endl;
        }
    }
    
    // Get the next frame data. Return 0 if success, -1 if failed.
    int getNextFrame(InputFrameDataWithGT &frame_data)
    {   
        frame_data.dataset_type = dataset_type_;

        // Check if the current frame id is out of range
        if(current_frame_id_ >= frame_num_){
            std::cout << "Warning: No more frames in the dataset! Use the last one." << std::endl;
            current_frame_id_ -= 1;
        }

        // Get the frame data
        int flag = 0;
        if(dataset_type_ == Dataset_Type::VIRTUAL_KITTI2){
            flag = getFrameVirtualKitti2(current_frame_id_, frame_data, visualize_);
        }else{
            std::cout << "Error: Unknown dataset type in Function getNextFrame()!" << std::endl;
            return -1;
        }

        if(flag == -1){
            std::cout << "Error: Failed to get the next frame!" << std::endl;
            return -1;
        }else{
            current_frame_id_ ++;
        }
        return 0;
    }

    // Get the frame data with the given frame id. Return 0 if success, -1 if failed.
    int getFrame(const int frame_id, InputFrameDataWithGT &frame_data)
    {   
        int flag = 0;
        if(dataset_type_ == Dataset_Type::VIRTUAL_KITTI2){
            flag = getFrameVirtualKitti2(frame_id, frame_data, visualize_);
        }else{
            std::cout << "Error: Unknown dataset type in Function getFrame()!" << std::endl;
            return -1;
        }

        if(flag == -1){
            std::cout << "Error: Failed to get the frame with id: " << frame_id << "!" << std::endl;
            return -1;
        }

        return 0;
    }

    // Get the total number of frames in the dataset
    int getFrameNum()
    {
        return frame_num_;
    }

    // Get the current frame id
    int getCurrentFrameId()
    {
        return current_frame_id_;
    }

    // Set the current frame id
    void setCurrentFrameId(const int frame_id)
    {
        current_frame_id_ = frame_id;
    }


private:

    int dataset_type_;
    std::string dataset_folder_path_;
    std::filesystem::path dataset_folder_path_fs_;

    InstanceNoiseMaker instance_noise_maker_;

    int current_frame_id_;
    bool visualize_;

    std::vector<std::string> rgb_image_path_;
    int frame_num_;

    /****** Data storage for Virtual Kitti2 dataset *****/
    std::vector<Eigen::Vector3d> camera_positions_all_frames_;
    std::vector<Eigen::Quaterniond> camera_orientations_all_frames_;

    std::vector<std::vector<Eigen::Vector3d>> tracked_object_positions_all_frames_;
    std::vector<std::vector<Eigen::Quaterniond>> tracked_object_orientations_all_frames_;
    std::vector<std::vector<int>> tracked_object_ids_all_frames_;
    std::unordered_map<int, int> tracked_object_id_to_label_id_map_;

private:

    /****** Initialize data reading for Virtual Kitti2 dataset *****/ 
    int initializeDataReadingVirtualKitti2(const std::string rgb_img_folder, const bool visualize = true)
    {
        /** Read RGB images Path **/    
        std::vector<std::string> rgb_image_paths;
        findImages(rgb_img_folder, rgb_image_paths, ".jpg");
        
        /** Read camera pose. **/
        std::filesystem::path rgb_img_folder_path = rgb_img_folder;
        std::filesystem::path dataset_folder_path = rgb_img_folder_path.parent_path().parent_path().parent_path();

        std::string camera_pose_file_path = dataset_folder_path.string() + "/gt/extrinsic.txt";
        std::cout << "Reading camera pose from " << camera_pose_file_path << std::endl;

        // Read camera_pose_file_path txt file
        std::ifstream camera_pose_file(camera_pose_file_path);
        if(!camera_pose_file.is_open())
        {
            std::cout << "Error: Failed to open the camera pose file!" << std::endl;
            return -1;
        }

        // Read the camera pose file line by line
        Eigen::Matrix4d T4test;
        int validation_seq = 0;

        std::string line;
        int valid_line_num = 0;
        while(std::getline(camera_pose_file, line))
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

            if(tokens[1] == "0") // Consider camera 0 only
            {
                Eigen::Matrix4d T;
                T << std::stod(tokens[2]), std::stod(tokens[3]), std::stod(tokens[4]), std::stod(tokens[5]),
                    std::stod(tokens[6]), std::stod(tokens[7]), std::stod(tokens[8]), std::stod(tokens[9]),
                    std::stod(tokens[10]), std::stod(tokens[11]), std::stod(tokens[12]), std::stod(tokens[13]),
                    0, 0, 0, 1;

                // Get global position and orientation of the camera
                Eigen::Matrix4d T_inv = T.inverse();

                Eigen::Vector3d t = T_inv.block<3,1>(0,3);
                Eigen::Quaterniond q(T_inv.block<3,3>(0,0));
                
                
                camera_positions_all_frames_.push_back(t);
                camera_orientations_all_frames_.push_back(q);

                if(valid_line_num == validation_seq){T4test = T;}

                valid_line_num ++;
            }
        }

        // Check if the number of camera poses is equal to the number of RGB images
        if(valid_line_num != rgb_image_paths.size())
        {
            std::cout << "Error: The number of camera poses is not equal to the number of RGB images!" << std::endl;
            return -1;
        }

        /** Read track id to label map **/
        std::string track_id_to_label_file_path = dataset_folder_path.string() + "/gt/info.txt";
        std::cout << "Reading track id to label map from " << track_id_to_label_file_path << std::endl;

        // Read track_id_to_label_file_path txt file
        std::ifstream track_id_to_label_file(track_id_to_label_file_path);
        if(!track_id_to_label_file.is_open()){
            std::cout << "Error: Failed to open the track id to label map file!" << std::endl;
            return -1;
        }

        // Read the track_id_to_label_file_path line by line
        while(std::getline(track_id_to_label_file, line))
        {
            // Skip the first line
            if(line.find("trackID") != std::string::npos){
                continue;
            }

            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

            int label_seq = 0;
            label_seq = label_id_map_default[tokens[1]];

            tracked_object_id_to_label_id_map_[std::stoi(tokens[0])] = label_seq;
        }
        

        /** Read Objects pose **/
        std::string object_pose_file_path = dataset_folder_path.string() + "/gt/pose.txt";
        std::cout << "Reading camera pose from " << object_pose_file_path << std::endl;

        // Read object_pose_file_path txt file
        std::ifstream object_pose_file(object_pose_file_path);
        if(!object_pose_file.is_open())
        {
            std::cout << "Error: Failed to open the object pose file!" << std::endl;
            return -1;
        }

        
        // Read the content of the object pose file into a vector of strings
        std::vector<std::string> object_pose_file_string;

        std::getline(object_pose_file, line); // Skip the first line
        int total_line_num = 0;
        while(std::getline(object_pose_file, line))
        {
            object_pose_file_string.push_back(line);
            total_line_num ++;
        }

        // Read the object_pose_file_string line by line
        int current_frame_id = 0;
        int line_id = 0;
        std::cout << "Reading object pose for each frame..." << std::endl;
        while(current_frame_id < rgb_image_paths.size())
        {
            std::vector<Eigen::Vector3d> tracked_object_positions;
            std::vector<Eigen::Quaterniond> tracked_object_orientations;
            std::vector<int> tracked_object_ids;

            std::vector<Eigen::Vector3d> tracked_object_camera_positions; //for test

            while(line_id < total_line_num) // Read the object_pose_file_string line by line.
            {   
                // Read a line and split the line into tokens
                std::istringstream iss(object_pose_file_string[line_id]);
                std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

                if(std::stoi(tokens[0]) == current_frame_id) // frame id
                {
                    if(tokens[1] != "0"){  // Consider camera 0 only
                        line_id ++;
                        continue;
                    }

                    Eigen::Vector3d position(std::stod(tokens[7]), std::stod(tokens[8]), std::stod(tokens[9]));
                    Eigen::Vector3d position_camera_frame(std::stod(tokens[13]), std::stod(tokens[14]), std::stod(tokens[15]));

                    double pitch = std::stod(tokens[10]);
                    double roll = std::stod(tokens[11]);
                    double yaw = std::stod(tokens[12]);
                    
                    // Transform to quaternion by Yaw, Pitch, Roll order
                    Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
                    

                    tracked_object_positions.push_back(position);
                    tracked_object_camera_positions.push_back(position_camera_frame);
                    tracked_object_orientations.push_back(q);
                    tracked_object_ids.push_back(std::stoi(tokens[2])); // track id

                    line_id ++;
                }else if(std::stoi(tokens[0]) > current_frame_id){
                    break;
                }else{
                    std::cout << "Error: The frame id in the object pose file is not in order!" << std::endl;
                }                
            }

            
            // Push back the tracked object poses of the current frame. If all lines are read, push back an empty vector.
            tracked_object_positions_all_frames_.push_back(tracked_object_positions);
            tracked_object_orientations_all_frames_.push_back(tracked_object_orientations);
            tracked_object_ids_all_frames_.push_back(tracked_object_ids);

            current_frame_id ++;
            // std::cout << "current_frame_id = " << current_frame_id << std::endl;
        }
        
        // Check if the number of camera poses is equal to the number of RGB images
        if(current_frame_id != rgb_image_paths.size()){
            std::cout << "current_frame_id: " << current_frame_id << std::endl;
            std::cout << "rgb_image_paths.size(): " << rgb_image_paths.size() << std::endl;
            std::cout << "Error: The number of frames for object poses is not equal to the number of RGB images!" << std::endl;
            return -1;
        }

        std::cout << "tracked_object_positions_all_frames_ size = " << tracked_object_positions_all_frames_.size() << std::endl;

        // Set Other data
        rgb_image_path_ = rgb_image_paths;
        frame_num_ = rgb_image_paths.size();

        return frame_num_;
    }


    /****** Read a frame from VirtualKitti2 dataset *****/ 
    int getFrameVirtualKitti2(const int frame_id, InputFrameDataWithGT& data, const bool visualize = true)
    {   
        /******* Initializae noise maps if noise is considered *********/
        static std::vector<int> ori_instance_id_list;
        static bool tracking_noise_initialized = false;

        if(!tracking_noise_initialized)
        {
            std::cout << "getFlagConsiderTrackingNoise is " << getFlagConsiderTrackingNoise() << std::endl;
            if(getFlagConsiderTrackingNoise())
            {   
                int max_id = 0;

                // Iterate tracked_object_id_to_label_id_map_ to get the ids of the instances in the dataset
                for(auto iter = tracked_object_id_to_label_id_map_.begin(); iter != tracked_object_id_to_label_id_map_.end(); iter++)
                {
                    int ori_instance_id = iter->first;
                    if(ori_instance_id > max_id) {max_id = ori_instance_id;}
                    
                    ori_instance_id_list.push_back(ori_instance_id);
                }

                instance_noise_maker_.generateTrackingNoiseMap(ori_instance_id_list, frame_id, tracked_object_ids_all_frames_.size()-1, max_id+1, 255);
            }
            tracking_noise_initialized = true;
        }

        // Get noise id map of the current frame
        std::unordered_map<int, int> noise_id_map_this_frame;
        if(getFlagConsiderTrackingNoise()){
            instance_noise_maker_.getNoiseIDMapOneFrame(frame_id, ori_instance_id_list, noise_id_map_this_frame);
        }

        /***************** Read RGB image *******************/
        std::filesystem::path rgb_image_path = rgb_image_path_[frame_id];
        std::string rgb_image_name = rgb_image_path.filename().string();
        std::string rgb_image_name_no_suffix = rgb_image_path.stem().string();

        // std::cout << "rgb_image_name_no_suffix: " << rgb_image_name_no_suffix << std::endl;

        cv::Mat rgb_image = cv::imread(rgb_image_path.string(), cv::IMREAD_COLOR);
        
        /*********** Get some names ********************/
        // Define a function to get the numeric part of the file names
        auto extractNumber = [](const std::string& s) {
            size_t i = 0;
            while (i < s.length() && !std::isdigit(s[i])) {
                i++;
            }
            size_t j = i;
            while (j < s.length() && std::isdigit(s[j])) {
                j++;
            }
            return std::stoi(s.substr(i, j - i));
        };
        int img_id = extractNumber(rgb_image_name);

        // Get the image id string
        std::string img_id_str;
        if(img_id < 10) img_id_str = "0000" + std::to_string(img_id);
        else if(img_id < 100) img_id_str = "000" + std::to_string(img_id);
        else if(img_id < 1000) img_id_str = "00" + std::to_string(img_id);
        else if(img_id < 10000) img_id_str = "0" + std::to_string(img_id);
        else img_id_str = std::to_string(img_id);
        
        std::filesystem::path folder_path = rgb_image_path.parent_path();
        std::filesystem::path parent_path = folder_path.parent_path();
        std::filesystem::path parent_parent_path = parent_path.parent_path();
        std::string camera_name = folder_path.filename().string();
        // std::cout << "camera_name = " << camera_name << std::endl;

        /************* Get depth, seg image and joints path **************/
        std::string gt_depth_img_path = parent_parent_path.string() + "/depth/" + camera_name + "/depth_" + img_id_str + ".png";

        std::string depth_img_path;

        if(getFlagConsiderDepthNoise()){
            depth_img_path = parent_parent_path.string() + "/depth/" + camera_name + "_noised/depth_" + img_id_str + ".png";
        }else{
            depth_img_path = gt_depth_img_path; 
        }

        std::string semantic_img_path = parent_parent_path.string() + "/semantic/" + camera_name + "/classgt_" + img_id_str + ".png";
        std::string instance_img_path = parent_parent_path.string() + "/instance/" + camera_name + "/instancegt_" + img_id_str + ".png";

        
        /********************* Read Depth ******************/
        // Convert the depth data to depth image for viusalization
        cv::Mat depth_img = cv::imread(depth_img_path, cv::IMREAD_ANYDEPTH);
        cv::Mat gt_depth_img = cv::imread(gt_depth_img_path, cv::IMREAD_ANYDEPTH);

        int height = depth_img.rows;
        int width = depth_img.cols;

        // std::cout << "depth_img type: " << depth_img.type() << std::endl;
        // std::cout << "Height: " << height << ", Width: " << width << std::endl;

        cv::Mat depth_uint8_mat(height, width, CV_8UC1);
        cv::Mat depth_value_mat(height, width, CV_32FC1);
        cv::Mat gt_depth_value_mat(height, width, CV_32FC1);

        for(int i=0; i< height; i++)
        {
            for(int j=0; j<width; j++)
            {
                uint16_t depth = depth_img.at<uint16_t>(i, j);
                depth_value_mat.at<float>(i,j) = depth/100.0;

                if(depth_value_mat.at<float>(i,j) > g_depth_range_max){
                    depth_uint8_mat.at<uint8_t>(i,j) = 0;
                }else{
                    depth_uint8_mat.at<uint8_t>(i,j) = depth_value_mat.at<float>(i,j) / g_depth_range_max * 255;
                }

                uint16_t gt_depth = gt_depth_img.at<uint16_t>(i, j);
                gt_depth_value_mat.at<float>(i,j) = gt_depth/100.0;
            }
        }

        cv::Mat color_depth;
        applyColorMap(depth_uint8_mat, color_depth, cv::COLORMAP_JET);

        /***************** Read Semantic Seg and Instance ******************/
        cv::Mat seg_img = cv::imread(semantic_img_path, cv::IMREAD_ANYCOLOR);
        png::image<png::index_pixel> instance_img(instance_img_path);

        // Read each pixel
        cv::Mat instance_value_mat(height, width, CV_16UC1);
        for(int i=0; i< height; i++)
        {
            for(int j=0; j<width; j++)
            {

                png::index_pixel instance = instance_img.get_pixel(j, i);
                uint16_t instance_uint = 65535; // 65535 means no instance
                uint16_t obj_label_id = 0;

                if(instance != 0){ // There is an instance.
                    instance_uint = instance-1; // Real Instance id = Pixel value - 1
                    obj_label_id = tracked_object_id_to_label_id_map_[instance_uint];

                }else{ // There is no instance. Check semantic seg.
                    cv::Vec3b seg = seg_img.at<cv::Vec3b>(i, j);

                    // Find the obj_label_id
                    for(auto &pair : label_color_map_default)
                    {
                        if(seg == pair.second){
                            obj_label_id = pair.first;
                            break;
                        }
                    }

                    // If obj_label_id is in label_id_map_static_reversed, set instance_uint to the corresponding instance id
                    if(label_id_map_static_reversed.find(obj_label_id) != label_id_map_static_reversed.end()){
                        instance_uint = label_to_instance_id_map_default[label_id_map_static_reversed[obj_label_id]];
                    }
                }

                instance_value_mat.at<uint16_t>(i,j) = instance_uint;

                
                // png::index_pixel instance = instance_img.get_pixel(j, i);
                // uint16_t instance_uint = 255; // 255 means no instance

                // if(instance != 0){
                //     instance_uint = instance-1; // Real Instance id = Pixel value - 1
                // }

                // // Add noise to the instance id
                // if(getFlagConsiderTrackingNoise() && instance_uint != 255){
                //     instance_uint = noise_id_map_this_frame[instance_uint];
                // }

                // cv::Vec3b seg = seg_img.at<cv::Vec3b>(i, j);

                // // Set the obj_label_id
                // uint16_t obj_label_id = 0;
                // for(auto &pair : label_color_map_default)
                // {
                //     if(seg == pair.second){
                //         obj_label_id = pair.first;
                //         break;
                //     }
                // }

                // instance_value_mat.at<uint16_t>(i,j) = obj_label_id << 8;
                // instance_value_mat.at<uint16_t>(i,j) += instance_uint;
                
                // if(obj_label_id == 4){
                //     std::cout << obj_label_id << ", " << instance_uint << ", ";
                //     std::cout << instance_value_mat.at<uint16_t>(i,j) << std::endl;
                // }
            }
        }
        
        /********** Set Camera Pose ************/
        data.camera_position = camera_positions_all_frames_[frame_id];
        // data.camera_orientation = camera_orientations_all_frames_[frame_id];

        data.camera_orientation = Eigen::Quaterniond(camera_orientations_all_frames_[frame_id].w(), camera_orientations_all_frames_[frame_id].x(), camera_orientations_all_frames_[frame_id].y(), camera_orientations_all_frames_[frame_id].z());

        /************* Set Objects Poses in tracking  *******************/
        data.gt_tracked_object_positions = tracked_object_positions_all_frames_[frame_id];
        data.gt_tracked_object_orientations = tracked_object_orientations_all_frames_[frame_id];
        data.gt_tracked_object_ids = tracked_object_ids_all_frames_[frame_id];

        // Get gt_tracked_object_labels
        for(int i=0; i<data.gt_tracked_object_ids.size(); i++){
            data.gt_tracked_object_labels.push_back(tracked_object_id_to_label_id_map_[data.gt_tracked_object_ids[i]]);
        }

        // Now apply noise to the tracked objects
        if(getFlagConsiderTrackingNoise()){
            for(int i=0; i<data.gt_tracked_object_ids.size(); i++){
                data.gt_tracked_object_ids[i] = noise_id_map_this_frame[data.gt_tracked_object_ids[i]];
            }
        }

        /**************** Visulization ***************/
        if(visualize){
            cv::imshow("rgb_img", rgb_image);
            cv::imshow("depth_img", color_depth);
            cv::imshow("seg_img", seg_img);
            // cv::imshow("instance_img", instance_img);
            // cv::waitKey(0);
            // Waitkey. If key is esc, exit. If key is space, continue.
            int key = cv::waitKey(100);
            if(key == 27){
                exit(0);
            }
        }
        
        /******************* Set the data ***************/
        data.frame_id = frame_id;
        data.rgb_image = rgb_image;
        // data.instance_image = instance_img;
        data.instance_value_mat = instance_value_mat;
        data.depth_image = color_depth;
        data.depth_value_mat = depth_value_mat;
        data.gt_depth_value_mat = gt_depth_value_mat;

        return 0;

    }



};






