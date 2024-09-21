/**
 * @file semantic_dsp_map.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This head file defines the class SemanticDSPMap, which is used to generate and update a semantic map at both object level and subobject level.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "object_layer.h"
#include "utils/pointcloud_tools.h"
#include "mc_ring/mt_operations.h"
#include "utils/tracking_result_handler.h"
#include "utils/object_info_handler.h"


class SemanticDSPMap
{
public:
    /// @brief Constructor
    SemanticDSPMap() :
        detection_probability_(0.95), 
        noise_number_(0.1), 
        nb_ptc_num_per_point_(3), 
        occupancy_threshold_(0.2), 
        max_obersevation_lost_time_(5),
        forgetting_rate_(1.0),
        max_forget_count_(5),
        id_transition_probability_(0.1),
        if_out_evaluation_format_(false),
        match_score_threshold_(0.3),
        beyesian_movement_distance_threshold_(0.1),
        beyesian_movement_probability_threshold_(0.69),
        beyesian_movement_increment_(0.1),
        beyesian_movement_decrement_(0.15),
        prediction_stddev_(0.05),
        depth_noise_model_first_order_(0.0),
        depth_noise_model_zero_order_(0.1)
    {
        // Initialize the color map for detected objects
        for(int i=0; i<256; ++i){
            color_map_int_256_.push_back(i);
        }
        color_map_int_256_ = shuffleVector(color_map_int_256_);

        // Initialize the color map for ground et.al. 
        for(int i=0; i<256; ++i){
            Eigen::Vector3i color;
            if(i < 64){
                color << 0, 0, i*4;
            }else if(i < 128){
                color << 0, (i-64)*4, 255;
            }else if(i < 192){
                color << (i-128)*4, 255, 255 - (i-128)*4;
            }else{
                color << 255, 255 - (i-192)*4, 0;
            }
            color_map_jet_256_.push_back(color);
        }
        
        // Initialize the gaussian random calculator
        gaussian_random_.calculateGaussianTable(prediction_stddev_); 
    };

    /// @brief Destructor
    ~SemanticDSPMap(){};


    /// @brief Clear the map
    void clear()
    {
        // Clear ring buffer
        op_mt_.clear();

        // Clear the object set
        object_set_.clear();
    }

    /// @brief Set template path for pcl tools
    /// @param template_path 
    void setTemplatePath(std::string template_path)
    {
        pt_tools_.setTemplate(template_path);
    }


    /// @brief Set the parameters of the DSP map
    /// @param detection_probability
    /// @param noise_number
    /// @param nb_ptc_num_per_point
    /// @param occupancy_threshold
    /// @param max_obersevation_lost_time
    /// @param forgetting_rate
    /// @param max_forget_count
    /// @param match_score_threshold
    /// @param id_transition_probability
    void setMapParameters(float detection_probability, float noise_number, int nb_ptc_num_per_point, float occupancy_threshold, int max_obersevation_lost_time, float forgetting_rate=1.f, int max_forget_count = 5, float match_score_threshold=0.5, float id_transition_probability = 0.1)
    {
        detection_probability_ = detection_probability;
        noise_number_ = noise_number;
        nb_ptc_num_per_point_ = nb_ptc_num_per_point;
        occupancy_threshold_ = occupancy_threshold;
        max_obersevation_lost_time_ = max_obersevation_lost_time;
        forgetting_rate_ = forgetting_rate;
        max_forget_count_ = max_forget_count;
        match_score_threshold_ = match_score_threshold;
        id_transition_probability_ = id_transition_probability;

        std::cout << "max_obersevation_lost_time_ = " << max_obersevation_lost_time_ << std::endl;
        std::cout << "id_transition_probability_ = " << id_transition_probability_ << std::endl;
    }


    /// @brief Set the parameters of the DSP map
    /// @param if_consider_depth_noise 
    /// @param if_use_independent_filter 
    void setMapOptions(bool if_consider_depth_noise, bool if_use_independent_filter)
    {
        setFlagConsiderDepthNoise(if_consider_depth_noise);
        setFlagUseIndependentFilter(if_use_independent_filter);
    }
    
    /// @brief Set visualization options
    /// @param visualize_with_zero_center If true, the visualization voxels will be centered at (0, 0, 0). Otherwise, the visualization will be centered at the camera position.
    /// @param if_out_evaluation_format If true, the output point cloud will be in evaluation format. No FOV check will be performed and the color will be based on the instance id.
    void setVisualizeOptions(bool visualize_with_zero_center, bool if_out_evaluation_format)
    {
        visualize_with_zero_center_ = visualize_with_zero_center;
        if_out_evaluation_format_ = if_out_evaluation_format;
    }


    /// @brief Set the parameters of the Beyesian movement filter used in object level update
    /// @param distance_threshold 
    /// @param probability_threshold 
    /// @param increment 
    /// @param decrement 
    void setBeyesianMovementParameters(double distance_threshold, double probability_threshold, double increment, double decrement)
    {
        beyesian_movement_distance_threshold_ = distance_threshold;
        beyesian_movement_probability_threshold_ = probability_threshold;
        beyesian_movement_increment_ = increment;
        beyesian_movement_decrement_ = decrement;
    }


    /// @brief The function to set the occupancy probability threshold. The voxel with occupancy probability higher than the threshold will be considered as occupied.
    /// @param threshold 
    void setOccupancyThreshold(float threshold)
    {
        occupancy_threshold_ = threshold;
    }

    /// @brief  The function to set the noise model of the depth sensor. Noise stddev = first_order * distance + zero_order
    /// @param first_order 
    /// @param zero_order 
    void setDepthNoiseModelParameters(float first_order, float zero_order)
    {
        depth_noise_model_first_order_ = first_order;
        depth_noise_model_zero_order_ = zero_order;
        std::cout << "Noise model is " << depth_noise_model_first_order_ << " * distance + " << depth_noise_model_zero_order_ << std::endl;
    }


    /// @brief Update the map with input pose, images and point cloud. Output the occupied point cloud.
    void update(cv::Mat &depth_value_mat, std::vector<MaskKpts> &ins_seg_result, Eigen::Vector3d &camera_position, Eigen::Quaterniond &camera_orientation, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &occupied_point_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &freespace_point_cloud, bool if_get_freespace=false, double time_stamp_double=0.0)
    {
        // Update time stamp, which is used in both object level and sub-object level update
        global_time_stamp += 1; 
#if VERBOSE_MODE == 1
        std::cout << "************ Step " << global_time_stamp << " ************" << std::endl;
        std::chrono::high_resolution_clock::time_point t1_obj = std::chrono::high_resolution_clock::now();
#endif
        // Reallocate Track ID if the track id is larger than the maximum movable object instance id
        for(int i=0; i<ins_seg_result.size(); ++i)
        {
            if(ins_seg_result[i].label != "static" && ins_seg_result[i].track_id > g_max_movable_object_instance_id)
            {
                std::cout << "Reach the maximum movable object instance id. ID reallocated." << std::endl;
                ins_seg_result[i].track_id = ins_seg_result[i].track_id % g_max_movable_object_instance_id;
            }
        }

        // Update the object set 
        if(g_consider_instance){
            objectLevelUpdate(ins_seg_result, camera_position, camera_orientation, time_stamp_double);
        }

#if VERBOSE_MODE == 1
        std::chrono::high_resolution_clock::time_point t2_obj = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_used_obj = std::chrono::duration_cast<std::chrono::duration<double>>(t2_obj - t1_obj);
        
        std::cout << "Time used for object level update: " << time_used_obj.count() << " s" << std::endl;

        static double time_used_obj_sum = 0.0;
        static int time_used_obj_count = 0;

        time_used_obj_sum += time_used_obj.count();
        time_used_obj_count += 1;

        std::cout << "Average time used for object level update: " << time_used_obj_sum / time_used_obj_count << " s" << std::endl;
#endif        

        // Generate Labeled Point Cloud
        pt_tools_.updateCameraPose(camera_position, camera_orientation);
        
        
#if BOOST_MODE == 0
        int cols = depth_value_mat.cols;
        int rows = depth_value_mat.rows;
#else
        int cols = depth_value_mat.cols * g_image_rescale;
        int rows = depth_value_mat.rows * g_image_rescale;
#endif

        // Define a vector to store the labeled point cloud
        std::vector<std::vector<LabeledPoint>> labeled_point_cloud(rows, std::vector<LabeledPoint>(cols));
        // Define a map to store the points of each tracked object
        std::unordered_map<uint16_t, std::vector<Eigen::Vector3d>> tracked_objects_points;
        std::unordered_map<int, int> track_to_label_id_map;

        // Generate the labeled point cloud
        pt_tools_.generateLabeledPointCloud(depth_value_mat, ins_seg_result, labeled_point_cloud, tracked_objects_points, track_to_label_id_map, depth_noise_model_first_order_, depth_noise_model_zero_order_);


#if VERBOSE_MODE == 1
        // Show the labeled point cloud using an image 
        cv::Mat labeled_point_cloud_image;
        pt_tools_.generateLabeledPointCloudInImage(labeled_point_cloud, labeled_point_cloud_image);
        cv::imshow("labeled_point_cloud_image", labeled_point_cloud_image);
        cv::waitKey(1);
#endif

        // Add a high_resolution_clock to measure the time used for sub-object level update
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /// Sub-object level update (Particle update)
        subObjectLevelUpdate(labeled_point_cloud, tracked_objects_points, track_to_label_id_map, depth_value_mat, ins_seg_result, camera_position, camera_orientation, occupied_point_cloud, freespace_point_cloud, if_get_freespace);
        
#if VERBOSE_MODE == 1
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Time used for sub-object level update: " << time_used.count() << " s" << std::endl;
        std::cout << "occupied_point_cloud size = " << occupied_point_cloud->size() << std::endl;
#endif

    }


private:
    ObjectSet object_set_; // The object set

    MTRingBufferOperations op_mt_; // The multi-thread ring buffer operations

    PointCloudTools pt_tools_; // The point cloud tools

    std::vector<int> color_map_int_256_; // A vector to store the color of each instance. 

    std::vector<Eigen::Vector3i> color_map_jet_256_; // A vector to store the color of jet colormap

    GaussianRandomCalculator gaussian_random_; // The gaussian random calculator

    std::unordered_map<int, std::vector<Eigen::Vector3d>> last_kpts_3d_map_; // A map to store the 3D keypoints of the last frame for each object for ZED2 setting
    std::unordered_map<int, double> last_kpts_3d_time_stamp_map_; // A map to store the time stamp of the last frame for each object for ZED2 setting

    std::unordered_map<int, std::vector<Eigen::Vector3d>> key_kpts_3d_map_; // A map to store the keypoints of the object for key frame setting
    std::unordered_map<int, double> key_kpts_3d_time_stamp_map_; // A map to store the time stamp of the key frame for each object for key frame setting


    float prediction_stddev_; // The standard deviation of the prediction
    float detection_probability_; // The detection probability of the sensor
    float noise_number_; // The noise strength of the sensor

    int nb_ptc_num_per_point_; // The number of new born particles from each point

    float occupancy_threshold_; // The threshold to determine if a voxel is occupied

    int max_obersevation_lost_time_; // The maximum number of observation lost time steps

    float forgetting_rate_; // The forgetting rate of the forgetting function for the update instances
    int max_forget_count_; // The maximum number of forgetting count
    float id_transition_probability_; // The probability of the transition from one instance to another

    float match_score_threshold_; // The threshold of the match score for template matching

    bool if_out_evaluation_format_; // If the output point ccloud is in evaluation format
    bool visualize_with_zero_center_; // If the visualization is centered at (0, 0, 0)

    float depth_noise_model_first_order_; // The first order coefficient of the depth noise model
    float depth_noise_model_zero_order_; // The zero order coefficient of the depth noise model

    double beyesian_movement_distance_threshold_;
    double beyesian_movement_probability_threshold_;
    double beyesian_movement_increment_;
    double beyesian_movement_decrement_;


    /// @brief Object level update with real tracking result
    /// @param ins_seg_result 
    /// @param camera_position
    /// @param camera_orientation
    void objectLevelUpdate(const std::vector<MaskKpts> &ins_seg_result, const Eigen::Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, double time_stamp_double)
    {
        // std::cout << "----- Object level update. -----" << std::endl;

        static double time_stamp_double_last = 0.0;

        // Iterate all the objects in ins_seg_result and update them
        std::unordered_set<int> object_ids_observed; // A set to store the object ids that are observed in this frame
        for(int i=0; i<ins_seg_result.size(); ++i)
        {
            // Ignore the static objects
            if(ins_seg_result[i].track_id > g_max_movable_object_instance_id || ins_seg_result[i].label == "static"){continue;}

#if VERBOSE_MODE == 1
            std::cout << "Object " << ins_seg_result[i].label << " is observed." << std::endl;
#endif

            // Get the object id of a movable object
            int object_id = ins_seg_result[i].track_id;
            object_ids_observed.insert(object_id);

            MJObject object;
            object.time_stamp = global_time_stamp;
            object.confidence = 1.f;
            
            // Check if the object is in the g_label_id_map_default
            if(g_label_id_map_default.find(ins_seg_result[i].label) == g_label_id_map_default.end()){
                std::cout << "Warning: Object " << object_id << " is not in the g_label_id_map_default. Ignore it." << std::endl;
                continue;
            }
            object.label = g_label_id_map_default[ins_seg_result[i].label]; 

#if SETTING == 1 || SETTING == 2
            const int minimum_required_keypoints = 5; // Minimum number of Superpoints for one object
#else
            const int minimum_required_keypoints = 4; // 3D object detection result represented by 3D keypoints. Should have exactly 4 keypoints.
#endif

            bool keypoint_method_success = false;
            // Three cases: 1. The object is newly observed. 2. The object is observed before and has enough keypoints for translation estimation. 3. The object is observed before but does not have enough keypoints.
            if(!object_set_.checkIfObjectExists(object_id)){
                // Case 1: The object is newly observed and is within the map range. Add to the object set.
                double closest_distance = std::numeric_limits<double>::max();
                for(auto &pt : ins_seg_result[i].kpts_current){
                    double distance = std::max(std::max(std::fabs(pt.x() - camera_position.x()), std::fabs(pt.y() - camera_position.y())), std::fabs(pt.z() - camera_position.z()));
                    if(distance < closest_distance){
                        closest_distance = distance;
                    }
                }

                static const double map_half_size_scaled = C_VOXEL_SIZE * (1 << (C_VOXEL_NUM_AXIS_N_BIGGEST - 1)) * 1.2; // The map size is 1.2 times the half size of the map. 1.2 as a buffer.
                if(closest_distance > map_half_size_scaled){ // The object is too far away. Ignore it.
                    // std::cout << "Object " << object_id << " is too far away. Ignore adding it." << std::endl;
                    continue;
                }

                // std::cout << "Case 1: Adding object " << object_id << std::endl;

                object_set_.addNewObject(object, object_id);

                keypoint_method_success = true;

#if SETTING == 3
                // Record the 3D keypoints for the object
                last_kpts_3d_map_[object_id] = ins_seg_result[i].kpts_current;
                last_kpts_3d_time_stamp_map_[object_id] = time_stamp_double;
                key_kpts_3d_map_[object_id] = ins_seg_result[i].kpts_current;
                key_kpts_3d_time_stamp_map_[object_id] = time_stamp_double;
#endif


            }else if(ins_seg_result[i].kpts_current.size() >= minimum_required_keypoints){
                // Case 2: The object is observed before and has enough keypoints for translation estimation. Calculate the translation and update the object.
                // std::cout << "Case 2: Updating object " << object_id << ", kpt size = " << ins_seg_result[i].kpts_current.size() << std::endl;

#if SETTING == 1 || SETTING == 2
                // Handle the case of Superpoint keypoints. These are matched keypoints in the tracking node.
                Eigen::MatrixXd last_kpts_3d(3, ins_seg_result[i].kpts_previous.size());
                Eigen::MatrixXd current_kpts_3d(3, ins_seg_result[i].kpts_current.size());

                // Add to the matrix
                for(int j=0; j<ins_seg_result[i].kpts_current.size(); ++j){
                    last_kpts_3d.col(j) = ins_seg_result[i].kpts_previous[j];
                    current_kpts_3d.col(j) = ins_seg_result[i].kpts_current[j];
                }

                // Calculate the transformation matrix and find inlier_indices
                std::vector<Eigen::Matrix4d> transformation_matrix_vec;
                std::vector<int> inlier_indices;

                Eigen::Matrix4d t_matrix;
                
                double mse = estimateTransformationRANSAC(last_kpts_3d, current_kpts_3d, t_matrix, inlier_indices, 100, 0.5f, true);
                // Check if the transformation is valid
                /// TODO: Improve the criteria for transformation validation for superpoint method
                if(mse > 0.2f || inlier_indices.size() < 5 || inlier_indices.size() / static_cast<double>(ins_seg_result[i].kpts_current.size()) < 0.5f){
                    std::cout << "Transformation is not valid.mse = " << mse << ", inlier_indices.size = " << inlier_indices.size() << std::endl;
                    keypoint_method_success = false;
                    // Won't update the object if the transformation is not valid.
                }else{
                    keypoint_method_success = true;
                }
#else           
                // Handle the case of 3D keypoints from 3D object detection
                Eigen::MatrixXd last_kpts_3d(3, 4); // 4 keypoints for 3D object detection
                Eigen::MatrixXd current_kpts_3d(3, 4); // 4 keypoints for 3D object detection

                std::vector<Eigen::Matrix4d> transformation_matrix_vec;
                Eigen::Matrix4d t_matrix;

                std::vector<int> inlier_indices;

                // Check if object ketpoints are out of fov, which means they are not reliable in 3D object detection
                bool out_of_fov = false;
                for(const Eigen::Vector3d &pt : ins_seg_result[i].kpts_current){
                    out_of_fov = isPointOutOfFOV(camera_position, camera_orientation, pt, 5); // 5 pixel as margin
                }

                double time_diff = time_stamp_double - last_kpts_3d_time_stamp_map_[object_id];
                int moved_observation = 0;

                // Check if last_kpts_3d_map_ has the 3D keypoints for the object
                if(out_of_fov){
                    // std::cout << "Warning: Object " << object_id << " keypoints are out of fov. Ignore updating it." << std::endl;
                    keypoint_method_success = false;
                }else if(last_kpts_3d_map_.find(object_id) == last_kpts_3d_map_.end()){
                    std::cout << "Error: No 3D keypoints for object " << object_id << " in last_kpts_3d_map_. Ignore updating it." << std::endl;
                    keypoint_method_success = false;
                    // Update the last_kpts_3d_map_ of the object
                    last_kpts_3d_map_[object_id] = ins_seg_result[i].kpts_current;
                    last_kpts_3d_time_stamp_map_[object_id] = time_stamp_double;
                    key_kpts_3d_map_[object_id] = ins_seg_result[i].kpts_current;
                    key_kpts_3d_time_stamp_map_[object_id] = time_stamp_double;

                }else{
                    // Add to the matrix
                    for(int j=0; j<ins_seg_result[i].kpts_current.size(); ++j){
                        last_kpts_3d.col(j) = last_kpts_3d_map_[object_id][j];
                        current_kpts_3d.col(j) = ins_seg_result[i].kpts_current[j];
                    }
                    
                    estimateTransformationRANSAC(last_kpts_3d, current_kpts_3d, t_matrix, inlier_indices, 2, 0.5f, false);
                    // estimateTransformationNoRotation(last_kpts_3d, current_kpts_3d, t_matrix);

                    // Update the key_kpts_3d_map_ of the object
                    double update_key_kpts_dist_threshold = beyesian_movement_distance_threshold_;
                    Eigen::Vector3d width_vector = current_kpts_3d.col(1) - current_kpts_3d.col(0);
                    double width = width_vector.norm();

                    if(update_key_kpts_dist_threshold < width){
                        update_key_kpts_dist_threshold = width;
                    }

                    double reference_point_moved_dist = (current_kpts_3d.col(0) - key_kpts_3d_map_[object_id][0]).norm();
                    // Check if the reference point has moved enough to be determined as moving
                    if(reference_point_moved_dist > update_key_kpts_dist_threshold){
#if VERBOSE_MODE == 1
                        std::cout << "reference_point_moved_dist=" << reference_point_moved_dist << ", update_key_kpts_dist_threshold=" << update_key_kpts_dist_threshold << std::endl;
#endif

                        moved_observation = 1;

                    }
                    
                    // Update the key_kpts_3d_map_ of the object every 2 seconds
                    if(time_stamp_double - key_kpts_3d_time_stamp_map_[object_id] > 2.0){
                        key_kpts_3d_map_[object_id] = ins_seg_result[i].kpts_current;
                        key_kpts_3d_time_stamp_map_[object_id] = time_stamp_double;
                    }

                    // Update the last_kpts_3d_map_ of the object
                    last_kpts_3d_map_[object_id] = ins_seg_result[i].kpts_current;
                    last_kpts_3d_time_stamp_map_[object_id] = time_stamp_double;

                    keypoint_method_success = true;
                }
#endif
               
                if(keypoint_method_success){
                    // For now, only one transformation matrix is used. Because the object is assumed to be rigid.
                    transformation_matrix_vec.push_back(t_matrix);
                    object.rigidbody_tmatrix_vec = transformation_matrix_vec;

                    double transformation_confidence = 1.0;

#if SETTING == 1 || SETTING == 2
                    // Use a inlier keypoint as a reference point
                    if(!inlier_indices.empty()){
                        object.reference_point = last_kpts_3d.col(inlier_indices[0]);
                        if(inlier_indices.size() < 3){
                            transformation_confidence = 0.0;
                        }
                    }else{
                        object.reference_point = last_kpts_3d.col(0);
                    }

                    // Update the object to estimate the velocity.
                    object_set_.updateObject(object_id, object, transformation_confidence, beyesian_movement_distance_threshold_, beyesian_movement_probability_threshold_, beyesian_movement_increment_, beyesian_movement_decrement_);
#else
                    // Use the first keypoint as a reference point
                    object.reference_point = last_kpts_3d.col(0);

                    // Update the object to estimate the velocity
                    object_set_.updateObject(object_id, object, transformation_confidence, beyesian_movement_distance_threshold_, beyesian_movement_probability_threshold_, beyesian_movement_increment_, beyesian_movement_decrement_, time_diff, moved_observation);
#endif
                    
                }
                               
            }

#if SETTING == 1 || SETTING == 2
            // Case 3: A dynamic object is observed before but does not have good keypoints currently. For textureless object or rematched objects after losing tracking. Use current point cloud to do rematching.
            if(!keypoint_method_success && object_set_.object_tracking_hash_map.at(object_id).object.rigidbody_moved_vec.size() > 0){
                if(object_set_.object_tracking_hash_map.at(object_id).object.rigidbody_moved_vec[0]){
                    // The object is moving.
                    /// Check if prediction is available
                    if(object_set_.object_tracking_hash_map.at(object_id).object.transformations.checkIfUpdated())
                    {
                        // std::cout << "Case 5: No much points but prediction available " << object_id << std::endl;
                        object_set_.predictAndSetTransformation(object_id);
                    }
                    else
                    {
                        /// NOTE: TODO: When the object is too close, adding a distance threshold to avoid matching may be helpful.
                        // std::cout << "Case 3: Rematching object " << object_id << std::endl;
                        object_set_.setFlagsUpdateByMatching(object_id);
                        // Matching function is in the sub-object level prediction.
                    }
                }
            }
#endif

        }

        // Prediction: Iterate the objects in object_tracking_hash_map that are not observed in this frame and update them with a prediction
        for(auto it = object_set_.object_tracking_hash_map.begin(); it != object_set_.object_tracking_hash_map.end(); ++it){
            // Check if the object is observed in this frame
            if(object_ids_observed.find(it->first) == object_ids_observed.end())
            {
                // Check if the object is newly created
                if(it->second.object.rigidbody_moved_vec.size() == 0){continue;}
                // Check if the object was moving. If not, ignore it.
                if(!it->second.object.rigidbody_moved_vec[0]){continue;}

#if VERBOSE_MODE == 1
                std::cout << "Predicting occluded object " << it->first << std::endl;
#endif
                // The object is not observed in this frame. Update it with a prediction.
                double time_diff = time_stamp_double - time_stamp_double_last;
                if(abs(time_diff) > 1.0){ 
                    time_diff = 1.0;
                }
                
                object_set_.predictAndSetTransformation(it->first, time_diff);
            }
        }

        time_stamp_double_last = time_stamp_double;

        // std::cout << "Object update done. Object set size = " << object_set_.object_tracking_hash_map.size() << std::endl;
    }


    /// @brief Sub-object level update (Particle update). Using the SMC-PHD filter.
    /// @param labeled_point_cloud: The labeled point cloud generated from the input data. Used for particle update and particle birth.
    /// @param tracked_objects_points: The points of each tracked object. Used to match the object point cloud and the template for additional particle birth.
    /// @param depth_value_mat: The depth value matrix generated from the input data
    /// @param camera_position: The camera position
    /// @param camera_orientation: The camera orientation
    /// @param occupied_point_cloud: The output occupied voxel point cloud
    void subObjectLevelUpdate(const std::vector<std::vector<LabeledPoint>> &labeled_point_cloud, const std::unordered_map<uint16_t, std::vector<Eigen::Vector3d>> &tracked_objects_points, std::unordered_map<int, int> &track_to_label_id_map,const cv::Mat &depth_value_mat,
                              const std::vector<MaskKpts> &ins_seg_result, const Eigen::Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &occupied_point_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &freespace_point_cloud, bool if_get_freespace=false)
    {
        /****************** Prediction ***********************/ 
        // Add a high_resolution_clock to measure the time used for prediction
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // Move ego center of the map
        Eigen::Vector3f ego_pose = camera_position.cast<float>();
        op_mt_.updateEgoCenterPos(ego_pose);

        // Move particles of the moving objects
        std::vector<int> objects_to_remove;
        std::vector<std::unordered_set<uint32_t>> ptc_indices_to_move;
        std::vector<Eigen::Matrix4f> t_matrices_to_move;
        std::vector<int> track_ids_to_move;

        for(auto it = object_set_.object_tracking_hash_map.begin(); it != object_set_.object_tracking_hash_map.end(); ++it)
        {
            // Ignore objects that are newly created
            if(it->second.object.rigidbody_moved_vec.size() == 0){continue;}
            // Check if the object moved. If not, ignore it.
            if(!it->second.object.rigidbody_moved_vec[0]){continue;}

            if(global_time_stamp - it->second.observation_time_step >= max_obersevation_lost_time_)
            {
#if VERBOSE_MODE == 1
                std::cout << "Dynamic object " << it->first << " deleted because has been seen for a long time." << std::endl;
#endif
                // Record the object id to be removed and remove it later after the loop
                objects_to_remove.push_back(it->first);
            }else{
                // Move the particles of the object in obj_ptc_hash_map.
                int track_id = it->second.track_id;
                // std::cout << "Checking track_id = " << track_id << std::endl;
                if(object_set_.obj_ptc_hash_map.checkIfObjectExists(track_id))
                {
                    auto obj_ptc_indices = object_set_.obj_ptc_hash_map.indices_map.at(track_id);
                    
                    // Get if using matching or not
                    bool if_use_matching = it->second.to_match_with_previous;
                    if(if_use_matching && getFlagUseTemplateMatching()){
                        // Find the observation of this object
                        for(int i=0; i<ins_seg_result.size(); ++i)
                        {
                            if(ins_seg_result[i].track_id == track_id){
                                // Get point cloud of this object using the ins_seg_result[i].mask
                                pcl::PointCloud<pcl::PointXYZ>::Ptr object_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                                // Iterate all the points in the mask
                                for(int row=0; row<ins_seg_result[i].mask.rows; ++row){
                                    for(int col=0; col<ins_seg_result[i].mask.cols; ++col){
                                        if(ins_seg_result[i].mask.at<uint8_t>(row, col) == 0){continue;}
                                        // Get the point position
                                        Eigen::Vector3d pt_pos;
                                        pt_pos << labeled_point_cloud[row][col].position.x(), labeled_point_cloud[row][col].position.y(), labeled_point_cloud[row][col].position.z();
                                        // Add the point to the point cloud
                                        pcl::PointXYZ pt;
                                        pt.x = pt_pos.x();
                                        pt.y = pt_pos.y();
                                        pt.z = pt_pos.z();
                                        object_point_cloud->points.push_back(pt);
                                    }
                                }

                                if(object_point_cloud->points.size() < 100){
                                    break;
                                }

                                // Get the particles of the object and save it to a point cloud
                                pcl::PointCloud<pcl::PointXYZ>::Ptr particles_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                                op_mt_.getParticlesAsPointCloud(obj_ptc_indices, particles_point_cloud, 0.1);

                                // Align the particles_point_cloud with object_point_cloud
                                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_particles_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                                Eigen::Vector3f camera_position_f = camera_position.cast<float>();
                                float match_score = pt_tools_.alignPointCloudToSource(object_point_cloud, camera_position_f, particles_point_cloud, aligned_particles_point_cloud);

                                // Check if the match is good enough. If good, remove the old particles and add aligned_particles_point_cloud to the ring buffer.
                                if(match_score > match_score_threshold_){
                                    // Remove the old particles
                                    op_mt_.deleteParticlesInSet(obj_ptc_indices);
                                    // Add the new particles
                                    std::unordered_set<uint32_t> new_ptc_indices;
                                    op_mt_.addMatchedParticles(aligned_particles_point_cloud, it->second.object.label, it->second.track_id, new_ptc_indices);
                                    // Update the hash map
                                    object_set_.obj_ptc_hash_map.updatePtcIndicesOfObj(track_id, new_ptc_indices);
                                    // std::cout << "Match score = " << match_score << ", good enough for matching." << std::endl;
                                }else{
                                    // std::cout << "Match score = " << match_score << ", not good enough for matching." << std::endl;
                                }

                                break;
                            }
                        }


                    }else{
                        auto tranformation_matrix = it->second.object.rigidbody_tmatrix_vec[0];
                        Eigen::Matrix4f tranformation_matrix_f = tranformation_matrix.cast<float>();

                        // Save all the particles to move and the corresponding transformation matrices. Move them later together to avoid overwriting useful particles.
                        ptc_indices_to_move.push_back(obj_ptc_indices);
                        t_matrices_to_move.push_back(tranformation_matrix_f);
                        track_ids_to_move.push_back(track_id);
#if VERBOSE_MODE == 1
                        std::cout << "Object " << track_id << " with points size = " << obj_ptc_indices.size() << " moved by transformation matrix." << std::endl;
                        std::cout << "tranformation_matrix = " << tranformation_matrix << std::endl;
#endif
                     }
                }
            }
            
        }


        // Now move the particles updated with transformation matrices
        std::vector<std::unordered_set<uint32_t>> newly_moved_ptc_indices;
        op_mt_.moveParticlesInSetsByTransformations(ptc_indices_to_move, t_matrices_to_move, newly_moved_ptc_indices);


        // Update the hash map
        for(int i=0; i<newly_moved_ptc_indices.size(); ++i){
            object_set_.obj_ptc_hash_map.updatePtcIndicesOfObj(track_ids_to_move[i], newly_moved_ptc_indices[i]);
        }

        // Remove the objects that are not observed for a long time
        for(auto &object_id : objects_to_remove)
        {
            object_set_.removeObjectByTrackID(object_id);

#if SETTING == 3
            last_kpts_3d_map_.erase(object_id); // Remove the stored 3D keypoints of the object
            last_kpts_3d_time_stamp_map_.erase(object_id); // Remove the stored time stamp of the object
#endif
        }

        // Test code begin
        /*** Don't know why there should be floating objects (with no particles) but removing them works. ***/
        // Check if there are floating particles by checking if obj_ptc_hash_map.indices_map have track_id that are not in object_tracking_hash_map
        std::vector<int> floating_objects;
        for(auto it = object_set_.obj_ptc_hash_map.indices_map.begin(); it != object_set_.obj_ptc_hash_map.indices_map.end(); ++it)
        {
            if(object_set_.object_tracking_hash_map.find(it->first) == object_set_.object_tracking_hash_map.end())
            {
                // std::cout << "!!!!!!!!! Floating particles of object " << it->first << " removed." << std::endl;
                // std::cout <<"number of particles = " << it->second.size() << std::endl;
                floating_objects.push_back(it->first);
            }
        }

        // Remove the floating objects
        for(auto &object_id : floating_objects)
        {
            object_set_.removeObjectByTrackID(object_id);
#if SETTING == 3
            last_kpts_3d_map_.erase(object_id); // Remove the stored 3D keypoints of the object
            last_kpts_3d_time_stamp_map_.erase(object_id); // Remove the stored time stamp of the object
            key_kpts_3d_map_.erase(object_id); // Remove the stored key 3D keypoints of the object
            key_kpts_3d_time_stamp_map_.erase(object_id); // Remove the stored time stamp of the key 3D keypoints of the object
#endif
        }
        // Test code end


        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        /****************** Update Weight ***********************/ 
        // Calculate the extrinsic matrix
        Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
        extrinsic.block<3,3>(0,0) = camera_orientation.cast<float>().toRotationMatrix();
        extrinsic.block<3,1>(0,3) = camera_position.cast<float>();
        extrinsic = extrinsic.inverse().eval();

        op_mt_.updateVisibleParitlcesWithBFS(extrinsic, depth_value_mat); // The time particle is ignored

        std::chrono::high_resolution_clock::time_point t2_1 = std::chrono::high_resolution_clock::now();

        // Visualize the pyramid image
#if VERBOSE_MODE == 1
        showSimplePyramidImage();
#endif

        // Update the weight of particles in the FOV
        if(getFlagUsePignisticProbability()){
            // updateParticlesWithFreePoint(labeled_point_cloud); // Disable the update with free points for now because Pigistic probability is not used.
            updateParticles(labeled_point_cloud); 
        }else{
            updateParticles(labeled_point_cloud); 
        }
        
        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

        /****************** Particle Birth & Resampling ***********************/
        // Generate new particles for each point in the point cloud
        uint32_t added_particle_num = 0;

        int points_rows = labeled_point_cloud.size();
        int points_cols = labeled_point_cloud[0].size();

        std::unordered_set<uint32_t> resampled_voxel_indices;

        // Iterate all the points in the point cloud in a sparse but all-covered way
        const int selection_interval = 3;
        for(int row_start=0; row_start<selection_interval; row_start++){
            for(int col_start=0; col_start<selection_interval; col_start++){
                for(int i=row_start; i<points_rows; i+=selection_interval){
                    for(int j=col_start; j<points_cols; j+=selection_interval){

                        auto &pt = labeled_point_cloud[i][j];
                        if(!pt.is_valid){ // Check if the point is valid
                            continue;
                        }

                        if(!getFlagConsiderDepthNoise()){ // Check if the depth noise is considered
                            // Add a new particle to the ring buffer. No noise. Resample if necessary.
                            addNewbornParticleAndResample(pt, added_particle_num, resampled_voxel_indices);
                        }else{
                            // Add new particles to the ring buffer with the consideration of noise. Resample if necessary.
                            addNewbornParticleWithNoiseAndResample(pt, added_particle_num, resampled_voxel_indices);
                        }
                        
                    }
                }
            }
        }
        

        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();


#if BOOST_MODE == 0
        /**************** Additional particle birth to known objects in tracking ******************/
        if(getFlagUseTemplateMatching()){
             // Iterate all the points in tracked_objects_points
            for(auto it = tracked_objects_points.begin(); it != tracked_objects_points.end(); ++it)
            {
                uint16_t instance_value = it->first;

                uint8_t label_id, track_id;
                
                track_id = instance_value;
                label_id = track_to_label_id_map[track_id];

                /// TODO: Change the matching condition. Only consider cars for now.
                if(label_id != g_label_id_map_default["Car"] || it->second.size() > 8000 || it->second.size() < 1500){ 
                    continue;
                }

                BBox3D bbox_object;
                getBoundingBox(it->second, bbox_object);

                if(bbox_object.size.x() < 1.0 || bbox_object.size.y() < 1.0){
                    continue;
                }

                // Check if the object exists in object_tracking_hash_map
                if(!object_set_.checkIfObjectExists(track_id)){
                    std::cout << "Object " << track_id << " does not exist in object_set_." << std::endl;
                    continue;
                }

                // Only consider objects that are to be matched with templates. These objects are newly observed.
                if(!object_set_.object_tracking_hash_map.at(track_id).to_match_with_templates){
                    continue;
                }

                // Match the point cloud of this object with template
                object_set_.object_tracking_hash_map.at(track_id).to_match_with_templates = false; // Set to false to avoid matching again

                pcl::PointCloud<pcl::PointXYZ>::Ptr object_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                for(auto &pt : it->second)
                {
                    pcl::PointXYZ pt_pcl;
                    pt_pcl.x = pt.x();
                    pt_pcl.y = pt.y();
                    pt_pcl.z = pt.z();
                    object_point_cloud->points.push_back(pt_pcl);
                }

                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_object_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                // float match_distance = pt_tools_.alignTemplatesToPointCloud(object_point_cloud, camera_position.cast<float>(), label_id, aligned_object_point_cloud);

                std::chrono::high_resolution_clock::time_point t5_1 = std::chrono::high_resolution_clock::now();            
                
                float match_score = pt_tools_.alignTemplatesToPointCloud(object_point_cloud, camera_position.cast<float>(), label_id, aligned_object_point_cloud);
                
                std::chrono::high_resolution_clock::time_point t5_2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_used_matching = std::chrono::duration_cast<std::chrono::duration<double>>(t5_2 - t5_1);

#if VERBOSE_MODE == 1
                std::cout << "Time used for matching: " << time_used_matching.count() << " s" << std::endl;
                std::cout << "match_score = " << match_score << std::endl;
#endif

                // Check if the match is good enough. If good, add new particles to the ring buffer.
                if(match_score < match_score_threshold_){
                    object_set_.object_tracking_hash_map.at(track_id).to_match_with_templates = true; // Set to true to match again
                    continue;
                }

                // Add new born particles to the object with aligned_object_point_cloud
                for(auto &pt : aligned_object_point_cloud->points)
                {
                    for(int kk=0; kk<3; kk++)
                    {
                        LabeledPoint pt_labeled;
                        static float noise_matched = 0.01;
                        pt_labeled.position << pt.x + gaussian_random_.queryNormalRandomZeroMean(noise_matched), pt.y + gaussian_random_.queryNormalRandomZeroMean(noise_matched), pt.z + gaussian_random_.queryNormalRandomZeroMean(noise_matched);
                        pt_labeled.label_id = label_id;
                        pt_labeled.track_id = track_id;

                        // Add a new particle to the ring buffer and obj_ptc_hash_map. No noise. Resample if necessary.
                        addGuessedParticle(pt_labeled);
                    }
                }
            }
        }

#endif

        std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

        /**************** Get occupancy result ******************/
        // Camera intrinsic matrix and extrinsic matrix
        Eigen::Matrix3f intrinsic_matrix;
        intrinsic_matrix << g_camera_fx, 0, g_camera_cx,
                            0.f, g_camera_fy, g_camera_cy,
                            0.f, 0.f, 1.f;
        Eigen::Matrix4f extrinsic_matrix = Eigen::Matrix4f::Identity();
        extrinsic_matrix.block<3,3>(0,0) = camera_orientation.cast<float>().toRotationMatrix();
        extrinsic_matrix.block<3,1>(0,3) = camera_position.cast<float>();
        extrinsic_matrix = extrinsic_matrix.inverse().eval();
        
        // Now get the occupancy result
        getOccupancyResult(occupied_point_cloud, freespace_point_cloud, extrinsic_matrix, intrinsic_matrix, if_get_freespace);

#if VERBOSE_MODE == 1
        std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();

        // Print the time used for each step
        std::chrono::duration<double> time_used_prediction = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::chrono::duration<double> time_used_projecttion = std::chrono::duration_cast<std::chrono::duration<double>>(t2_1 - t2);
        std::chrono::duration<double> time_used_update_weight = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
        std::chrono::duration<double> time_used_particle_birth = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);
        std::chrono::duration<double> time_used_additional_particle_birth = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4);
        std::chrono::duration<double> time_used_occupancy_result = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5);

        std::cout << "Time used for prediction: " << time_used_prediction.count() << " s" << std::endl;
        std::cout << "Time used for projecttion: " << time_used_projecttion.count() << " s" << std::endl;
        std::cout << "Time used for update weight: " << time_used_update_weight.count() << " s" << std::endl;
        std::cout << "Time used for particle birth: " << time_used_particle_birth.count() << " s" << std::endl;
        std::cout << "Time used for additional particle birth: " << time_used_additional_particle_birth.count() << " s" << std::endl;
        std::cout << "Time used for occupancy result: " << time_used_occupancy_result.count() << " s" << std::endl;

        static double time_used_total_prediction = 0.0;
        static double time_used_total_projection = 0.0;
        static double time_used_total_update_weight = 0.0;
        static double time_used_total_particle_birth = 0.0;
        static double time_used_total_additional_particle_birth = 0.0;
        static double time_used_total_occupancy_result = 0.0;
        static int time_used_total_count = 0;

        time_used_total_prediction += time_used_prediction.count();
        time_used_total_projection += time_used_projecttion.count();
        time_used_total_update_weight += time_used_update_weight.count();
        time_used_total_particle_birth += time_used_particle_birth.count();
        time_used_total_additional_particle_birth += time_used_additional_particle_birth.count();
        time_used_total_occupancy_result += time_used_occupancy_result.count();
        time_used_total_count += 1;

        std::cout << "Average time used for prediction: " << time_used_total_prediction / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for projecttion: " << time_used_total_projection / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for update weight: " << time_used_total_update_weight / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for particle birth: " << time_used_total_particle_birth / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for additional particle birth: " << time_used_total_additional_particle_birth / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for occupancy result: " << time_used_total_occupancy_result / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for all steps: " << (time_used_total_prediction + time_used_total_update_weight + time_used_total_particle_birth + time_used_total_additional_particle_birth + time_used_total_occupancy_result) / time_used_total_count << " s" << std::endl;
#endif
    
    }


    /// @brief Update the weight of particles in the FOV based on the SMC-PHD filter. 
    /// @param labeled_point_cloud: The labeled point cloud generated from the input data. 
    void updateParticles(const std::vector<std::vector<LabeledPoint>> &labeled_point_cloud)
    {
        // Calculate the neighbor size
        /// TODO: use adaptive neighbor size
#if BOOST_MODE == 0
        static std::vector<std::vector<int>> neighbor_width_half_array(g_image_height, std::vector<int>(g_image_width, 5));
        static std::vector<std::vector<int>> neighbor_height_half_array(g_image_height, std::vector<int>(g_image_width, 5));
#else
        static std::vector<std::vector<int>> neighbor_width_half_array(g_image_height, std::vector<int>(g_image_width, 3));
        static std::vector<std::vector<int>> neighbor_height_half_array(g_image_height, std::vector<int>(g_image_width, 3));
#endif

        // Calculate CK + kappa first.
        float ck_kappa_array[g_image_height][g_image_width];
        for(int i=0; i<g_image_height; ++i)
        {
            for(int j=0; j<g_image_width; ++j)
            {   
                // Check if the point is valid. Mask out the invalid points
                if(!labeled_point_cloud[i][j].is_valid){ 
                    continue;
                }

                int neighbor_width_half = neighbor_width_half_array[i][j]; 
                int neighbor_height_half = neighbor_height_half_array[i][j];
                float sigma_this_pixel = labeled_point_cloud[i][j].sigma;

                // Calculate the ck + kappa value of the pixel with neighbors
                float ck_this_pixel = 0.f;
                for(int m=-neighbor_height_half; m<=neighbor_height_half; ++m)
                {
                    for(int n=-neighbor_width_half; n<=neighbor_width_half; ++n)
                    {
                        int neighbor_i = i + m;
                        int neighbor_j = j + n;

                        if(neighbor_i < 0 || neighbor_i >= g_image_height || neighbor_j < 0 || neighbor_j >= g_image_width){
                            continue;
                        }

                        // Iterate all particles in the pixel
                        int particle_num_this_pixel = particle_to_pixel_num_array[neighbor_i][neighbor_j];

                        if(particle_num_this_pixel > 0){
                            int id = neighbor_i*g_image_width + neighbor_j;
                            for(int l=0; l<particle_num_this_pixel; ++l)
                            {
                                auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][l]];

                                if(getFlagUseIndependentFilter()){ //|| particle->track_id > g_max_movable_object_instance_id
                                    // Skip the particles that don't belong to the same object
                                    if(particle->track_id != labeled_point_cloud[i][j].track_id){
                                        continue;
                                    }
                                }
                                
                                float gk = gaussian_random_.queryNormalPDF(particle->pos.x, labeled_point_cloud[i][j].position.x(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.y, labeled_point_cloud[i][j].position.y(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.z, labeled_point_cloud[i][j].position.z(), sigma_this_pixel);

                                if(!getFlagUseIndependentFilter())
                                {
                                    gk *= getForgettingFactor(particle->forget_count, forgetting_rate_, max_forget_count_);
                                    if(particle->track_id != labeled_point_cloud[i][j].track_id)
                                    {
                                        gk *= id_transition_probability_; // ID transition probability. 
                                    }
                                }

                                ck_this_pixel += particle->pos.weight * gk;
                            }
                        }
                    }
                }

                ck_kappa_array[i][j] = ck_this_pixel * detection_probability_ + noise_number_;
            }
        }


        // Update the weight of particles in the FOV. Consider forget_count update at the same time
        for(int i=0; i<g_image_height; ++i)
        {
            for(int j=0; j<g_image_width; ++j)
            { 
                int neighbor_width_half = neighbor_width_half_array[i][j]; 
                int neighbor_height_half = neighbor_height_half_array[i][j];
                float sigma_this_pixel = labeled_point_cloud[i][j].sigma;

                // Update particles in each pixel
                int particle_num_this_pixel = particle_to_pixel_num_array[i][j];                
                if(particle_num_this_pixel > 0)
                {
                    int id = i*g_image_width + j;
                    for(int l=0; l<particle_num_this_pixel; ++l)
                    {
                        auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][l]];
                        float acc_this_particle = 0.f;

                        bool updated_with_right_id = false;

                        for(int m=-neighbor_height_half; m<=neighbor_height_half; ++m)
                        {
                            for(int n=-neighbor_width_half; n<=neighbor_width_half; ++n)
                            {
                                int neighbor_i = i + m;
                                int neighbor_j = j + n;

                                if(neighbor_i < 0 || neighbor_i >= g_image_height || neighbor_j < 0 || neighbor_j >= g_image_width){
                                    continue;
                                }

                                if(!labeled_point_cloud[neighbor_i][neighbor_j].is_valid){ // Check if the point is valid
                                    continue;
                                }

                                if(getFlagUseIndependentFilter()) //|| particle->track_id > g_max_movable_object_instance_id
                                {
                                    // Skip the point that don't belong to the same object
                                    if(labeled_point_cloud[neighbor_i][neighbor_j].track_id != particle->track_id){
                                        continue;
                                    }
                                }
                                
                                
                                float gk = gaussian_random_.queryNormalPDF(particle->pos.x, labeled_point_cloud[neighbor_i][neighbor_j].position.x(), sigma_this_pixel)
                                            * gaussian_random_.queryNormalPDF(particle->pos.y, labeled_point_cloud[neighbor_i][neighbor_j].position.y(), sigma_this_pixel)
                                            * gaussian_random_.queryNormalPDF(particle->pos.z, labeled_point_cloud[neighbor_i][neighbor_j].position.z(), sigma_this_pixel);

                                if(!getFlagUseIndependentFilter())
                                {
                                    if(particle->track_id != labeled_point_cloud[neighbor_i][neighbor_j].track_id){
                                        gk *= id_transition_probability_; // ID transition probability.
                                    }else{
                                        if(gk > c_min_rightly_updated_pdf){updated_with_right_id = true;}
                                    }
                                    gk *= getForgettingFactor(particle->forget_count, forgetting_rate_, max_forget_count_);
                                }
                                
                                acc_this_particle += gk / ck_kappa_array[neighbor_i][neighbor_j];
                            }
                        }

                        particle->pos.weight *= (acc_this_particle * detection_probability_ + 1.f - detection_probability_);
                        particle->status = Particle_Status::UPDATED;
                        particle->time_stamp = global_time_stamp;

                        if(!getFlagUseIndependentFilter()){
                            if(updated_with_right_id){
                                particle->forget_count = 0;
                            }else{
                                if(particle->forget_count < 5){particle->forget_count += 1;}
                            }
                        }              

                    }
                }

            }
        }

    }


    /// @brief Add a Guessed Particle from the template matching result
    /// @param pt 
    inline void addGuessedParticle(const LabeledPoint &pt)
    {
        // Add a new particle to the ring buffer. No noise.
        uint32_t voxel_idx, ptc_idx;
        op_mt_.addGuessedParticles(pt.position, pt.label_id, pt.track_id, voxel_idx, ptc_idx);
        if(ptc_idx != INVALID_PARTICLE_INDEX)
        {   
            // Add the new particle to the object. 
            int track_id = static_cast<int>(pt.track_id);
            int label_id = static_cast<int>(pt.label_id);

            // Check if label_id is in g_movable_object_label_ids
            if(track_id <= g_max_movable_object_instance_id){
                object_set_.obj_ptc_hash_map.addParticleToObj(track_id, ptc_idx);
            }
        }
    }

    /// @brief Add a new born particle to the ring buffer and resample if necessary
    /// @param pt 
    /// @param added_particle_num 
    /// @param resampled_voxel_indices 
    inline void addNewbornParticleAndResample(const LabeledPoint &pt, uint32_t &added_particle_num, std::unordered_set<uint32_t> &resampled_voxel_indices)
    {
        // Add a new particle to the ring buffer. No noise.
        uint32_t voxel_idx, ptc_idx;
        op_mt_.addNewParticleWithSemantics(pt.position, pt.label_id, pt.track_id, voxel_idx, ptc_idx);
        if(ptc_idx != INVALID_PARTICLE_INDEX)
        {   
            // Add the new particle to the object. 
            int track_id = static_cast<int>(pt.track_id);
            int label_id = static_cast<int>(pt.label_id);
            added_particle_num += 1;

            if(track_id <= g_max_movable_object_instance_id){
                object_set_.obj_ptc_hash_map.addParticleToObj(track_id, ptc_idx);
            }
        }

        // Resample if necessary
        if(voxel_idx != INVALID_PARTICLE_INDEX && resampled_voxel_indices.count(voxel_idx) == 0){
            if(resampleParticlesInVoxel(voxel_idx, object_set_.obj_ptc_hash_map)){
                resampled_voxel_indices.insert(voxel_idx);
            }
        }
    }

    /// @brief Add new particles to the ring buffer with the consideration of noise and resample if necessary
    /// @param pt 
    /// @param added_particle_num 
    /// @param resampled_voxel_indices 
    inline void addNewbornParticleWithNoiseAndResample(const LabeledPoint &pt, uint32_t &added_particle_num, std::unordered_set<uint32_t> &resampled_voxel_indices)
    {
        // Add new particles to the ring buffer with noise
        for(int n=0; n<nb_ptc_num_per_point_; ++n)
        {
            Eigen::Vector3f noise;
            if(nb_ptc_num_per_point_ == 1){
                noise << 0.f, 0.f, 0.f;
            }else{
                float sigma_this_pixel = pt.sigma;
                noise << gaussian_random_.queryNormalRandomZeroMean(sigma_this_pixel), gaussian_random_.queryNormalRandomZeroMean(sigma_this_pixel), gaussian_random_.queryNormalRandomZeroMean(sigma_this_pixel);
            }

            uint32_t voxel_idx, ptc_idx;
            op_mt_.addNewParticleWithSemantics(pt.position + noise, pt.label_id, pt.track_id, voxel_idx, ptc_idx);

            if(ptc_idx != INVALID_PARTICLE_INDEX) // Particle is added successfully
            {   
                // Add the new particle to the object. 
                int track_id = static_cast<int>(pt.track_id);
                int label_id = static_cast<int>(pt.label_id);
                added_particle_num += 1;

                // Only add the particle to the object if it is a movable object. CHG. May need to change this.
                if(track_id <= g_max_movable_object_instance_id){
                    object_set_.obj_ptc_hash_map.addParticleToObj(track_id, ptc_idx);
                }

            }else{
                // Resampling if necessary
                if(voxel_idx != INVALID_PARTICLE_INDEX && resampled_voxel_indices.count(voxel_idx) == 0){
                    if(resampleParticlesInVoxel(voxel_idx, object_set_.obj_ptc_hash_map)){ // Resample the particles in the voxel
                        resampled_voxel_indices.insert(voxel_idx);

                        // Try to add the particle again
                        op_mt_.addNewParticleWithSemantics(pt.position + noise, pt.label_id, pt.track_id, voxel_idx, ptc_idx); // Try to add the particle again

                        if(ptc_idx != INVALID_PARTICLE_INDEX) // Particle is added successfully
                        {   
                            // Add the new particle to the object. 
                            int track_id = static_cast<int>(pt.track_id);
                            int label_id = static_cast<int>(pt.label_id);
                            added_particle_num += 1;

                            // Only add the particle to the object if it is a movable object. CHG. May need to change this.
                            if(track_id <= g_max_movable_object_instance_id){
                                object_set_.obj_ptc_hash_map.addParticleToObj(track_id, ptc_idx);
                            }
                        }
                    }
                }
            }
        }
    }


    /// @brief Get the occupancy result. The occupied voxel will be colored and those which are in the FOV will be colored with higher brightness.
    /// @param occupied_point_cloud 
    /// @param freespace_point_cloud
    /// @param extrinsic_matrix 
    /// @param intrinsic_matrix 
    /// @param get_freespace if true, the freespace point cloud will be generated
    inline void getOccupancyResult(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &occupied_point_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &freespace_point_cloud, const Eigen::Matrix4f &extrinsic_matrix, const Eigen::Matrix3f &intrinsic_matrix, bool get_freespace = false)
    {   
        Eigen::Vector3f camera_position = extrinsic_matrix.inverse().eval().block<3,1>(0,3);

        int invalid_voxel_num = 0;
        for(uint32_t i=0; i<C_VOXEL_NUM_TOTAL; ++i){
            uint16_t track_id;
            uint8_t label_id;

            /// TODO:improve the efficiency of this function: determineIfVoxelOccupied
            int occ_result_this_voxel;
            if(getFlagUsePignisticProbability()){
                /// Pignistic probability is aborted for now. TOO SLOW and no obvious improvement.
                // occ_result_this_voxel = op_mt_.determineIfVoxelOccupiedConsiderFreePoint(i, label_id, track_id, occupancy_threshold_);
                occ_result_this_voxel = op_mt_.determineIfVoxelOccupied(i, label_id, track_id, occupancy_threshold_);
            }else{
                occ_result_this_voxel = op_mt_.determineIfVoxelOccupied(i, label_id, track_id, occupancy_threshold_);
            }

            if(occ_result_this_voxel > 0){  //&& occ_result_this_voxel <= occupied_max_flag    
                Eigen::Vector3f voxel_pos;
                op_mt_.getVoxelGlobalPosition(i, voxel_pos);
                pcl::PointXYZRGB pt;
                
                if(visualize_with_zero_center_){
                    pt.x = voxel_pos.x() - camera_position.x();
                    pt.y = voxel_pos.y() - camera_position.y();
                    pt.z = voxel_pos.z() - camera_position.z();
                }else{
                    pt.x = voxel_pos.x();
                    pt.y = voxel_pos.y();
                    pt.z = voxel_pos.z();
                }


                if(occ_result_this_voxel == 1){ // Occupied
                    // Using the color map
                    static int background_id = g_label_id_map_default["Background"];
                    if(label_id == background_id){ 
                        // Color by z axis. Map to color_map_jet_256_
#if SETTING != 3
                        // Adjust to display the color map in the right axis
                        int color_index = std::min(std::max(static_cast<int>((-pt.z+2.f)*51.2f), 0), 255);
#else
                        int color_index = std::min(std::max(static_cast<int>((pt.y+2.f)*51.2f), 0), 255);
#endif
            
                        pt.r = color_map_jet_256_[color_index](0);
                        pt.g = color_map_jet_256_[color_index](1);
                        pt.b = color_map_jet_256_[color_index](2);

                        if(if_out_evaluation_format_){
                            pt.r = 0;
                            pt.g = 0;
                            pt.b = 0;
                        }
                        
                    }
#if SETTING == 0 // For KITTI 360 semantic slam test. Color represents the label id.
                    else{
                        cv::Vec3b color = g_label_color_map_default[label_id];
                        pt.r = color[2];
                        pt.g = color[1];
                        pt.b = color[0];
                    }
#else 
                    else if(track_id > g_max_movable_object_instance_id){ // Static objects
                        cv::Vec3b color = g_label_color_map_default[label_id];
                        pt.r = color[2];
                        pt.g = color[1];
                        pt.b = color[0];
                    }else{  // Cars et.al.
                        if(if_out_evaluation_format_)
                        {
                            pt.r = label_id;
                            pt.g = track_id >> 8;
                            pt.b = track_id & 0xFF;
                        }else{
                            pt.r = 160;
                            pt.g = color_map_int_256_[track_id];
                            pt.b = color_map_int_256_[label_id];
                        }
                    }
#endif
                    
                    
                }else{ // Guessed to be occupied
                    // Use white for guessed result
                    pt.r = 255;
                    pt.g = 255;
                    pt.b = 255;
                    // std::cout << "Matched point ";
                }

                if(!if_out_evaluation_format_){
                    // Turn to HSV space
                    cv::Mat rgb(1, 1, CV_8UC3, cv::Scalar(pt.r, pt.g, pt.b));
                    cv::Mat hsv;
                    cv::cvtColor(rgb, hsv, cv::COLOR_RGB2HSV);
                    
                    // Check if the point is in the FOV
                    if(!op_mt_.checkIfPointInFrustum(voxel_pos, extrinsic_matrix, intrinsic_matrix, g_image_width, g_image_height))
                    {
                        hsv.at<cv::Vec3b>(0,0)[2] *= 0.7f;
                    }
                    // Turn back to RGB space
                    cv::Mat rgb2;
                    cv::cvtColor(hsv, rgb2, cv::COLOR_HSV2RGB);

                    pt.r = rgb2.at<cv::Vec3b>(0,0)[0];
                    pt.g = rgb2.at<cv::Vec3b>(0,0)[1];
                    pt.b = rgb2.at<cv::Vec3b>(0,0)[2];
                }

                occupied_point_cloud->points.push_back(pt);
            }else if(get_freespace && occ_result_this_voxel == 0){
                Eigen::Vector3f voxel_pos;
                op_mt_.getVoxelGlobalPosition(i, voxel_pos);

                pcl::PointXYZRGB pt;

                if(visualize_with_zero_center_)
                {
                    pt.x = voxel_pos.x() - camera_position.x();
                    pt.y = voxel_pos.y() - camera_position.y();
                    pt.z = voxel_pos.z() - camera_position.z();
                }else{
                    pt.x = voxel_pos.x();
                    pt.y = voxel_pos.y();
                    pt.z = voxel_pos.z();
                }

                pt.r = 0;
                pt.g = 255;
                pt.b = 0;

                freespace_point_cloud->points.push_back(pt);
            }

        }

#if VERBOSE_MODE == 1
        std::cout << "Occupied voxel num = " << occupied_point_cloud->points.size() << ", Free voxel num = " << freespace_point_cloud->points.size() << "." << std::endl;
#endif
    }

    /// @brief Get the 3d bounding box of the object composed of points or particles
    /// @param points Input points
    /// @param bbox3d Result
    inline void getBoundingBox(const std::vector<Eigen::Vector3d> &points, BBox3D &bbox3d)
    {
        Eigen::Vector3d min_pt, max_pt;
        min_pt << 1000.f, 1000.f, 1000.f;
        max_pt << -1000.f, -1000.f, -1000.f;

        for(auto &pt : points)
        {
            if(pt.x() < min_pt.x()){min_pt.x() = pt.x();}
            if(pt.y() < min_pt.y()){min_pt.y() = pt.y();}
            if(pt.z() < min_pt.z()){min_pt.z() = pt.z();}

            if(pt.x() > max_pt.x()){max_pt.x() = pt.x();}
            if(pt.y() > max_pt.y()){max_pt.y() = pt.y();}
            if(pt.z() > max_pt.z()){max_pt.z() = pt.z();}
        }

        bbox3d.center.x() = (min_pt.x() + max_pt.x()) / 2.f;
        bbox3d.center.y() = (min_pt.y() + max_pt.y()) / 2.f;
        bbox3d.center.z() = (min_pt.z() + max_pt.z()) / 2.f;

        bbox3d.size.x() = max_pt.x() - min_pt.x();
        bbox3d.size.y() = max_pt.y() - min_pt.y();
        bbox3d.size.z() = max_pt.z() - min_pt.z();
    }

    
    /// @brief Check if the point is out of the FOV
    /// @param camera_position Camera position in the global frame
    /// @param camera_orientation Camera orientation in the global frame
    /// @param point_global Point position in the global frame
    /// @param margin Margin of the image to be considered. Default is 0.
    /// @return True if the point is out of the FOV
    bool isPointOutOfFOV(const Eigen::Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, const Eigen::Vector3d &point_global, int margin = 0)
    {
        // Transform point to camera frame
        Eigen::Vector3d point_camera = camera_orientation.inverse() * (point_global - camera_position);

        // Check if the point is in front of the camera
        if (point_camera.z() <= 0) {
            return true;
        }

        // Project the point onto the image plane
        double u = g_camera_fx * (point_camera.x() / point_camera.z()) + g_camera_cx;
        double v = g_camera_fy * (point_camera.y() / point_camera.z()) + g_camera_cy;

        // Check if the point is within the image bounds
        if (u < margin || u >= g_image_width - margin || v < margin || v >= g_image_height - margin) {
            return true;
        }

        // The point is within the FOV
        return false;
    }


    /// @brief Resample the particles in a voxel to avoid particle degenaration
    /// @param voxel_index
    /// @return true if resampled 
    inline bool resampleParticlesInVoxel(const uint32_t &voxel_index, ObjectParticleHashMap &obj_ptc_hash_map)
    {
        uint32_t start_ptc_seq = voxel_index << C_MAX_PARTICLE_NUM_PER_VOXEL_N;

        // Calculate the weight summation and the updated particle number
        float weight_sum = 0.f;
        uint32_t updated_particle_num = 0;
        for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i)
        {
            if(PARTICLE_ARRAY[start_ptc_seq+i].status == Particle_Status::UPDATED){
                weight_sum += PARTICLE_ARRAY[start_ptc_seq+i].pos.weight;
                ++ updated_particle_num;
            }
        }

        static uint32_t resample_triger_ptc_num = C_MAX_PARTICLE_NUM_PER_VOXEL >> 1;
        if(updated_particle_num > resample_triger_ptc_num) // Resample
        {
            // Remove all the updated particles in the voxel if the weight summation is too small
            if(weight_sum < 0.01f)
            {
                for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i)
                {
                    uint32_t particle_index = start_ptc_seq+i;
                    if(PARTICLE_ARRAY[particle_index].status == Particle_Status::UPDATED){
                        int particle_track_id = PARTICLE_ARRAY[particle_index].track_id;
                        // Remove the particle from the particle array
                        PARTICLE_ARRAY[particle_index].status = Particle_Status::INVALID;
                        // Remove the particle from the object particle hash map
                        obj_ptc_hash_map.removeParticleFromObj(particle_track_id, particle_index);
                    }
                }
                return true;
            }

            // Calculate the weight of each particle after resampling
            float weight_per_particle = weight_sum / resample_triger_ptc_num;
            if(weight_per_particle > 1.f){
                weight_per_particle = 1.f;
            }

            // Rejection sampling
            float particle_weight_sum = 0.f;
            float particle_weight_sum_threshold = weight_per_particle;
            for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i)
            {                
                uint32_t particle_index = start_ptc_seq+i;
                if(PARTICLE_ARRAY[particle_index].status == Particle_Status::UPDATED){
                    int particle_track_id = PARTICLE_ARRAY[particle_index].track_id;
                    particle_weight_sum += PARTICLE_ARRAY[particle_index].pos.weight;

                    if(particle_weight_sum < particle_weight_sum_threshold){
                        // Remove the particle from the particle array
                        PARTICLE_ARRAY[particle_index].status = Particle_Status::INVALID;
                        // Remove the particle from the object particle hash map
                        obj_ptc_hash_map.removeParticleFromObj(particle_track_id, particle_index);
                    }else{
                        PARTICLE_ARRAY[particle_index].pos.weight = weight_per_particle;
                        particle_weight_sum_threshold += weight_per_particle;
                        // Copy the particle to a vacant position if the weight of the particle is very large
                        while(particle_weight_sum > particle_weight_sum_threshold){
                            particle_weight_sum_threshold += weight_per_particle;
                        }
                    }
                }
            }

            return true;
        }else{
            return false;
        }
    }


    /// @brief Visualize the pyramid image containing the particles
    void showSimplePyramidImage()
    {
        // Make a image with the particles in the pyramids and show
        cv::Mat pyramid_image_weight = cv::Mat::zeros(g_image_height, g_image_width, CV_8UC1);
        cv::Mat pyramid_image_number = cv::Mat::zeros(g_image_height, g_image_width, CV_8UC1);

        for(int i=0; i<g_image_height; ++i)
        {
            for(int j=0; j<g_image_width; ++j)
            {   
                // Use particle number
                uint8_t particle_num = static_cast<uint8_t>(particle_to_pixel_num_array[i][j]);
                if(particle_num > 0){
                    pyramid_image_number.at<uint8_t>(i, j) = std::min(particle_num * 20 + 80, 255);
                }else{
                    pyramid_image_number.at<uint8_t>(i, j) = 0;
                }

                // Use particle weight sum
                float weight_sum = 0.f;
                int particle_num_this_pixel = particle_to_pixel_num_array[i][j];

                if(particle_num_this_pixel > 0){
                    int id = i*g_image_width + j;
                    for(int m=0; m<particle_num_this_pixel; ++m)
                    {
                        auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][m]];
                        weight_sum += particle->pos.weight;
                    }
                }

                if(weight_sum > 0.f){
                    pyramid_image_weight.at<uint8_t>(i, j) = std::min(weight_sum * 255.f, 255.f);
                }

            }
        }

        // Give a color the image
        cv::applyColorMap(pyramid_image_weight, pyramid_image_weight, cv::COLORMAP_JET);
        cv::applyColorMap(pyramid_image_number, pyramid_image_number, cv::COLORMAP_JET);

        // Show the image
        cv::imshow("pyramid_image_weight", pyramid_image_weight);
        cv::imshow("pyramid_image_number", pyramid_image_number);

        cv::waitKey(1);
    }
};

