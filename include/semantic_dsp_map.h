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
#include "pointcloud_tools.h"
#include "mc_ring/mt_operations.h"
#include "tracking_result_handler.h"
#include "visualization_tools.h"

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
        if_out_evaluation_format_(false),
        match_score_threshold_(0.3),
        beyesian_movement_distance_threshold_(0.1),
        beyesian_movement_probability_threshold_(0.69),
        beyesian_movement_increment_(0.1),
        beyesian_movement_decrement_(0.15)
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
        gaussian_random_.initialize(); 
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

    /// @brief Set the flag to output evaluation format point cloud
    /// @param if_consider_depth_noise
    void useEvaluationFormat(bool if_out_evaluation_format)
    {
        if_out_evaluation_format_ = if_out_evaluation_format;
    }


    /// @brief Set the parameters of the DSP map
    /// @param detection_probability
    /// @param noise_number
    /// @param nb_ptc_num_per_point
    /// @param occupancy_threshold
    /// @param max_obersevation_lost_time
    /// @param forgetting_rate
    void setMapParameters(float detection_probability, float noise_number, int nb_ptc_num_per_point, float occupancy_threshold, int max_obersevation_lost_time, float forgetting_rate, float match_score_threshold)
    {
        detection_probability_ = detection_probability;
        noise_number_ = noise_number;
        nb_ptc_num_per_point_ = nb_ptc_num_per_point;
        occupancy_threshold_ = occupancy_threshold;
        max_obersevation_lost_time_ = max_obersevation_lost_time;
        forgetting_rate_ = forgetting_rate;
        match_score_threshold_ = match_score_threshold;
    }


    /// @brief Set the parameters of the DSP map
    /// @param if_consider_depth_noise 
    /// @param if_use_pignistic_probability 
    /// @param if_use_independent_filter 
    void setMapOptions(bool if_consider_depth_noise, bool if_consider_tracking_noise, bool if_use_pignistic_probability, bool if_use_independent_filter, bool if_use_template_matching)
    {
        setFlagConsiderDepthNoise(if_consider_depth_noise);
        setFlagConsiderTrackingNoise(if_consider_tracking_noise);
        setFlagUsePignisticProbability(if_use_pignistic_probability);
        setFlagUseIndependentFilter(if_use_independent_filter);
        setFlagUseTemplateMatching(if_use_template_matching);
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

    
    /// @brief Update the map with input pose, images and point cloud. Output the occupied point cloud.
    void update(const cv::Mat &depth_value_mat, const std::vector<MaskKpts> &ins_seg_result, const Eigen::Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &occupied_point_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &freespace_point_cloud, bool if_get_freespace=false)
    {
        // Update time stamp, which is used in both object level and sub-object level update
        global_time_stamp += 1; 

        // Update the object set 
        std::chrono::high_resolution_clock::time_point t1_obj = std::chrono::high_resolution_clock::now();
        objectLevelUpdate(ins_seg_result, depth_value_mat, camera_position, camera_orientation);
        
        std::chrono::high_resolution_clock::time_point t2_obj = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_used_obj = std::chrono::duration_cast<std::chrono::duration<double>>(t2_obj - t1_obj);
        std::cout << "Time used for object level update: " << time_used_obj.count() << " s" << std::endl;

        static double time_used_obj_sum = 0.0;
        static int time_used_obj_count = 0;

        time_used_obj_sum += time_used_obj.count();
        time_used_obj_count += 1;

        std::cout << "Average time used for object level update: " << time_used_obj_sum / time_used_obj_count << " s" << std::endl;

        
        // Generate Labeled Point Cloud
        pt_tools_.updateCameraPose(camera_position, camera_orientation);
        
        int cols = depth_value_mat.cols;
        int rows = depth_value_mat.rows;
        // Define a vector to store the labeled point cloud
        std::vector<std::vector<LabeledPoint>> labeled_point_cloud(rows, std::vector<LabeledPoint>(cols));
        // Define a map to store the points of each tracked object
        std::unordered_map<uint16_t, std::vector<Eigen::Vector3d>> tracked_objects_points;
        std::unordered_map<int, int> track_to_label_id_map;

        // Generate the labeled point cloud
        pt_tools_.generateLabeledPointCloud(depth_value_mat, ins_seg_result, labeled_point_cloud, tracked_objects_points, track_to_label_id_map);

        // Add a high_resolution_clock to measure the time used for sub-object level update
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /// Sub-object level update (Particle update)
        subObjectLevelUpdate(labeled_point_cloud, tracked_objects_points, track_to_label_id_map, depth_value_mat, ins_seg_result, camera_position, camera_orientation, occupied_point_cloud, freespace_point_cloud, if_get_freespace);
        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Time used for sub-object level update: " << time_used.count() << " s" << std::endl;
        std::cout << "occupied_point_cloud size = " << occupied_point_cloud->size() << std::endl;
    }


private:
    ObjectSet object_set_; // The object set

    MTRingBufferOperations op_mt_; // The multi-thread ring buffer operations

    PointCloudTools pt_tools_; // The point cloud tools

    std::vector<int> color_map_int_256_; // A vector to store the color of each instance. 

    std::vector<Eigen::Vector3i> color_map_jet_256_; // A vector to store the color of jet colormap

    GaussianRandomCalculator gaussian_random_; // The gaussian random calculator

    float detection_probability_; // The detection probability of the sensor

    float noise_number_; // The noise strength of the sensor

    int nb_ptc_num_per_point_; // The number of new born particles from each point

    float occupancy_threshold_; // The threshold to determine if a voxel is occupied

    int max_obersevation_lost_time_; // The maximum number of observation lost time steps

    float forgetting_rate_; // The forgetting rate of the forgetting function for the update instances

    float match_score_threshold_; // The threshold of the match score for template matching

    bool if_out_evaluation_format_; // If the output point ccloud is in evaluation format


    double beyesian_movement_distance_threshold_;
    double beyesian_movement_probability_threshold_;
    double beyesian_movement_increment_;
    double beyesian_movement_decrement_;


    /// @brief Object level update with real tracking result
    /// @param ins_seg_result 
    /// @param depth_value_mat
    /// @param camera_position
    /// @param camera_orientation
    void objectLevelUpdate(const std::vector<MaskKpts> &ins_seg_result, const cv::Mat &depth_value_mat, const Eigen::Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation)
    {
        std::cout << "----- Object level update. -----" << std::endl;

        // Iterate all the objects in ins_seg_result and update them
        std::unordered_set<int> object_ids_observed; // A set to store the object ids that are observed in this frame
        for(int i=0; i<ins_seg_result.size(); ++i)
        {
            // Ignore the static objects
            if(ins_seg_result[i].track_id == 65535 || ins_seg_result[i].label == "static"){continue;}

            // Get the object id of a movable object
            int object_id = ins_seg_result[i].track_id;
            object_ids_observed.insert(object_id);

            MJObject object;
            object.time_stamp = global_time_stamp;
            object.confidence = 1.f;
            
            // object.label = label_id_map_default["Car"]; //ins_seg_result[i].label

            // Check if the object is in the label_id_map_default
            if(label_id_map_default.find(ins_seg_result[i].label) == label_id_map_default.end()){
                std::cout << "Warning: Object " << object_id << " is not in the label_id_map_default. Ignore it." << std::endl;
                continue;
            }
            object.label = label_id_map_default[ins_seg_result[i].label]; 


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

                static double map_half_size = C_VOXEL_SIZE * (1 << (C_VOXEL_NUM_AXIS_N - 1));

                if(closest_distance > map_half_size + 5.0){ // The object is too far away. Ignore it.
                    std::cout << "Object " << object_id << " is too far away. Ignore adding it." << std::endl;
                    continue;
                }

                std::cout << "Case 1: Adding object " << object_id << std::endl;
                object_set_.addNewObject(object, object_id);
                keypoint_method_success = true;

            }else if(ins_seg_result[i].kpts_current.size() >= 5){
                // Case 2: The object is observed before and has enough keypoints for translation estimation. Calculate the translation and update the object.
                std::cout << "Case 2: Updating object " << object_id << ", kpt size = " << ins_seg_result[i].kpts_current.size() << std::endl;
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
                /// TODO: Improve the criteria for transformation validation
                if(mse > 0.2f || inlier_indices.size() < 5 || inlier_indices.size() / static_cast<double>(ins_seg_result[i].kpts_current.size()) < 0.5f){
                    std::cout << "Transformation is not valid.mse = " << mse << ", inlier_indices.size = " << inlier_indices.size() << std::endl;
                    keypoint_method_success = false;
                    // Won't update the object if the transformation is not valid.
                }else{
                    // For now only one transformation matrix is used because the object is rigid.
                    transformation_matrix_vec.push_back(t_matrix);
                    object.rigidbody_tmatrix_vec = transformation_matrix_vec;

                    double transformation_confidence = 1.0;
                    // Use a inlier keypoint as a reference point
                    if(!inlier_indices.empty()){
                        object.reference_point = last_kpts_3d.col(inlier_indices[0]);
                        if(inlier_indices.size() < 3){
                            transformation_confidence = 0.0;
                        }
                    }else{
                        object.reference_point = last_kpts_3d.col(0);
                    }
                    
                    std::cout << "inlier_indices.size = " << inlier_indices.size() << std::endl;

                    // Update the object to estimate the velocity
                    object_set_.updateObject(object_id, object, transformation_confidence, beyesian_movement_distance_threshold_, beyesian_movement_probability_threshold_, beyesian_movement_increment_, beyesian_movement_decrement_);

                    keypoint_method_success = true;
                }
                               
            }

            // Case 3: A dynamic object is observed before but does not have good keypoints currently. For textureless object or rematched objects after losing tracking. Use current point cloud to do rematching.
            if(!keypoint_method_success && object_set_.object_tracking_hash_map.at(object_id).object.rigidbody_moved_vec.size() > 0){
                if(object_set_.object_tracking_hash_map.at(object_id).object.rigidbody_moved_vec[0]){
                    // The object is moving.
                    /// Check if prediction is available
                    if(object_set_.object_tracking_hash_map.at(object_id).object.transformations.checkIfUpdated())
                    {
                        std::cout << "Case 5: No much points but prediction available " << object_id << std::endl;
                        object_set_.predictAndSetTransformation(object_id);
                    }
                    else
                    {
                        /// NOTE: TODO: When the object is too close, adding a distance threshold to avoid matching may be helpful.
                        std::cout << "Case 3: Rematching object " << object_id << std::endl;
                        object_set_.setFlagsUpdateByMatching(object_id);
                        // Matching function is in the sub-object level prediction.
                    }
                }
            }
        }

        // Case 4: Iterate the objects in object_tracking_hash_map that are not observed in this frame and update them with a prediction
        for(auto it = object_set_.object_tracking_hash_map.begin(); it != object_set_.object_tracking_hash_map.end(); ++it){
            // Check if the object is observed in this frame
            if(object_ids_observed.find(it->first) == object_ids_observed.end())
            {
                // Check if the object is newly created
                if(it->second.object.rigidbody_moved_vec.size() == 0){continue;}
                // Check if the object was moving. If not, ignore it.
                if(!it->second.object.rigidbody_moved_vec[0]){continue;}

                std::cout << "Case 4: Predicting object " << it->first << std::endl;
                // The object is not observed in this frame. Update it with a prediction.
                object_set_.predictAndSetTransformation(it->first);
            }
        }

        std::cout << "Object update done. Object set size = " << object_set_.object_tracking_hash_map.size() << std::endl;
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
            // Ignore objects that is newly created
            if(it->second.object.rigidbody_moved_vec.size() == 0){continue;}
            // Check if the object moved. If not, ignore it.
            if(!it->second.object.rigidbody_moved_vec[0]){continue;}

            if(global_time_stamp - it->second.observation_time_step >= max_obersevation_lost_time_)
            {
                std::cout << "Dynamic object " << it->first << " deleted because has been seen for a long time." << std::endl;
                // Record the object id to be removed and remove it later after the loop
                objects_to_remove.push_back(it->first);
            }else{
                // Move the particles of the object in obj_ptc_hash_map.
                int track_id = it->second.track_id;
                std::cout << "Checking track_id = " << track_id << std::endl;
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

                                std::cout << "Object point cloud size = " << object_point_cloud->points.size() << std::endl;
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
                                    std::cout << "Match score = " << match_score << ", good enough for matching." << std::endl;
                                }else{
                                    std::cout << "Match score = " << match_score << ", not good enough for matching." << std::endl;
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

                        // // Show tranformation_matrix_f
                        // std::cout << "tranformation_matrix_f = " << tranformation_matrix_f << std::endl;
                        // std::cout << "obj_ptc_indices size = " << obj_ptc_indices.size() << std::endl;

                        // // Move particles in the ring buffer
                        // std::unordered_set<uint32_t> obj_ptc_indices_new;
                        // op_mt_.moveParticlesInSetByTransformation(obj_ptc_indices, tranformation_matrix_f, obj_ptc_indices_new);
                        
                        // // Update the hash map
                        // object_set_.obj_ptc_hash_map.updatePtcIndicesOfObj(track_id, obj_ptc_indices_new);

                        // std::cout << "Transformation overflowed particle num = " << obj_ptc_indices.size() - obj_ptc_indices_new.size() << std::endl;
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
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        /****************** Update Weight ***********************/ 
        // Calculate the extrinsic matrix
        Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
        extrinsic.block<3,3>(0,0) = camera_orientation.cast<float>().toRotationMatrix();
        extrinsic.block<3,1>(0,3) = camera_position.cast<float>();
        extrinsic = extrinsic.inverse().eval();

        op_mt_.updateVisibleParitlcesWithBFS(extrinsic, depth_value_mat); // The time particle is ignored

        // Visualize the pyramid image
        showSimplePyramidImage();

        // Update the weight of particles in the FOV
        std::cout << "Update weight of particles in the FOV..............." << std::endl;
        if(getFlagUsePignisticProbability()){
            updateParticlesWithFreePoint(labeled_point_cloud);
        }else{
            updateParticles(labeled_point_cloud); 
        }
        std::cout << "Update weight done." << std::endl;
        
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

        std::cout << "*** added_particle_num = " << added_particle_num << std::endl;

        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

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
                if(label_id != label_id_map_default["Car"] || it->second.size() > 8000 || it->second.size() < 1500){ 
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
                std::cout << "Time used for matching: " << time_used_matching.count() << " s" << std::endl;
                std::cout << "match_score = " << match_score << std::endl;

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
                        pt_labeled.color_h = 0;

                        // Add a new particle to the ring buffer and obj_ptc_hash_map. No noise. Resample if necessary.
                        addGuessedParticle(pt_labeled);
                    }

                    // LabeledPoint pt_labeled;
                    // pt_labeled.position << pt.x, pt.y, pt.z;
                    // pt_labeled.label_id = label_id;
                    // pt_labeled.track_id = track_id;
                    // pt_labeled.color_h = 0;

                    // Add a new particle to the ring buffer and obj_ptc_hash_map. No noise. Resample if necessary.
                    // addGuessedParticle(pt_labeled);
                }
            }
        }

        std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();

        /**************** Get occupancy result ******************/
        // Camera intrinsic matrix and extrinsic matrix
        Eigen::Matrix3f intrinsic_matrix;
        intrinsic_matrix << g_camera_fx, 0.f, g_camera_cx,
                            0.f, g_camera_fy, g_camera_cy,
                            0.f, 0.f, 1.f;
        Eigen::Matrix4f extrinsic_matrix = Eigen::Matrix4f::Identity();
        extrinsic_matrix.block<3,3>(0,0) = camera_orientation.cast<float>().toRotationMatrix();
        extrinsic_matrix.block<3,1>(0,3) = camera_position.cast<float>();
        extrinsic_matrix = extrinsic_matrix.inverse().eval();
        
        // Now get the occupancy result
        getOccupancyResult(occupied_point_cloud, freespace_point_cloud, extrinsic_matrix, intrinsic_matrix, if_get_freespace);


        std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();


        // Print the time used for each step
        std::chrono::duration<double> time_used_prediction = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::chrono::duration<double> time_used_update_weight = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
        std::chrono::duration<double> time_used_particle_birth = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);
        std::chrono::duration<double> time_used_additional_particle_birth = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4);
        std::chrono::duration<double> time_used_occupancy_result = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5);

        std::cout << "Time used for prediction: " << time_used_prediction.count() << " s" << std::endl;
        std::cout << "Time used for update weight: " << time_used_update_weight.count() << " s" << std::endl;
        std::cout << "Time used for particle birth: " << time_used_particle_birth.count() << " s" << std::endl;
        std::cout << "Time used for additional particle birth: " << time_used_additional_particle_birth.count() << " s" << std::endl;
        std::cout << "Time used for occupancy result: " << time_used_occupancy_result.count() << " s" << std::endl;

        static double time_used_total_prediction = 0.0;
        static double time_used_total_update_weight = 0.0;
        static double time_used_total_particle_birth = 0.0;
        static double time_used_total_additional_particle_birth = 0.0;
        static double time_used_total_occupancy_result = 0.0;
        static int time_used_total_count = 0;

        time_used_total_prediction += time_used_prediction.count();
        time_used_total_update_weight += time_used_update_weight.count();
        time_used_total_particle_birth += time_used_particle_birth.count();
        time_used_total_additional_particle_birth += time_used_additional_particle_birth.count();
        time_used_total_occupancy_result += time_used_occupancy_result.count();
        time_used_total_count += 1;

        std::cout << "Average time used for prediction: " << time_used_total_prediction / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for update weight: " << time_used_total_update_weight / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for particle birth: " << time_used_total_particle_birth / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for additional particle birth: " << time_used_total_additional_particle_birth / time_used_total_count << " s" << std::endl;
        std::cout << "Average time used for occupancy result: " << time_used_total_occupancy_result / time_used_total_count << " s" << std::endl;


        /**************** Save the Object Points for Template ******************/
        // std::cout << "Object number = " << object_set_.object_tracking_hash_map.size() << std::endl;
        // // Iterate all the objects in object_tracking_hash_map
        // for(auto it = object_set_.object_tracking_hash_map.begin(); it != object_set_.object_tracking_hash_map.end(); ++it)
        // {
        //     // Consider newly updated objects only
        //     if(it->second.observation_time_step != global_time_stamp){continue;}
        //     int object_id = it->first;

        //     // Check if the object exists in obj_ptc_hash_map
        //     if(!object_set_.obj_ptc_hash_map.checkIfObjectExists(object_id)){
        //         std::cout << "Object " << object_id << " does not exist in obj_ptc_hash_map." << std::endl;
        //         continue;
        //     }

        //     std::cout << "Object " << object_id << " has " << object_set_.obj_ptc_hash_map.indices_map.at(object_id).size() << " particles." << std::endl;
            
        //     // Get the particles of object_id in obj_ptc_hash_map and save it to pcl point cloud
        //     pcl::PointCloud<pcl::PointXYZ>::Ptr object_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        //     for(std::unordered_set<uint32_t>::iterator it2 = object_set_.obj_ptc_hash_map.indices_map.at(object_id).begin(); it2 != object_set_.obj_ptc_hash_map.indices_map.at(object_id).end(); ++it2)
        //     {
        //         Eigen::Vector3f ptc_pos;
        //         float occ_weight, free_weight;
        //         op_mt_.getParticlePosWeightByIndex(*it2, ptc_pos, occ_weight, free_weight);

        //         pcl::PointXYZ pt;
        //         pt.x = ptc_pos.x();
        //         pt.y = ptc_pos.y();
        //         pt.z = ptc_pos.z();

        //         object_point_cloud->points.push_back(pt);
        //     }

        //     // Save the point cloud if the point number is larger than 100
        //     if(object_point_cloud->size() > 100)
        //     {   
        //         // Use global_time_stamp + object_id as the file name
        //         object_point_cloud->width = object_point_cloud->size();
        //         object_point_cloud->height = 1;
        //         std::string file_name = std::to_string(global_time_stamp) + "_" + std::to_string(object_id) + ".pcd";
        //         std::string full_path = "/home/clarence/ros_ws/semantic_dsp_ws/src/Semantic_DSP_Map/data/VirtualKitti2/pcd_constructed/" + file_name;
        //         pcl::io::savePCDFileASCII(full_path, *object_point_cloud);
        //     }
            
        // }
        
    }

    /// @brief Add a Guessed Particle from the template matching result
    /// @param pt 
    inline void addGuessedParticle(const LabeledPoint &pt)
    {
        // Add a new particle to the ring buffer. No noise.
        uint32_t voxel_idx, ptc_idx;
        op_mt_.addGuessedParticles(pt.position, pt.label_id, pt.track_id, pt.color_h, voxel_idx, ptc_idx);
        if(ptc_idx != INVALID_PARTICLE_INDEX)
        {   
            // Add the new particle to the object. 
            int track_id = static_cast<int>(pt.track_id);
            int label_id = static_cast<int>(pt.label_id);

            if(label_id > c_movable_object_label_id_start){
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
        op_mt_.addNewParticleWithSemantics(pt.position, pt.label_id, pt.track_id, pt.color_h, voxel_idx, ptc_idx);
        if(ptc_idx != INVALID_PARTICLE_INDEX)
        {   
            // Add the new particle to the object. 
            int track_id = static_cast<int>(pt.track_id);
            int label_id = static_cast<int>(pt.label_id);
            added_particle_num += 1;

            if(label_id >= c_movable_object_label_id_start){
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
            float sigma_this_pixel = pt.sigma;
            noise << gaussian_random_.queryNormalRandomZeroMean(sigma_this_pixel), gaussian_random_.queryNormalRandomZeroMean(sigma_this_pixel), gaussian_random_.queryNormalRandomZeroMean(sigma_this_pixel);
            
            uint32_t voxel_idx, ptc_idx;
            op_mt_.addNewParticleWithSemantics(pt.position + noise, pt.label_id, pt.track_id, pt.color_h, voxel_idx, ptc_idx);

            if(ptc_idx != INVALID_PARTICLE_INDEX)
            {   
                // Add the new particle to the object. 
                int track_id = static_cast<int>(pt.track_id);
                int label_id = static_cast<int>(pt.label_id);
                added_particle_num += 1;

                // Only add the particle to the object if it is a movable object
                if(label_id >= c_movable_object_label_id_start){
                    object_set_.obj_ptc_hash_map.addParticleToObj(track_id, ptc_idx);
                }
            }

            // Resampling if necessary
            if(voxel_idx != INVALID_PARTICLE_INDEX && resampled_voxel_indices.count(voxel_idx) == 0){
                if(resampleParticlesInVoxel(voxel_idx, object_set_.obj_ptc_hash_map)){
                    resampled_voxel_indices.insert(voxel_idx);
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
        int invalid_voxel_num = 0;
        for(uint32_t i=0; i<C_VOXEL_NUM_TOTAL; ++i){
            uint16_t track_id;
            uint8_t label_id;
            /// TODO:improve the efficiency of this function: determineIfVoxelOccupied
            int occ_result_this_voxel;
            if(getFlagUsePignisticProbability()){
                occ_result_this_voxel = op_mt_.determineIfVoxelOccupiedConsiderFreePoint(i, label_id, track_id, occupancy_threshold_);
            }else{
                occ_result_this_voxel = op_mt_.determineIfVoxelOccupied(i, label_id, track_id, occupancy_threshold_);
            }

            // int occupied_max_flag = 10;
            // if(if_out_evaluation_format_){ //Keep only the occupied voxel
            //     occupied_max_flag = 1;
            // }

            // // Count the number of invalid voxels
            // if(occ_result_this_voxel < 0){
            //     invalid_voxel_num ++;
            //     continue;
            // }

            if(occ_result_this_voxel > 0){  //&& occ_result_this_voxel <= occupied_max_flag    
                Eigen::Vector3f voxel_pos;
                op_mt_.getVoxelGlobalPosition(i, voxel_pos);

                pcl::PointXYZRGB pt;
                pt.x = voxel_pos.x();
                pt.y = voxel_pos.y();
                pt.z = voxel_pos.z();

                if(occ_result_this_voxel == 1){ // Occupied
                    // Using the color map
                    static int background_id = label_id_map_default["Background"];
                    static int tree_id = label_id_map_default["Tree"];
                    if(label_id == background_id){  // Ground et.al.
                        // Color by y axis. Map -1 to 5 to color_map_jet_256_
                        float y = voxel_pos.y();
                        int color_index = std::min(std::max(static_cast<int>((y+1.f)*51.2f), 0), 255);
                        pt.r = color_map_jet_256_[color_index](0);
                        pt.g = color_map_jet_256_[color_index](1);
                        pt.b = color_map_jet_256_[color_index](2);

                        if(if_out_evaluation_format_){
                            pt.r = label_id;
                        }

                    }else if(label_id == tree_id){
                        //Dark green
                        pt.r = 0;
                        if(if_out_evaluation_format_){
                            pt.r = label_id;
                        }
                        pt.g = 200;
                        pt.b = 0;
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
                    Eigen::Vector3f pt_vector(pt.x, pt.y, pt.z);
                    if(!op_mt_.checkIfPointInFrustum(pt_vector, extrinsic_matrix, intrinsic_matrix, g_image_width, g_image_height))
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
                pt.x = voxel_pos.x();
                pt.y = voxel_pos.y();
                pt.z = voxel_pos.z();

                pt.r = 0;
                pt.g = 255;
                pt.b = 0;

                freespace_point_cloud->points.push_back(pt);
            }
        }

        // std::cout << "invalid_voxel_num = " << invalid_voxel_num << std::endl;
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

    /// @brief Update the particles considering the free point. To be used for Pignistic Probability
    void updateParticlesWithFreePoint(const std::vector<std::vector<LabeledPoint>> &labeled_point_cloud)
    {
        // Calculate the neighbor size
        /// TODO: use adaptive neighbor size
        int neighbor_width_half_array[g_image_height][g_image_width];
        int neighbor_height_half_array[g_image_height][g_image_width];

        for(int i=0; i<g_image_height; ++i)
        {
            for(int j=0; j<g_image_width; ++j)
            { 
                neighbor_width_half_array[i][j] = 5;
                neighbor_height_half_array[i][j] = 5;
            }
        }

        // Calculate CK + kappa first
        float ck_kappa_array[g_image_height][g_image_width];
        float ck_kappa_free_array[g_image_height][g_image_width];
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
                float sigma_this_pixel_free = sigma_this_pixel * 0.1f;

                // Calculate the ck + kappa value of the pixel with neighbors
                float ck_this_pixel = 0.f;
                float ck_this_pixel_free = 0.f;

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
                        int max_particle_num_in_array = std::min(particle_num_this_pixel, C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID);
                        for(int l=0; l<max_particle_num_in_array; ++l)
                        {
                            auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_array[neighbor_i][neighbor_j][l]];

                            // Skip the particles that don't belong to the same object
                            if(particle->track_id != labeled_point_cloud[i][j].track_id){
                                continue;
                            }

                            float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[i][j].position.x(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[i][j].position.y(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[i][j].position.z(), sigma_this_pixel);

                            float gk_free = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[i][j].free_position.x(), sigma_this_pixel_free)
                                            * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[i][j].free_position.y(), sigma_this_pixel_free)
                                            * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[i][j].free_position.z(), sigma_this_pixel_free);

                            ck_this_pixel += particle->occ_weight * gk;
                            ck_this_pixel_free += particle->free_weight * gk_free;
                        }

                        // Consider the overflowed particles stored in the map
                        int overflowed_particle_num = particle_num_this_pixel - C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID;
                        if(overflowed_particle_num > 0){
                            int id = neighbor_i*g_image_width + neighbor_j;
                            for(int l=0; l<overflowed_particle_num; ++l)
                            {
                                auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][l]];
                             
                                // Skip the particles that don't belong to the same object
                                if(particle->track_id != labeled_point_cloud[i][j].track_id){
                                    continue;
                                }

                                float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[i][j].position.x(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[i][j].position.y(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[i][j].position.z(), sigma_this_pixel);

                                float gk_free = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[i][j].free_position.x(), sigma_this_pixel_free)
                                                * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[i][j].free_position.y(), sigma_this_pixel_free)
                                                * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[i][j].free_position.z(), sigma_this_pixel_free);

                                ck_this_pixel += particle->occ_weight * gk;
                                ck_this_pixel_free += particle->free_weight * gk_free;
                            }
                        }
                    }
                }

                ck_kappa_array[i][j] = ck_this_pixel * detection_probability_ + noise_number_;
                ck_kappa_free_array[i][j] = ck_this_pixel_free * detection_probability_ + noise_number_;
            }
        }

        // Update the weight of particles in the FOV
        for(int i=0; i<g_image_height; ++i)
        {
            for(int j=0; j<g_image_width; ++j)
            { 
                int neighbor_width_half = neighbor_width_half_array[i][j]; 
                int neighbor_height_half = neighbor_height_half_array[i][j];
                float sigma_this_pixel = labeled_point_cloud[i][j].sigma;
                float sigma_this_pixel_free = sigma_this_pixel * 0.1f;

                // Update particles in each pixel
                int particle_num_this_pixel = particle_to_pixel_num_array[i][j];
                int max_particle_num_in_array = std::min(particle_num_this_pixel, C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID);
                for(int l=0; l<max_particle_num_in_array; ++l)
                {
                    auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_array[i][j][l]];
                    float acc_this_particle = 0.f;
                    float acc_free_this_particle = 0.f;
                    for(int m=-neighbor_height_half; m<=neighbor_height_half; ++m)
                    {
                        for(int n=-neighbor_width_half; n<=neighbor_width_half; ++n)
                        {
                            int neighbor_i = i + m;
                            int neighbor_j = j + n;

                            if(neighbor_i < 0 || neighbor_i >= g_image_height || neighbor_j < 0 || neighbor_j >= g_image_width){
                                continue;
                            }

                            // Check if the point is valid
                            if(!labeled_point_cloud[neighbor_i][neighbor_j].is_valid){ 
                                continue;
                            }
                            // Skip the point that don't belong to the same object
                            if(labeled_point_cloud[neighbor_i][neighbor_j].track_id != particle->track_id){
                                continue;
                            }

                            float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[neighbor_i][neighbor_j].position.x(), sigma_this_pixel)
                                        * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[neighbor_i][neighbor_j].position.y(), sigma_this_pixel)
                                        * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[neighbor_i][neighbor_j].position.z(), sigma_this_pixel);
                            
                            float gk_free = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[neighbor_i][neighbor_j].free_position.x(), sigma_this_pixel_free)
                                            * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[neighbor_i][neighbor_j].free_position.y(), sigma_this_pixel_free)
                                            * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[neighbor_i][neighbor_j].free_position.z(), sigma_this_pixel_free);
            

                            acc_this_particle += gk / ck_kappa_array[neighbor_i][neighbor_j];
                            acc_free_this_particle += gk_free / ck_kappa_free_array[neighbor_i][neighbor_j];
                        }
                    }

                    static const float c_one_minus_detection_probability = 1.f - detection_probability_;

                    particle->occ_weight *= (acc_this_particle * detection_probability_ + c_one_minus_detection_probability);
                    particle->free_weight *= (acc_free_this_particle * detection_probability_ + c_one_minus_detection_probability);
                    particle->status = Particle_Status::UPDATED;
                    particle->time_stamp = global_time_stamp;
                }

                // Consider the overflowed particles stored in the map
                int overflowed_particle_num = particle_num_this_pixel - C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID;
                if(overflowed_particle_num > 0)
                {
                    int id = i*g_image_width + j;
                    for(int l=0; l<overflowed_particle_num; ++l)
                    {
                        auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][l]];
                        float acc_this_particle = 0.f;
                        float acc_free_this_particle = 0.f;

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

                                float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[neighbor_i][neighbor_j].position.x(), sigma_this_pixel)
                                            * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[neighbor_i][neighbor_j].position.y(), sigma_this_pixel)
                                            * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[neighbor_i][neighbor_j].position.z(), sigma_this_pixel);
                                
                                float gk_free = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[neighbor_i][neighbor_j].free_position.x(), sigma_this_pixel_free)
                                            * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[neighbor_i][neighbor_j].free_position.y(), sigma_this_pixel_free)
                                            * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[neighbor_i][neighbor_j].free_position.z(), sigma_this_pixel_free);
            
                                acc_this_particle += gk / ck_kappa_array[neighbor_i][neighbor_j];
                                acc_free_this_particle += gk_free / ck_kappa_free_array[neighbor_i][neighbor_j];
                            }
                        }

                        static const float c_one_minus_detection_probability = 1.f - detection_probability_;

                        particle->occ_weight *= (acc_this_particle * detection_probability_ + c_one_minus_detection_probability);
                        particle->free_weight *= (acc_free_this_particle * detection_probability_ + c_one_minus_detection_probability);

                        particle->status = Particle_Status::UPDATED;
                        particle->time_stamp = global_time_stamp;
                    }
                }

            }
        }

    }


    /// @brief Update the weight of particles in the FOV based on the SMC-PHD filter. 
    /// @param labeled_point_cloud: The labeled point cloud generated from the input data. 
    void updateParticles(const std::vector<std::vector<LabeledPoint>> &labeled_point_cloud)
    {
        // Calculate the neighbor size
        /// TODO: use adaptive neighbor size
        int neighbor_width_half_array[g_image_height][g_image_width];
        int neighbor_height_half_array[g_image_height][g_image_width];

        for(int i=0; i<g_image_height; ++i)
        {
            for(int j=0; j<g_image_width; ++j)
            { 
                neighbor_width_half_array[i][j] = 3;  //5
                neighbor_height_half_array[i][j] = 3;
            }
        }

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
                        int max_particle_num_in_array = std::min(particle_num_this_pixel, C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID);
                        for(int l=0; l<max_particle_num_in_array; ++l)
                        {
                            auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_array[neighbor_i][neighbor_j][l]];

                            if(getFlagUseIndependentFilter()){ //|| particle->track_id > c_max_movable_object_instance_id
                                // Skip the particles that don't belong to the same object
                                if(particle->track_id != labeled_point_cloud[i][j].track_id){
                                    continue;
                                }
                            }
                            
                            float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[i][j].position.x(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[i][j].position.y(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[i][j].position.z(), sigma_this_pixel);

                            if(!getFlagUseIndependentFilter())
                            {
                                /// TODO: Use a better ID transition probability
                                gk *= getForgettingFactor(particle->forget_count, forgetting_rate_);
                                if(particle->track_id != labeled_point_cloud[i][j].track_id)
                                {
                                    gk *= 0.5f; // ID transition probability
                                }
                            }

                            ck_this_pixel += particle->occ_weight * gk;
                        }

                        // Consider the overflowed particles stored in the map
                        int overflowed_particle_num = particle_num_this_pixel - C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID;
                        if(overflowed_particle_num > 0){
                            int id = neighbor_i*g_image_width + neighbor_j;
                            for(int l=0; l<overflowed_particle_num; ++l)
                            {
                                auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][l]];

                                if(getFlagUseIndependentFilter()){ //|| particle->track_id > c_max_movable_object_instance_id
                                    // Skip the particles that don't belong to the same object
                                    if(particle->track_id != labeled_point_cloud[i][j].track_id){
                                        continue;
                                    }
                                }
                                
                                float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[i][j].position.x(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[i][j].position.y(), sigma_this_pixel)
                                       * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[i][j].position.z(), sigma_this_pixel);

                                if(!getFlagUseIndependentFilter())
                                {
                                    /// TODO: Use a better ID transition probability
                                    gk *= getForgettingFactor(particle->forget_count, forgetting_rate_);
                                    if(particle->track_id != labeled_point_cloud[i][j].track_id)
                                    {
                                        gk *= 0.5f; // ID transition probability
                                    }
                                }

                                ck_this_pixel += particle->occ_weight * gk;
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
                int max_particle_num_in_array = std::min(particle_num_this_pixel, C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID);
                for(int l=0; l<max_particle_num_in_array; ++l)
                {
                    auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_array[i][j][l]];
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

                            // Check if the point is valid
                            if(!labeled_point_cloud[neighbor_i][neighbor_j].is_valid){ 
                                continue;
                            }

                            if(getFlagUseIndependentFilter() ){ //|| particle->track_id > c_max_movable_object_instance_id
                                // Skip the point that don't belong to the same object
                                if(labeled_point_cloud[neighbor_i][neighbor_j].track_id != particle->track_id){
                                    continue;
                                }
                            }

                            float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[neighbor_i][neighbor_j].position.x(), sigma_this_pixel)
                                        * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[neighbor_i][neighbor_j].position.y(), sigma_this_pixel)
                                        * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[neighbor_i][neighbor_j].position.z(), sigma_this_pixel);
                            
                            if(!getFlagUseIndependentFilter())
                            {
                                /// TODO: Use a better ID transition probability
                                if(particle->track_id != labeled_point_cloud[neighbor_i][neighbor_j].track_id){
                                    gk *= 0.5f; // ID transition probability
                                }else{
                                    if(gk > c_min_rightly_updated_pdf){updated_with_right_id = true;}
                                }
                                gk *= getForgettingFactor(particle->forget_count, forgetting_rate_);
                            }
                            
                            acc_this_particle += gk / ck_kappa_array[neighbor_i][neighbor_j];
                        }
                    }

                    particle->occ_weight *= (acc_this_particle * detection_probability_ + 1.f - detection_probability_);
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

                // Consider the overflowed particles stored in the map
                int overflowed_particle_num = particle_num_this_pixel - C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID;
                if(overflowed_particle_num > 0)
                {
                    int id = i*g_image_width + j;
                    for(int l=0; l<overflowed_particle_num; ++l)
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

                                if(getFlagUseIndependentFilter()) //|| particle->track_id > c_max_movable_object_instance_id
                                {
                                    // Skip the point that don't belong to the same object
                                    if(labeled_point_cloud[neighbor_i][neighbor_j].track_id != particle->track_id){
                                        continue;
                                    }
                                }
                                
                                float gk = gaussian_random_.queryNormalPDF(particle->pos.x(), labeled_point_cloud[neighbor_i][neighbor_j].position.x(), sigma_this_pixel)
                                            * gaussian_random_.queryNormalPDF(particle->pos.y(), labeled_point_cloud[neighbor_i][neighbor_j].position.y(), sigma_this_pixel)
                                            * gaussian_random_.queryNormalPDF(particle->pos.z(), labeled_point_cloud[neighbor_i][neighbor_j].position.z(), sigma_this_pixel);

                                if(!getFlagUseIndependentFilter())
                                {
                                    /// TODO: Use a better ID transition probability
                                    if(particle->track_id != labeled_point_cloud[neighbor_i][neighbor_j].track_id){
                                        gk *= 0.5f; // ID transition probability
                                    }else{
                                        if(gk > c_min_rightly_updated_pdf){updated_with_right_id = true;}
                                    }
                                    gk *= getForgettingFactor(particle->forget_count, forgetting_rate_);
                                }
                                
                                acc_this_particle += gk / ck_kappa_array[neighbor_i][neighbor_j];
                            }
                        }

                        particle->occ_weight *= (acc_this_particle * detection_probability_ + 1.f - detection_probability_);
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
                weight_sum += PARTICLE_ARRAY[start_ptc_seq+i].occ_weight;
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
                    particle_weight_sum += PARTICLE_ARRAY[particle_index].occ_weight;

                    if(particle_weight_sum < particle_weight_sum_threshold){
                        // Remove the particle from the particle array
                        PARTICLE_ARRAY[particle_index].status = Particle_Status::INVALID;
                        // Remove the particle from the object particle hash map
                        obj_ptc_hash_map.removeParticleFromObj(particle_track_id, particle_index);
                    }else{
                        PARTICLE_ARRAY[particle_index].occ_weight = weight_per_particle;
                        particle_weight_sum_threshold += weight_per_particle;
                        // Copy the particle to a vacant position if the weight of the particle is very large
                        while(particle_weight_sum > particle_weight_sum_threshold){
                            // Find a vacant position in the voxel
                            for(int j=1; j<C_MAX_PARTICLE_NUM_PER_VOXEL; ++j){
                                uint32_t copied_particle_index = start_ptc_seq+j;
                                if(PARTICLE_ARRAY[copied_particle_index].status == Particle_Status::INVALID){
                                    PARTICLE_ARRAY[copied_particle_index] = PARTICLE_ARRAY[particle_index]; // Copy
                                    PARTICLE_ARRAY[copied_particle_index].status = Particle_Status::COPIED; // Set status as COPIED in case next resampling will still consider this particle
                                    // Add the particle to the object particle hash map
                                    obj_ptc_hash_map.addParticleToObj(particle_track_id, copied_particle_index);
                                    break;
                                }
                            }
                            particle_weight_sum_threshold += weight_per_particle;
                        }
                    }
                }
            }

            return true;
        }else{
            return false;
        }

        /// TODO: Further Verify the resampling result
        
        // For Test: Count the number of particles in the voxel
        // uint32_t particle_num = 0;
        // for(int i=1; i<C_MAX_PARTICLE_NUM_PER_VOXEL; ++i)
        // {
        //     if(PARTICLE_ARRAY[start_ptc_seq+i].status == Particle_Status::UPDATED || PARTICLE_ARRAY[start_ptc_seq+i].status == Particle_Status::COPIED){
        //         ++ particle_num;
        //     }
        // }
        // if(updated_particle_num > 0){
        //     std::cout << "Original Updated Particle number " << updated_particle_num << ". Resampled Particle number: " << particle_num << std::endl;
        // }
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
                int particle_num_vector = std::min(particle_num_this_pixel, C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID);
                int overflowed_particle_num = particle_num_this_pixel - C_ESTIMATED_PARTICLE_NUM_PER_PYRAMID;

                for(int m=0; m<particle_num_vector; ++m)
                {
                    auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_array[i][j][m]];
                    weight_sum += particle->occ_weight;
                }

                if(overflowed_particle_num > 0){
                    int id = i*g_image_width + j;
                    for(int m=0; m<overflowed_particle_num; ++m)
                    {
                        auto *particle = &PARTICLE_ARRAY[particle_to_pixel_index_map[id][m]];
                        weight_sum += particle->occ_weight;
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
    }
};

