/**
 * @file object_layer.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief Update the object layer in the map. Unfinished. Only use ground truth tracking now.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#pragma once

#include "data_base.h"
#include "basic_algorithms.h"
#include "mc_ring/buffer.h"

/// @brief A simple class to store which object has which particle to build the
class ObjectParticleHashMap
{
public:
    ObjectParticleHashMap(){};
    ~ObjectParticleHashMap(){};

    inline void updatePtcIndicesOfObj(int &track_id, std::unordered_set<uint32_t> &particle_indices){
        indices_map[track_id] = particle_indices;
    }

    inline void addParticleToObj(int &track_id, uint32_t &particle_id){
        indices_map[track_id].insert(particle_id);
    }
    
    inline void removeParticleFromObj(int &track_id, uint32_t &particle_id){
        indices_map[track_id].erase(particle_id);
    }
    
    inline bool doesObjectHaveParticle(int &track_id, uint32_t &particle_id){
        return indices_map[track_id].count(particle_id) > 0;
    }

    inline bool checkIfObjectExists(int &track_id){
        return indices_map.count(track_id) > 0;
    }

    void clear(){
        indices_map.clear();
    }
    
    // A map to store which object has which particle. The key is the object track_id. The value is a set of particle indices.
    std::unordered_map<int, std::unordered_set<uint32_t>> indices_map;
};



/// @brief This class is used to estimate the 6d motion of an object
class MotionEstimation {
public:
    Eigen::Vector3d translationVelocity;
    Eigen::Vector3d angularVelocity;

    MotionEstimation() {
        translationVelocity = Eigen::Vector3d::Zero();
        angularVelocity = Eigen::Vector3d::Zero();
    }

    ~MotionEstimation() {};

    /// @brief This function computes the centroid (average) of a set of points. It's used to find the center of mass of the group of points.
    /// @param points 
    /// @return 
    Eigen::Vector3d computeCentroid(const std::vector<Eigen::Vector3d>& points) {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& point : points) {
            centroid += point;
        }
        return centroid / static_cast<double>(points.size());
    }

    /// @brief This function computes the relative rotation between two points in the form of a quaternion. The two points are interpreted as vectors from the origin, and the quaternion representing the rotation from p1 to p2 is computed.
    /// @param p1 
    /// @param p2 
    /// @return 
    Eigen::Quaterniond computeRelativeQuaternion(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
        Eigen::Quaterniond q1(0, p1.x(), p1.y(), p1.z());
        Eigen::Quaterniond q2(0, p2.x(), p2.y(), p2.z());
        return q2 * q1.inverse();
    }

    /// @brief This function estimates the translational and angular velocities of an object using a set of transformation matrices that describe the rigid body transformations between two consecutive time steps.
    /// @param T_matrices 
    /// @param time_deltas
    /// @param reference_point 
    void estimateByTransformations(const std::vector<Eigen::Matrix4d> &T_matrices, const std::vector<double>& time_deltas, const std::vector<Eigen::Vector3d> &reference_points)
    {
        // Check if the number of transformation matrices is equal to the number of time deltas
        if(T_matrices.size() != time_deltas.size()){
            std::cout << "The number of transformation matrices is not equal to the number of time deltas!" << std::endl;
            return;
        }

        // Create the points_groups. Each point group contains one reference point and two constructed points.
        std::vector<std::vector<Eigen::Vector3d>> points_groups;
        std::vector<std::vector<Eigen::Vector3d>> points_groups_frame_last;

        for(int i=0; i<T_matrices.size(); ++i)
        {
            std::vector<Eigen::Vector3d> points;
            points.push_back(reference_points[i]);

            // Add two more points to the points
            Eigen::Vector3d reference_point_x_plus = reference_points[i] + Eigen::Vector3d(1,0,0);
            points.push_back(reference_point_x_plus);
            Eigen::Vector3d reference_point_y_plus = reference_points[i] + Eigen::Vector3d(0,1,0);
            points.push_back(reference_point_y_plus);
            
            // Apply the transformation matrix to the points
            std::vector<Eigen::Vector3d> points_transformed;
            for(int j=0; j<points.size(); ++j)
            {
                Eigen::Vector4d transformed_point = T_matrices[i] * Eigen::Vector4d(points[j].x(), points[j].y(), points[j].z(), 1);
                points_transformed.push_back(Eigen::Vector3d(transformed_point.x(), transformed_point.y(), transformed_point.z()));
            }
            points_groups.push_back(points_transformed);
            points_groups_frame_last.push_back(points);
        }


        // Estimate the translational and angular velocities with the points_groups and time_deltas_new
        estimate(points_groups, points_groups_frame_last, time_deltas);
    }


    /// @brief This is the core function where the translational and angular velocities are estimated
    /// @param points_groups 
    /// @param time_deltas 
    void estimate(const std::vector<std::vector<Eigen::Vector3d>>& points_groups, const std::vector<std::vector<Eigen::Vector3d>>& points_groups_last_frame, const std::vector<double>& time_deltas) {
        /// TODO: IMPROVE the estimation. CHG.
        // Check if the size of points_groups is equal to the size of points_groups_last_frame
        if(points_groups.size() != points_groups_last_frame.size()){
            std::cout << "The size of points_groups is not equal to the size of points_groups_last_frame!" << std::endl;
            return;
        }

        Eigen::Vector3d translationVelocitySum = Eigen::Vector3d::Zero();
        Eigen::Vector3d angularVelocitySum = Eigen::Vector3d::Zero();

        for(int i=0; i<points_groups.size(); ++i){
            Eigen::Vector3d centroidPrev = computeCentroid(points_groups_last_frame[i]);
            Eigen::Vector3d centroidCurr = computeCentroid(points_groups[i]);

            translationVelocitySum += (centroidCurr - centroidPrev) / time_deltas[i];

            Eigen::Quaterniond relativeQ = computeRelativeQuaternion(centroidPrev, centroidCurr);
            Eigen::AngleAxisd rotation(relativeQ);
            angularVelocitySum += rotation.axis() * rotation.angle() / time_deltas[i];
        }

        translationVelocity = translationVelocitySum / (points_groups.size() - 1);
        angularVelocity = angularVelocitySum / (points_groups.size() - 1);

        /// TEST CODE. CHG.
        angularVelocity = Eigen::Vector3d::Zero();
        translationVelocity.y() = 0;

    }

    /// @brief This function predicts the future positions of a set of points using the previously estimated translational and angular velocities
    /// @param last_points 
    /// @param delta_time 
    /// @return 
    std::vector<Eigen::Vector3d> predict(const std::vector<Eigen::Vector3d>& last_points, double delta_time) {
        Eigen::Vector3d predictedTranslation = translationVelocity * delta_time;

        Eigen::AngleAxisd rotation(delta_time * angularVelocity.norm(), angularVelocity.normalized());
        Eigen::Quaterniond q(rotation);

        // std::cout << "translationVelocity: " << translationVelocity.transpose() << std::endl;
        // std::cout << "angularVelocity: " << angularVelocity.transpose() << std::endl;

        std::vector<Eigen::Vector3d> predicted_points;
        for (const auto& point : last_points) {
            Eigen::Quaterniond p(0, point.x(), point.y(), point.z());
            Eigen::Quaterniond movedP = q * p * q.inverse();
            Eigen::Vector3d movedPoint(movedP.x(), movedP.y(), movedP.z());
            predicted_points.push_back(movedPoint + predictedTranslation);
        }

        return predicted_points;
    }

    /// @brief This function predicts the future transformation matrix of an object using the previously estimated translational and angular velocities
    Eigen::Matrix4d predictTransformationMatrix(double delta_time)
    {
        Eigen::Matrix4d T_predicted = Eigen::Matrix4d::Identity();

        Eigen::Vector3d predictedTranslation = translationVelocity * delta_time;

        /// TODO: CHECK. rotation seems to be wrong with two angularVelocity
        Eigen::AngleAxisd rotation(delta_time * angularVelocity.norm(), angularVelocity.normalized());
        Eigen::Quaterniond q(rotation);

        T_predicted.block<3,3>(0,0) = q.toRotationMatrix();
        T_predicted.block<3,1>(0,3) = predictedTranslation;

        return T_predicted;
    }
};


/// @brief This class is to store the object transformations
class ObjectTransformations
{
public:
    ObjectTransformations():max_window_size(5), updated_(false){};
    ~ObjectTransformations(){};

    Eigen::Matrix4d t_matrix_curr;
    Eigen::Vector3d reference_point_curr;
    double delta_t_curr;
    int max_window_size;

    /// @brief Update the object transformations
    /// @param t_matrix 
    /// @param delta_t 
    /// @param reference_point 
    void update(const Eigen::Matrix4d &t_matrix, const double delta_t, const Eigen::Vector3d &reference_point)
    {
        t_matrix_curr = t_matrix;
        delta_t_curr = delta_t;
        reference_point_curr = reference_point;

        t_matrix_vec_.push_back(t_matrix);
        delta_t_vec_.push_back(delta_t);
        reference_point_vec_.push_back(reference_point);
        t_matrix_time_stamp_vec_.push_back(global_time_stamp);

        // Remove too old transformations
        while(reference_point_vec_.size() > 0)
        {
            /// TODO: Make the too old time step a variable. CHG
            if(global_time_stamp - t_matrix_time_stamp_vec_[0] > 10){
                t_matrix_vec_.erase(t_matrix_vec_.begin());
                delta_t_vec_.erase(delta_t_vec_.begin());
                t_matrix_time_stamp_vec_.erase(t_matrix_time_stamp_vec_.begin());
                reference_point_vec_.erase(reference_point_vec_.begin());
            }else{
                break;
            }
        }

        // Keep the window size.
        if(t_matrix_vec_.size() > max_window_size){
            t_matrix_vec_.erase(t_matrix_vec_.begin());
            delta_t_vec_.erase(delta_t_vec_.begin());
            t_matrix_time_stamp_vec_.erase(t_matrix_time_stamp_vec_.begin());
            reference_point_vec_.erase(reference_point_vec_.begin());
        }

        // The object is updated (can be used for prediction) only if the number of transformations is no less than 2
        if(t_matrix_vec_.size() < 3){
            updated_ = false;
            return;
        }

        estimator_.estimateByTransformations(t_matrix_vec_, delta_t_vec_, reference_point_vec_);
        updated_ = true;
    }


    /// @brief Predict the future position of the reference point
    /// @param delta_t
    /// @return predict T matrix
    bool predictTMatrix(double delta_t, Eigen::Matrix4d &t_matrix)
    {
        if(updated_){
            t_matrix = estimator_.predictTransformationMatrix(delta_t);
            return true;
        }else{
            return false;
        }
    }

    /// @brief Clear the object transformations data
    void clear(){
        t_matrix_vec_.clear();
        delta_t_vec_.clear();
        reference_point_vec_.clear();
        t_matrix_time_stamp_vec_.clear();
        updated_ = false;
    }

    /// @brief Check if the object is updated. If not, the object cannot be used for prediction.
    bool checkIfUpdated(){
        return updated_;
    }

private:
    std::vector<Eigen::Matrix4d> t_matrix_vec_;
    std::vector<uint32_t> t_matrix_time_stamp_vec_;
    std::vector<double> delta_t_vec_;
    std::vector<Eigen::Vector3d> reference_point_vec_;

    MotionEstimation estimator_;
    bool updated_;
};


/// TODO: Remove MJ Representation. Use simple object representation. CHG
/// @brief A struct to store the data of an object with multiple joints (MJ) / rigidbodies, e.g., human, manipulator
class MJObject
{   
public:
    MJObject():moved_probability(0.5){};

    ~MJObject(){};

    ///< Object label id
    int label;
    ///< Object confidence from the detector
    double confidence;

    ///< Updated time stamp
    uint32_t time_stamp;

    ///< ObjectTransformations
    ObjectTransformations transformations;

    ///< Reference point for transformation. Use a valid keypoint/joint. CHG
    Eigen::Vector3d reference_point;

    ///< Position 
    Eigen::Vector3d position;
    ///< Orientation
    Eigen::Quaterniond orientation;

    ///< Joints vector 
    std::vector<Eigen::Vector3d> joints; 
    ///< Joints rigidbodies
    std::vector<std::vector<int>> rigidbody_joint_seq;
    ///< Rigitbody transformation matrix
    std::vector<Eigen::Matrix4d> rigidbody_tmatrix_vec;
    ///< If the rigidbody moved in this step. Check by gt pose or by the change of the joints/features.
    std::vector<bool> rigidbody_moved_vec;

    double moved_probability;

    ///< Last joints vector. For visualization only.
    std::vector<Eigen::Vector3d> joints_last_vis; 
};


/// @brief A class to store the objects in tracking and do common operations for object-level tracking and updating
class ObjectSet
{
public:
    ObjectSet():new_track_id_(0){};
    ~ObjectSet(){};

private:
    // A struct to store the object in tracking
    struct ObjectInTracking
    {
        int track_id;
        int observation_time_step; 
        int observation_count;

        float completeness;

        bool to_match_with_templates;
        bool to_match_with_previous;

        MJObject object;
    };

    // The new track id for a new object
    int new_track_id_;

    
public:
    // The hash map to store the objects in tracking. The key is the track_id. The value is the object.
    std::unordered_map<int, ObjectInTracking> object_tracking_hash_map;

    // The map to store which object has which particle
    ObjectParticleHashMap obj_ptc_hash_map;

    /// @brief: A function to clear the object_tracking_hash_map and obj_ptc_hash_map
    void clear(){
        object_tracking_hash_map.clear();
        
        // Clear the hash map
        obj_ptc_hash_map.clear();
    }

    /// @brief: A function to add a new object to the object_tracking_hash_map
    /// @param object: The new object
    void addNewObject(const MJObject &object, int track_id = -1){
        ObjectInTracking obj;
        if(track_id == -1){
            obj.track_id = new_track_id_;
            new_track_id_++;
            if(new_track_id_ > 1000000){
                new_track_id_ = 0;
            }
        }else{
            obj.track_id = track_id;
        }
        
        obj.observation_time_step = global_time_stamp;
        obj.observation_count = 1;

        obj.object = object;
        obj.to_match_with_templates = true; // Match with templates for the newly added objects
        obj.to_match_with_previous = false; // Do not match with previous objects for the newly added objects
                
        object_tracking_hash_map[obj.track_id] = obj; // Add the object to the hash map
    }

    /// @brief: A function to remove an object from the object_tracking_hash_map by its track_id
    /// @param track_id
    inline void removeObjectByTrackID(int track_id){
        // Make the flags of particles of the object invalid
        for (const auto& particle_id : obj_ptc_hash_map.indices_map[track_id]) {
            PARTICLE_ARRAY[particle_id].status = Particle_Status::INVALID;
        }

        // Remove the particle indices of the object from obj_ptc_hash_map 
        obj_ptc_hash_map.indices_map.erase(track_id);

        // Remove the object from object_tracking_hash_map
        object_tracking_hash_map.erase(track_id);
    }

    /// @brief: A function to remove the objects that are not observed for a long time
    /// @param step_threshold: The threshold of the time steps
    void removeOldObjects(int step_threshold = 100){
        int count = 0;
        std::vector<int> keys_to_remove;

        // Iterate over each object in object_tracking_hash_map
        for (const auto& pair : object_tracking_hash_map) {
            if (global_time_stamp - pair.second.observation_time_step > step_threshold) {
                keys_to_remove.push_back(pair.first);
            }
        }
        // Step 2: Remove them.
        for (const int key : keys_to_remove) {
            removeObjectByTrackID(key);
            ++count;
        }

        std::cout << "global_time_stamp: " << global_time_stamp << std::endl;
        std::cout << "Remove " << count << " old objects." << std::endl;
        std::cout << "The number of objects in the tracking set is " << object_tracking_hash_map.size() << std::endl;
    }

    /// @brief Check if an object exists in the object_tracking_hash_map by its track_id
    /// @param track_id 
    /// @return 
    bool checkIfObjectExists(int track_id){
        return object_tracking_hash_map.count(track_id) > 0;
    }

    /// @brief: Update the object in the tracking set with the new observation using ground truth tracking
    // void updateWithGroundTruthTracking(const int object_in_tracking_id, const MJObject &new_observation)
    // {
    //     auto *object_in_tracking = &object_tracking_hash_map[object_in_tracking_id];

    //     // Calculate the rigid body transformation matrix. Only one rigid body is considered here.
    //     std::vector<Eigen::Matrix4d> transformation_matrix_vec;

    //     Eigen::Quaterniond q1 = object_in_tracking->object.orientation;
    //     Eigen::Quaterniond q2 = new_observation.orientation;
    //     Eigen::Vector3d p1 = object_in_tracking->object.position;
    //     Eigen::Vector3d p2 = new_observation.position;

    //     Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

    //     transformation_matrix.block<3,3>(0,0) = q1.toRotationMatrix().transpose() * q2.toRotationMatrix();
    //     transformation_matrix.block<3,1>(0,3) = p2 - transformation_matrix.block<3,3>(0,0) * p1;
    //     transformation_matrix_vec.push_back(transformation_matrix);

    //     // Check if the transformation is correct
    //     Eigen::Vector3d p1_new = transformation_matrix_vec[0].block<3,3>(0,0) * p1 + transformation_matrix_vec[0].block<3,1>(0,3);
    //     // std::cout << "p1_new = " << p1_new.transpose() << std::endl;
    //     // std::cout << "p2 = " << p2.transpose() << std::endl;

    //     // Check if the object moved
    //     std::vector<bool> rigidbody_moved_vec;
    //     bool object_moved = false;
    //     if((p2 - p1).norm() > 0.01){
    //         object_moved = true;
    //     }else if((q1.toRotationMatrix().transpose() * q2.toRotationMatrix() - Eigen::Matrix3d::Identity()).norm() > 0.01){
    //         object_moved = true;
    //     }
    //     rigidbody_moved_vec.push_back(object_moved);

    //     object_in_tracking->object = new_observation;
    //     object_in_tracking->observation_time_step = global_time_stamp;
    //     object_in_tracking->observation_count++;
    //     object_in_tracking->object.rigidbody_tmatrix_vec = transformation_matrix_vec;
    //     object_in_tracking->object.rigidbody_moved_vec = rigidbody_moved_vec;

    //     /// Note: these conditions should be transferred. to_match_with_templates has been transferred to addtional new-born part.
    //     // object_in_tracking->to_match_with_previous = false;  // Do not match with old updates for the objects with ground truth tracking
    //     // object_in_tracking->to_match_with_templates = false; // Do not match with templates for the objects not newly added
    // }


    /// @brief Update the object in the tracking set with the new observation using external tracking
    /// @param object_in_tracking_id Tracking id of the object to be updated
    /// @param new_observation The new observation
    /// @param transformation_confidence Confidence of the transformation matrix
    /// @param movement_distance_threshold The distance threshold to determine if the object moved as a condition in the Beyesian filter
    /// @param movement_probability_threshold The probability threshold to determine if the object moved as a condition in the Beyesian filter
    /// @param movement_increment The increment of the moved probability in the Beyesian filter
    /// @param movement_decrement The decrement of the moved probability in the Beyesian filter
    void updateObject(const int object_in_tracking_id, const MJObject &new_observation, double transformation_confidence = 1.0, double movement_distance_threshold = 0.1, double movement_probability_threshold = 0.69, double movement_increment = 0.1, double movement_decrement = 0.15)
    {
        auto *object_in_tracking = &object_tracking_hash_map[object_in_tracking_id];

        // Check if the object moved by checking the transformation matrix
        std::vector<bool> rigidbody_moved_vec;

        bool object_moved = false;
        Eigen::Matrix4d transformation_matrix = new_observation.rigidbody_tmatrix_vec[0];

        Eigen::Vector4d reference_point_transformed = transformation_matrix * Eigen::Vector4d(new_observation.reference_point.x(), new_observation.reference_point.y(), new_observation.reference_point.z(), 1);
        Eigen::Vector3d reference_point_transition = Eigen::Vector3d(reference_point_transformed.x(), reference_point_transformed.y(), reference_point_transformed.z()) - new_observation.reference_point;

        // std::cout << "** object_in_tracking_id = " << object_in_tracking_id << std::endl;
        // std::cout << "reference_point = " << new_observation.reference_point.transpose() << std::endl;
        std::cout << "reference_point_transition = " << reference_point_transition.transpose() << std::endl;

        /// Use a Beyesian filter to determine if the object moved to filter the noise.
        /// TODO: Improve the moving determination. CHG. Consider accumulation if the transformation is small.
        if(reference_point_transition.norm() > movement_distance_threshold && transformation_confidence > 0.8){
            object_in_tracking->object.moved_probability += movement_increment;
        }else{
            object_in_tracking->object.moved_probability -= movement_decrement;
        }
        
        if(object_in_tracking->object.moved_probability > movement_probability_threshold){
            object_moved = true;  // The object moved
        }else{
            object_moved = false; // The object did not move
        }
        
        // Limit the moved probability to [0,1]
        object_in_tracking->object.moved_probability = std::min(1.0, std::max(0.0, object_in_tracking->object.moved_probability));

        rigidbody_moved_vec.push_back(object_moved);

        // Update the object with the new observation
        object_in_tracking->object.time_stamp = new_observation.time_stamp;
        object_in_tracking->object.label = new_observation.label;
        object_in_tracking->object.confidence = new_observation.confidence;
        object_in_tracking->object.rigidbody_tmatrix_vec = new_observation.rigidbody_tmatrix_vec;

        object_in_tracking->observation_time_step = global_time_stamp;
        object_in_tracking->observation_count++;

        object_in_tracking->to_match_with_previous = false;
        ///Note: To_match_with_templates has been transferred to addtional new-born part.
        // object_in_tracking->to_match_with_templates = false;

        object_in_tracking->object.rigidbody_moved_vec = rigidbody_moved_vec;

        /// TODO: Update for later prediction. Use a non-fixed time step. CHG
        if(object_moved){
            object_in_tracking->object.transformations.update(transformation_matrix, 0.2, new_observation.reference_point);
        }
        
    }

    /// @brief Set the flags for the object to be updated by matched with the previous object's point template.
    /// @param object_in_tracking_id 
    void setFlagsUpdateByMatching(const int object_in_tracking_id)
    {
        auto *object_in_tracking = &object_tracking_hash_map[object_in_tracking_id];

        object_in_tracking->observation_time_step = global_time_stamp;

        /// NOTE: Clear the transformations is not necessary!!! CHG
        // object_in_tracking->object.transformations.clear();
        object_in_tracking->to_match_with_previous = true;
        object_in_tracking->to_match_with_templates = false;
    }


    /// @brief Predict the tramnsformation matrix of the object in tracking if the object is not observed in this step
    /// @param object_in_tracking_id 
    void predictAndSetTransformation(const int object_in_tracking_id)
    {
        auto *object_in_tracking = &object_tracking_hash_map[object_in_tracking_id];

        // Update only the rigidbody_tmatrix_vec for particle layer prediction. Keep the rigidbody_moved_vec.
        std::vector<Eigen::Matrix4d> transformation_matrix_vec;
        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
        
        /// Use prediction if the object was observed so that the transformation estimation is valid.
        if(object_in_tracking->object.transformations.predictTMatrix(0.2, transformation_matrix)){
            // Print the predicted transformation matrix
            std::cout << "transformation_matrix = " << transformation_matrix << std::endl;
            transformation_matrix_vec.push_back(transformation_matrix);
        }else{
            std::cout << "No prediction for object_in_tracking_id = " << object_in_tracking_id << std::endl;
            // Use the last transformation matrix if the object was not observed in this step
            transformation_matrix_vec = object_in_tracking->object.rigidbody_tmatrix_vec;
        }

        object_in_tracking->to_match_with_previous = false;        
        object_in_tracking->object.rigidbody_tmatrix_vec = transformation_matrix_vec;
    }


    /// @brief: A function to remove the objects that have few particles. This function should be used after prediction.
    void removeObjectsWithFewParticles(int threshold = 0)
    {
        ///TODO: Use this function somewhere. CHG
        std::vector<int> removed_obj_id_vec;
        for (const auto& pair : object_tracking_hash_map) {
            if (obj_ptc_hash_map.indices_map[pair.first].size() < threshold) {
                removed_obj_id_vec.push_back(pair.first);
            }
        }

        // Remove the objects from obj_ptc_hash_map
        for(int i=0; i < removed_obj_id_vec.size(); ++i){
            removeObjectByTrackID(removed_obj_id_vec[i]);
        }
    }
    

private:
   
    /// @brief: A function to compute the rigidbody_tmatrix_vec of the current object
    int objectRigidbodiesTranlationMatrixCal(const MJObject &ob_new_observation, const MJObject &ob_ori, std::vector<Eigen::Matrix4d> &transformation_matrix_vec)
    {   
        // Check if the number of rigid bodies is equal
        int rigidbody_num = ob_new_observation.rigidbody_joint_seq.size();
        if(rigidbody_num > 0){
            if(rigidbody_num != ob_ori.rigidbody_joint_seq.size())
            {
                std::cout << "The number of rigid bodies in the current object is not equal to the number of rigid bodies in the tracking object!" << std::endl;
                return -1;
            }

            // Iterate over each rigid body
            for(int i=0; i < rigidbody_num; ++i){
                int joint_num = ob_new_observation.rigidbody_joint_seq[i].size();
                
                if(joint_num != ob_ori.rigidbody_joint_seq[i].size())
                {
                    std::cout << "The number of joints in the current rigid body is not equal to the number of joints in the tracking rigid body!" << std::endl;
                    return -1;
                }

                Eigen::MatrixXd P(3, joint_num);
                Eigen::MatrixXd Q(3, joint_num);
                // Iterate over keypoints/joints in each rigid body
                for(int j=0; j < joint_num; ++j){
                    int joint_id = ob_new_observation.rigidbody_joint_seq[i][j];
                    P.col(j) = ob_ori.joints[joint_id];
                    Q.col(j) = ob_new_observation.joints[joint_id];
                }

                Eigen::Matrix4d transformation_matrix = estimateTransformation(P, Q);
                transformation_matrix_vec.push_back(transformation_matrix);
            }
            std::cout << "transformation_matrix_vec size =" << transformation_matrix_vec.size() << std::endl;
        }

        return 0;
    }


};


