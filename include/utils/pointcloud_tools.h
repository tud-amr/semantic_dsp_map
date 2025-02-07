#pragma once

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <pcl/filters/voxel_grid.h>
#include "../settings/external_settings.h"
#include "../settings/settings.h"
#include "tracking_result_handler.h"
#include <sys/stat.h> 

class PointCloudTools
{
public:
    PointCloudTools(){
        // Set camera parameters
        depth_truncate_distance_max_ = g_depth_range_max;
        depth_truncate_distance_min_ = g_depth_range_min;
        camera_pos_global_frame_ << 0.f, 0.f, 0.f;
        camera_orientation_global_frame_ = Eigen::Quaterniond::Identity();
        camera_intrinsics_matrix_ << g_camera_fx, 0, g_camera_cx,
                                     0, g_camera_fy, g_camera_cy,
                                     0, 0, 1;        
    };
    ~PointCloudTools(){};

    /// @brief Read exiting point cloud models
    /// @param template_path
    void setTemplate(std::string template_path)
    {
        readExistingPointCloudTemplates(template_path);

        pcd_save_path_ = template_path + "/../pcd_normal_orientation_size";

        // Create a folder pcd_save_path_ if it doesn't exist
        struct stat buffer;   
        if(stat (pcd_save_path_.c_str(), &buffer) != 0)
            mkdir(pcd_save_path_.c_str(), 0777); // creates the directory with the specified path

    }


    /// @brief Set the camera position and orientation in the global frame
    /// @param camera_pos_global_frame
    /// @param camera_orientation_global_frame
    void updateCameraPose(const Eigen::Vector3d &camera_pos_global_frame, const Eigen::Quaterniond &camera_orientation_global_frame)
    {
        camera_pos_global_frame_ = camera_pos_global_frame;
        camera_orientation_global_frame_ = camera_orientation_global_frame;
    }


    /// @brief Generate a image of labeled point cloud based on the sigma of the points for visualization
    /// @param cloud 
    /// @param image 
    void generateLabeledPointCloudInImage(const std::vector<std::vector<LabeledPoint>> &cloud, cv::Mat &image)
    {
        image = cv::Mat::zeros(cloud.size(), cloud[0].size(), CV_8UC3);
        for(int i=0; i<cloud.size(); ++i)
        {
            for(int j=0; j<cloud[i].size(); ++j)
            {
                if(cloud[i][j].is_valid)
                {
                    // image.at<uchar>(i,j) = std::min(255, (int)(cloud[i][j].sigma * 400.f) + 50); 
                    
                    // Check if cloud[i][j].label_id exists in g_label_color_map_default
                    if(g_label_color_map_default.find(cloud[i][j].label_id) == g_label_color_map_default.end()){
                        continue;
                    }
                    image.at<cv::Vec3b>(i,j) = g_label_color_map_default[cloud[i][j].label_id];
                }
            }
        }
    }

    /// @brief Generate a semantic point cloud using the depth image and instance masks. 
    /// @param depth_value_mat Depth value matrix. Float.
    /// @param ins_seg_result Instance segmentation result.
    /// @param cloud The generated semantic point cloud.
    /// @param tracked_objects_points Generated points of tracked objects.
    /// @param track_to_label_id_map Generated track id to label id map for tracked objects.
    /// @return 
    int generateLabeledPointCloud(cv::Mat &depth_value_mat, std::vector<MaskKpts> &ins_seg_result,
                                  std::vector<std::vector<LabeledPoint>> &cloud, std::unordered_map<uint16_t, std::vector<Eigen::Vector3d>> &tracked_objects_points, 
                                  std::unordered_map<int, int> &track_to_label_id_map, float noise_model_first_order = 0.f, float noise_model_zero_order = 0.1f, bool use_global_coordinate = true)
    {
        if(depth_value_mat.empty()){
            std::cerr << "Error: depth image is empty." << std::endl;
            return -1;
        }

        
#if BOOST_MODE == 1
        // Resize the depth image if in boost mode
        // cv::resize(depth_value_mat, depth_value_mat, cv::Size(), g_image_rescale, g_image_rescale, cv::INTER_LINEAR);
        manualResize<float>(depth_value_mat, depth_value_mat, g_image_rescale);
#endif

        // Calculate transformation matrix from camera frame to global frame
        Eigen::Matrix4d camera_to_global_matrix = Eigen::Matrix4d::Identity();
        if(use_global_coordinate){
            Eigen::Matrix3d camera_orientation_matrix = camera_orientation_global_frame_.toRotationMatrix();
            camera_to_global_matrix.block<3,3>(0,0) = camera_orientation_matrix;
            camera_to_global_matrix.block<3,1>(0,3) = camera_pos_global_frame_;
        }

        // Calculate the inverse of the camera intrinsics matrix
        Eigen::Matrix3d camera_intrinsics_matrix_inverse_ = camera_intrinsics_matrix_.inverse().eval();

        // Define a track id mask image to store the track id of the pixels
        cv::Mat track_id_mask = cv::Mat::zeros(depth_value_mat.rows, depth_value_mat.cols, CV_16UC1);


        // Consider static objects first if there is a static object mask
        bool has_static_object = false;

        for(int i=0; i<ins_seg_result.size(); ++i)
        {
            if(ins_seg_result[i].label == "static")
            {
#if BOOST_MODE == 1
                // Resize the mask if in boost mode
                // cv::resize(ins_seg_result[i].mask, ins_seg_result[i].mask, cv::Size(), g_image_rescale, g_image_rescale, cv::INTER_LINEAR);
                manualResize<uchar>(ins_seg_result[i].mask, ins_seg_result[i].mask, g_image_rescale);
#endif
                auto *ins_seg_this = &ins_seg_result[i];
                // Merge the mask into track_id_mask by letting the pixels whose value is one to be the track id
                for(int j=0; j<ins_seg_this->mask.rows; ++j)
                {
                    for(int k=0; k<ins_seg_this->mask.cols; ++k)
                    {
                        int pixel_label = static_cast<int>(ins_seg_this->mask.at<uchar>(j,k)) + 1;
                        track_id_mask.at<uint16_t>(j, k) = g_label_to_instance_id_map_default[g_label_id_map_reversed[pixel_label]];
                    }
                }
                has_static_object = true;
                break; //There should be only one static object mask
            }
        }

        if(!has_static_object){
            // If there is no static object mask, set track id mask to 65535
            for(int i=0; i<depth_value_mat.rows; ++i)
            {
                for(int j=0; j<depth_value_mat.cols; ++j)
                {
                    track_id_mask.at<uint16_t>(i, j) = 65535;
                }
            }
        }


#if SETTING == 3 // ZED2
        std::unordered_map<uint16_t, double> track_id_min_x_map, track_id_max_x_map, track_id_min_y_map, track_id_max_y_map, track_id_min_z_map, track_id_max_z_map;
#endif

        if(g_consider_instance){
            // Consider movable objects
            for(int i=0; i<ins_seg_result.size(); ++i)
            {
                if(ins_seg_result[i].label != "static")
                {
                    track_to_label_id_map[ins_seg_result[i].track_id] = g_label_id_map_default[ins_seg_result[i].label];

                    auto *ins_seg_this = &ins_seg_result[i];

#if BOOST_MODE == 1
                    // Resize the mask if in boost mode
                    // cv::resize(ins_seg_this->mask, ins_seg_this->mask, cv::Size(), g_image_rescale, g_image_rescale, cv::INTER_LINEAR);
                    manualResize<uchar>(ins_seg_this->mask, ins_seg_this->mask, g_image_rescale);
#endif

#if SETTING == 3 // ZED2
                    // Get the min and max x, y, z of the instance. Global frame.
                    double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::min();
                    double min_y = std::numeric_limits<double>::max(), max_y = std::numeric_limits<double>::min();
                    double min_z = std::numeric_limits<double>::max(), max_z = std::numeric_limits<double>::min();

                    for(const auto &kpt : ins_seg_this->kpts_current)
                    {
                        if(kpt.x() < min_x) min_x = kpt.x();
                        if(kpt.x() > max_x) max_x = kpt.x();
                        if(kpt.y() < min_y) min_y = kpt.y();
                        if(kpt.y() > max_y) max_y = kpt.y();
                        if(kpt.z() < min_z) min_z = kpt.z();
                        if(kpt.z() > max_z) max_z = kpt.z();
                    }
                    const double margin = 1.0;
                    track_id_min_x_map[ins_seg_this->track_id] = min_x - margin;
                    track_id_max_x_map[ins_seg_this->track_id] = max_x + margin;
                    track_id_min_y_map[ins_seg_this->track_id] = min_y - margin;
                    track_id_max_y_map[ins_seg_this->track_id] = max_y + margin;
                    track_id_min_z_map[ins_seg_this->track_id] = min_z - margin;
                    track_id_max_z_map[ins_seg_this->track_id] = max_z + margin;
#endif

                    // Merge the mask into track_id_mask by letting the pixels whose value is one to be the track id
                    for(int j=0; j<ins_seg_this->mask.rows; ++j)
                    {
                        for(int k=0; k<ins_seg_this->mask.cols; ++k)
                        {
                            if(ins_seg_this->mask.at<uchar>(j,k) > 0)
                            {
                                track_id_mask.at<uint16_t>(j, k) = ins_seg_this->track_id;
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate the 3D point cloud
        int width = depth_value_mat.cols;
        int height = depth_value_mat.rows;
        for(int i=0; i<height; ++i) // Row
        {
            for(int j=0; j<width; ++j) // Col
            {
                LabeledPoint point;
                float depth_value = depth_value_mat.at<float>(i, j);

                // Invalid depth value
                if (std::isnan(depth_value) || depth_value < depth_truncate_distance_min_ || depth_value > depth_truncate_distance_max_) {
                    point.is_valid = false;
                    cloud[i][j] = point;
                    continue;
                }

#if SETTING == 3 // ZED2
                // Invalid track id. Sky should be excluded.
                if(track_id_mask.at<uint16_t>(i,j) == g_label_to_instance_id_map_default["Sky"]){
                    point.is_valid = false;
                    cloud[i][j] = point;
                    continue;
                }
#endif

                Eigen::Vector3d point_3d; // point_3d_free_point; //CHG
                point_3d = camera_intrinsics_matrix_inverse_ * Eigen::Vector3d(j, i, 1) * depth_value;

                // Transform the point to the global frame
                if(use_global_coordinate){
                    point_3d = camera_to_global_matrix.block<3,3>(0,0) * point_3d + camera_to_global_matrix.block<3,1>(0,3);
                }

                // Get the instance value
                uint16_t instance_id = track_id_mask.at<uint16_t>(i,j);
                
#if SETTING == 3 // ZED2
                // Check if point_3d.x, y, z is in the range of the instance. If not, treat it as background to filter out the segmentation noise
                if(g_consider_instance && instance_id < g_max_movable_object_instance_id)
                {
                    if(point_3d.x() < track_id_min_x_map[instance_id] || point_3d.x() > track_id_max_x_map[instance_id] ||
                       point_3d.y() < track_id_min_y_map[instance_id] || point_3d.y() > track_id_max_y_map[instance_id] ||
                       point_3d.z() < track_id_min_z_map[instance_id] || point_3d.z() > track_id_max_z_map[instance_id])
                    {
                        point.position.x() = point_3d.x();
                        point.position.y() = point_3d.y();
                        point.position.z() = point_3d.z();
                        point.label_id = 0; // Background
                        point.track_id = 65535; 
                        point.is_valid = true;

                        cloud[i][j] = point;
                        continue;
                    }
                }
#endif

                // Get the label id. Here the efficiency can be improved bacause the input contains the label id
                uint8_t label_id;
                if(instance_id > g_max_movable_object_instance_id)
                {
                    label_id = g_label_id_map_static[g_instance_id_to_label_map_default[instance_id]];
                }else{
                    label_id = track_to_label_id_map[instance_id];
                }
                
                // Set the sigma of the point
                if(getFlagConsiderDepthNoise()){
                    point.sigma = noise_model_zero_order + noise_model_first_order * depth_value;
                }else{
                    point.sigma = 0.1f;
                }
                
                // Add a point to tracked_objects_points if the depth value is in the map
                static const float max_depth_for_guessed_points = (1 << C_VOXEL_NUM_AXIS_N_BIGGEST) * C_VOXEL_SIZE * 0.5f; 
                if(depth_value < max_depth_for_guessed_points && instance_id <= g_max_movable_object_instance_id) // Consider only movable objects
                {
                    tracked_objects_points[instance_id].push_back(point_3d); //CHG
                }

                point.position.x() = point_3d.x();
                point.position.y() = point_3d.y();
                point.position.z() = point_3d.z();
                point.label_id = label_id;
                point.track_id = instance_id; 
                point.is_valid = true;

                cloud[i][j] = point;
            }
        }

        return 0;
    }

    /// @brief 
    /// @param source_cloud 
    /// @param source_camera_position 
    /// @param cloud_to_align 
    /// @param aligned_cloud 
    /// @return 
    float alignPointCloudToSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const Eigen::Vector3f source_camera_position, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_align, pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_cloud)
    {
        alignPointCloudToXYZCenter(cloud_to_align);
        float score = findBestTemplateTransformationPHD(source_cloud, source_camera_position, cloud_to_align, aligned_cloud, false);

        return score;
    }


    /// @brief Find the best template point cloud to align with the source point cloud
    /// @param source_cloud Source point cloud
    /// @param source_camera_position Camera position when the source point cloud is captured
    /// @param label_id Label id of the object
    /// @param aligned_cloud The aligned point cloud, which is the template point cloud after transformation
    /// @return The smallest Modified Hausdorff distance between the source point cloud and the templates
    float alignTemplatesToPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const Eigen::Vector3f source_camera_position, int label_id, pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_cloud)
    {
        // Check if the label_id exists in existing_point_cloud_templates_map_
        if(existing_point_cloud_templates_map_.find(label_id) == existing_point_cloud_templates_map_.end()){
            std::cerr << "Error: label_id " << label_id << " does not exist in existing_point_cloud_templates_map_." << std::endl;
            return -1.0f;
        }

        // Get the template point clouds
        std::vector<pcl::PointCloud<pcl::PointXYZ>> template_clouds = existing_point_cloud_templates_map_[label_id];

        // Find the best transformation between the source point cloud and each template point cloud
        // float smallest_mhd = std::numeric_limits<float>::max();
        float max_score = std::numeric_limits<int>::min();
        for(auto &template_cloud : template_clouds)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
            
            float score = findBestTemplateTransformationPHDFSConsidered(source_cloud, source_camera_position, template_cloud.makeShared(), aligned_cloud_temp);
            
            if(score > max_score){
                max_score = score;
                aligned_cloud = aligned_cloud_temp;
            }
        }

        return max_score;
    }


    /// @brief Find the x,y,z range center of the point and move the point cloud so that the center is at the origin
    /// @param cloud: Point cloud to be moved
    void alignPointCloudToXYZCenter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
    {
        // Find the x,y,z range center of the template cloud
        Eigen::Vector3f center;
        findCloudXYZCenter(cloud, center);

        // Move the template cloud so that the center is at the origin
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            cloud->points[i].x -= center[0];
            cloud->points[i].y -= center[1];
            cloud->points[i].z -= center[2];
        }
    }

    /// @brief Read existing point cloud templates from a folder and save them to existing_point_cloud_templates_map_
    /// @param existing_templates_folder_path 
    /// @return the number of added templates
    int readExistingPointCloudTemplates(std::string existing_templates_folder_path)
    {
        // Search for all the files with extension ".pcd" in the folder and its subfolders
        std::vector<std::string> pcd_file_paths;
        std::string file_extension = ".pcd";
        int added_model_count = 0;
        
        if(findFilesWithExtension(existing_templates_folder_path, pcd_file_paths, file_extension)){
            // Read the point clouds and save it to existing_point_cloud_templates_map_
            for(auto &pcd_file : pcd_file_paths)
            {
                // Read the point cloud
                pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *point_cloud) == -1) {
                    std::cerr << "Error: Cannot read file " << pcd_file << std::endl;
                    return added_model_count;
                }

                // Split the pcd_file with '/' and get the last but one string
                std::vector<std::string> strs;
                boost::split(strs, pcd_file, boost::is_any_of("/"));
                std::string label_id_str = strs[strs.size()-2];
                int label_id = std::stoi(label_id_str);
                
                std::cout << "Read point cloud model: " << pcd_file << std::endl;
                std::cout << "label_id: " << label_id << std::endl;

                // Add the point cloud to existing_point_cloud_templates_map_
                existing_point_cloud_templates_map_[label_id].push_back(*point_cloud);
                added_model_count++;
            }

        }else{
            std::cerr << "Error: No .pcd files found in the folder." << std::endl;
            return 0;
        }

        return added_model_count;
    }
    
    /// @brief The function to remove outliers from a point cloud
    /// @param cloud 
    /// @param cloud_filtered 
    void removeOutliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_filtered)
    {
       // Use a voxel filter first to downsample the source cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);
        voxel_grid.filter(*cloud_downsampled);

        std::cout << "cloud_downsampled->points.size()=" << cloud_downsampled->points.size() << std::endl;

        // Cluster the downsampled cloud and keep the largest cluster
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclidean_cluster;
        euclidean_cluster.setClusterTolerance(0.2f);
        euclidean_cluster.setMinClusterSize(100);
        euclidean_cluster.setMaxClusterSize(20000);
        euclidean_cluster.setInputCloud(cloud_downsampled);
        euclidean_cluster.extract(cluster_indices);

        // Find the largest cluster
        int largest_cluster_index = 0;
        int largest_cluster_size = 0;
        for(int i=0; i<cluster_indices.size(); ++i)
        {
            if(cluster_indices[i].indices.size() > largest_cluster_size){
                largest_cluster_size = cluster_indices[i].indices.size();
                largest_cluster_index = i;
            }
        }

        std::cout << "largest_cluster_size=" << largest_cluster_size << std::endl;

        // Extract the largest cluster as cloud_filtered
        for(int i=0; i<cluster_indices[largest_cluster_index].indices.size(); ++i)
        {
            cloud_filtered->points.push_back(cloud_downsampled->points[cluster_indices[largest_cluster_index].indices[i]]);
        }
        cloud_filtered->width = cloud_filtered->points.size();
        cloud_filtered->height = 1;

        std::cout << "cloud_filtered->points.size()=" << cloud_filtered->points.size() << std::endl;
    }


private:
    Eigen::Matrix3d camera_intrinsics_matrix_;
    double depth_truncate_distance_max_;
    double depth_truncate_distance_min_;

    Eigen::Vector3d camera_pos_global_frame_;
    Eigen::Quaterniond camera_orientation_global_frame_;

    std::string pcd_save_path_;

    std::map<int, std::vector<pcl::PointCloud<pcl::PointXYZ>>> existing_point_cloud_templates_map_; // <label_id, point_cloud>


    /// @brief Compute the Modified (mean) Hausdorff distance between two point clouds using KdTree
    /// @param source: Source point cloud
    /// @param target: Target point cloud
    /// @param kdtree: KdTree of the target point cloud
    /// @return float: Modified Hausdorff distance
    float computeMHD(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source, const pcl::PointCloud<pcl::PointXYZ>::Ptr &target, const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree) {
        float total_distance = 0.0f;
        
        for (size_t i = 0; i < source->points.size(); ++i) {
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            
            if (kdtree.nearestKSearch(source->points[i], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                total_distance += sqrt(pointNKNSquaredDistance[0]);
            }
        }
        
        return total_distance / source->points.size();
    }

    /// @brief Find the x,y,z range center of the point cloud
    /// @param cloud 
    /// @param center 
    void findCloudXYZCenter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Vector3f &center)
    {
        // Find the maximum and minimum x, y, z coordinates of the template cloud
        float max_x = std::numeric_limits<float>::min();
        float max_y = std::numeric_limits<float>::min();
        float max_z = std::numeric_limits<float>::min();
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();

        for (size_t i = 0; i < cloud->points.size(); ++i) {
            if (cloud->points[i].x > max_x) {
                max_x = cloud->points[i].x;
            }
            if (cloud->points[i].y > max_y) {
                max_y = cloud->points[i].y;
            }
            if (cloud->points[i].z > max_z) {
                max_z = cloud->points[i].z;
            }
            if (cloud->points[i].x < min_x) {
                min_x = cloud->points[i].x;
            }
            if (cloud->points[i].y < min_y) {
                min_y = cloud->points[i].y;
            }
            if (cloud->points[i].z < min_z) {
                min_z = cloud->points[i].z;
            }
        }

        // Calculate the center of the template cloud
        center[0] = (max_x + min_x) / 2.0f;
        center[1] = (max_y + min_y) / 2.0f;
        center[2] = (max_z + min_z) / 2.0f;
    }

    /// @brief Find the mass center of the point cloud
    /// @param cloud 
    /// @param center 
    void findCloudMassCenter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Vector3f &center)
    {
        // Find the maximum and minimum x, y, z coordinates of the template cloud
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;

        for (size_t i = 0; i < cloud->points.size(); ++i) {
            sum_x += cloud->points[i].x;
            sum_y += cloud->points[i].y;
            sum_z += cloud->points[i].z;
        }

        // Calculate the center of the template cloud
        center[0] = sum_x / cloud->points.size();
        center[1] = sum_y / cloud->points.size();
        center[2] = sum_z / cloud->points.size();
    }



    /// @brief Find the best transformation between a source point cloud and a template point cloud using a simple sampling method.
    /// @param source_cloud: source point cloud
    /// @param source_camera_position: the camera position when the source point cloud is captured
    /// @param aligned_cloud: the aligned point cloud, which is the template point cloud after transformation 
    /// @return float: The smallest Modified Hausdorff distance between the source point cloud and the aligned point cloud
    float findBestTemplateTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const Eigen::Vector3f &source_camera_position, const pcl::PointCloud<pcl::PointXYZ>::Ptr &template_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_cloud)
    {
        //Start a high-resolution timer
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        pcl::PointXYZ camera_position_point;
        camera_position_point.x = source_camera_position[0];
        camera_position_point.y = source_camera_position[1];
        camera_position_point.z = source_camera_position[2];

        // Find the point cloest to the camera position in source_cloud
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(source_cloud);
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        kdtree.nearestKSearch(camera_position_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance);

        pcl::PointXYZ closest_point = source_cloud->points[pointIdxNKNSearch[0]];
        // std::cout << "Cloest point: " << closest_point << std::endl;

        // Make a new point cloud by moving the source cloud so that the cloest point is at the origin
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_moved(new pcl::PointCloud<pcl::PointXYZ>);
        
        pcl::copyPointCloud(*source_cloud, *source_cloud_moved);
        for (size_t i = 0; i < source_cloud_moved->points.size(); ++i) {
            source_cloud_moved->points[i].x -= closest_point.x;
            source_cloud_moved->points[i].y -= closest_point.y;
            source_cloud_moved->points[i].z -= closest_point.z;
        }
        /// TODO: If source_cloud contains too many points, we can downsample it here

        
        Eigen::Vector3f camera_position_to_closest_point_vec, camera_position_to_closest_point_vec_normalized;
        camera_position_to_closest_point_vec[0] = closest_point.x - source_camera_position[0];
        camera_position_to_closest_point_vec[1] = closest_point.y - source_camera_position[1];
        camera_position_to_closest_point_vec[2] = closest_point.z - source_camera_position[2];

        // Normalize the vector
        camera_position_to_closest_point_vec_normalized = camera_position_to_closest_point_vec;
        camera_position_to_closest_point_vec_normalized.normalize();

        // Initialize the kdtree with the new source_cloud_moved
        kdtree.setInputCloud(source_cloud_moved);

        // Rotate the template cloud about the y-axis and find the best angle
        float best_angle = 0.0f;
        Eigen::Vector3f best_template_cloud_move_vec;
        float smallest_mhd = std::numeric_limits<float>::max();
        
        for (float angle = 0.0f; angle <= 350.0f; angle += 10.0f) {
            // Move the template cloud by camera_position_to_closest_point_vec * 10.0f. We assume the object's half size is no more than 10.0f
            Eigen::Vector3f template_cloud_move_vec = camera_position_to_closest_point_vec_normalized * 10.0f;
            
            // Rotate template_cloud about the y-axis and then move by template_cloud_move_vec
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.rotate(Eigen::AngleAxisf(angle * M_PI / 180.0f, Eigen::Vector3f::UnitY()));
            transform.translation() << template_cloud_move_vec[0], template_cloud_move_vec[1], template_cloud_move_vec[2];
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*template_cloud, *transformed_cloud, transform);

            // Find the point cloest to origin (0,0,0) in transformed_cloud
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
            kdtree2.setInputCloud(transformed_cloud);
            pcl::PointXYZ origin_point;
            origin_point.x = 0.0f;
            origin_point.y = 0.0f;
            origin_point.z = 0.0f;
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            kdtree2.nearestKSearch(origin_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance);

            pcl::PointXYZ template_closest_point = transformed_cloud->points[pointIdxNKNSearch[0]];
            // std::cout << "Template cloest point: " << template_closest_point << std::endl;

            // Move the transformed_cloud so that the template_closest_point is at the origin
            for (size_t i = 0; i < transformed_cloud->points.size(); ++i) {
                transformed_cloud->points[i].x -= template_closest_point.x;
                transformed_cloud->points[i].y -= template_closest_point.y;
                transformed_cloud->points[i].z -= template_closest_point.z;
            }

            float mhd = computeMHD(transformed_cloud, source_cloud, kdtree);
            if (mhd < smallest_mhd) {
                smallest_mhd = mhd;
                best_angle = angle;
                Eigen::Vector3f template_second_move_vec(-template_closest_point.x, -template_closest_point.y, -template_closest_point.z);
                best_template_cloud_move_vec = template_cloud_move_vec + template_second_move_vec + camera_position_to_closest_point_vec + source_camera_position;
            }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
#if VERBOSE_MODE == 1
        std::cout << "Matching time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds" << std::endl;
        // std::cout << "Best angle: " << best_angle << " degrees with Modified Hausdorff Distance: " << smallest_mhd << std::endl;
        // std::cout << "Best template cloud move vector: " << best_template_cloud_move_vec << std::endl;
#endif
        // Rotate template_cloud by best_angle and move by best_template_cloud_move_vec
        Eigen::Affine3f best_transform = Eigen::Affine3f::Identity();
        best_transform.rotate(Eigen::AngleAxisf(best_angle * M_PI / 180.0f, Eigen::Vector3f::UnitY()));
        best_transform.translation() << best_template_cloud_move_vec[0], best_template_cloud_move_vec[1], best_template_cloud_move_vec[2];
        pcl::transformPointCloud(*template_cloud, *aligned_cloud, best_transform);

        return smallest_mhd;
    }



    /// @brief Find the best transformation between a source point cloud and a template point cloud using a voxel-based method based on the PHD.
    /// @param source_cloud 
    /// @param source_camera_position 
    /// @param template_cloud 
    /// @param aligned_cloud 
    /// @return A score that indicates the quality of the alignment. 0 - 1. 1 means the best.
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_filtered;
    float findBestTemplateTransformationPHD(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const Eigen::Vector3f &source_camera_position, const pcl::PointCloud<pcl::PointXYZ>::Ptr &template_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_cloud, bool filter_source_cloud = false)
    {
        //Start a high-resolution timer
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // Filter the source_cloud with a voxel grid filter and cluster the points with a euclidean cluster extractor. Keep only the largest cluster
        source_cloud_filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if(filter_source_cloud){
            removeOutliers(source_cloud, source_cloud_filtered);
        }else{
            source_cloud_filtered = source_cloud;
        }

        // Set camera position
        pcl::PointXYZ camera_position_point;
        camera_position_point.x = source_camera_position[0];
        camera_position_point.y = source_camera_position[1];
        camera_position_point.z = source_camera_position[2];

        // Find the point cloest to the camera position and the maximum, minimum , x, y, z coordinates in source_cloud
        float max_x = std::numeric_limits<float>::min();
        float max_y = std::numeric_limits<float>::min();
        float max_z = std::numeric_limits<float>::min();
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float closest_point_dist_squre = std::numeric_limits<float>::max();
        pcl::PointXYZ closest_point;

        for(const auto &pt : source_cloud_filtered->points) {
            if (pt.x > max_x) {max_x = pt.x;}
            if (pt.y > max_y) {max_y = pt.y;}
            if (pt.z > max_z) {max_z = pt.z;}
            if (pt.x < min_x) {min_x = pt.x;}
            if (pt.y < min_y) {min_y = pt.y;}
            if (pt.z < min_z) {min_z = pt.z;}
            float point_dist_squre = (pt.x-source_camera_position[0])* (pt.x-source_camera_position[0]) + (pt.y-source_camera_position[1])* (pt.y-source_camera_position[1]) + (pt.z-source_camera_position[2])* (pt.z-source_camera_position[2]);
            if (point_dist_squre < closest_point_dist_squre) {
                closest_point_dist_squre = point_dist_squre;
                closest_point = pt;
            }
        }

        Eigen::Vector3f camera_position_to_closest_point_vec, camera_position_to_closest_point_vec_normalized;
        camera_position_to_closest_point_vec[0] = closest_point.x - source_camera_position[0];
        camera_position_to_closest_point_vec[1] = closest_point.y - source_camera_position[1];
        camera_position_to_closest_point_vec[2] = closest_point.z - source_camera_position[2];

        // Normalize the vector
        camera_position_to_closest_point_vec_normalized = camera_position_to_closest_point_vec;
        camera_position_to_closest_point_vec_normalized.normalize();

        // Save the source_cloud_filtered in voxel form in a std::vector
        float voxel_size = 0.2f;
        int voxel_num_x = (max_x - min_x) / voxel_size + 1;
        int voxel_num_y = (max_y - min_y) / voxel_size + 1;
        int voxel_num_z = (max_z - min_z) / voxel_size + 1;
        std::vector<std::vector<std::vector<bool>>> voxel_grid(voxel_num_x, std::vector<std::vector<bool>>(voxel_num_y, std::vector<bool>(voxel_num_z, false)));


        for (size_t i = 0; i < source_cloud_filtered->points.size(); ++i) {
            int x_idx = (source_cloud_filtered->points[i].x - min_x) / voxel_size;
            int y_idx = (source_cloud_filtered->points[i].y - min_y) / voxel_size;
            int z_idx = (source_cloud_filtered->points[i].z - min_z) / voxel_size;
            voxel_grid[x_idx][y_idx][z_idx] = true;
        }

        // Count the number of occupied voxels separately because one voxel can have multiple points
        int occupied_voxel_num = 0;
        for (size_t i = 0; i < voxel_num_x; ++i) {
            for (size_t j = 0; j < voxel_num_y; ++j) {
                for (size_t k = 0; k < voxel_num_z; ++k) {
                    if (voxel_grid[i][j][k]) {
                        occupied_voxel_num++;
                    }
                }
            }
        }
        
        // Rotate the template cloud about the y-axis and find the best angle
        float best_angle = 0.0f;
        Eigen::Vector3f best_template_cloud_move_vec;
        int biggest_intersection_num = 0;
        
        for (float angle = 0.0f; angle <= 350.0f; angle += 10.0f) {
            // Move the template cloud by camera_position_to_closest_point_vec * 10.0f. We assume the object's half size is no more than 10.0f
            Eigen::Vector3f template_cloud_move_vec = camera_position_to_closest_point_vec_normalized * 10.0f;
            
            // Rotate template_cloud about the y-axis and then move by template_cloud_move_vec
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.rotate(Eigen::AngleAxisf(angle * M_PI / 180.0f, Eigen::Vector3f::UnitY()));
            transform.translation() << template_cloud_move_vec[0], template_cloud_move_vec[1], template_cloud_move_vec[2];
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*template_cloud, *transformed_cloud, transform);

            // Find the point cloest to origin (0,0,0) in transformed_cloud
            float closest_point_dist_squre = std::numeric_limits<float>::max();
            pcl::PointXYZ template_closest_point;
            for (size_t i = 0; i < transformed_cloud->points.size(); ++i) {
                float point_dist_squre = transformed_cloud->points[i].x * transformed_cloud->points[i].x + transformed_cloud->points[i].y * transformed_cloud->points[i].y + transformed_cloud->points[i].z * transformed_cloud->points[i].z;
                if (point_dist_squre < closest_point_dist_squre) {
                    closest_point_dist_squre = point_dist_squre;
                    template_closest_point = transformed_cloud->points[i];
                }
            }

            Eigen::Vector3f correct_transition_vec;
            correct_transition_vec[0] = closest_point.x - template_closest_point.x;
            correct_transition_vec[1] = closest_point.y - template_closest_point.y;
            correct_transition_vec[2] = closest_point.z - template_closest_point.z;

            
            // Check x +- voxel_size, y +- voxel_size
            const int noise_range = 1;
            for(int x=-noise_range; x<=noise_range; x++)
            {
                for(int y=-noise_range; y<=noise_range; y++)
                {
                    Eigen::Vector3f correct_transition_vec_temp = correct_transition_vec;
                    correct_transition_vec_temp[0] += x * voxel_size;
                    correct_transition_vec_temp[1] += y * voxel_size;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::copyPointCloud(*transformed_cloud, *transformed_cloud_copy);

                    // Move the transformed_cloud so that the template_closest_point is at the closest_point of the source cloud
                    for (size_t i = 0; i < transformed_cloud_copy->points.size(); ++i) {
                        transformed_cloud_copy->points[i].x += correct_transition_vec_temp[0];
                        transformed_cloud_copy->points[i].y += correct_transition_vec_temp[1];
                        transformed_cloud_copy->points[i].z += correct_transition_vec_temp[2];
                    }

                    // Add a timer
                    // std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
                    
                    // Calculate the intersection number
                    std::vector<std::vector<std::vector<bool>>> voxel_calculated(voxel_num_x, std::vector<std::vector<bool>>(voxel_num_y, std::vector<bool>(voxel_num_z, false)));

                    int intersection_num = 0;
                    for (size_t i = 0; i < transformed_cloud_copy->points.size(); ++i) {
                        int x_idx = (transformed_cloud_copy->points[i].x - min_x) / voxel_size;
                        int y_idx = (transformed_cloud_copy->points[i].y - min_y) / voxel_size;
                        int z_idx = (transformed_cloud_copy->points[i].z - min_z) / voxel_size;

                        if(x_idx < 0 || x_idx >= voxel_num_x || y_idx < 0 || y_idx >= voxel_num_y || z_idx < 0 || z_idx >= voxel_num_z) {
                            continue;
                        }

                        if (voxel_grid[x_idx][y_idx][z_idx] && !voxel_calculated[x_idx][y_idx][z_idx]) {
                            intersection_num++;
                            voxel_calculated[x_idx][y_idx][z_idx] = true;
                        }
                    }
                    
                    if (intersection_num > biggest_intersection_num) {
                        biggest_intersection_num = intersection_num;
                        best_angle = angle;
                        best_template_cloud_move_vec = template_cloud_move_vec + correct_transition_vec_temp;
                    }
                }
            }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
#if VERBOSE_MODE == 1
        std::cout << "Align Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds" << std::endl;
        // std::cout << "Best angle: " << best_angle << " degrees with biggest_intersection_num: " << biggest_intersection_num << std::endl;
        // std::cout << "Best template cloud move vector: " << best_template_cloud_move_vec << std::endl;
#endif


        // Rotate template_cloud by best_angle and move by best_template_cloud_move_vec
        Eigen::Affine3f best_transform = Eigen::Affine3f::Identity();
        best_transform.rotate(Eigen::AngleAxisf(best_angle * M_PI / 180.0f, Eigen::Vector3f::UnitY()));
        best_transform.translation() << best_template_cloud_move_vec[0], best_template_cloud_move_vec[1], best_template_cloud_move_vec[2];
        pcl::transformPointCloud(*template_cloud, *aligned_cloud, best_transform);

        // Use the percentage of the intersection number to the occupied voxel number as the score
        return (float)biggest_intersection_num / occupied_voxel_num;
    }


    /// @brief Find the best transformation between a source point cloud and a template point cloud using a voxel-based method based on the PHD.
    /// @param source_cloud 
    /// @param source_camera_position 
    /// @param template_cloud 
    /// @param aligned_cloud 
    /// @return A score that indicates the quality of the alignment. 0 - 1. 1 means the best.
    float findBestTemplateTransformationPHDFSConsidered(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, const Eigen::Vector3f &source_camera_position, const pcl::PointCloud<pcl::PointXYZ>::Ptr &template_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_cloud, bool filter_source_cloud = false)
    {
        //Start a high-resolution timer
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        // Filter the source_cloud with a voxel grid filter and cluster the points with a euclidean cluster extractor. Keep only the largest cluster
        source_cloud_filtered = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if(filter_source_cloud){
            removeOutliers(source_cloud, source_cloud_filtered);
        }else{
            source_cloud_filtered = source_cloud;
        }

        // Set camera position
        pcl::PointXYZ camera_position_point;
        camera_position_point.x = source_camera_position[0];
        camera_position_point.y = source_camera_position[1];
        camera_position_point.z = source_camera_position[2];

        // Find the point cloest to the camera position and the maximum, minimum , x, y, z coordinates in source_cloud
        float max_x = std::numeric_limits<float>::min();
        float max_y = std::numeric_limits<float>::min();
        float max_z = std::numeric_limits<float>::min();
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float closest_point_dist_squre = std::numeric_limits<float>::max();
        pcl::PointXYZ closest_point;

        for(const auto &pt : source_cloud_filtered->points) {
            if (pt.x > max_x) {max_x = pt.x;}
            if (pt.y > max_y) {max_y = pt.y;}
            if (pt.z > max_z) {max_z = pt.z;}
            if (pt.x < min_x) {min_x = pt.x;}
            if (pt.y < min_y) {min_y = pt.y;}
            if (pt.z < min_z) {min_z = pt.z;}
            float point_dist_squre = (pt.x-source_camera_position[0])* (pt.x-source_camera_position[0]) + (pt.y-source_camera_position[1])* (pt.y-source_camera_position[1]) + (pt.z-source_camera_position[2])* (pt.z-source_camera_position[2]);
            if (point_dist_squre < closest_point_dist_squre) {
                closest_point_dist_squre = point_dist_squre;
                closest_point = pt;
            }
        }

        Eigen::Vector3f camera_position_to_closest_point_vec, camera_position_to_closest_point_vec_normalized;
        camera_position_to_closest_point_vec[0] = closest_point.x - source_camera_position[0];
        camera_position_to_closest_point_vec[1] = closest_point.y - source_camera_position[1];
        camera_position_to_closest_point_vec[2] = closest_point.z - source_camera_position[2];

        // Normalize the vector
        camera_position_to_closest_point_vec_normalized = camera_position_to_closest_point_vec;
        camera_position_to_closest_point_vec_normalized.normalize();

        // Save the source_cloud_filtered in voxel form in a std::vector
        float voxel_size = 0.2f;
        int voxel_num_x = (max_x - min_x) / voxel_size + 1;
        int voxel_num_y = (max_y - min_y) / voxel_size + 1;
        int voxel_num_z = (max_z - min_z) / voxel_size + 1;

        /// TODO: Reduce the expand num
        const int expand_num = 10;
        int expanded_voxel_num_x = voxel_num_x + expand_num * 2;
        int expanded_voxel_num_y = voxel_num_y + expand_num * 2;
        int expanded_voxel_num_z = voxel_num_z + expand_num * 2;

        min_x -= expand_num * voxel_size;
        min_y -= expand_num * voxel_size;
        min_z -= expand_num * voxel_size;

        
        // Make a vector to store the state of each voxel. 0: Occluded/Unknown, 1: occupied, 2: Free
        std::vector<std::vector<std::vector<int>>> voxel_grid(expanded_voxel_num_x, std::vector<std::vector<int>>(expanded_voxel_num_y, std::vector<int>(expanded_voxel_num_z, 0)));    

        for (size_t i = 0; i < source_cloud_filtered->points.size(); ++i) {
            int x_idx = (source_cloud_filtered->points[i].x - min_x) / voxel_size;
            int y_idx = (source_cloud_filtered->points[i].y - min_y) / voxel_size;
            int z_idx = (source_cloud_filtered->points[i].z - min_z) / voxel_size;
            voxel_grid[x_idx][y_idx][z_idx] = 1; // occupied

            // Now do ray casting to mark the occluded voxels
            Eigen::Vector3f voxel_point(source_cloud_filtered->points[i].x, source_cloud_filtered->points[i].y, source_cloud_filtered->points[i].z);
            Eigen::Vector3f ray_direction = voxel_point - source_camera_position;
            float step = voxel_size;
            float max_length = ray_direction.norm();

            if(max_length < 1e-6){
                continue;
            }

            ray_direction = ray_direction / max_length;

            float current_length = 0.0f;
            while(current_length < max_length){
                voxel_point = source_camera_position + current_length * ray_direction;
                int x_idx = (voxel_point[0] - min_x) / voxel_size;
                int y_idx = (voxel_point[1] - min_y) / voxel_size;
                int z_idx = (voxel_point[2] - min_z) / voxel_size;

                if(x_idx < 0 || x_idx >= expanded_voxel_num_x || y_idx < 0 || y_idx >= expanded_voxel_num_y || z_idx < 0 || z_idx >= expanded_voxel_num_z){
                    current_length += step;
                    continue; // Not in range. But still can be later.
                }

                if(voxel_grid[x_idx][y_idx][z_idx] == 1){ // If the voxel is occupied, stop the ray
                    break;
                }

                voxel_grid[x_idx][y_idx][z_idx] = 2; // Free
                current_length += step;
            }
        }

        // Count the number of occupied voxels separately because one voxel can have multiple points
        int occupied_voxel_num = 0;
        for (size_t i = 0; i < expanded_voxel_num_x; ++i) {
            for (size_t j = 0; j < expanded_voxel_num_y; ++j) {
                for (size_t k = 0; k < expanded_voxel_num_z; ++k) {
                    if (voxel_grid[i][j][k] == 1) {
                        occupied_voxel_num++;
                    }
                }
            }
        }
        

        // Rotate the template cloud about the y-axis and find the best angle
        float best_angle = 0.0f;
        Eigen::Vector3f best_template_cloud_move_vec;
        int biggest_intersection_num = 0;
        
        for (float angle = 0.0f; angle <= 350.0f; angle += 10.0f) {
            // Move the template cloud by camera_position_to_closest_point_vec * 10.0f. We assume the object's half size is no more than 10.0f
            Eigen::Vector3f template_cloud_move_vec = camera_position_to_closest_point_vec_normalized * 10.0f;
            
            // Rotate template_cloud about the y-axis and then move by template_cloud_move_vec
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.rotate(Eigen::AngleAxisf(angle * M_PI / 180.0f, Eigen::Vector3f::UnitY()));
            transform.translation() << template_cloud_move_vec[0], template_cloud_move_vec[1], template_cloud_move_vec[2];
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*template_cloud, *transformed_cloud, transform);

            // Find the point cloest to origin (0,0,0) in transformed_cloud
            float closest_point_dist_squre = std::numeric_limits<float>::max();
            pcl::PointXYZ template_closest_point;
            for (size_t i = 0; i < transformed_cloud->points.size(); ++i) {
                float point_dist_squre = transformed_cloud->points[i].x * transformed_cloud->points[i].x + transformed_cloud->points[i].y * transformed_cloud->points[i].y + transformed_cloud->points[i].z * transformed_cloud->points[i].z;
                if (point_dist_squre < closest_point_dist_squre) {
                    closest_point_dist_squre = point_dist_squre;
                    template_closest_point = transformed_cloud->points[i];
                }
            }

            Eigen::Vector3f correct_transition_vec;
            correct_transition_vec[0] = closest_point.x - template_closest_point.x;
            correct_transition_vec[1] = closest_point.y - template_closest_point.y;
            correct_transition_vec[2] = closest_point.z - template_closest_point.z;
            
            // Check x +- voxel_size, y +- voxel_size
            const int noise_range = 1;
            for(int x=-noise_range; x<=noise_range; x++)
            {
                for(int y=-noise_range; y<=noise_range; y++)
                {
                    Eigen::Vector3f correct_transition_vec_temp = correct_transition_vec;
                    correct_transition_vec_temp[0] += x * voxel_size;
                    correct_transition_vec_temp[1] += y * voxel_size;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::copyPointCloud(*transformed_cloud, *transformed_cloud_copy);

                    // Move the transformed_cloud so that the template_closest_point is at the closest_point of the source cloud
                    for (size_t i = 0; i < transformed_cloud_copy->points.size(); ++i) {
                        transformed_cloud_copy->points[i].x += correct_transition_vec_temp[0];
                        transformed_cloud_copy->points[i].y += correct_transition_vec_temp[1];
                        transformed_cloud_copy->points[i].z += correct_transition_vec_temp[2];
                    }

                    // Add a timer
                    // std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
                    
                    // Calculate the intersection number. Make a vector to store if the voxel is calculated to avoid double counting.
                    std::vector<std::vector<std::vector<bool>>> voxel_calculated(expanded_voxel_num_x, std::vector<std::vector<bool>>(expanded_voxel_num_y, std::vector<bool>(expanded_voxel_num_z, false)));

                    int intersection_num = 0;
                    for (size_t i = 0; i < transformed_cloud_copy->points.size(); ++i) {
                        int x_idx = (transformed_cloud_copy->points[i].x - min_x) / voxel_size;
                        int y_idx = (transformed_cloud_copy->points[i].y - min_y) / voxel_size;
                        int z_idx = (transformed_cloud_copy->points[i].z - min_z) / voxel_size;

                        if(x_idx < 0 || x_idx >= expanded_voxel_num_x || y_idx < 0 || y_idx >= expanded_voxel_num_y || z_idx < 0 || z_idx >= expanded_voxel_num_z) {
                            continue;
                        }

                        if (voxel_grid[x_idx][y_idx][z_idx]==1 && !voxel_calculated[x_idx][y_idx][z_idx]) { // Source Occupied
                            intersection_num++;
                            voxel_calculated[x_idx][y_idx][z_idx] = true; // Mark as calculated
                        }else if(voxel_grid[x_idx][y_idx][z_idx]==2 && !voxel_calculated[x_idx][y_idx][z_idx]){ // Source free
                            intersection_num--;
                            voxel_calculated[x_idx][y_idx][z_idx] = true; // Mark as calculated
                        }
                    }
                    
                    // std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
                    // std::cout << "intersection Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << std::endl;

                    if (intersection_num > biggest_intersection_num) {
                        biggest_intersection_num = intersection_num;
                        best_angle = angle;
                        best_template_cloud_move_vec = template_cloud_move_vec + correct_transition_vec_temp;
                    }
                }
            }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        // Rotate template_cloud by best_angle and move by best_template_cloud_move_vec
        Eigen::Affine3f best_transform = Eigen::Affine3f::Identity();
        best_transform.rotate(Eigen::AngleAxisf(best_angle * M_PI / 180.0f, Eigen::Vector3f::UnitY()));
        best_transform.translation() << best_template_cloud_move_vec[0], best_template_cloud_move_vec[1], best_template_cloud_move_vec[2];
        pcl::transformPointCloud(*template_cloud, *aligned_cloud, best_transform);

        // Use the percentage of the intersection number to the occupied voxel number as the score
        return (float)biggest_intersection_num / occupied_voxel_num;
    }


    /// @brief Resize a cv::Mat manually because cv::resize is not working on Jetson Boards.
    /// @param src Source cv::Mat
    /// @param dst Destination cv::Mat
    /// @param scale Scale factor
    template<typename T>
    void manualResize(const cv::Mat& src, cv::Mat& dst, float scale) {
        // Create temporary Mat if src and dst are the same
        cv::Mat temp;
        const cv::Mat* src_ptr = &src;
        
        if (&src == &dst) {
            temp = src.clone();
            src_ptr = &temp;
        }
        
        int new_rows = static_cast<int>(src.rows * scale);
        int new_cols = static_cast<int>(src.cols * scale);
        dst = cv::Mat(new_rows, new_cols, src.type());

        float scale_inv = 1.f / scale;

        for (int i = 0; i < new_rows; i++) {
            for (int j = 0; j < new_cols; j++) {
                int src_i = static_cast<int>(i * scale_inv);
                int src_j = static_cast<int>(j * scale_inv);
                
                // Ensure we don't go out of bounds
                src_i = std::min(src_i, src_ptr->rows - 1);
                src_j = std::min(src_j, src_ptr->cols - 1);
                
                dst.at<T>(i, j) = src_ptr->at<T>(src_i, src_j);
            }
        }
    }

};

