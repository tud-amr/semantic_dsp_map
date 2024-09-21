/**
 * @file basic_algorithms.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This file contains some basic math or datastructure algorithms used in the map
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "data_base.h"
#include <random>
#include <math.h>
#include <boost/filesystem.hpp>

#ifndef M_PIf32
#define M_PIf32                3.14159265358979323846        /* pi */
#endif

#ifndef M_PI_2f32
#define M_PI_2f32                1.57079632679489661923        /* pi/2 */
#endif


/// @brief TODO: Rewrite this function with a better implementation
/// @param forget_count 
/// @param stability_factor 
/// @return 
inline float getForgettingFactor(const int &forget_count, float stability_factor=2.f, int max_forget_count=3){
    static float forgetting_function[5];
    static bool initialized = false;
    if(!initialized){
        for(int i = 0; i < 5; ++i){
            forgetting_function[i] = pow(2.5, -i/stability_factor);
            // std::cout << "forgetting_function[" << i << "]=" << forgetting_function[i] << std::endl;
        }
        initialized = true;
    }

    if(forget_count < max_forget_count){
        return forgetting_function[forget_count];
    }else{
        return 0.f;
    }
}



/// @brief Compute the transformation matrix that aligns the two sets of points
/// @param P Last set of points
/// @param Q New set of points
/// @return Transformation matrix
Eigen::Matrix4d estimateTransformation(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q) {
    // Ensure the matrices P and Q are of size 3xN
    assert(P.rows() == 3 && Q.rows() == 3);
    assert(P.cols() == Q.cols());
    
    int N = P.cols();

    // Compute centroids of P and Q
    Eigen::Vector3d centroid_P = P.rowwise().mean();
    Eigen::Vector3d centroid_Q = Q.rowwise().mean();

    // Center the data
    Eigen::MatrixXd P_centered = P.colwise() - centroid_P;
    Eigen::MatrixXd Q_centered = Q.colwise() - centroid_Q;

    // Compute cross-covariance matrix H
    Eigen::MatrixXd H = P_centered * Q_centered.transpose();

    // Compute the Singular Value Decomposition of H
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    // Compute the rotation matrix R
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0) {
        V.col(2) *= -1;  // Invert the third column of V
        R = V * U.transpose();
    }

    // Compute the translation vector t
    Eigen::Vector3d t = centroid_Q - R * centroid_P;

    // Construct the transformation matrix T
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;

    return T;
}

/// @brief Compute the transformation matrix that aligns the two sets of points (using RANSAC to reject outliers)
/// @param P Last set of points
/// @param Q New set of points
/// @param result_t_matrix The transformation matrix
/// @param inlier_indices The indices of inliers
/// @param max_iterations The maximum number of iterations
/// @param threshold The threshold to determine whether a point is an inlier
/// @param recomupte_with_inliers Whether to recompute the transformation matrix using only the inliers
/// @return The MSE of inliers
double estimateTransformationRANSAC(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q, Eigen::Matrix4d &result_t_matrix, std::vector<int> &inlier_indices, int max_iterations=100, double threshold=0.5, bool recomupte_with_inliers=false) {
    int N = P.cols();
    int max_inliers = -1;
    Eigen::Matrix4d best_T;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, N - 1);

    for (int i = 0; i < max_iterations; i++) {
        std::vector<int> indices;
        /// TODO: Improve the efficiency of this part. Randomly sampling can be unnecessary if the number of points is small.
        while (indices.size() < 3) {  // sample 3 random indices
            int random_index = dist(rng);
            if (std::find(indices.begin(), indices.end(), random_index) == indices.end()) {
                indices.push_back(random_index);
            }
        }

        Eigen::MatrixXd P_sample(3, 3);
        Eigen::MatrixXd Q_sample(3, 3);
        for (int j = 0; j < 3; j++) {
            P_sample.col(j) = P.col(indices[j]);
            Q_sample.col(j) = Q.col(indices[j]);
        }

        Eigen::Matrix4d T = estimateTransformation(P_sample, Q_sample);  // Use the previous SVD-based function
        Eigen::MatrixXd P_transformed = (T.block<3, 3>(0, 0) * P).colwise() + T.block<3, 1>(0, 3);

        int inliers = 0;
        std::vector<int> inlier_indices_temp;
        for (int j = 0; j < N; j++) {
            double error = (P_transformed.col(j) - Q.col(j)).norm();
            if (error < threshold) {
                inlier_indices_temp.push_back(j);
                inliers++;
            }
        }

        if (inliers > max_inliers) {
            max_inliers = inliers;
            best_T = T;
            inlier_indices = inlier_indices_temp;
            // std::cout << "RANSAC iteration " << i << ": " << max_inliers << " inliers" << std::endl;
        }

        // Break if over 90% points are inliers
        if (max_inliers > 0.9 * N) {
            break;
        }
    }

    // Recompute transformation using all inlier points
    if(recomupte_with_inliers && inlier_indices.size() >= 3){
        
        Eigen::MatrixXd P_inliers(3, inlier_indices.size());
        Eigen::MatrixXd Q_inliers(3, inlier_indices.size());
        for (size_t j = 0; j < inlier_indices.size(); j++) {
            P_inliers.col(j) = P.col(inlier_indices[j]);
            Q_inliers.col(j) = Q.col(inlier_indices[j]);
        }

        // Recompute transformation using only the inliers
        Eigen::Matrix4d refined_T = estimateTransformation(P_inliers, Q_inliers);

        result_t_matrix = refined_T;
    }else{
        result_t_matrix = best_T;
    }

    // Calculate and print the MSE of inlines and all points
    Eigen::MatrixXd P_transformed_refined = (result_t_matrix.block<3, 3>(0, 0) * P).colwise() + result_t_matrix.block<3, 1>(0, 3);
    double total_error = 0;
    double total_error_inliers = 0;
    for (int j = 0; j < N; j++) {
        double error = (P_transformed_refined.col(j) - Q.col(j)).squaredNorm();
        total_error += error;
        if (std::find(inlier_indices.begin(), inlier_indices.end(), j) != inlier_indices.end()) {
            total_error_inliers += error;
        }
    }

#if VERBOSE_MODE == 1
    std::cout << "Inliers: " << inlier_indices.size() << " / " << N << std::endl; 
    std::cout << "MSE of all points: " << total_error / N << std::endl;
    std::cout << "MSE of inliers: " << total_error_inliers / inlier_indices.size() << std::endl;
#endif

    return total_error_inliers / inlier_indices.size();
}


/// @brief Compute the transformation matrix that aligns the two sets of points. Consider only translation.
/// @param P Last set of points
/// @param Q Current set of points
/// @param result_t_matrix Result transformation matrix. The rotation part is identity.
/// @return 
bool estimateTransformationNoRotation(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Q, Eigen::Matrix4d &result_t_matrix)
{
    if(P.cols() != Q.cols() || P.cols() == 0){
        std::cerr << "Error: The number of points in P and Q are not equal." << std::endl;
        return false;
    }

    Eigen::Vector3d total_movement = Eigen::Vector3d::Zero();
    for(int i = 0; i < P.cols(); ++i){
        total_movement += Q.col(i) - P.col(i);
    }

    Eigen::Vector3d average_movement = total_movement / P.cols();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 1>(0, 3) = average_movement;

    result_t_matrix = T;
    return true;
}


/// @brief Generate a random transformation matrix for testing
/// @return transformation matrix
Eigen::Matrix4d generateRandomTransformation()
{
    // Generate a random rotation matrix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    Eigen::Matrix3d R;
    R << d(gen), d(gen), d(gen),
         d(gen), d(gen), d(gen),
         d(gen), d(gen), d(gen);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();

    // Generate a random translation vector
    Eigen::Vector3d t(d(gen), d(gen), d(gen));

    // Construct the transformation matrix from R and t
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;

    return T;
}

/// @brief  Test the transformation computation
/// @return 
int transformationComputeTest(){
    // Generate random 3D points for matrix P
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    int n_points = 2;

    Eigen::MatrixXd P(3, n_points);
    for (int i = 0; i < n_points; ++i) {
        P.col(i) << d(gen), d(gen), d(gen);
    }

    // Generate a random transformation matrix
    Eigen::Matrix4d T_ori = generateRandomTransformation();

    // Calculate matrix Q
    Eigen::MatrixXd Q = T_ori.block<3, 3>(0, 0) * P + T_ori.block<3, 1>(0, 3) * Eigen::MatrixXd::Ones(1, P.cols());

    // Add noise to P and Q
    double noise_level = 0.1;
    P += Eigen::MatrixXd::Random(3, n_points) * noise_level;
    Q += Eigen::MatrixXd::Random(3, n_points) * noise_level;

    // Compute the transformation matrix
    Eigen::Matrix4d T = estimateTransformation(P, Q);

    std::cout << "T=" << std::endl << T << std::endl;
    std::cout << "T_ori=" << std::endl << T_ori << std::endl;
    
    // Transform P using the transformation matrix
    Eigen::MatrixXd P_transformed = T.block<3, 3>(0, 0) * P + T.block<3, 1>(0, 3) * Eigen::MatrixXd::Ones(1, P.cols());


    std::cout << "P=" << std::endl << P << std::endl;
    std::cout << "Q=" << std::endl << Q << std::endl;
    std::cout << "P_transformed=" << std::endl << P_transformed << std::endl;

    return 0;
}

/// @brief Calculate the distance between a point and a line segment
/// @param p: point
/// @param a: line segment start point
/// @param b: line segment end point
/// @return 
double pointToLineSegmentDistance(const Eigen::Vector3d &p, const Eigen::Vector3d &a, const Eigen::Vector3d &b){
    Eigen::Vector3d ab = b - a;
    Eigen::Vector3d ap = p - a;
    Eigen::Vector3d bp = p - b;

    double e = ap.dot(ab);
    if(e <= 0){
        return ap.norm();
    }

    double f = ab.dot(ab);
    if(e >= f){
        return bp.norm();
    }

    return ap.norm() * sin(acos(e/f));
}


/// @brief Find the files with the specified extension in the folder and its subfolders (optional)
/// @param folder_path: The folder path
/// @param file_paths: The file paths that are found
/// @param file_extension: The file extension
/// @param search_subfolder: Whether to search the subfolders
/// @return true if successful, false otherwise
bool findFilesWithExtension(std::string folder_path, std::vector<std::string> &file_paths, std::string file_extension = ".pcd", bool search_subfolder = true){
    // Check if the folder path is valid
    if(!boost::filesystem::exists(folder_path)){
        std::cerr << "Error: folder path " << folder_path << " does not exist." << std::endl;
        return false;
    }
    // Check if the folder path is a folder
    if(!boost::filesystem::is_directory(folder_path)){
        std::cerr << "Error: folder path " << folder_path << " is not a folder." << std::endl;
        return false;
    }

    // Iterate through the folder
    boost::filesystem::directory_iterator end_itr;
    for(boost::filesystem::directory_iterator itr(folder_path); itr != end_itr; ++itr){
        // Check if it is a file
        if(boost::filesystem::is_regular_file(itr->path())){
            // Check if the file extension is correct
            if(itr->path().extension() == file_extension){
                file_paths.push_back(itr->path().string());
            }
        }
        // Check if it is a folder
        else if(boost::filesystem::is_directory(itr->path())){
            // Check if the subfolder should be searched
            if(search_subfolder){
                findFilesWithExtension(itr->path().string(), file_paths, file_extension, search_subfolder);
            }
        }
    }

    return true;
}

/// Make shuffleVector a template function
template<typename T>
std::vector<T> shuffleVector(const std::vector<T>& input)
{
    std::vector<T> shuffled = input;
    int n = shuffled.size();
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = n - 1; i > 0; --i)
    {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(gen);
        std::swap(shuffled[i], shuffled[j]);
    }

    return shuffled;
}

/** Storage for Gaussian randoms and Gaussian PDF**/
#define GAUSSIAN_RANDOM_NUM 1000000
#define GAUSSIAN_PDF_NUM 20000

std::vector<float> gaussian_randoms(GAUSSIAN_RANDOM_NUM); ///< Store Gaussian randoms with center 0 and predefined stddev
std::vector<float> standard_gaussian_pdf(GAUSSIAN_PDF_NUM); ///< Store Standard Gaussian PDF. Range[-10, 10]; 10 sigma

/// @brief A class to generate Gaussian randoms and Gaussian PDF
class GaussianRandomCalculator
{
public:
    /// @brief Construct a new Gaussian Random Calculator object
    GaussianRandomCalculator() : gaussian_random_seq_(0) {};

    /// @brief Destroy the Gaussian Random Calculator object
    ~GaussianRandomCalculator() {};

    /// @brief Initialize the Gaussian Random Calculator. This will fill the Gaussian randoms and Gaussian PDF. Don't forget to call this function before using the calculator.  Also only call this function once.
    void calculateGaussianTable(double stddev = 1.0)
    {
        // Generate Gaussian randoms with center 0 and stddev and store in gaussian_randoms
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, stddev);
        for (int i = 0; i < GAUSSIAN_RANDOM_NUM; ++i){
            gaussian_randoms[i] = d(gen);
        }
        
        // Generate standard Gaussian PDF and store in standard_gaussian_pdf
        for(int i = 0; i < GAUSSIAN_PDF_NUM; ++i){
            standard_gaussian_pdf[i] = calStandardGaussianPDF((float) (i - GAUSSIAN_PDF_NUM/2) * 0.001f); // range[-10, 10]; 10 sigma
        }

        std::cout << "Gaussian randoms and PDF initialized." << std::endl;
    }

    /// @brief Query the Gaussian PDF. Calculate the PDF of a value with center mu and stddev sigma.
    /// @param x: The value to calculate the PDF
    /// @param mu: The center of the Gaussian
    /// @param sigma: The stddev of the Gaussian
    /// @return The PDF of the value
    inline float queryNormalPDF(const float &x, const float &mu, const float &sigma)
    {
        float corrected_x = (x-mu)/sigma;
        if(corrected_x>9.9f || corrected_x<-9.9f) return 1e-9f;
        return standard_gaussian_pdf[static_cast<int>(corrected_x*1000+10000)];
    }

    /// @brief Query the Gaussian randoms with the predefined stddev
    /// @return A Gaussian random
    inline float queryNormalRandomPresetSD(){
        // float delt = gaussian_randoms[gaussian_random_seq_];
        gaussian_random_seq_ += 1;
        if(gaussian_random_seq_ >= GAUSSIAN_RANDOM_NUM){
            gaussian_random_seq_ = 0;
        }
        return gaussian_randoms[gaussian_random_seq_];
    }

    /// @brief Query the Gaussian randoms with center 0 and stddev sigma
    /// @param sigma Stddev of the Gaussian
    /// @return A Gaussian random
    inline float queryNormalRandomZeroMean(float &sigma){
        return sigma * queryNormalRandomPresetSD();
    }

    /// @brief Query the Gaussian randoms with center mu and stddev sigma
    /// @param mu: The center of the Gaussian
    /// @param sigma: The stddev of the Gaussian
    /// @return A Gaussian random
    inline float queryNormalRandom(float &mu, float &sigma){
        return mu + sigma * queryNormalRandomPresetSD();
    }

private:
    int gaussian_random_seq_; ///< The sequence of the Gaussian randoms in query sequence

    /// @brief Calculate the standard Gaussian PDF
    /// @param value: The value to calculate the PDF
    /// @return The PDF of the value
    inline float calStandardGaussianPDF(float value)
    {
        float fx = (1.f/(sqrtf(2.f*M_PI_2f32)))*expf(-powf(value,2)/(2));
        return fx;
    }
};

