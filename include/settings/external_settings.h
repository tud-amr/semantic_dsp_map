/**
 * @file external_settings.h
 * @author Clarence Chen (g-ch@github.com)
 * @brief This head file is to define the parameters of the RGB-D camera. Set the parameters according to your camera.
 * @version 0.1
 * @date 2023-06-28
 * 
 * @copyright Copyright (c) 2023
 */


#pragma once

inline bool g_if_consider_depth_noise  = false; ///< Whether to consider the noise in the depth image
inline bool g_if_use_pignistic_probability = false; ///< Whether to use pignistic probability to determine the occupancy probability
inline bool g_if_use_independent_filter = false; ///< Whether to use independent SMC-PHD filters for each instance
inline bool g_if_use_template_matching = false; ///< Whether to use template matching to determine the occupancy probability

/******** Get the Flags for the usage of somewhere else ***********/

/// @brief Get the flag of considering the noise in the depth image
/// @return 
inline bool getFlagConsiderDepthNoise() {return g_if_consider_depth_noise;}

/// @brief Get the flag of using pignistic probability to determine the occupancy probability
/// @return
inline bool getFlagUsePignisticProbability() {return g_if_use_pignistic_probability;}

/// @brief Get the flag of using independent SMC-PHD filters for each instance
/// @return
inline bool getFlagUseIndependentFilter() {return g_if_use_independent_filter;}

/// @brief Get the flag of using template matching to enhance the mapping reaction
inline bool getFlagUseTemplateMatching() {return g_if_use_template_matching;}


/******** Set the Flags from somewhere else ***********/

/// @brief Set the flag of considering the noise in the depth image
/// @param flag
inline void setFlagConsiderDepthNoise(bool flag) {g_if_consider_depth_noise = flag;}

/// @brief Set the flag of using pignistic probability to determine the occupancy probability
/// @param flag
inline void setFlagUsePignisticProbability(bool flag) {g_if_use_pignistic_probability = flag;}

/// @brief Set the flag of using independent SMC-PHD filters for each instance
/// @param flag
inline void setFlagUseIndependentFilter(bool flag) {g_if_use_independent_filter = flag;}

/// @brief Set the flag of using template matching to enhance the mapping reaction
/// @param flag
inline void setFlagUseTemplateMatching(bool flag) {g_if_use_template_matching = flag;}


