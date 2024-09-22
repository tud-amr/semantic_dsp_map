# Parameter Table

## Overview
There are two types of parameters in Semantic DSP map. 

- __Static parameters__ are in ```include/settings/settings.h```. These parameters are related to data structure or compilation. Any change to these parameters requires recompilation of the code. 

- __Dynamic parameters__ are stored in the ```yaml``` file in ```cfg``` and will be loaded when the mapping node starts. Changes to these parameters do not require recompilation. A ```csv``` file storing the label_id,label,instance_id,color_b,color_g,color_r is also loaded when the mapping node starts. Set instance_id a big fixed number for static objects and -1 (auto-assgin) for movable objects and the rest should be the same as what you use in the instance/semantic segmentation model.


__NOTE:__ When a different camera is used, parameters start with "g_camera" need to be modified according to the right camera intrinsics. The following parameters should be increased if a depth camera with heavy noise is used.
```
depth_noise_model_first_order  depth_noise_model_zero_order	 noise_number
```

## Static parameters

| Name  |  Meaning | Default  |
|---|---|---|
| STORAGE_TYPE  |  Storage type for particles. 0: row major, 1: column major, 2: morton code. Morton code is a bit more efficient but needs C_VOXEL_NUM_AXIS_X_N==C_VOXEL_NUM_AXIS_Y_N==C_VOXEL_NUM_AXIS_Z_N (or C_VOXEL_NUM_AXIS_Z_N=C_VOXEL_NUM_AXIS_X_N-1). The row major and column major are more flexible.  | 0  |
| VERBOSE_MODE  | 1: show std::cout and cv::imshow. 0: otherwise  | 0 |
| SETTING  | Different working mode. 0: KITTI_360 (static), 1:CODA (similar to VIRTUAL_KITTI2), 2:VIRTUAL_KITTI2 (dynamic, use superpoints for tracking and transformation estimation), 3:ZED2 (dynamic, use ZED2 SDK __Recommended__) |  3 |
| BOOST_MODE  | 1: use BOOST MODE. The input image will be scaled by 0.5 to accelerate. 0: not use | 1 if SETTING==3 else 0   |
| C_VOXEL_NUM_AXIS_X_N  | Set number of voxels along X axis to be 2^C_VOXEL_NUM_AXIS_X_N.  | -  |
| C_VOXEL_NUM_AXIS_Y_N  | Set number of voxels along Y axis to be 2^C_VOXEL_NUM_AXIS_Y_N.  | -   |
| C_VOXEL_NUM_AXIS_Z_N  | Set number of voxels along Z axis to be 2^C_VOXEL_NUM_AXIS_Z_N.  | -   |
| C_MAX_PARTICLE_NUM_PER_VOXEL_N  | Set maximum particle number in each voxel to be 2^ C_MAX_PARTICLE_NUM_PER_VOXEL_N. | 2 or 3  |
| C_VOXEL_SIZE  | Voxel Size in meter.  | -  |
| g_camera_fx_set  | Camera intrinsics: fx  | -  |
| g_camera_fy_set  | Camera intrinsics: fy  | -  |
| g_camera_cx_set  | Camera intrinsics: cx  | -  |
| g_camera_cy_set  | Camera intrinsics: cy  | -  |
| g_image_width_set  | Input depth image size: width  | -  |
| g_image_height_set  | Input depth image size: height  | -  |
| g_consider_instance  | Whether to consider instance in mapping. true or false  | - |
| g_depth_range_min  | Minimum depth range threshold. Below this threshold will be considered as invalid depth. | 0.3  |
| g_depth_range_max  | Max depth range threshold. Above this threshold will be considered as invalid depth | 10 to 30  |


__NOTE__: C_VOXEL_NUM_AXIS_X_N + C_VOXEL_NUM_AXIS_Y_N + C_VOXEL_NUM_AXIS_Z_N + C_MAX_PARTICLE_NUM_PER_VOXEL_N should be no more than 31

## Dynamic parameters
| Name  |  Meaning | Default  |
|---|---|---|
| depth_image_topic  | Depth image topic name to receive. | -  |
| camera_pose_topic  | Camera pose topic name to receive.  |   |
| mask_group_topic  | Mask group topic (from simple_zed_wrapper or single_camera_tracking) topic name to receive.  | /mask_group_super_glued |
| beyesian_movement_distance_threshold  |  Distance changing to have an ```moving``` observation for one object. I.E. One object moves more than 0.3 m will have an observation ```moving```, otherwise ```static```. | 0.2  |
| beyesian_movement_decrement | Logwise moving probability to reduce if one ```static``` observation is obtained. | 0.05  |
| beyesian_movement_increment  | Logwise moving probability to add if one ```moving``` observation is obtained. | 0.2 |
| beyesian_movement_probability_threshold  | Logwise moving probability threshold to determine if an object is moving. If logwise moving probability is bigger than this threshold, the object will be labeled as moving. If the threshold is below 0.5, objects are considered dynamic initially. | 0.75 or 0.3 |
| if_consider_depth_noise  | Whether to consider depth noise.  | true  |
| if_out_evaluation_format  | Whether to output evaluation_format color. If false, FOV will be shown with with bright color. |  false |
| if_output_freespace  | Whether to free space point cloud.  | false  |
| if_use_independent_filter  | Whether to use independent filter mode in the mapping.  |  false |
| if_use_pignistic_probability  | Whether to use  pignistic probability for better accuracy. | false (too slow with little accuracy improvement. Aborted in code.)  |
| if_use_template_matching  | Whether to use template matching for newly observed objects.  | false  |
| match_score_threshold  | Matching score threshold if template matching is used.  | 0.6  |
| max_obersevation_lost_time  | Maximum steps of prediction to perform if an object is occcluded or is out of FOV.  | 10  |
| nb_ptc_num_per_point  | Number of particles generated from one observation point.  | 1  |
|  noise_number  | Number/Heavy Percentage of noise (0 to 1)  | 0.3  |
| detection_probability | Probability that one visible point can be detected in the depth image. | 0.6  |
| occupancy_threshold  |   | 0.1  |
| depth_noise_model_first_order  | Depth noise model. Noise = depth_noise_model_first_order * distance + depth_noise_model_zero_order  | 0.02  |
| depth_noise_model_zero_order  |  Depth noise model. Noise = depth_noise_model_first_order * distance + depth_noise_model_zero_order  | 0.2  |
| forgetting_rate | Forgetting rate used in the forgetting function. Positive. Smaller result in faster forgetting.  | 1  |
| max_forget_count | For truncation in forgetting function. If an particle is updated with only points that have a different label for more than this value, the particle will be deleted. Positive int.  | 5  |
| id_transition_probability | For ID transition function, when a particle is updated with a different labeled point, the transition is scaled with this number. Range (0,1)  | 0.5  |

__Note__: we use default values from ```options_zed2.yaml```. For different ```SETTING```, the default value can be different.