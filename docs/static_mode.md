## Description
This file contains instructions of using Semantic DSP map with only static semantic mapping. Since our map is an ego-centric local map, a simple global mapping node is also provided. Tests were conducted with KITTI 360 dataset.

## Installation
1. Download the code and build
    ```
    mkdir -p semantic_map_ws/src
    cd semantic_map_ws/src
    git clone git@github.com:g-ch/mask_kpts_msgs.git
    git clone -b external_tracker --recursive git@github.com:g-ch/Semantic_DSP_Map.git
    catkin build
    ```

2. Download global mapping node
   ```
   cd semantic_map_ws/src
   git@github.com:g-ch/dsp_global_mapping.git
   catkin build
   ```

## Set Key Parameters
In ```include/settings/setting.h```, set ```#define SETTING 0``` and change the camera intrinsics and image size if you are using a dataset other than KITTI 360.

```
#if SETTING == 0 ///< KITTI_360
    ...
    // Kitti-360 settings
    constexpr float g_camera_fx_set = 552.554261;
    constexpr float g_camera_fy_set = 552.554261;
    constexpr float g_camera_cx_set = 682.049453;
    constexpr float g_camera_cy_set = 238.769549;

    constexpr int g_image_width_set = 1408;
    constexpr int g_image_height_set = 376;
```

After the above parameters are set, run ```catkin build```.
   
Information about other optional parameters can be found in [Parameter Table](parameter_table.md).


## Usage
- Run mapping
```
roslaunch semantic_dsp_map kitti360.launch
```

- Run global mapping. If you want to run global mapping node, please first clone and build [dsp_global_map](git@github.com:g-ch/dsp_global_mapping.git) and check its readme. 
```
roslaunch dsp_global_mapping global_mapping.launch out_file:=path_to_save_ply_global_map write_color:=1
```

Then feed data to topics 
```
mask_kpts_msgs::MaskGroup /mask_group_super_glued
sensor_msgs::Image  /camera_depth_image
geometry_msgs::PoseStamped /camera_pose
```
where in ```/mask_group_super_glued``` topic, only one mask with label "static" is required. Pixel value (one channel) in the mask stands for label id. This mask can be acquired by tools like MMSeg. The feature points are not required.
