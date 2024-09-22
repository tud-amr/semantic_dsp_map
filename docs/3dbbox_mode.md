## Description
This file contains instructions of using Semantic DSP map with 3D BBOX from [ZED2 Camera SDK](https://www.stereolabs.com/docs).  According to ZED's website, this SDK supports ZED2, ZED2I and ZED mini. The objects that can be detected are listed [here](https://www.stereolabs.com/docs/api/group__Object__group.html#ga13b0c230bc8fee5bbaaaa57a45fa1177).

## Installation
1. Download the code and build
    ```
    mkdir -p semantic_map_ws/src
    cd semantic_map_ws/src
    git clone git@github.com:g-ch/mask_kpts_msgs.git
    git clone --recursive git@github.com:g-ch/Semantic_DSP_Map.git
    catkin build
    ```

2. Enable ZED2 Camera Usage
   
   Follow instructions from [official website](https://www.stereolabs.com/en-nl/developers/release) to install SDK. ZED's SDK provides Object Detection and Instance Segmentation ability. To recognize road, trees, etc., install mmseg as Step 3 describes.
   
   Then clone and build our simple_zed2_wrapper by:
   ```
   cd semantic_map_ws/src
   git clone -b semantic_dsp git@github.com:g-ch/simple_zed2_wrapper.git
   catkin build
   ```

3. MMSeg for Semantic Segmentation
   
   To run semantic segmentation with [mmseg](https://github.com/open-mmlab/mmsegmentation) and use ros at the same time, install a conda environment with the following steps. (Skip to 3.5 if you already have one environment that works for both mmseg and ros.)
   
   3.1 Install Anaconda or Miniconda following the official [instructions](https://docs.anaconda.com/miniconda/miniconda-install/). Then create an environment with a name you want, for example, dspmap. And then install Mamba. (Let's use mamba otherwise the environment solving step takes forever :)
   ```
   conda create --name dspmap python=3.11 -y
   conda activate dspmap
   conda install mamba -c conda-forge
   ```

   3.2 Enable ros usage with [robostack](https://robostack.github.io/GettingStarted.html). 
   ```
   conda config --env --add channels conda-forge
   conda config --env --add channels robostack-staging
   conda config --env --remove channels defaults
   mamba install ros-noetic-desktop
   ```
   __NOTE__: Enable ros by "pip install -U rospkg" is also an option but in some cases the cv_bridge doesn't work.
   For robostack, minimum ros version is Noetic. When using this environment, do __not__ source the system ROS environment, i.e., ```source /opt/ros/noetic/setup.bash```, check [here](https://robostack.github.io/GettingStarted.html) for details.

   3.3 Install [pytorch](https://pytorch.org/get-started/locally/) with a version compatible with your hardware. For example
   ```
   mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

   3.4 Follow the Installation steps in [instructions](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to install mmengine, mmcv, and MMSegmentation. You will clone mmsegmentation in this step.
   
   Verify the installation by following [this](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#verify-the-installation).
   You may also need to install ftfy and regex with
   ```
   mamba install ftfy regex
   ```
   to fix running issues.

   3.5 Download DDRNET pretrained model in Cityscapes dataset from [here](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/ddrnet). Modify ```script/mmseg/image_demo_ros.py``` by replacing the default config with the path of mmsegmentation you cloned and checkpoint with the path of the model file you downloaded. 
   
   __Note__: mmsegmentation offers a lot of different segmentation methods. We chose DDRNET because it's fast. Choose a different method if want to have higher accuracy or train with your dataset following the official instructions of mmsegmentaion.


## Set Key Parameters
In ```include/settings/setting.h```, set ```#define SETTING 3``` and change the camera intrinsics and image size the same as your ZED2 camera.

```
#elif SETTING == 3 ///< ZED2
    ...
    constexpr float g_camera_fx_set = 527.8191528320312;
    constexpr float g_camera_fy_set = 527.8191528320312;
    constexpr float g_camera_cx_set = 633.9357299804688;
    constexpr float g_camera_cy_set = 366.3338623046875;

    constexpr int g_image_width_set = 1280;
    constexpr int g_image_height_set = 720;
```
By default, our simple_zed2_wrapper use image size 1280*720. Default camera intrinsics can be found in a ros topic whose name contains ```camera_info```.

After the above parameters are set, run ```catkin build```.

Information about other optional parameters can be found in [Parameter Table](parameter_table.md).

## Usage
Three nodes needs to be run. The running order doesn't matter.

1. ZED Node
```
#Source your workspace
roslaunch simple_zed2_wrapper zed2_semantic_dsp.launch external_semantic_seg_on:=true
```
Set "external_semantic_seg_on:=false" if you don't plan to run semantic segmentation (the next step).


2. Semantic segmentation Node
```
# Do not source your workspace or ROS!!!
conda activate dspmap
cd Semantic_DSP_Map/script/mmseg
python image_demo_ros.py
```

3. Mapping Node
```
#Source your workspace
roslaunch semantic_dsp_map zed2.launch
```

The topic will be transmitted as the following graph shows
<p align="center">
<img src="../assets/zed2_mode_rosgraph.svg" alt="isolated" width="1000">
</p>


## Real-time Performance

With RTX 2060 mq GPU + AMD R7-4800 CPU, the mapping node runs at 10 Hz when the following map size parameters are used
```
constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 7;
constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 5;  // Y is the height. Use smaller range to reduce the computation.
constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 7;

constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 2;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
constexpr float C_VOXEL_SIZE = 0.15f; ///< Voxel size. Unit: meter
```
which means the map size is: length (Z), width (X) and height (Y) ```(19.2, 19.2, 4.8)```m (```2^X_N * C_VOXEL_SIZE```). There could be at most 4  ```=(2^2)``` particles in each voxel.

The delay is bigger than 100 ms because segmentation step also takes time. We are planning to compensate the delay by predicting dynamic objects' future occupancy.

When the following settings are used, the mapping node runs at 20 Hz.
```
constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 6;
constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 5;  // Y is the height. Use smaller range to reduce the computation.
constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 6;

constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 2;  ///< Number of bits used to store the particle index in a voxel. At least 1. The first particle is the time particle. The max normal particle number in a voxel is then 2^C_MAX_PARTICLE_NUM_PER_VOXEL_N - 1
constexpr float C_VOXEL_SIZE = 0.25f; ///< Voxel size. Unit: meter
```


__What can further make it faster?__

The easiest way is to reduce the map size. Furthermore, parallel computing and improving the particle projection algorithm etc. can be technical solutions, which haven't been improved yet.


