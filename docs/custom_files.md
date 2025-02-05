# object_csv
A csv file that defines a table that describes the predefined objects' id, color, et al. For example, ```cfg/object_info_zed2.csv```.

Each line should have six elements:
```
label_id,label,instance_id(-1 means not fixed),color_b,color_g,color_r
```
- label_id: uint_8 0-255
- label: string
- instance_id: uint_16 0-65535 or -1. If -1, it means the object is of interest and being tracked. A dynamic track id will be allocated.
- color_b, color_g, color_r: bgr color 0-255 of the object in segmentation. The colors will also be used for visualization.

Please modify this table according to your instance/semantic segmentation model. This will also be used in the global mapping node.
In our code, Class ObjectInfoHandler in ```include/utils/object_info_handler.h``` is used to read the table.

# mask_kpts_msgs
This is a custom message defined [here](https://github.com/g-ch/mask_kpts_msgs). All three modes of semantic DSP map use [MaskGroup](https://github.com/g-ch/mask_kpts_msgs/blob/master/msg/MaskGroup.msg) message as inputs. Therefore, the output of segmentation and tracking result should be in this type of message. For example, in our [zed2_node](https://github.com/g-ch/simple_zed2_wrapper/blob/semantic_dsp/src/zed2_node.cpp) for ZED2 mode and [single_camera_tracking](https://github.com/g-ch/single_camera_tracking/tree/main) for Superpoint mode, we publish this message with topic name "/mask_group_super_glued".

* If you are not using ROS or don't want to use ROS message, the Class MaskKpts in ```include/utils/tracking_result_handler.h``` can be used to give the input directly.

## Message Explanation
[MaskGroup](https://github.com/g-ch/mask_kpts_msgs/blob/master/msg/MaskGroup.msg) message consists of an array of 
submessage type [MaskKpts](https://github.com/g-ch/mask_kpts_msgs/blob/master/msg/MaskKpts.msg). Each MaskKpts can represent one dynamic object or all the background objects.

```
string label
uint16 track_id
sensor_msgs/Image mask
Keypoint bbox_tl
Keypoint bbox_br
Keypoint[] kpts_curr
Keypoint[] kpts_last
```
## Dynamic objects
For dynamic objects or objects of interest, set label and track_id following the rules you define in [object_csv](#object_csv). The mask is a gray scale image with 255 for the pixels of the object and 0 otherwise. bbox_tl and bbox_br define the top left and bottom right bounding box position (pixel) of the object (z is not required). 

Specially,
- kpts_last is an position array of 3D key points on the object of the last frame in the global coordinate.
- kpts_curr is an position array of 3D key points on the object of the current frame in the global coordinate.

The size of kpts_last and kpts_curr should be the same and larger than three. kpts_last[n] and kpts_curr [n] should ideally be the same point on the object physically.
We will use these keypoints to estimate a transformation of the object between two frames. In the ZED2 mode, these keypoints are corners of the 3D bboxs. In the superpoint mode, these keypoints are superpoints and are paired with superglue (a learning-based matching algorithm).

## Static objects
For static or background objects, only one mask image is required. Objects of different semantics should be colored the same as in [object_csv](#object_csv). The label should be "static" and track_id should be 65535. Bbox and keypoints are not required.


