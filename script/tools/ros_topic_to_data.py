#!/usr/bin/env python3

import rospy
import message_filters
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

# Initialize CvBridge to convert ROS images to OpenCV format
bridge = CvBridge()

# Directories to save images
folder1 = "/media/clarence/Clarence/semantic_dsp_bags/data_for_Symphonies/zed2/depth"
folder2 = "/media/clarence/Clarence/semantic_dsp_bags/data_for_Symphonies/zed2/rgb"

# Create directories if they do not exist
os.makedirs(folder1, exist_ok=True)
os.makedirs(folder2, exist_ok=True)

def callback(depth_msg, rgb_msg):
    # Convert depth image to a NumPy array (float32) and save to .npy
    try:
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        # resize the image to Kitti-360 size
        depth_image = cv2.resize(depth_image, (1408, 376), interpolation=cv2.INTER_NEAREST)

        depth_filename = os.path.join(folder1, f"depth_{depth_msg.header.stamp.to_nsec()}.npy")
        np.save(depth_filename, depth_image)
        rospy.loginfo(f"Depth image saved to {depth_filename}")
    except Exception as e:
        rospy.logerr(f"Failed to save depth image: {e}")

    # Convert RGB image to an OpenCV format (BGR) and save to .png
    try:
        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

        # resize the image to Kitti-360 size
        rgb_image = cv2.resize(rgb_image, (1408, 376))

        rgb_filename = os.path.join(folder2, f"rgb_{rgb_msg.header.stamp.to_nsec()}.png")
        cv2.imwrite(rgb_filename, rgb_image)
        rospy.loginfo(f"RGB image saved to {rgb_filename}")
    except Exception as e:
        rospy.logerr(f"Failed to save RGB image: {e}")


def main():
    rospy.init_node('image_saver', anonymous=True)

    # Subscribing to the depth and rgb topics
    depth_sub = message_filters.Subscriber("/camera/depth_repub", Image)
    rgb_sub = message_filters.Subscriber("/zed2/left/rgb/image", Image)

    # Time synchronizer to ensure messages are received together
    ts = message_filters.ApproximateTimeSynchronizer([depth_sub, rgb_sub], queue_size=10, slop=0.1)
    ts.registerCallback(callback)

    rospy.loginfo("Started synchronized image saver node")
    
    rospy.spin()

if __name__ == '__main__':
    main()
