#!/usr/bin/env python
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import fnmatch
import argparse
import rospy
import tf.transformations as tfm
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

def generate_point_cloud(depth_image, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rows, cols = depth_image.shape
    x = np.linspace(0, cols-1, cols)
    y = np.linspace(0, rows-1, rows)
    x, y = np.meshgrid(x, y)

    x = (x - cx) * depth_image / fx
    y = (y - cy) * depth_image / fy
    z = depth_image

    point_cloud = np.stack([x, y, z], axis=-1)
    points = point_cloud.reshape(-1, 3)

    return points


def generate_point_cloud_with_rgb(depth_image, rgb_image, K, translation=None, quaternion=None, max_depth=1000.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rows, cols = depth_image.shape
    x = np.linspace(0, cols-1, cols)
    y = np.linspace(0, rows-1, rows)
    x, y = np.meshgrid(x, y)

    x = (x - cx) * depth_image / fx
    y = (y - cy) * depth_image / fy
    z = depth_image

    # Create the point cloud (x, y, z)
    point_cloud = np.stack([x, y, z], axis=-1)
    points = point_cloud.reshape(-1, 3)

    # Create a mask to filter out points with depth greater than max_depth
    valid_mask = z.flatten() <= max_depth

    # Filter points and corresponding RGB values
    points = points[valid_mask]

    if translation is not None and quaternion is not None:
        # Convert the quaternion to rotation matrix
        R = tfm.quaternion_matrix(quaternion)[:3, :3]
        t = translation

        # Transform the point cloud to world frame
        points = np.dot(R, points.T).T + t


    rgb_image = rgb_image.reshape(-1, 3)[valid_mask]

    # Pack the RGB values into a single 32-bit integer. cv2.imread reads the image in BGR format
    r = rgb_image[:, 2].astype(np.uint32)
    g = rgb_image[:, 1].astype(np.uint32)
    b = rgb_image[:, 0].astype(np.uint32)

    rgb_packed = (r << 16) | (g << 8) | b
    rgb_packed = rgb_packed.view(np.float32)  # Cast to float for compatibility with PointCloud2

    # Combine the points and packed RGB values
    rgb_points = np.column_stack((points, rgb_packed))

    return rgb_points
    


# Function to remove outliers from point cloud
def remove_outliers(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)
    
    return np.asarray(pcd.points)


def visualize_point_cloud(points, point_size=1.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


def scan_files_with_ext(dir, ext):
    files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if fnmatch.fnmatch(file, ext):
                files.append(os.path.join(root, file))

    return files        

def read_pose_txt(pose_txt):
    with open(pose_txt, 'r') as f:
        lines = f.readlines()
        poses = [] # frame_idx, pose x y z qx qy qz qw
        for line in lines:
            # Each line has 17 numbers, the first number is an integer denoting the frame index. The rest is a 4x4 matrix denoting the rigid body transform from the rectified perspective camera coordinates to the world coordinate system.
            pose = line.split()
            

            # Check if line has 13 elements. If so, add 0 0 0 1 to the pose
            if len(pose) == 13:
                frame_idx = int(pose[0])
                # Add 0 0 0 1 to the pose
                pose += ['0', '0', '0', '1']
                imu_to_world = np.array(pose[1:], dtype=np.float32).reshape(4, 4)

                # camera to imu transformation 0.0371783278 -0.0986182135 0.9944306009 1.5752681039 0.9992675562 -0.0053553387 -0.0378902567 0.0043914093 0.0090621821 0.9951109327 0.0983468786 -0.6500000000 
                camera_to_imu = np.array([
                    [ 0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039],
                    [ 0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093],
                    [ 0.0090621821,  0.9951109327, 0.0983468786, -0.6500000000],
                    [0, 0, 0, 1]
                ])

                cam0_to_world = np.dot(imu_to_world, camera_to_imu)

            elif len(pose) == 17:
                frame_idx = int(pose[0])
                cam0_to_world = np.array(pose[1:], dtype=np.float32).reshape(4, 4)
            else:
                raise ValueError("Invalid number of elements in pose")
            
            translation = cam0_to_world[:3, 3]
            quaternion = tfm.quaternion_from_matrix(cam0_to_world)

            poses.append([frame_idx, translation, quaternion])
            
    return poses


if __name__ == '__main__':
    rospy.init_node('kitti360_data_reader', anonymous=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--rgb_dir', type=str, default='/media/cc/Elements/KITTI-360/data_2d_test_slam/test_2/2013_05_28_drive_0004_sync/image_00/data_rect')
    parser.add_argument('--depth_dir', type=str, default='/media/cc/Elements/KITTI-360/data_2d_test_slam/depth_sgm/test_2/2013_05_28_drive_0004_sync/depth')
    parser.add_argument('--semantic_seg_dir', type=str, default='/media/cc/Elements/KITTI-360/data_2d_test_slam/segmentation_cmnext/test_2/2013_05_28_drive_0004_sync')
    parser.add_argument('--pose_txt', type=str, default='/media/cc/Elements/KITTI-360/data_2d_test_slam/poses/test_2_poses.txt') #orb_slam2

    parser.add_argument('--starting_frame_idx', type=int, default=0)
    parser.add_argument('--stop_frame_idx', type=int, default=1000000)

    parser.add_argument('--rgb_image_topic', type=str, default='/kitti360/cam0/rgb')
    parser.add_argument('--depth_image_topic', type=str, default='/kitti360/cam0/depth')
    parser.add_argument('--camera_pose_topic', type=str, default='/kitti360/pose_cam')
    parser.add_argument('--semantic_seg_image_topic', type=str, default='/kitti360/cam0/semantic')
    parser.add_argument('--semantic_point_cloud_topic', type=str, default='/kitti360/semantic_point')

    parser.add_argument('--loop_rate', type=int, default=0.4)
    parser.add_argument('--publish_semantic_seg', type=bool, default=True)
    parser.add_argument('--publish_semantic_pointcloud', type=bool, default=True)

    parser.add_argument('--repeat_first_frame', type=int, default=2)

    args = parser.parse_args()

    rgb_image_pub = rospy.Publisher(args.rgb_image_topic, Image, queue_size=1)
    depth_image_pub = rospy.Publisher(args.depth_image_topic, Image, queue_size=1)
    camera_pose_pub = rospy.Publisher(args.camera_pose_topic, PoseStamped, queue_size=1)
    semantic_seg_image_pub = rospy.Publisher(args.semantic_seg_image_topic, Image, queue_size=1)
    semantic_point_cloud_pub = rospy.Publisher(args.semantic_point_cloud_topic, PointCloud2, queue_size=1)
    

    # Read the pose data. Only publish the image with the corresponding pose
    pose_data = read_pose_txt(args.pose_txt)

    print("Number of poses = ", len(pose_data))

    # Loop through the pose data and publish the image and pose
    publish_pose_idx = 0
    rate = rospy.Rate(args.loop_rate)

    first_frame_repeat_count = 0

    while publish_pose_idx < len(pose_data):
        pose = pose_data[publish_pose_idx]
        
        # Skip the frame if the frame index is less than the starting frame index
        if publish_pose_idx < args.starting_frame_idx:
            publish_pose_idx += 1
            continue
        
        # Break if the frame index is greater than the stop frame index
        if publish_pose_idx > args.stop_frame_idx:
            break
        
        # Repeat the first frame 3 times. For initialisation
        if publish_pose_idx == args.starting_frame_idx + 1 and first_frame_repeat_count < args.repeat_first_frame:
            publish_pose_idx -= 1
            first_frame_repeat_count += 1

        frame_idx, translation, quaternion = pose

        rgb_image_path = os.path.join(args.rgb_dir, str(frame_idx).zfill(10) + '.png')
        depth_image_path = os.path.join(args.depth_dir, str(frame_idx).zfill(10) + '.npy')

        rgb_image = cv2.imread(rgb_image_path)
        depth_image = np.load(depth_image_path)

        time = rospy.get_rostime()

        # Publish the image
        rgb_image_msg = Image()
        depth_image_msg = Image()
        if rgb_image is not None:
            rgb_image_msg = CvBridge().cv2_to_imgmsg(rgb_image)
            rgb_image_msg.header.stamp = time
            rgb_image_pub.publish(rgb_image_msg)
        else:
            # Raise an exception
            raise ValueError("RGB Image is None")
        
        if depth_image is not None:
            depth_image_msg = CvBridge().cv2_to_imgmsg(depth_image, encoding="32FC1")
            depth_image_msg.header.stamp = time
            depth_image_pub.publish(depth_image_msg)
        else:
            # Raise an exception
            raise ValueError("Depth Image is None")
        
        # Publish the camera pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = time
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        camera_pose_pub.publish(pose_msg)

        if args.publish_semantic_seg:
            semantic_seg_image_path = os.path.join(args.semantic_seg_dir, str(frame_idx).zfill(10) + '.png')
            semantic_seg_image = cv2.imread(semantic_seg_image_path) 

            semantic_seg_image_msg = Image()
            if semantic_seg_image is not None:
                semantic_seg_image_msg = CvBridge().cv2_to_imgmsg(semantic_seg_image)

                semantic_seg_image_msg.header.stamp = time
                semantic_seg_image_pub.publish(semantic_seg_image_msg)
            else:
                # Raise an exception
                raise ValueError("Semantic Segmentation Image is None")

        
            # Turn the float 32 depth image to a 3 channel image. Max depth is 30m
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            # cv2.imshow('Depth Image', depth_colored)
            # cv2.waitKey(1)

            overlay = cv2.addWeighted(semantic_seg_image, 0.7, depth_colored, 0.3, 0)
            cv2.imshow('Overlay', overlay)
            cv2.waitKey(1)

        
            if args.publish_semantic_pointcloud:

                # Kitti 360 camera intrinsic matrix
                # camera_fx: 552.554261
                # camera_fy: 552.554261
                # camera_cx: 682.049453
                # camera_cy: 238.769549
                K = np.array([[552.554261, 0.000000, 682.049453],
                            [0.000000, 552.554261, 238.769549],
                            [0.000000, 0.000000, 1.000000]])
                            
                
                points = generate_point_cloud_with_rgb(depth_image, semantic_seg_image, K, translation, quaternion, max_depth=30.0)

                # Publish the point cloud
                header = Header()
                header.stamp = time
                header.frame_id = "map"
                # Define fields
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgb', 12, PointField.FLOAT32, 1)
                ]
                pc2_msg = pc2.create_cloud(header, fields, points)
                semantic_point_cloud_pub.publish(pc2_msg)



        if publish_pose_idx % 10 == 0:
            print("Progress: ", publish_pose_idx, "/", len(pose_data))
        
        publish_pose_idx += 1

        if rospy.is_shutdown():
            break

        rate.sleep()



