from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model
import numpy as np
import matplotlib.pyplot as plt
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from mask_kpts_msgs.msg import ImageWithID

file_path = os.path.dirname(os.path.abspath(__file__))

#### Don't change this part. These are from the original cityscapes.py
# Define the label id for the object
label_id_map_default = {
    'road': 1,
    'sidewalk': 2,
    'building': 3,
    'wall': 4,
    'fence': 5,
    'pole': 6,
    'traffic_light': 7,
    'traffic_sign': 8,
    'vegetation': 9,
    'terrain': 10,
    'sky': 11,
    'person': 12,
    'rider': 13,
    'car': 14,
    'truck': 15,
    'bus': 16,
    'train': 17,
    'motorcycle': 18,
    'bicycle': 19
}

# Define the color for each label. BGR
label_color_map_default = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (232, 35, 244),
    3: (70, 70, 70),
    4: (156, 102, 102),
    5: (153, 153, 190),
    6: (153, 153, 153),
    7: (30, 170, 250),
    8: (0, 220, 220),
    9: (35, 142, 107),
    10: (152, 251, 152),
    11: (180, 130, 70),
    12: (60, 20, 220),
    13: (0, 0, 255),
    14: (142, 0, 0),
    15: (70, 0, 0),
    16: (100, 60, 0),
    17: (100, 80, 0),
    18: (230, 0, 0),
    19: (32, 11, 119)
}


used_labels = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def create_color_mapping(used_labels, label_id_map_default, label_color_map_default):
    color_mapping = np.zeros((len(used_labels), 3), dtype=np.uint8)
    for i, label in enumerate(used_labels):
        if label in label_id_map_default:
            color_mapping[i] = label_color_map_default[label_id_map_default[label]]
    return color_mapping


class Inferencer:
    def __init__(self, args):
        print("Initializing ROS node")

        print("Subscribing to image topic: ", args.image_topic)
        
        self.model = self.init_model(args.config, args.checkpoint, args.device)

        # Create a mapping from label IDs to colors
        self.color_mapping = create_color_mapping(used_labels, label_id_map_default, label_color_map_default)

        if(not args.use_image_with_id):
            self.image_sub = rospy.Subscriber(args.image_topic, Image, self.image_callback)
        else:
            self.image_sub = rospy.Subscriber(args.image_topic, ImageWithID, self.image_with_id_callback)
        
        self.image_pub = rospy.Publisher(args.output_topic, Image, queue_size=1)
        self.image_with_id_pub = rospy.Publisher(args.output_with_id_topic, ImageWithID, queue_size=1)

        self.bridge = CvBridge()
        rospy.spin()


    def init_model(self, config, checkpoint, device='cuda:0'):
        model = init_model(config, checkpoint, device=device)
        if device == 'cpu':
            model = revert_sync_batchnorm(model)

        print("Model initialized")
        return model
    
    def image_callback(self, msg):
        # Convert ROS image to OpenCV image and do the inference
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        image_height, image_width = img.shape[:2]

        # Resize the image to 1024x512
        img = cv2.resize(img, (1024, 512))
        
        time_start = time.time()
        mask, __ = self.inference(img)
        time_end = time.time()
        print("Inference time: ", time_end - time_start)

        # Resize the mask back to the original size
        mask = cv2.resize(mask, (image_width, image_height))

        # Convert OpenCV image to ROS image
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='bgr8')
        mask_msg.header = msg.header
        self.image_pub.publish(mask_msg)

    
    def image_with_id_callback(self, msg):
        # Convert ROS image to OpenCV image and do the inference
        try:
            img = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        image_height, image_width = img.shape[:2]

        # Resize the image to 1024x512
        img = cv2.resize(img, (1024, 512))
        
        time_start = time.time()
        mask, label_img = self.inference(img)
        time_end = time.time()
        print("Inference time: ", time_end - time_start)

        # Resize the mask back to the original size
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        # Convert OpenCV image to ROS image and publish
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='bgr8')
        mask_msg.header = msg.image.header
        self.image_pub.publish(mask_msg)

        label_img_msg = self.bridge.cv2_to_imgmsg(label_img, encoding='mono8')
        label_img_msg.header = msg.image.header
        image_with_id_msg = ImageWithID()
        image_with_id_msg.id = msg.id
        image_with_id_msg.image = label_img_msg
        self.image_with_id_pub.publish(image_with_id_msg)


    def inference(self, img):
        result = inference_model(self.model, img)
        segmentation_mask = result.pred_sem_seg

        tensor = segmentation_mask.data  # Access the tensor
        segmentation_mask_array = tensor.cpu().numpy()

        # Apply the color mapping
        mask_img = self.color_mapping[segmentation_mask_array]

        mask_img = np.squeeze(mask_img, axis=0)

        label_id_img = np.squeeze(segmentation_mask_array, axis=0).astype(np.uint8)
        
        # Show the mask image
        cv2.imshow('mask', mask_img)
        cv2.imshow('label_id', label_id_img)
        cv2.waitKey(1)

        return mask_img, label_id_img


if __name__ == '__main__':
    rospy.init_node('image_demo_ros', anonymous=True)

    parser = ArgumentParser()
    parser.add_argument('--image_topic', help='Image topic', default='zed2/left/rgb/image_with_id')
    parser.add_argument('--output_topic', help='Output topic', default='zed2/left/rgb/image_mask')
    parser.add_argument('--output_with_id_topic', help='Output topic with image id', default='zed2/left/rgb/image_mask_with_id')

    parser.add_argument('--use_image_with_id', help='Use image with id', default=True)

    parser.add_argument('--config', help='Config file', default='/home/clarence/git/mmsegmentation/configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default=os.path.join(file_path, 'model/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth'))
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    inferencer = Inferencer(args)

