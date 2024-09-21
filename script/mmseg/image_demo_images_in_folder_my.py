from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv


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

# Define the color for each label. RGB
label_color_map_default = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32)
}

used_labels = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def find_images(directory, image_extensions=['.jpg', '.jpeg', '.png'], exclude_keywords=None):
    """
    Recursively finds all images in a directory and its subdirectories.

    :param directory: The root directory to search for images.
    :param image_extensions: A list of image file extensions to look for.
    :return: A list of paths to image files.
    """
    images = []
    for extension in image_extensions:
        # Using glob to find files with the specified extensions
        images.extend(glob.glob(os.path.join(directory, '**/*' + extension), recursive=True))

    # Remove images that contain the exclude keywords
    if exclude_keywords is not None:
        images_excluded = [image for image in images if exclude_keywords not in image]
    else:
        images_excluded = images

    return images_excluded


def create_color_mapping(used_labels, label_id_map_default, label_color_map_default):
    color_mapping = np.zeros((len(used_labels), 3), dtype=np.uint8)
    for i, label in enumerate(used_labels):
        if label in label_id_map_default:
            color_mapping[i] = label_color_map_default[label_id_map_default[label]]
    return color_mapping

def main():
    parser = ArgumentParser()
    parser.add_argument('--folder', help='Image folder')
    parser.add_argument('--output', help='Output folder')

    parser.add_argument('--config', help='Config file', default='/home/cc/git/mmsegmentation/configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='/home/cc/git/mmsegmentation/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference', choices=['cuda:0', 'cpu'])
    
    args = parser.parse_args()


    # Initialize the model
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Get all the images in the folder and its subfolders (recursively)
    images = find_images(args.folder, exclude_keywords="classmmseg")

    print("Found", len(images), "images")

    # For each image, get the segmentation mask
    for image in images:
        result = inference_model(model, image)
        segmentation_mask = result.pred_sem_seg

        tensor = segmentation_mask.data  # Access the tensor
        segmentation_mask_array = tensor.cpu().numpy()

        # Create a mapping from label IDs to colors
        color_mapping = create_color_mapping(used_labels, label_id_map_default, label_color_map_default)

        # Apply the color mapping
        mask_img = color_mapping[segmentation_mask_array]

        mask_img = np.squeeze(mask_img, axis=0)

        # Split the image name by "_" and get the last part
        # image_name = "classmmseg_" + image.split("_")[-1].split(".")[0] + ".png"
        image_name = image.split("/")[-1].split(".")[0] + ".png"

        # Save the mask image with the image name and use the same directory
        # path = image.split("/")
        # path[-1] = image_name
        # path = "/".join(path)
        
        path = os.path.join(args.output, image_name)
        plt.imsave(path, mask_img)

        print("Saved", path, " progress:", images.index(image) + 1, "/", len(images))

    # Show the mask image
    # plt.imshow(mask_img)
    # plt.show()

if __name__ == '__main__':
    main()
