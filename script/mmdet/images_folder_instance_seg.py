import os
import cv2
import numpy as np
import torch

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdet.core.visualization import imshow_det_bboxes


# function to get all image file paths with suffix ".jpg" or ".png"
def get_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths



#Function to load the model
def load_model(config, checkpoint, device='cuda:0'):
    # Load the config
    config = mmcv.Config.fromfile(config)

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    return model


# Main function
if __name__ == '__main__':
    # Set the folder where the images are located
    # folder = '/mnt/data/dataset/UTKinect/RGB'   # No "/" at the end
    folder = '/home/cc/git/mmsegmentation/demo/ut_campus'

    # Set the device to be used for evaluation
    device='cuda:0'

    # Set the confidence threshold
    confidence_threshold = 0.4

    # Set the flag to show the results
    if_show = False

    # Find all images in the folder
    image_paths = get_image_paths(folder)
    print("Found {} images".format(len(image_paths)))

    if len(image_paths) == 0:
        print("No images found in the folder {}".format(folder))
        exit()

    # Get the parent directory of the folder
    parent_dir = os.path.dirname(folder)

    # Make a new folder called instance_seg_results in the parent directory if the folder does not exist
    if not os.path.exists(os.path.join(parent_dir, 'instance_seg_results')):
        os.makedirs(os.path.join(parent_dir, 'instance_seg_results'))
    
    results_folder = os.path.join(parent_dir, 'instance_seg_results')
    print("Results will be saved in {}".format(results_folder))


    # Load the model

    # Mask R-CNN
    # config = '/home/cc/git/mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'
    # checkpoint = '/home/cc/git/mmdetection/mymodels/mask_rcnn/res_101_fpn/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

    # Mask2former
    config = '/home/cc/git/mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
    checkpoint = '/home/cc/git/mmdetection/mymodels/mask2former/ins_resnet_50/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

    model = load_model(config, checkpoint, device)

    # Iterate through all images and perform inference
    # counter = 0
    for img in image_paths:
        # Get the image name and scene name
        img_name = os.path.basename(img).split('.')[0]
        scene_name = img.split('/')[-2]

        # Create a new folder for the image if it does not exist
        if not os.path.exists(os.path.join(results_folder, scene_name, img_name)):
            os.makedirs(os.path.join(results_folder, scene_name, img_name))
        result_folder_this_img = os.path.join(results_folder, scene_name, img_name)

        # Inference the image
        result = inference_detector(model, img)

        # Show the results
        if if_show:
            show_result_pyplot(model, img, result, score_thr=confidence_threshold)
        
        # Process the results
        assert isinstance(result, tuple)
        bbox_result, mask_result = result
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        if len(labels) == 0:
            bboxes = np.zeros([0, 5])
            masks = np.zeros([0, 0, 0])

        # draw segmentation masks
        else:
            masks = mmcv.concat_list(mask_result)

            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).detach().cpu().numpy()
            else:
                masks = np.stack(masks, axis=0)
            # dummy bboxes
            if bboxes[:, :4].sum() == 0:
                num_masks = len(bboxes)
                x_any = masks.any(axis=1)
                y_any = masks.any(axis=2)
                for idx in range(num_masks):
                    x = np.where(x_any[idx, :])[0]
                    y = np.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        bboxes[idx, :4] = np.array(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1],
                            dtype=np.float32)

        # Remove the bboxes with low confidence
        high_confidence_idx_array = np.where(bboxes[:, -1] > confidence_threshold)[0]
        print(high_confidence_idx_array)

        # Change the True/False values in the masks to 255/0
        masks = masks.astype(np.uint8)
        masks *= 255

        # Save labels and bboxes to csv file.
        with open(os.path.join(result_folder_this_img, os.path.basename(img).split('.')[0].split('/')[-1] + '.csv'), 'w') as label_file:
            with open(os.path.join(result_folder_this_img, os.path.basename(img).split('.')[0].split('/')[-1] + '_bboxes.csv'), 'w') as bbox_file:
                
                # Save the results
                seq = 0
                for idx in high_confidence_idx_array:
                    # Save the labels and confidence to csv file
                    label_file.write(str(labels[idx]) + ',' + str(bboxes[idx, -1]) + '\n')

                    # Save the bboxes with int pixel position to csv file
                    bbox_file.write(str(int(bboxes[idx, 0])) + ',' + str(int(bboxes[idx, 1])) + ',' + str(int(bboxes[idx, 2])) + ',' + str(int(bboxes[idx, 3])) + '\n')

                    # Save the masks to png files
                    cv2.imwrite(os.path.join(result_folder_this_img, os.path.basename(img).split('.')[0].split('/')[-1] + '_' + str(seq) + '.png'), masks[idx, :, :])
                    seq += 1
                    
                    # print(bboxes[idx, :])
                    # print(labels[idx])
                    # print(masks[idx, :, :])
       
        # counter += 1
        # if counter > 2:
        #     break


