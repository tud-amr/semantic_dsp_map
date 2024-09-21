import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector


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
    # Load the image
    img = '/mnt/data/dataset/UTKinect/RGB/s01_e01/colorImg190.jpg'
    # Choose to use a config and initialize the detector

    # Mask R-CNN
    # config = '/home/cc/git/mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'
    # checkpoint = '/home/cc/git/mmdetection/mymodels/mask_rcnn/res_101_fpn/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

    # Mask2former
    config = '/home/cc/git/mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
    checkpoint = '/home/cc/git/mmdetection/mymodels/mask2former/ins_resnet_50/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

    # Set the device to be used for evaluation
    device='cuda:0'

    model = load_model(config, checkpoint, device)


    # Inference the image
    result = inference_detector(model, img)

    # Show the results
    show_result_pyplot(model, img, result, score_thr=0.3)

