
import cv2
import os


def apply_feature_detectors(image):
    # FAST Detector
    fast = cv2.FastFeatureDetector_create(threshold=25)
    keypoints_fast = fast.detect(image, None)
    image_fast = cv2.drawKeypoints(image, keypoints_fast, None, color=(255, 0, 0))

    # ORB Detector
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints_orb = orb.detect(image, None)
    image_orb = cv2.drawKeypoints(image, keypoints_orb, None, color=(0, 255, 0))

    # Shi-Tomasi Detector
    corners = cv2.goodFeaturesToTrack(image, maxCorners=500, qualityLevel=0.01, minDistance=10)
    keypoints_shi_tomasi = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in corners]
    image_shi_tomasi = cv2.drawKeypoints(image, keypoints_shi_tomasi, None, color=(0, 0, 255))

    return image_fast, image_orb, image_shi_tomasi


def display_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_fast, img_orb, img_shi_tomasi = apply_feature_detectors(img)

    cv2.imshow('FAST Features', img_fast)
    cv2.imshow('ORB Features', img_orb)
    cv2.imshow('Shi-Tomasi Features', img_shi_tomasi)

    return cv2.waitKey(0)


if __name__ == '__main__':
    folder_path = '/home/clarence/ros_ws/semantic_dsp_ws/src/Semantic_DSP_Map/data/VirtualKitti2/rgb/Camera_0'
    images = [img for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Optional, to sort the files alphabetically

    print('Number of images: ', len(images))

    current_index = 0

    while True:
        print('Current image: ', images[current_index])
        image_path = os.path.join(folder_path, images[current_index])
        key = display_image(image_path)

        if key == 27:  # ESC key to break the loop
            break
        elif key == 65 or key == 97:  # A
            current_index = max(0, current_index - 1)  # go to previous image
        elif key == 68 or key == 100:  # D
            current_index = min(len(images) - 1, current_index + 1)  # go to next image

    cv2.destroyAllWindows()
