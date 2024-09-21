import cv2
import numpy as np
import os
import argparse

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = image[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--image", help="Path to the image file", default="/media/cc/Elements/KITTI-360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic_rgb/0000000251.png")
    args = arg.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)

    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB',mouseRGB)

    #Do until esc pressed
    while(1):
        cv2.imshow('mouseRGB',image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    #if esc pressed, finish.
    cv2.destroyAllWindows()