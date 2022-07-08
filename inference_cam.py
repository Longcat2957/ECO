import torch
import time
import cv2
from model.eco_lite import ECO_Lite


if __name__ == '__main__':
    WINDOW_NAME = 'ECO_torch_cv_demo'
    model = ECO_Lite(5)

    cam = cv2.VideoCapture(0) # 0 : cam, or video files
    
    while cv2.waitKey(1) < 1:
        tic = time.time()
        (grabbed, img) = cam.read()
        if not grabbed:
            exit()