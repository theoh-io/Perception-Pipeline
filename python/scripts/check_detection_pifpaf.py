from time import sleep
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
if str(os.getcwd())[-7:] == "scripts":
    os.chdir("..")
print(f"Current WD: {os.getcwd()}")

from dlav22.detectors import pifpaf_detectors
from dlav22.utils.utils import Utils

from dlav22.utils.utils import FrameGrab

import logging


if __name__ == "__main__":

    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    VERBOSE = False
    
    pifpafDetect = pifpaf_detectors.PifPafDetector(verbose=VERBOSE)
    colDetect = pifpaf_detectors.ColorDetector(verbose=VERBOSE,dim_img=(250,170))
    poseColFuseDetect = pifpaf_detectors.PoseColorGuidedDetector(dim_img=(250,170))
    grab = FrameGrab(mode="webcam")

    while(1):

        image = grab.read_cap() # This outputs BGR format

        tic = time.perf_counter()
        dim_original = image.shape
        dim_process = (200,150)
        # print(dim_original,dim_process)
        image_original = image.copy()

        image = Utils.get_resized_img(image,new_dim=dim_process)
        bbox_pifpaf = pifpafDetect.predict(image)
        toc1 = time.perf_counter()
        bbox_color = colDetect.predict(image)
        toc2 = time.perf_counter()

        # if bbox_pifpaf is not None:
        #     cv2.rectangle(image,bbox_pifpaf[0],bbox_pifpaf[1],(0,255,0),2)
        # if bbox_color is not None:
        #     cv2.rectangle(image,bbox_color[0],bbox_color[1],(255,0,0),2)

        # FIXME Get the right format to test it here!
        bbox_comb = poseColFuseDetect.predict(image)
        if bbox_comb is not None:
            bbox = bbox_comb[0]
            top_left=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2))  #top-left corner
            bot_right= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
            print(top_left)
            print(bot_right)
            cv2.rectangle(image,top_left,bot_right,(0,0,255),2)

        toc = time.perf_counter()
        # print(f"Elapsed time: {toc-tic:.4f}")

        image = Utils.get_resized_img(image,new_dim=(dim_original[1],dim_original[0]))
        cv2.imshow('result', image)

        
        # if bbox_comb is not None: #FIXME Plot that
        #     print(bbox_comb)
        #     x_left = int(np.round((bbox_comb[0][0]+1) * dim_original[0] / dim_process[0])) - 1
        #     y_up = int(np.round((bbox_comb[0][1]+1) * dim_original[1] / dim_process[1])) - 1
        #     x_right = int(np.round((bbox_comb[1][0]+1) * dim_original[0] / dim_process[0])) - 1 
        #     y_down =int(np.round((bbox_comb[1][1]+1) * dim_original[1] / dim_process[1])) - 1
        #     bbox_comb[0] = (x_left, y_up)
        #     bbox_comb[1] = (x_right, y_down)
        #     print(bbox_comb)
        #     cv2.rectangle(image_original,bbox_comb[0],bbox_comb[1],(0,0,255),2)
        # cv2.imshow('ori', image_original)

        #cv2.waitKey(0)    
        k = cv2.waitKey(10) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
        
        sleep(0.3)
    
    cv2.destroyAllWindows()
    del grab






# Other
        # if bbox_color is not None and bbox_pifpaf is not None:
        #     IoU = Utils.bb_intersection_over_union(bbox_pifpaf,bbox_color)
        #     print(IoU)
        #     if IoU > 0.4:
        #         print(f"DETECTED")
        #         #FIXME Get all blue bboxes larger than a specific area and check with all

        
        # pifpaf.plot_key_points(image)
        # plt.imshow(image)
        # plt.show()
