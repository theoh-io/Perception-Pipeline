from time import sleep
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import logging
import importlib
import os 
if str(os.getcwd())[-5:] == "loomo":
    os.chdir("python")
if str(os.getcwd())[-7:] == "scripts":
    os.chdir("..")

print(f"Current WD: {os.getcwd()}")

import torch

from dlav22.deep_sort.deep_sort import DeepSort
from dlav22.utils.utils import FrameGrab

from dlav22.perception import perception
from dlav22.utils.utils import Utils

from dlav22.deep_sort.utils.parser import get_config

if __name__ == "__main__":

    verbose = False #FIXME Change that to logging configuration

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    logger_pifpaf = logging.getLogger("openpifpaf.predictor")
    logger_pifpaf.setLevel(logging.WARNING)


    detector = perception.DetectorG16(verbose=verbose)
    # start streaming video from webcam
    grab = FrameGrab(mode="video", video="../Benchmark/Loomo/Demo3/theo_Indoor.avi")

    # Change the logging level
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    bboxes_to_save = []
    iters = 0
    while(True):
        iters += 1

        img = grab.read_cap()
        if img is None:
            print("Stop reading.")
            break

        bbox = detector.forward(img)

        bboxes_to_save.append(bbox)
        ###################
        #  Visualization  #
        ###################
        if bbox is not None:
            if verbose is True: print("Visualization bbox:", bbox)
            top_left=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2))  #top-left corner
            bot_right= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
            bbox_array = cv2.rectangle(img,top_left,bot_right,(255,0,0),2)
        else:
            pass
            # print("no visualization:", bbox)
        cv2.imshow('result', img)
  
        k = cv2.waitKey(10) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
        
        sleep(0.05)

    bboxes_to_save = [b if b is not None else -np.ones(4) for b in bboxes_to_save]
    bboxes_to_save = np.array(bboxes_to_save, dtype=np.int16)

    folder_str = detector.cfg.PERCEPTION.FOLDER_FOR_PREDICTION
    save_str = f"{folder_str}/{detector.cfg.PERCEPTION.BENCHMARK_FILE.replace('.','').replace('/','').replace('_','')}_tracker_{detector.cfg.TRACKER.TRACKER_CLASS[-11:]}.txt"
    print(f"Saving predicted bboxes to {save_str}.")
    np.savetxt(save_str, bboxes_to_save, fmt='%.i',delimiter=' , ')

    cv2.destroyAllWindows()
    del grab
