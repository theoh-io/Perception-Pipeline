from time import sleep
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

from pathlib import Path
import yaml
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
from dlav22.utils.utils import YamlInteract

from dlav22.deep_sort.utils.parser import get_config

if __name__ == "__main__":

    verbose = False #FIXME Change that to logging configuration

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    logger_pifpaf = logging.getLogger("openpifpaf.predictor")
    logger_pifpaf.setLevel(logging.WARNING)

    detector = perception.DetectorG16(verbose=verbose)

    detector.cfg.DEEPSORT.MAX_DIST = 0.5
    detector.initialize_detector()

    # start streaming video from webcam
    grab = FrameGrab(mode="video", video="../Benchmark/Loomo/Demo3/theo_Indoor.avi")

    # Change the logging level
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    bboxes_to_save = []
    iters = 0
    while(True):
        iters += 1
        # if iters > 5:
        #     break

        img = grab.read_cap()

        # if iters > 10 and iters < 30:
        #     img = np.zeros_like(img)

        if img is None:
            print("Stop reading.")
            break

        tic = time.perf_counter()
        bbox = detector.forward(img)
        toc = time.perf_counter()
        print(f"Elapsed time for whole fordward pass: {(toc-tic)*1e3:.1f}ms")

        bboxes_to_save.append(bbox)
        ###################
        #  Visualization  #
        ###################
        if bbox is not None:
            if verbose is True:
                print("Visualization bbox:", bbox)
            top_left=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2))  #top-left corner
            bot_right= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
            bbox_array = cv2.rectangle(img, top_left, bot_right, (255, 0, 0), 2)
        else:
            pass

        print("bbox:", bbox)
        cv2.imshow('result', img)

        k = cv2.waitKey(10) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break

        sleep(0.05)

    if detector.cfg.PERCEPTION.SAVE_RESULTS:
        bboxes_to_save = [b if b is not None else -np.ones(4) for b in bboxes_to_save]
        bboxes_to_save = np.array(bboxes_to_save, dtype=np.int16)

        folder_str = detector.cfg.PERCEPTION.FOLDER_FOR_PREDICTION
        # save_str = f"{folder_str}/{detector.cfg.PERCEPTION.BENCHMARK_FILE.replace('.','').replace('/','').replace('_','')}_tracker_{detector.cfg.TRACKER.TRACKER_CLASS[-11:]}.txt"
        save_str = f"{folder_str}/ID_{detector.cfg.PERCEPTION.EXPID:04d}_prediction"
        path = Path(f"{save_str}.txt")
        if path.is_file():
            logging.warning("File already exists. Did not store it.")
        else:
            print(f"Saving predicted bboxes to {save_str}.txt.")
            np.savetxt(f"{save_str}.txt", bboxes_to_save, fmt='%.i',delimiter=' , ')

            # FIXME Specify all parameters that are varied for a specif configuration
            config_dict = {"EXP_ID": detector.cfg.PERCEPTION.EXPID, "DS_MAX_DIST": detector.cfg.DEEPSORT.MAX_DIST}

            file = open(f"{save_str}.yaml", "w")
            yaml.dump(config_dict,file)
            file.close()
            detector.store_elapsed_time()




    cv2.destroyAllWindows()
    del grab
