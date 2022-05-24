
import numpy as np
import logging

import numpy as np
import torch
import os

from dlav22.deep_sort.deep_sort import DeepSort
from dlav22.utils.utils import FrameGrab, Utils
from dlav22.deep_sort.utils.parser import get_config
from dlav22.detectors import yolo_detector, pifpaf_detectors

from dlav22.trackers import custom_trackers, reid_tracker

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

class PifPafYOLODetector():
    def __init__(self, verbose: bool = False) -> None:
        self.detector = yolo_detector.YoloDetector(verbose=verbose) #simple yolo
        self.first_detector = pifpaf_detectors.PoseColorGuidedDetector() #detector combining color detection and PifPaf
        self.start = True

    def predict(self, img: np.ndarray):
        if self.start:
            bbox_list = self.first_detector.predict(img)
            if bbox_list is not None and bbox_list[0] is not None:
                self.start = False
        else:
            bbox_list= self.detector.predict_multiple(img)
        
        return bbox_list