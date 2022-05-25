
import logging

import numpy as np

from dlav22.detectors import pose_detectors, yolo_detector

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

class PoseYoloDetector():
    def __init__(self, verbose: bool = False) -> None:
        self.detector = yolo_detector.YoloDetector(verbose=verbose) #simple yolo
        self.first_detector = pose_detectors.PoseDetector() #detector combining color detection and PifPaf
        self.start = True

    def predict(self, img: np.ndarray):
        if self.start:
            bbox_list = self.first_detector.predict(img)
            if bbox_list is not None and bbox_list[0] is not None:
                self.start = False
        else:
            bbox_list= self.detector.predict_multiple(img)
        
        return bbox_list