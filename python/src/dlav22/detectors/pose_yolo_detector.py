
import numpy as np

from dlav22.detectors import pose_detectors, yolo_detector
from dlav22.utils.utils import Utils
import dlav22
class PoseYoloDetector():
    def __init__(self, cfg, verbose: bool = False) -> None:
        verbose = cfg.PERCEPTION.VERBOSE
        self.detector = yolo_detector.YoloDetector(cfg) #simple yolo
        self.first_detector = Utils.import_from_string(cfg.DETECTOR.POSE_DETECTOR_CLASS)() #pose_detectors.PoseDetector() #detector combining color detection and PifPaf
        self.start = True

    def predict(self, img: np.ndarray):
        if self.start:
            bbox_list = self.first_detector.predict(img)
            if bbox_list is not None and bbox_list[0] is not None:
                self.start = False
        else:
            bbox_list= self.detector.predict_multiple(img)
        
        return bbox_list