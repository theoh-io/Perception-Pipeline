
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
            self.start = False
        else:
            bbox_list= self.detector.predict_multiple(img)
        
        return bbox_list

class DetectorG16():

    def __init__(self, verbose=False) -> None:

        self.detector=yolo_detector.YoloDetector(verbose=verbose) #simple yolo
        self.first_detector = pifpaf_detectors.PoseColorGuidedDetector() #detector combining color detection and PifPaf
        
        current_path=os.getcwd()
        ReIDpath=current_path+"/src/dlav22/trackers/ReID_model.pth.tar"
        tracker=reid_tracker.ReID_Tracker(ReIDpath, 'cosine', 0.87, verbose=verbose)

        cfg = get_config(config_file="src/dlav22/deep_sort/configs/deep_sort.yaml")

        deep_sort_model = cfg.DEEPSORT.MODEL_TYPE
        desired_device = ''    
        cpu = 'cpu' == desired_device
        cuda = not cpu and torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')

        self.ds_tracker = DeepSort(
            deep_sort_model,
            device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )

        self.ds_reid_tracker = custom_trackers.Custom_ReID_with_Deepsort(self.ds_tracker, tracker)

    def forward(self, img: np.ndarray):
        pass