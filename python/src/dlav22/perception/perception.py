import numpy as np
import os 
import logging
import time
# from dlav22.detectors.pose_detectors import PoseDetector
from dlav22.detectors.pose_yolo_detector import PoseYoloDetector
from dlav22.trackers.fused_ds_reid import FusedDsReid
from dlav22.utils.utils import Utils
from dlav22.deep_sort.utils.parser import YamlParser

class DetectorG16():

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.loaf_cfg()
        self.initialize_detector()

    def loaf_cfg(self):
        file_path = os.path.dirname(os.path.realpath(__file__))

        cfg = YamlParser(config_file=file_path + "/../configs/cfg_perception.yaml")
        cfg.merge_from_file(file_path + "/../configs/cfg_detector.yaml")
        cfg.merge_from_file(file_path + "/../configs/cfg_tracker.yaml")

        self.cfg = cfg

    def initialize_detector(self):

        self.use_img_transform = True

        self.detector = Utils.import_from_string(self.cfg.DETECTOR.DETECTOR_CLASS)(self.cfg) #PoseYoloDetector(verbose=verbose)
        print(f"-> Using {self.cfg.DETECTOR.DETECTOR_CLASS} as detector.")

        self.tracker = Utils.import_from_string(self.cfg.TRACKER.TRACKER_CLASS)(self.cfg) #FusedDsReid(cfg)
        print(f"-> Using {self.cfg.TRACKER.TRACKER_CLASS} as tracker.")
        try:
            if callable(getattr(self.tracker, "image_preprocessing", None)):
                self.img_processing = self.tracker.image_preprocessing
            elif callable(getattr(self.tracker.reid_tracker, "image_preprocessing", None)):
                self.img_processing = self.tracker.reid_tracker.image_preprocessing
        except:
            logging.warning("image_preprocessing seems not to be implemented in this class. Do not use it")
            self.use_img_transform = False

    def forward(self, img: np.ndarray):

        # Detection
        tic = time.perf_counter()
        bbox_list = self.detector.predict(img)
        toc = time.perf_counter()
        print(f"Elapsed time for detector forward pass: {(toc - tic) * 1e3:.1f}ms")

        if bbox_list is not None and bbox_list[0] is not None:
            if self.use_img_transform:
                cut_imgs = Utils.crop_img_parts_from_bboxes(bbox_list,img,self.img_processing)
                if self.verbose and cut_imgs is not None: print("in preprocessing: ", bbox_list)
            else:
                cut_imgs = None
        else:
            #if self.verbose is True: 
            print("no detection")

        # Tracking
        if bbox_list is not None and len(bbox_list)!=0 and bbox_list[0] is not None:
            tic = time.perf_counter()
            bbox = self.tracker.track(cut_imgs, bbox_list,img)
            toc = time.perf_counter()
            print(f"Elapsed time for tracker forward pass: {(toc - tic) * 1e3:.1f}ms")
        else:
            invert_op = getattr(self, "tracker.increment_ds_ages", None)
            if callable(invert_op):
                self.tracker.increment_ds_ages()
                print('Not existing')
            bbox = None
        return bbox