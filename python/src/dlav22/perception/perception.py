import numpy as np
import os 
import logging
import time
from typing import List
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

        self.fac_bbox_downscale = self.cfg.PERCEPTION.BBOX_FACTOR

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

        self.log_elapsed_times = []

        self.count_none_tracked = 0

    def store_elapsed_time(self):
        # Save logs of elapsed time
        self.log_elapsed_times = np.array(self.log_elapsed_times)
        folder_str = self.cfg.PERCEPTION.REPORT_ELPASED_TIME_FOLDER
        save_str = f"{folder_str}/ID_{self.cfg.PERCEPTION.EXPID:04d}_elapsed_time_detector_tracker"
        print(self.log_elapsed_times)
        np.savetxt(f"{save_str}.txt", self.log_elapsed_times*1e3, fmt='%.3f', delimiter=' , ')
        print(f"Saved log files to {save_str}.txt")

    def scale_bbox_size(self, bbox: List[float]):
        bbox[2] *= self.fac_bbox_downscale
        bbox[3] *= self.fac_bbox_downscale
        return bbox

    def forward(self, img: np.ndarray):

        if self.count_none_tracked > self.cfg.PERCEPTION.REINIT_WITH_PIFPAF_AFTER_TIMES:
            print("Reinitialize with PifPaf...")
            self.detector.start = True # Relevant for Pifpaf YOLO detector
            self.count_none_tracked = 0

        # Detection
        tic = time.perf_counter()
        bbox_list = self.detector.predict(img)
        toc = time.perf_counter()
        self.log_elapsed_times.append(toc-tic)
        if self.verbose:
            print(f"Elapsed time for detector forward pass: {(toc - tic) * 1e3:.1f}ms")

        if bbox_list is not None and bbox_list[0] is not None:
            if self.use_img_transform:
                cut_imgs = Utils.crop_img_parts_from_bboxes(bbox_list,img,self.img_processing)
                if self.verbose and cut_imgs is not None: print("in preprocessing: ", bbox_list)
            else:
                cut_imgs = None
        else:
            #if self.verbose is True: 
            print("No person detected.")

        # Tracking
        tic = time.perf_counter()
        if bbox_list is not None and len(bbox_list)!=0 and bbox_list[0] is not None:
            bbox = self.tracker.track(cut_imgs, bbox_list,img)
            if self.verbose:
                print(f"Elapsed time for tracker forward pass: {(toc - tic) * 1e3:.1f}ms")
        else:
            invert_op = getattr(self, "tracker.increment_ds_ages", None)
            if callable(invert_op):
                self.tracker.increment_ds_ages()
                print('Not existing')
            bbox = None

        # FIXME Add threshold -> calculate what should be the minium distance before changing this.
        if bbox is not None:
            bbox = self.scale_bbox_size(bbox)
        else:
            self.count_none_tracked += 1

        toc = time.perf_counter()
        self.log_elapsed_times.append(toc - tic)

        return bbox
