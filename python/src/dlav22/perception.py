import numpy as np
# from dlav22.detectors.pose_detectors import PoseDetector
from dlav22.detectors.pose_yolo_detector import PoseYoloDetector
from dlav22.trackers.fused_ds_reid import FusedDsReid
from dlav22.utils.utils import Utils

class DetectorG16():

    def __init__(self, verbose=False) -> None:
        self.pif_yolo_detector = PoseYoloDetector(verbose=verbose)
        self.ds_reid_tracker = FusedDsReid(verbose=verbose)
        self.verbose = verbose

    def forward(self, img: np.ndarray):

        # Detection
        bbox_list = self.pif_yolo_detector.predict(img)
        if bbox_list is not None and bbox_list[0] is not None:
            tensor_img = Utils.crop_img_parts_from_bboxes(bbox_list,img,self.ds_reid_tracker.reid_tracker.image_preprocessing)
            if self.verbose and tensor_img is not None: print("in preprocessing: ", bbox_list)
        else:
            #if self.verbose is True: 
            print("no detection")

        # Tracking
        if bbox_list is not None and len(bbox_list)!=0 and bbox_list[0] is not None:
            bbox = self.ds_reid_tracker.track(tensor_img, bbox_list,img)
        else:
            self.ds_reid_tracker.increment_ds_ages()
            bbox = None
        return bbox