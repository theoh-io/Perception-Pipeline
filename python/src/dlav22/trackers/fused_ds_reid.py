import torch
import numpy as np
import os

from dlav22.utils.utils import Utils

from dlav22.trackers.reid_tracker import ReID_Tracker
from dlav22.deep_sort.deep_sort import DeepSort
from dlav22.deep_sort.utils.parser import get_config

class FusedDsReid():

    def __init__(self, verbose: bool = False) -> None:
        
        current_path=os.getcwd()
        ReIDpath=current_path+"/src/dlav22/trackers/ReID_model.pth.tar"
        try:
            self.reid_tracker=ReID_Tracker(ReIDpath, 'cosine', 0.87, verbose=verbose)
        except:
            ReIDpath = "/home/group16/dlav22_16/DLAV-Project-Loomo/python/src/dlav22/trackers/ReID_model.pth.tar"
            self.reid_tracker=ReID_Tracker(ReIDpath, 'cosine', 0.87, verbose=verbose)

        try:
            cfg = get_config(config_file="src/dlav22/deep_sort/configs/deep_sort.yaml")
        except:
            cfg = get_config(config_file="/home/group16/dlav22_16/DLAV-Project-Loomo/python/src/dlav22/deep_sort/configs/deep_sort.yaml")
        

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
        self.current_ds_track_idx = None
        self.start = True
        
    def first_tracking(self, cut_imgs: list, detections: list, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        idx_=self.reid_tracker.track(cut_imgs)
        bbox = None
        if idx_ is not None:
            #cut_img = cut_imgs[idx_]
            bbox = self.update_deepsort(detections, idx_, img)
        return bbox

    def track(self, cut_imgs: list, detections: list, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        if self.start:
            self.start = False
            return self.first_tracking(cut_imgs, detections, img)
        else:
            self.track_with_deepsort(detections,img)
            track_ids = [track.track_id for track in self.ds_tracker.tracker.tracks]
            if self.current_ds_track_idx in track_ids:
                idx_ = track_ids.index(self.current_ds_track_idx)
            else:
                idx_ = self.reid_tracker.track(cut_imgs)
                if idx_ is None: # FIXME Why is this happening?? -> Did not find a similar obj
                    return None
                bbox = self.update_deepsort(detections, idx_, img)
            bbox = detections[idx_]
            return bbox

    def update_deepsort(self,detections, idx_, img):
        bbox=detections[idx_]
        self.track_with_deepsort([bbox], img)
        self.current_ds_track_idx = self.ds_tracker.tracker.tracks[0].track_id
        print(f"Updating DeepSort with ReID with ID {self.current_ds_track_idx}.")
        return bbox

    def track_with_deepsort(self, bboxes: list, img: np.ndarray):
        confs = list(np.zeros(len(bboxes)))
        classes = list(np.zeros(len(bboxes)))
        for i, bbox_ in enumerate(bboxes):
            bbox_ = [int(b) for b in bbox_] #FIXME more efficient with numpy
            bboxes[i] = bbox_
        bboxes = torch.tensor(bboxes)
        confs = torch.tensor(confs)
        classes = torch.tensor(classes)
        self.ds_tracker.update(bboxes, confs, classes, img)
    
    def increment_ds_ages(self):
        self.ds_tracker.increment_ages()

