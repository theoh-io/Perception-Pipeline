import torch
import numpy as np
import os

from dlav22.utils.utils import Utils

from dlav22.trackers.reid_tracker import ReIdTracker
from dlav22.deep_sort.deep_sort import DeepSort
from dlav22.deep_sort.utils.parser import get_config

class DsCustomReid():

    def __init__(self, cfg) -> None:

        deep_sort_model = cfg.DEEPSORT.MODEL_TYPE
        #add a line to print the model type with verbose
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
        
    def track(self, cut_imgs: list, detections: list, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        self.track_with_deepsort(detections,img)
        track_ids = [track.track_id for track in self.ds_tracker.tracker.tracks]
        if self.current_ds_track_idx in track_ids:
            idx_ = track_ids.index(self.current_ds_track_idx)
        else:
            #FIXME how to handle the case when we hae lost of ID
            print("loss of ID")
            idx_ = 0
        bbox = detections[idx_]
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

