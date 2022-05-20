import torch
import numpy as np

from dlav22.utils.utils import Utils

from dlav22.trackers.reid_tracker import ReID_Tracker
from dlav22.trackers.kalman_filter import KalmanFilter
from dlav22.trackers.track import Track
from dlav22.deep_sort.deep_sort import DeepSort

class Custom_ReID_with_Deepsort():

    def __init__(self, ds_tracker: DeepSort, reid_tracker: ReID_Tracker) -> None:
        self.ds_tracker = ds_tracker
        self.reid_tracker = reid_tracker
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
            return self.first_tracking(cut_imgs, detections, img)
        else:
            self.track_with_deepsort(detections,img)
            track_ids = [track.track_id for track in self.ds_tracker.tracker.tracks]
            print('IDs',track_ids)
            print('ID',self.current_ds_track_idx)
            if self.current_ds_track_idx in track_ids:
                idx_ = track_ids.index(self.current_ds_track_idx)
            else:
                idx_ = self.reid_tracker.track(cut_imgs)
                bbox = self.update_deepsort(detections, idx_, img)
            bbox = detections[idx_]
            return bbox

    def update_deepsort(self,detections, idx_, img):
        print("Updating...")
        bbox=detections[idx_]
        self.track_with_deepsort([bbox], img)
        self.current_ds_track_idx = self.ds_tracker.tracker.tracks[0].track_id
        return bbox

    def track_with_deepsort(self, bboxes: list, img: np.ndarray):
        confs = list(np.zeros(len(bboxes)))
        classes = list(np.zeros(len(bboxes)))
        # print(bboxes) #[array([312, 276, 515, 363])]
        for i, bbox_ in enumerate(bboxes):
            bbox_ = Utils.get_bbox_tlwh_from_xcent_ycent_w_h(bbox_)
            bbox_ = Utils.get_xyah_from_tlwh(bbox_)
            bbox_ = [int(b) for b in bbox_] #FIXME more efficient with numpy
            bboxes[i] = bbox_
        # print(bboxes)
        #bboxes = np.array(bboxes)
        bboxes = torch.tensor(bboxes)
        confs = torch.tensor(confs)
        classes = torch.tensor(classes)
        # img = torch.tensor(img)
        self.ds_tracker.update(bboxes, confs, classes, img)
    

class ReID_with_KF():

    def __init__(self, ReID: ReID_Tracker) -> None:
        self.reid_tracker = ReID
        self.kf = KalmanFilter()
        # FIXME Setting up the KF needs a lot of efforts... (to have it clean)
        # Simple KF is not sufficient since one has to include functionality when to stop tracking

    def initiate_track(self, detection):
        # Execite after first detection and when the obj enters the state again
        detection = Utils.get_bbox_tlwh_from_xcent_ycent_w_h(detection)
        detection = Utils.get_xyah_from_tlwh(detection)
        mean, covariance = self.kf.initiate(detection)
        conf = 1.0
        n_init = 0
        max_age = 1000
        feature = None
        self.track = Track(mean, covariance, 0, 0, conf, n_init, max_age, feature)

    def track(self, bboxes: list, img_detection: np.ndarray):
        
        idx_ = self.reid_tracker.track(img_detection)
        bbox_detect = bboxes[idx_]
        self.predict()
        self.update(bbox_detect) #confidences
    
    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

    def update(self,detection):
        detection = Utils.get_bbox_tlwh_from_xcent_ycent_w_h(detection)
        detection = Utils.get_xyah_from_tlwh(detection)
        # self.conf = conf
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection)
        self.time_since_update = 0
