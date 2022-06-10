import torch
import numpy as np
from mmtrack.apis import inference_sot, init_model

from perceptionloomo.utils.utils import Utils


class SotaTracker():
    def __init__(self, cfg) -> None:
        '''
        init_model parameters: path to config, path to checkpoints_weights, desired device to specify cpu
        '''
        #add a line to print the model type with verbose
        desired_device = cfg.MMTRACKING.DEVICE
        path_config=cfg.MMTRACKING.CONFIG
        path_model=cfg.MMTRACKING.MODEL
        cpu = 'cpu' == desired_device
        cuda = not cpu and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.tracker = init_model(path_config, path_model, self.device) 
        #prog_bar = mmcv.ProgressBar(len(imgs))
        self.conf_thresh=cfg.MMTRACKING.CONF
        self.frame=0


    def track(self, cut_imgs: list, detections: list, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        if self.frame==0:
            #print(f"list of detection{detections[0]}")
            init_bbox=detections[0]
            print(f"bbox xcenter format {init_bbox}")
            # init_bbox[2] += init_bbox[0]
            # init_bbox[3] += init_bbox[1]
            #convert from (xcenter,y_center, width, height) to (x1,y1,x2,y2)
            # offset_x=int(init_bbox[2]/2)
            # offset_y=int(init_bbox[3]/2)
            # self.new_bbox=[0, 0, 0, 0]
            # self.new_bbox[0]=init_bbox[0]-offset_x
            # self.new_bbox[1]=init_bbox[1]-offset_y
            # self.new_bbox[2]=init_bbox[0]+offset_x
            # self.new_bbox[3]=init_bbox[1]+offset_y
            self.new_bbox=Utils.bbox_xcentycentwh_to_x1y1x2y2(init_bbox)
            print(f"bbox x2y2 format {self.new_bbox}")

        #input of the bbox format is x1, y1, x2, y2
        result = inference_sot(self.tracker, img, self.new_bbox, frame_id=self.frame)
        
        self.frame+=1
        track_bbox=result['track_bboxes']
        #remove last index -1
        confidence=track_bbox[4]
        print(f"conf: {confidence}")
        bbox=track_bbox[:4]#[test_bbox[0], test_bbox[1], test_bbox[2]-test_bbox[0], test_bbox[3]-test_bbox[1]]
        
        if confidence>self.conf_thresh:
            #changing back format from (x1, y1, x2, y2) to (xcenter, ycenter, width, height) before writing
            bbox=Utils.bbox_x1y1x2y2_to_xcentycentwh(bbox)
            bbox = [int(x) for x in bbox]
        else:
            print("!! Under Tracking threshold")
            bbox=[0, 0, 0, 0]

        return bbox

