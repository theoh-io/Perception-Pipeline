import numpy as np

import argparse
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from pathlib import Path
import sys
import os

#CWD is Perception-Pipeline/python
#print(f"cwd is :{os.getcwd()}")
path_yolov7=os.path.join((Path(os.getcwd()).parents[0]), "libs/yolov7")
sys.path.append(path_yolov7)
#print(sys.path)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Yolov7Detector():
    def __init__(self, cfg, model='default', verbose = False):
        verbose = cfg.PERCEPTION.VERBOSE
        #yolo_version = cfg.DETECTOR.YOLO.MODEL_VERSION
        #self.model.classes=0 #running only person detection
        self.detection=np.array([0, 0, 0, 0])
        self.verbose=verbose
        print(f"Created YOLO detector with verbose={verbose}.")

        # Load model
        #self.device=1 #'cpu'
        self.classes=0
        self.weights="yolov7.pt"

        # Initialize
        set_logging()
        self.device = select_device()
        #print(f"device selected: {torch.device}, device is {self.device}")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        with torch.no_grad():
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz= 640
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
            self.model = TracedModel(self.model, self.device, self.imgsz)
            if self.half:
                self.model.half()  # to FP16
            self.init=True
        

    def bbox_format(self):
        #detection format xmin, ymin, xmax,ymax, conf, class, 'person'
        #bbox format xcenter, ycenter, width, height
        det_new=np.array([0, 0, 0, 0])
        if(self.detection.shape[0]==1):
            #image has been transposed before
            self.detection=np.squeeze(self.detection)
            xmin=self.detection[0]
            ymin=self.detection[1]
            xmax=self.detection[2]
            ymax=self.detection[3]
            x_center=(xmin+xmax)/2
            y_center=(ymin+ymax)/2
            width=xmax-xmin
            height=ymax-ymin
            bbox=[x_center, y_center, width, height]
            bbox=np.expand_dims(bbox, axis=0)
            return bbox
        else:
            if self.verbose is True: print(self.detection.shape)
            bbox_list=[]
            for i in range(self.detection.shape[0]):
                xmin=self.detection[i][0]
                ymin=self.detection[i][1]
                xmax=self.detection[i][2]
                ymax=self.detection[i][3]
                x_center=(xmin+xmax)/2
                y_center=(ymin+ymax)/2
                width=xmax-xmin
                height=ymax-ymin
                bbox_unit=np.array([x_center, y_center, width, height])
                if self.verbose is True: print(bbox_unit)
                bbox_list.append(bbox_unit)
            bbox_list=np.vstack(bbox_list)
            #bbox_list=bbox_list.tolist()
            if self.verbose is True: print("final bbox", bbox_list)
            return bbox_list

            

    def predict(self, image, thresh=0.01):            
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()

         # Run inference
        with torch.no_grad():
            iou_thresh=0
            #image
            print(f"input shape be4 preproc: {image.shape}")

            # Padded resize (preprocessing in yolov7/datasets.py LoadIages __next__())
            img = letterbox(image, self.imgsz, stride=self.stride)[0]
            #print(f"shape after padding {img.shape}")
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            #print(f"shape after transpose {img.shape}")
            img = np.ascontiguousarray(img)

            #for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()

            print("bug 2")
            print(f"input shape: {img.shape}")
            #print(summary(self.model, (1,3,240,320)))

            pred = self.model(img)[0]
            # Apply NMS
            print("bug 3")
            #exit()
            pred = non_max_suppression(pred, thresh, iou_thresh, classes=0)#self.classes)
            t2 = time_synchronized()

            #gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    print(det)
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    det=det[:,:4].cpu().detach().numpy()
                    self.detection=det
                    print(f"detection shape {self.detection.shape}, {self.detection}")
                    self.detection=self.bbox_format()
                    print(self.detection)
                    return self.detection
                else:
                    return None



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)