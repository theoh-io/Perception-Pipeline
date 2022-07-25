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
            self.imgsz=640
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
            self.model = TracedModel(self.model, self.device, self.imgsz)
            if self.half:
                self.model.half()  # to FP16

            self.init=True
        

    def bbox_format(self):
        #detection format xmin, ymin, xmax,ymax, conf, class, 'person'
        #bbox format xcenter, ycenter, width, height
        if(self.detection.shape[0]==1):
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
        # #threshold for confidence detection
        # # Inference
        # results = self.model(image) #might need to specify the size

        # #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
        # detect_pandas=results.pandas().xyxy

        # self.detection=np.array(detect_pandas)
        # if self.verbose is True: print("shape of the detection: ", self.detection.shape)
        # #print("detection: ",self.detection)

        # if (self.detection.shape[1]!=0):
        #     if self.verbose is True: print("DETECTED SOMETHING !!!")
        #     #save resuts
        #     #results.save()
            
        #     #use np.squeeze to remove 0 dim from the tensor
        #     self.detection=np.squeeze(self.detection,axis=0) 
        #     if self.verbose is True: print("bbox before format: ", self.detection)
        #     #modify the format of detection for bbox
        #     bbox=self.bbox_format()
        #     if self.verbose is True: print("bbox after format: ", bbox)
        #     return bbox
        # return None

        #initialization
        if self.init:
            # Second-stage classifier
            self.classify = False
            if self.classify:
                modelc = load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

            #Set Dataloader
            #dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        
        if self.device.type != 'cpu':
            #self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
            self.model(torch.zeros(1, 3, self.imgsz, 640).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()

         # Run inference
        with torch.no_grad():
            iou_thresh=0
            #image
            #print(f"input shape be4 preproc: {image.shape}")
            img=np.transpose(image)
            #print(f"input shape after transpose: {img.shape}")
            #for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, thresh, iou_thresh, classes=0)#self.classes)
            t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, img)

            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    print(det)
                    #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    det=det[0].cpu().detach().numpy()
                    print(det)
                    det=det[:4]
                    print(det)
                    det_old=det
                    det_new=np.array([0, 0, 0, 0])
                    det_new[0]=det_old[1]
                    det_new[1]=det_old[0]
                    det_new[2]=det_old[3]
                    det_new[3]=det_old[2]
                    print(f"det_old is {det_old} and det is {det_new}")
                    self.detection=np.expand_dims(det_new, 0)
                    print(f"detection shape {self.detection.shape}, {self.detection}")
                    self.detection=self.bbox_format()
                    print(self.detection)
                    return self.detection