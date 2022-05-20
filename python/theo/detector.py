import torch
import cv2
from PIL import Image
import numpy as np
#import pandas


#Inference Settings
#model.conf = 0.25  # NMS confidence threshold
#      iou = 0.45  # NMS IoU threshold
#      agnostic = False  # NMS class-agnostic
#      multi_label = False  # NMS multiple labels per box
#      classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#      max_det = 1000  # maximum number of detections per image
#      amp = False  # Automatic Mixed Precision (AMP) inference
#results = model(imgs, size=320)  # custom inference size


class YoloDetector(object):
    def __init__(self, model, thresh, verbose):
        self.model = torch.hub.load('ultralytics/yolov5', model)
        self.model.classes=0 #running only person detection
        self.detection=np.array([0, 0, 0, 0])
        self.yolo_thresh=thresh
        self.verbose=verbose

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
            arr=[x_center, y_center, width, height]
            bbox=np.array(arr)
            bbox=np.expand_dims(bbox, 0)
            return bbox
        else:
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
                arr=[x_center, y_center, width, height]
                bbox_unit=np.array(arr)
                bbox_list.append(bbox_unit)
                #bbox=np.append(bbox_unit, axis=0)
            bbox=np.vstack(bbox_list)
            return bbox

            

    def best_detection(self):
        N=self.detection.shape[0]
        if(N != 1):
            if self.verbose is True: print("multiple persons detected")
            #extracting the detection with max confidence
            idx=np.argmax(self.detection[range(N),4])
            self.detection=self.detection[idx]
        else: #1 detection
            self.detection=np.squeeze(self.detection)


    def predict(self, image):
        #threshold for confidence detection
        
        # Inference
        results = self.model(image) #might need to specify the size

        #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
        detect_pandas=results.pandas().xyxy

        self.detection=np.array(detect_pandas)
        #print("shape of the detection: ", self.detection.shape)
        if self.verbose is True: print("all detections: ",self.detection)

        if (self.detection.shape[1]!=0):          
            #use np.squeeze to remove 0 dim from the tensor
            self.detection=np.squeeze(self.detection,axis=0) 

            #class function to decide which detection to keep
            self.best_detection()
            if (self.detection[4]>self.yolo_thresh):
                #modify the format of detection for bbox
                self.detection=np.expand_dims(self.detection, axis=0)
                bbox=np.squeeze(self.bbox_format())
                return bbox
            
        return None

    def detection_confidence(self):
        # for i in range(self.detection.shape[0]):
        #     if self.detection[i][4]<thresh:
        #         print("yolo under thresh")
        #         self.detection=np.delete(i, self.detection)

        thresh=self.yolo_thresh
        #self.detection= np.delete(self.detection, np.where(self.detection[range(self.detection.shape[0]),4] <thresh))
        det_conf=self.detection[range(self.detection.shape[0]),4].astype(float)
        idx=np.argwhere(det_conf<thresh)
        self.detection= np.delete(self.detection, idx, axis=0)
        #print("after cleaning")
        #print("new shape:", self.detection.shape)
        #if self.verbose is True: print("yolo conf", self.detection)
        if self.detection.size==0:
            return False
        else:
            return True
    

        

    def predict_multiple(self, image):
        #threshold for confidence detection
        
        # Inference
        results = self.model(image) #might need to specify the size

        #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
        detect_pandas=results.pandas().xyxy

        self.detection=np.array(detect_pandas)
        #if self.verbose is True: print("all detections: ",self.detection)

        if (self.detection.shape[1]!=0):
            #print("DETECTED SOMETHING !!!")
            #save resuts
            #results.save()
            
            #use np.squeeze to remove 0 dim from the tensor
            self.detection=np.squeeze(self.detection,axis=0)

            #Handling the case of poor confidence detection using yolo 
            #print("self detection shape:", self.detection.shape)
            #print("self detect confi ??", self.detection)
            if not self.detection_confidence():
                return [0.0, 0.0, 0.0, 0.0],False            

            #modify the format of detection for bbox
            bbox=self.bbox_format()
            if self.verbose is True: print("list of detections:", bbox)
            return bbox
        return None