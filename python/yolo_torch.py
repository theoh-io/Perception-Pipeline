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

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model.classes=0

# # Image
# img1 = Image.open('yolo_imgs/crosswalk.jpg')
# img2 = cv2.imread('yolo_imgs/Cr796.jpeg')[..., ::-1]  # OpenCV image (BGR to RGB)
# imgs = [img1, img2]  # batch of images


# # Inference
# results = model(imgs, (96)) #might specify the size
# print("type of result", type(results))
# results.print()
# results.save()




# print(results.xyxy[0])
# #xmin, ymin, xmax,ymax, conf, class
# print(results.pandas().xyxy[0])

class YoloDetector(object):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.classes=0 #running only person detection
        self.detection=np.array([0,0])

# def yolo_model(self, classe=0):
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#     model.classes=classe
#     return model


    def bbox_format(self):
        #detection format xmin, ymin, xmax,ymax, conf, class, 'person'
        #bbox format xcenter, ycenter, width, height
        xmin=self.detection[0]
        ymin=self.detection[1]
        xmax=self.detection[2]
        ymax=self.detection[3]
        x_center=(xmin+xmax)/2
        y_center=(ymin+ymax)/2
        width=xmax-xmin
        height=ymax-ymin
        bbox=[x_center, y_center, width, height]
        return bbox



    def yolo_to_loomo(self):
        N=self.detection.shape[0]
        if(N != 1):
            print("multiple persons detected")
            #extracting the detection with max confidence
            idx=np.argmax(self.detection[range(N),4])
            self.detection=self.detection[idx]
        else: #1 detection
            self.detection=np.squeeze(self.detection)


    def yolo_predict(self, image):
        #threshold for confidence detection
        thresh=0.01
        # Inference
        results = self.model(image) #might need to specify the size
        #results.xyxy: [xmin, ymin, xmax, ymax, conf, class]
        detect_pandas=results.pandas().xyxy
        #detect_np=detect_tensor.numpy()
        #print(detect_np)
        self.detection=np.array(detect_pandas)
        print("shape of the detection: ", self.detection.shape)
        print("detection: ",self.detection)
        #if (self.detection.shape[0]==0):
            #return [], False
        if (self.detection.shape[1]!=0):
            print("DETECTED SOMETHING !!!")
            #save resuts for my demo
            results.save()
            #use np.squeeze to remove 0 dim from the tensor
            self.detection=np.squeeze(self.detection,axis=0) 
            self.yolo_to_loomo()
            if(self.detection[4]>thresh):
                label=True
            #modify the format of detection for bbox
            bbox=self.bbox_format()
            return bbox, label

        
        return [0.0, 0.0, 0.0, 0.0],False
