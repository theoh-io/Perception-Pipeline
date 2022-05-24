import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

from dlav22.detectors.final_detector import DetectorG16

        
class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()
        self.detector = DetectorG16(verbose=False)

    def forward(self, pil_image: np.ndarray):  

        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

        bbox = self.detector.forward(opencvImage)
        
        #FIXME What is the label?
        pred_y_label = 1

        return bbox, pred_y_label

