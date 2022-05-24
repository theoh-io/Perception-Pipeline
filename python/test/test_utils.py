#!/usr/bin/env python

import os
import cv2
from distutils.command.config import config
from dlav22.utils.utils import FrameGrab
import numpy as np

import time

def test_bbox_conversion():
    
    assert True

def test_img_graber():
    grab = FrameGrab(mode="video")
    i = 1
    while i<10:
        i += 1
        image = grab.read_cap()
        cv2.imshow('result', image)
        k = cv2.waitKey(10) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
        
        time.sleep(0.2)

    assert True

if __name__ == "__main__":
    
    test_img_graber()