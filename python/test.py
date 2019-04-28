import cv2
import socket
import sys
import numpy
# import struct
# import binascii

from PIL import Image
from detector import Detector


img = cv2.imread('1.png')
cv2.imshow('Test window',img)

# Set up detector
detector = Detector()
detector.load('./saved_model.pth')

# 
print(img.shape)
bbox, bbox_label = detector.forward(img)

print(bbox)
print(bbox_label)


cv2.waitKey(0)
cv2.destroyAllWindows()
