# VITA, EPFL 
from xmlrpc.client import boolean
import cv2
import socket
import sys
import numpy as np
import struct
import time
import torch
import argparse
from PIL import Image

from perceptionloomo.detectors.pose_detectors import PoseColorGuidedDetector
from perceptionloomo.perception.perception import DetectorG16
from perceptionloomo.utils.utils import Utils

if str(os.getcwd())[-7:] == "scripts":
    os.chdir("..")

detector=DetectorG16()
#General Configs
print(detector.cfg)
verbose=detector.cfg.PERCEPTION.VERBOSE
ip=detector.cfg.LOOMO.IP
downscale=detector.cfg.LOOMO.DOWNSCALE
rec= detector.cfg.LOOMO.RECORDING

print("verbose :", verbose)
print("value of downscale parameter :", downscale)
print("ip adress of Loomo: ", ip)
if rec is not None:
    print("path for the recorded video :", rec)
#Detector configs

#Tracker configs
print(detector.cfg.TRACKER.TRACKER_CLASS)

##### IP Address of server #########
host = ip #local : 127.0.0.1  # The server's hostname or IP address
####################################
port = 8081        # The port used by the server


# create socket  FIXME can be added to utils
print('# Creating socket')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
    print('Failed to create socket')
    sys.exit()
print('# Getting remote IP address') 
try:
    remote_ip = socket.gethostbyname( host )
except socket.gaierror:
    print('Hostname could not be resolved. Exiting')
    sys.exit()

# Connect to remote server
print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
s.connect((remote_ip , port))


# image data
width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels

#Video writersize
framerate=3.5 #3.3 for downscale 2 and 5 for downscale 4
if rec is not None:
    path_vid=rec
    output_vid = cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'MJPG'), framerate, (width, height)) #try without specifying width and height
    

#function to Warn the user in case of wrong downscale factor
def size_adjust():
    if(mismatch==1 and time.time()>timeout):
        print("Warning: Image Size Mismatch: ", sz_image, "  ",net_recvd_length)

mismatch=1 #FSM for avoiding checking size once it has been verified one time
timeout=time.time()+0.2 #variable used to avoid printing warning on size mismatch for initialization

#Image Receiver 
net_recvd_length = 0
recvd_image = b''
print("cuda : ", torch.cuda.is_available())
while True:
    # Receive data
    reply = s.recv(sz_image)
    recvd_image += reply
    net_recvd_length += len(reply)
    # if verbose is True:
    #     print("Size info: ", sz_image, "  ",net_recvd_length)
    if net_recvd_length == sz_image:
        pil_image = Image.frombytes('RGB', (width, height), recvd_image)
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

        net_recvd_length = 0
        recvd_image = b''
        mismatch=0

        bbox=detector.forward(opencvImage)

        #######################
        # Visualization
        #######################
        #Video recording adding the bounding boxes
        if rec is not None:
            output_vid.write(opencvImage)
            
        Utils.visualization(opencvImage, bbox)

        



        #######################
        # Socket
        #######################
        # https://pymotw.com/3/socket/binary.html
        if bbox is None:
            bbox=[0,0,0,0]
            bbox_label=False
        else:
            bbox_label=True

        values = (bbox[0], bbox[1], bbox[2], bbox[3], float(bbox_label))
        packer = struct.Struct('f f f f f')
        packed_data = packer.pack(*values)
        # Send data
        send_info = s.send(packed_data)

        k=cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
    else:
        size_adjust()

#detector.store_elapsed_time()
