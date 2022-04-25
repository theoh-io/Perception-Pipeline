# VITA, EPFL 

import cv2
import socket
import sys
import numpy as np
import struct
import binascii
import time

from PIL import Image
#from detector import Detector
from detector import YoloDetector
from ReID import ReID_Tracker

import torch
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('-c', '--checkpoint',
                    help=('directory to load checkpoint'))
parser.add_argument('-i','--ip-address',
                    help='IP Address of robot')
parser.add_argument('--instance-threshold', default=0.0, type=float,
                    help='Defines the threshold of the detection score')
parser.add_argument('-d', '--downscale', default=2, type=int,
                    help=('downscale of the received image'))


args = parser.parse_args()

##### IP Address of server #########
host = args.ip_address #local : 127.0.0.1  # The server's hostname or IP address
####################################
port = 8081        # The port used by the server

# image data
downscale = args.downscale
width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels



# create socket
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

# Set up detector
#arguments = ["--checkpoint",args.checkpoint,"--pif-fixed-scale", "1.0", "--instance-threshold",args.instance_threshold]
path=args.checkpoint
#detector = Detector() 
detector=YoloDetector()
tracker=ReID_Tracker()
tracker.load_pretrained(path)


#Image Receiver 
net_recvd_length = 0
recvd_image = b''

#function to automatically adjust the downscale factor
def size_adjust():
    global sz_image, timer_start, mismatch
    if(mismatch==1):
        print("Warning: Image Size Mismatch: ", sz_image, "  ",net_recvd_length)
        timer_start=time.time()
    #if(net_recvd_length==sz_image/4):
        #sz_image=sz_image/4
    

#Test Controller
direction = -1
cnt = 0
#out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480)) #creating video file

mismatch=1

while True:
    
    # Receive data
    reply = s.recv(sz_image)
    recvd_image += reply
    net_recvd_length += len(reply)
    if net_recvd_length == sz_image:
        pil_image = Image.frombytes('RGB', (width, height), recvd_image)
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

        net_recvd_length = 0
        recvd_image = b''

        mismatch=0

        #######################
        # Detect
        #######################
        #bbox format xcenter, ycenter, width, height

        bbox, bbox_label = detector.predict_multiple(opencvImage)  #detection using yolo detector
        
        if bbox_label:
            print(bbox)
        else:
            print("False")

        ###########################################
        #   Image Cropping and preprocessing      #
        ###########################################

        #crop_img = img[y:y+h, x:x+w]
        img_list=[]
        if bbox_label==True:
            for i in range(bbox.shape[0]):
                bbox_indiv=bbox[i]
                crop_img=np.array(opencvImage[int((bbox_indiv[1]-bbox_indiv[3]/2)):int((bbox_indiv[1]+bbox_indiv[3]/2)), int((bbox_indiv[0]-bbox_indiv[2]/2)):int((bbox_indiv[0]+bbox_indiv[2]/2))])
                #to apply the normalization need a PIL image
                # PIL RGB while CV is BGR.
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img = Image.fromarray(crop_img)
                tensor_img_indiv=tracker.image_preprocessing(crop_img)
                tensor_img_indiv=torch.unsqueeze(tensor_img_indiv, 0)
                img_list.append(tensor_img_indiv)
            tensor_img=torch.cat(img_list)

            #print("writing the file")
            #cv2.imwrite("cropped/crop.jpg", crop_img)
        else:
            print("no detection")

        ############
        # Tracking #
        ############
        #handle no detection case
        #generate embedding
        if bbox_label==True:
        #generate embedding
        #print(tensor_img.shape)
            idx=tracker.embedding_comparator_mult(tensor_img, 'L2')
            #select the bbox corresponding to correct detection
            #print(bbox.size)
            if idx!=None:
                bbox=bbox[idx]
            else:
                #print("empty bbox")
                bbox_label= False
                bbox=[0, 0, 0, 0]
        else:
            #print("no detection 2")
            #print("empty bbox")
            bbox=[0, 0, 0, 0]

        #######################
        # Visualization
        #######################
        start=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2))  #top-left corner
        stop= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
        cv2.rectangle(opencvImage, start, stop, (0,0,255), 1)

        cv2.imshow('Camera Loomo',opencvImage)
        cv2.waitKey(1)
        #out.write(opencvImage) #writing the video file

        #######################
        # Socket
        #######################

        # https://pymotw.com/3/socket/binary.html
        values = (bbox[0], bbox[1], bbox[2], bbox[3], float(bbox_label))
        # values = (50.0, 30.0, 10.0, 10.0, 1.0)

        # #Test Controller
        # cnt = cnt + 1
        # if cnt > 50:
        #     direction = - direction
        #     cnt = 0
        # values = (40.0 + direction * 20.0, 30.0, 10.0, 20.0, 1.0)
        
        packer = struct.Struct('f f f f f')
        packed_data = packer.pack(*values)

        # Send data
        send_info = s.send(packed_data)
    else:
        size_adjust()
