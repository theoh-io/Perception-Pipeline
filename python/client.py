# VITA, EPFL 
import cv2
import socket
import sys
import numpy as np
import struct
import time
import torch
import argparse
from PIL import Image

from detector import YoloDetector
from tracker import ReID_Tracker

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-c', '--checkpoint', default=False,
                    help=('directory to load checkpoint'))
parser.add_argument('-i','--ip-address',
                    help='IP Address of robot')
parser.add_argument('-yt','--yolo-threshold', default=0.4, type=float,
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

# Initialize Detector and Tracker
conf_thresh=args.yolo_threshold
path=args.checkpoint
detector=YoloDetector(conf_thresh)
if path != False:
    tracker=ReID_Tracker()
    tracker.load_pretrained(path)
else:
    print("No tracking as no model as been provided with argument -c")

#Image Receiver 
net_recvd_length = 0
recvd_image = b''

#function to Warn the user in case of wrong downscale factor
def size_adjust():
    if(mismatch==1 and time.time()>timeout):
        print("Warning: Image Size Mismatch: ", sz_image, "  ",net_recvd_length)

# def checking_size():
#     global net_recvd_length
#     timeout=time.time()+0.1
#     start=time.time()
#     print(time.time())
#     while time.time()<timeout:
#         print(time.time())
#         reply = s.recv(sz_image)
#         net_recvd_length += len(reply)
#         if net_recvd_length==sz_image:
#             print("time_needed ", time.time()-start)
#             net_recvd_length = 0
#             break
#         time.sleep(1)



#Test Controller
#direction = -1
#cnt = 0
#out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480)) #creating video file

mismatch=1 #FSM for avoiding checking size once it has been verified one time
timeout=time.time()+0.1 #variable used to avoid printing warning on size mismatch for initialization

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
        if path != False:
            bbox, bbox_label = detector.predict_multiple(opencvImage)
        else:
            bbox, bbox_label = detector.predict(opencvImage)
        if bbox_label:
            print("bbox candidate(s):", bbox)
        else:
            print("no bbox candidate")

        ###########################################
        #   Image Cropping and preprocessing      #
        ###########################################
        #crop_img = img[y:y+h, x:x+w]
        img_list=[]
        if bbox_label==True and path!=False:
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
        elif bbox_label==False:
            print("no detection")

        ############
        # Tracking #
        ############
        #handle no detection case
        #generate embedding
        if bbox_label==True and path!=False:
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
        elif bbox_label==False:
            #print("no detection 2")
            #print("empty bbox")
            bbox=[0, 0, 0, 0]

        #######################
        # Visualization
        #######################
        #plotting the bounding boxes on laptop video stream
        start=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2)) #top-left corner
        stop= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
        cv2.rectangle(opencvImage, start, stop, (0,0,255), 2)
        cv2.imshow('Camera Loomo',opencvImage)
        cv2.waitKey(1)

        #######################
        # Socket
        #######################
        # https://pymotw.com/3/socket/binary.html
        values = (bbox[0], bbox[1], bbox[2], bbox[3], float(bbox_label))
        packer = struct.Struct('f f f f f')
        packed_data = packer.pack(*values)
        # Send data
        send_info = s.send(packed_data)
    else:
        size_adjust()
