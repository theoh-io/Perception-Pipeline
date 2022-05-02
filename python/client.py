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

from detector import YoloDetector
from tracker import ReID_Tracker

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-c', '--checkpoint', default=False,
                    help=('directory to load checkpoint'))
parser.add_argument('-i','--ip-address',
                    help='IP Address of robot')
parser.add_argument('-d', '--downscale', default=2, type=int,
                    help=('downscale of the received image'))
parser.add_argument('-v', '--verbose', default=True,
                    help=('Silent Mode if Verbose is False'))
#Detector Arguments
parser.add_argument('-ym','--yolo-model', default='yolov5s',
                    help='Yolo model to import: v5n, v5s, v5m, v5l (in order of size)')
parser.add_argument('-yt','--yolo-threshold', default=0.4, type=float,
                    help='Defines the threshold for the detection score')
#Tracker arguments
parser.add_argument('--dist-metric', default='cosine',
                    help=('Distance metric used to compare distance between embeddings in the tracker'))
parser.add_argument('-tt','--tracker-threshold', default=0.87, type=float,
                    help='Defines the threshold for the distance between embeddings refering to the same target')
parser.add_argument('--ref-emb', default='multiple',
                    help='Defines the method to get the reference embedding in tracker (simple, mulitple, smart)')
parser.add_argument('--nb-ref', default='8', type=int,
                    help='number of embeddings to keep for the computation of the average embedding')
parser.add_argument('--av-method', default='standard',
                    help='Averaging method to use on the list of ref embeddings')
parser.add_argument('--intra-dist', default='6', type=float,
                    help='Used for smart embedding comparision, L2 distance threshold for high diversity embeddings')


args = parser.parse_args()

##### IP Address of server #########
host = args.ip_address #local : 127.0.0.1  # The server's hostname or IP address
####################################
port = 8081        # The port used by the server

verbose=args.verbose == 'True' or args.verbose=='true'
print("verbose = ", args.verbose)

# image data
downscale = args.downscale
width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels

# create socket
if (verbose is True): print('# Creating socket')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
    print('Failed to create socket')
    sys.exit()
if verbose is True: print('# Getting remote IP address') 
try:
    remote_ip = socket.gethostbyname( host )
except socket.gaierror:
    print('Hostname could not be resolved. Exiting')
    sys.exit()

# Connect to remote server
if verbose is True: print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
s.connect((remote_ip , port))

# Initialize Detector and Tracker
model, conf_thresh=args.yolo_model, args.yolo_threshold
path, dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist=args.checkpoint, args.dist_metric, args.tracker_threshold, args.ref_emb, args.nb_ref, args.av_method, args.intra_dist


detector=YoloDetector(model, conf_thresh, verbose)
if path != False:
    tracker=ReID_Tracker(path, dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist, verbose)
    #tracker.load_pretrained(path)
else:
    if verbose is True: print("No tracking as no model as been provided with argument -c")

#Image Receiver 
net_recvd_length = 0
recvd_image = b''

#function to Warn the user in case of wrong downscale factor
def size_adjust():
    if(mismatch==1 and time.time()>timeout):
        print("Warning: Image Size Mismatch: ", sz_image, "  ",net_recvd_length)

mismatch=1 #FSM for avoiding checking size once it has been verified one time
timeout=time.time()+0.2 #variable used to avoid printing warning on size mismatch for initialization

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
        #if bbox_label:
            #if verbose: print("bbox candidate(s):", bbox)
        #else:
            #if verbose: print("no bbox candidate")

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

        elif bbox_label==False:
            if verbose is True: print("no detection")

        ############
        # Tracking #
        ############
        #handle no detection case
        #generate embedding
        if bbox_label==True and path!=False:
        #generate embedding
        #print(tensor_img.shape)
            #idx=tracker.embedding_comparator_mult(tensor_img, 'L2')
            idx=tracker.track(tensor_img)
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
