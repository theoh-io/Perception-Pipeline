# VITA, EPFL 
from xmlrpc.client import boolean
import cv2
import sys
import numpy as np
import time
import torch
import argparse
from PIL import Image

from detector import YoloDetector
from tracker import ReID_Tracker

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
#General Arguments
parser.add_argument('-s','--source', default='None',
                    help='Path of the video to use for inference')
parser.add_argument('-c', '--checkpoint', default=False,
                    help=('directory to load checkpoint'))
parser.add_argument('-v', '--verbose', default=True,
                    help=('Silent Mode if Verbose is False'))
#Detector Arguments
parser.add_argument('-ym','--yolo-model', default='yolov5s',
                    help='Yolo model to import: v5n, v5s, v5m, v5l (in order of size)')
parser.add_argument('-yt','--yolo-threshold', default=0, type=float,
                    help='Defines the threshold for the detection score')
#Tracker arguments
parser.add_argument('--dist-metric', default='cosine',
                    help=('Distance metric used to compare distance between embeddings in the tracker'))
parser.add_argument('-tt','--tracker-threshold', default=0.87, type=float,
                    help='Defines the threshold for the distance between embeddings refering to the same target')
parser.add_argument('--ref-emb', default='multiple',
                    help='Defines the method to get the reference embedding in tracker (simple, multiple, smart)')
parser.add_argument('--nb-ref', default='8', type=int,
                    help='number of embeddings to keep for the computation of the average embedding')
parser.add_argument('--av-method', default='standard',
                    help='Averaging method to use on the list of ref embeddings (standard, linear, exponential')
parser.add_argument('--intra-dist', default='6', type=float,
                    help='Used for smart embedding comparision, L2 distance threshold for high diversity embeddings')
#recording option argument
parser.add_argument('--recordingbb', default=None, help='If a path is provided will save the video from loomo camera with bounding boxes')
#visualisation argument
parser.add_argument('--visu', default=True, help='Will create a window to see the inference in real time')
args = parser.parse_args()


def inference_vid():
    global cap, frame_number, path, detector, tracker, output_vid_bb ,verbose, visu
    # Capture frame-by-frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_number+=1
            if verbose is True: print(frame_number)
            opencvImage = frame #cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
            #Writing the frame number
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(opencvImage, str(frame_number), (4,height-4), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            if visu is True:
                cv2.imshow('Camera Loomo',opencvImage)
                cv2.waitKey(1)

            #Video recording adding the bounding boxes
            if recbb is True:
                output_vid_bb.write(opencvImage)
        else:
            break
    sys.exit()


path_source= args.source
print("source video: ", path_source)
try:
    cap=cv2.VideoCapture(path_source)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 3
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4
    fps=cap.get(cv2.CAP_PROP_FPS)           # 5
    frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)   # 7
    print("input video format: width = ", width, " height = ", height )
except:
    print("path provided to the video source is not Working. Exiting")
    sys.exit()

verbose=args.verbose == 'True' or args.verbose=='true'
print("verbose = ", verbose)
visu= args.visu
print("visu = ", visu)
recbb=(args.recordingbb != None)
if recbb is True:
    print("recording with bounding boxes")
else:
    print("no path provided for the output video")


#Video writersize
if recbb is True:
    path_vidbb=args.recordingbb
    output_vid_bb = cv2.VideoWriter(path_vidbb, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))


# Initialize Detector and Tracker
model, conf_thresh=args.yolo_model, args.yolo_threshold
path, dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist=args.checkpoint, args.dist_metric, args.tracker_threshold, args.ref_emb, args.nb_ref, args.av_method, args.intra_dist


detector=YoloDetector(model, conf_thresh, verbose)
if path != False:
    tracker=ReID_Tracker(path, dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist, verbose)
    #tracker.load_pretrained(path)
else:
    if verbose is True: print("No tracking as no model as been provided with argument -c")

mismatch=1 #FSM for avoiding checking size once it has been verified one time
timeout=time.time()+0.2 #variable used to avoid printing warning on size mismatch for initialization




#measuring time between frames for perfromance of inference ??
start_time=time.time()
mean_fr=np.empty((1))
frame_number=0
inference_vid()


