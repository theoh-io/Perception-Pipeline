# VITA, EPFL 
from xmlrpc.client import boolean
import cv2
import sys
import os
import numpy as np
import time
import torch
import argparse
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from detector import YoloDetector
from tracker import ReID_Tracker

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    description="Benchmark of Single Person Tracking compatible with vieo and MOTChallenge Images sequences"
)
#General Arguments
parser.add_argument('-s','--source', default='None',
                    help='Path of the video to use for inference')
parser.add_argument('-c', '--checkpoint', default=False,
                    help=('directory to load checkpoint'))
parser.add_argument('-v', '--verbose', action='store_true',
                    help=('Silent Mode if Verbose is False'))
#Detector Arguments
parser.add_argument('-ym','--yolo-model', default='yolov5s',
                    help='Yolo model to import: v5n, v5s, v5m, v5l (in order of size)')
parser.add_argument('-yt','--yolo-threshold', default=0, type=float,
                    help='Defines the threshold for the detection score')
parser.add_argument('--init-det', default=None, type=int,
                    help='In case of multiple detections automatically select')
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
                    help='Averaging method to use on the list of ref embeddings (standard, linear, exp')
parser.add_argument('--intra-dist', default='6', type=float,
                    help='Used for smart embedding comparision, L2 distance threshold for high diversity embeddings')
#recording option argument
parser.add_argument('--recordingbb', default=False, help='If a path is provided will save the video from loomo camera with bounding boxes')
#visualisation argument
parser.add_argument('--novisu', action='store_true', help='By default will create a window to see the inference in real time')
#Benchmarking argument: 
parser.add_argument('--ground-truth', default=None, help='If a path is provided activate benchmarking and compare results with ground truth')
parser.add_argument('--deepsort', default=None, help='If a path is provided will plot the boundingboxes coming from deepsort')
parser.add_argument('--loss', default='IoU',
                  help='Function used to compute performance of the tracker. By default use IoU, can use L2 dist from bbox prediction center to ground truth')
parser.add_argument('--stats', default=None, help='If a path is provided write down the statistics: for Example IoU')
args = parser.parse_args()


def getargs():
    global args 
    global path_source, path, verbose
    global model, conf_thresh, init_det
    global dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist
    global path_ground_truth, path_deepsort, loss_method, path_stats
    global recordingbb, visu
    #getting all the parser arguments into variables
    #General
    path_source, path, verbose=args.source, args.checkpoint, args.verbose
    #Detector
    model, conf_thresh, init_det=args.yolo_model, args.yolo_threshold, args.init_det
    #Tracker
    dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist = args.dist_metric, args.tracker_threshold, args.ref_emb, args.nb_ref, args.av_method, args.intra_dist
    #Benchmarking
    path_ground_truth, path_deepsort, loss_method, path_stats = args.ground_truth, args.deepsort, args.loss, args.stats
    #Others
    recordingbb, visu = args.recordingbb, not args.novisu 

def print_config():
    global verbose, visu, recordingbb
    
    print("verbose = ", verbose)
    if visu is False:
        print("Visualization is deactivated")
    recbb=(recordingbb != None)
    if recbb is True:
        print("recording with bounding boxes")
    else:
        print("no path provided for the output video")

def img_seq2vid():
    global path_seq, path_source, seq_vid_fps, path_vid, verbose
    path_vid=os.path.join(path_source, "video.avi")
    #check if the video from img sequence has already been created
    exists=os.path.exists(path_vid)
    #TO BE SOLVED: if no source argument is passed path source is None but still goes in the if
    if path_source is not None:
        if not exists:
            if verbose is True: print("Using Image sequence as Input")
            path_seq=path_source + '/img'
            sequences = os.listdir(path_seq)
            sequences=sorted(sequences)
            #if the video from the sequence of images doesn't exist => create it
            if verbose is True: print("init image", os.path.join(path_seq, sequences[0]))
            init_img=cv2.imread(os.path.join(path_seq, sequences[0]))
            height, width, layers =init_img.shape
            size=(width, height)
            if verbose is True: print("size of the input seq:", size)
            seq_vid=cv2.VideoWriter(path_vid, cv2.VideoWriter_fourcc(*'MJPG'), seq_vid_fps, size)
            for sequence in sequences:
                #print("Running sequence %s" % sequence)
                sequence_dir = os.path.join(path_seq, sequence)
                cvimg=cv2.imread(sequence_dir)
                seq_vid.write(cvimg)
            seq_vid.release()
    else:
        print("path source provided is not working")
        sys.exit()

def openvideo():
    global path_vid, sys, cap, width, height, fps, frame_count
    #use video as input
    print("source video: ", path_vid)
    try:
        cap=cv2.VideoCapture(path_vid)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 3
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4
        fps=cap.get(cv2.CAP_PROP_FPS)           # 5
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 7
        print("input video format: width = ", width, " height = ", height )
    except:
        print("path provided to the video source is not Working. Exiting")
        sys.exit()


def videowriter():
    global recordingbb, fps, width, height, output_vid_bb, dir_out
    #Video writersize
    if recordingbb is not False:
        path_vidbb=recordingbb
        dir_out, file_name =os.path.split(path_vidbb)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        output_vid_bb = cv2.VideoWriter(path_vidbb, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

def filewriter():
    global results_file, path_source, path_stats
    #txt file to write the performance of the benchmarked algorithm
    if path_stats is not None:
        results_file = open(path_stats, "w+")

    


def init():
    global model, conf_thresh, path, dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist, verbose, detector, tracker, path_stats, results_file
    # Initialize Detector and Tracker
    detector=YoloDetector(model, conf_thresh, verbose)
    if path != False:
        tracker=ReID_Tracker(path, dist_metric, dist_thresh, ref_method, nb_ref, av_method, intra_dist, verbose)
        #tracker.load_pretrained(path)
    else:
        if verbose is True: print("No tracking as no model as been provided with argument -c")


def inference_vid(truth_df=None, deepsort_df=None):
    global cap, frame_number, frame_count, init_det, path, detector, tracker, output_vid_bb, results_file ,verbose, visu, nodet_counter
    # Capture frame-by-frame
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_number+=1

            if verbose is True: print("\n New frame: ",frame_number, "/", frame_count)
            opencvImage = frame #cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #######################
            # Detect
            #######################
            #bbox format xcenter, ycenter, width, height
            if path != False:
                bbox = detector.predict_multiple(opencvImage)
            else:
                bbox = detector.predict(opencvImage)
    
            ###########################################
            #   Image Cropping and preprocessing      #
            ###########################################
            #crop_img = img[y:y+h, x:x+w]
            img_list=[]
            if bbox is not None and path!=False:
                print(bbox)
                ###Ajoute la condition initialisation avec plusieurs detections
                nb_ref=tracker.ref_emb.nelement()
                nb_det=bbox.shape[0]  
                if nb_det!=1 and nb_ref == 0:
                    bbox=first_detection_selection(bbox, opencvImage, init_det)
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

            elif bbox is None:
                if verbose is True: print("no detection")
                

            ############
            # Tracking #
            ############
            #handle no detection case
            #generate embedding
            if bbox is not None and path!=False:
            #generate embedding
            #print(tensor_img.shape)
                #idx=tracker.embedding_comparator_mult(tensor_img, 'L2')
                idx=tracker.track(tensor_img)
                #select the bbox corresponding to correct detection
                #print(bbox.size)
                if idx!=None:
                    bbox=bbox[idx]
                else:
                    bbox=None
                    print("Tracking didn't work")

            ##############
            # Benchmarking
            ##############
            if truth_df is not None:
                #print("ground truth provided")
                truth=truth_df[frame_number-1]
                #formatting of the bbox :[xtopleft, ytopleft, width, height] to [xcenter, ycenter, width, height]
                truth=[truth[0]+truth[2]/2,truth[1]+truth[3]/2, truth[2], truth[3]]
                if bbox is not None:
                    print("truth :", truth)
                    print("bbox :", bbox)
                    dist_to_truth=loss(truth, bbox)
                else:
                    dist_to_truth=-1
                #if verbose is True: print("ground truth dist = ", dist_to_truth)
                if path_stats is not None:
                    results_file.write(str(dist_to_truth)+'\n')
            if bbox is None:
                nodet_counter+=1
            

            #######################
            # Visualization
            #######################
            #plotting the bounding boxes on laptop video stream
            if bbox is not None:
                start=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2)) #top-left corner
                stop= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
                cv2.rectangle(opencvImage, start, stop, (0,0,255), 1)

            #plotting ground_truth bounding boxes
            if truth_df is not None:
                t_start=(int(truth[0]-truth[2]/2), int(truth[1]+truth[3]/2)) #top-left corner
                t_stop= (int(truth[0]+truth[2]/2), int(truth[1]-truth[3]/2)) #bottom right corner
                # t_start=(int(truth[0]), int(truth[1])) #top-left corner
                # t_stop= (int(truth[0]+truth[2]), int(truth[1]+truth[3])) #bottom right corner
                cv2.rectangle(opencvImage, t_start, t_stop, (0,255,0), 1)
                #score_bbox(truth_df, bbox)
            
            #plotting deepsort bounding boxes
            if deepsort_df is not None:
                deepsort_bbox=deepsort_df[frame_number]
                #formatting of the bbox :[xtopleft, ytopleft, width, height] to [xcenter, ycenter, width, height]
                #deepsort_bbox=[deepsort_bbox[0]+deepsort_bbox[2]/2,deepsort_bbox[1]+deepsort_bbox[3]/2, deepsort_bbox[2], deepsort_bbox[3]]
                t_start=(int(deepsort_bbox[0]), int(deepsort_bbox[1])) #top-left corner
                t_stop= (int(deepsort_bbox[0]+deepsort_bbox[2]), int(deepsort_bbox[1]+deepsort_bbox[3])) #bottom right corner
                # t_start=(int(truth[0]), int(truth[1])) #top-left corner
                # t_stop= (int(truth[0]+truth[2]), int(truth[1]+truth[3])) #bottom right corner
                cv2.rectangle(opencvImage, t_start, t_stop, (255,0,0), 1)
                
            
            
            #Writing the frame number
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(opencvImage, str(frame_number)+"/"+str(frame_count), (4,height-4), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            if visu is True:
                cv2.imshow('Camera Loomo',opencvImage)
                cv2.waitKey(1)

            #Video recording adding the bounding boxes
            if recordingbb is not False: 
                output_vid_bb.write(opencvImage)
        else:
            break
    if verbose is True: 
        print("number of missdetection is: ", nodet_counter)
    #printhistogram()
    sys.exit()

def load_groundtruth():
    global path_ground_truth, verbose
    try:
        path_current=os.getcwd()
        path_txt=os.path.join(path_current, path_ground_truth)      
        data = pd.read_csv(path_txt, header=None, names= ["x_center", "y_center", "width", "height"], index_col=None)  
        print(data)
        data.index = np.arange(1, len(data) + 1)  #start frame index at 1
        #if verbose is True: print(data)
        data=data.to_numpy()
        #only if needed bbox format
        #for i in data.shape[0]:
        #    bbox=data[i]
        #    data[i]=[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[1]-bbox[3]]
        
        if verbose is True: print("in load gt :", data)
    except:
       print("path provided to ground_truth is not Working.")
       sys.exit()
    return data

def load_deepsort():
    global path_deepsort, verbose
    try:
        path_current=os.getcwd()
        path_txt=os.path.join(path_current, path_deepsort)     
        data = pd.read_csv(path_txt, sep=" ", header=None, names=["frame+1", "id", "xtl", "ytl", "w", "h", "6", "7", "8", "class"], index_col=False) #names= ["frame", "det", "x_center", "y_center", "width", "height", "_", "_","_", "class"]
        #drop the unneceessary columns
        data.drop(['id','6', '7', '8', 'class'] , inplace=True, axis=1)
        #fill the missing values
        first_frame=data["frame+1"][0]
        for i in range(first_frame+1):
            data.loc[-1] = [0, 0, 0, 0, 0]  # adding a row
            data.index = data.index + 1  # shifting index
            data = data.sort_index()  # sorting by index
        print(data)
        data=data.iloc[:,1:5]
        print(data)
        #data=data["x_center", "y_center", "width", "height"] 
        data.index = np.arange(1, len(data) + 1)  #start frame index at 1
        #if verbose is True: print(data)
        data=data.to_numpy()
        
        
        #only if needed bbox format
        #for i in data.shape[0]:
        #    bbox=data[i]
        #    data[i]=[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[1]-bbox[3]]
        #if verbose is True: print("in load deepsort :", data)
    except:
       print("path provided to deepsort is not Working.")
       sys.exit()
    return data

def loss(truthbbox, det_bbox):
    global loss_method, verbose
    if loss_method=="IoU":
        #convert format from [xcenter, ycenter, width, height] to top left, bottom right
        boxA=[truthbbox[0]-truthbbox[2]/2, truthbbox[1]+truthbbox[3]/2, truthbbox[0]+truthbbox[2]/2, truthbbox[1]-truthbbox[3]/2]
        boxB=[det_bbox[0]-det_bbox[2]/2, det_bbox[1]+det_bbox[3]/2, det_bbox[0]+det_bbox[2]/2, det_bbox[1]-det_bbox[3]/2]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yA - yB + 1)
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[1] - boxA[3] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[1] - boxB[3] + 1)
        # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        if verbose is True: print("iou value:", iou)
        return iou

    elif loss_method== "L2":
        truth_center=np.array([truthbbox[0], truthbbox[1]])
        det_center=np.array([det_bbox[0], det_bbox[1]])
        dist = np.linalg.norm(truth_center-det_center)
        if verbose is True: print("L2 distance ground_truth: ", dist)
        return dist
    else:
        print("loss_method: ",loss_method, " is not implemented" )

def first_detection_selection(bboxes, opencvImage, init_det):       
    i=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors=[(128, 128,128), (128, 0,0), (128,128, 0), (0, 128, 0), (128,0,128 ), (0,0,255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192)]
    for bbox in bboxes:
        i+=1
        print("detection nb ", i, ": ", bbox)
        start=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2)) #top-left corner
        stop= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
        cv2.rectangle(opencvImage, start, stop, colors[i], 2)
        cv2.putText(opencvImage, str(i), start, font, 2, colors[i], 2, cv2.LINE_AA)
    
    #value= int(cv2.waitKey(0))-1
    if init_det is None:
        cv2.imshow('Camera Loomo',opencvImage)
        cv2.waitKey(0)
        value=int(input('Enter your input:'))-1
    else:
        value=init_det-1
        print("argument to choose init det =", value)
    
    #value=cv2.waitKey(10000)
    #value = int(input("Please select detection:\n"))
    bbox=np.expand_dims(bboxes[value], 0)
    print("key pressed by user: ", value)
    print("selected bbox: ", bbox)
    return bbox


if __name__ == "__main__":
    #arbitrary parameters
    seq_vid_fps=10
    frame_number=0
    #statistics for comparisions of trackers
    nodet_counter=0
    #error_bbox=np.array([])
    #cumulated_error=0

    getargs()
    print_config()
    img_seq2vid()
    openvideo()
    videowriter()
    filewriter()
    init()

    if path_ground_truth is not None:
        truth_df=load_groundtruth()
        if path_deepsort is not None:
            deepsort_df=load_deepsort()
            inference_vid(truth_df, deepsort_df)
        else:
            inference_vid(truth_df)
    else:
        if path_deepsort is not None:
            deepsort_df=load_deepsort()
            inference_vid(deepsort_df)
        else:
            inference_vid()
    



