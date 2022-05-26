from time import sleep
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import logging
import importlib
import os 

if str(os.getcwd())[-7:] == "scripts":
    os.chdir("..")
print(f"Current WD: {os.getcwd()}")

import torch

from dlav22.detectors import pose_detectors, yolo_detector
from dlav22.trackers import custom_trackers, reid_tracker
from dlav22.utils.utils import Utils
from dlav22.utils.utils import FrameGrab


if __name__ == "__main__":

    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    
    verbose = False
    #create instance of the detector
    detector=yolo_detector.YoloDetector(verbose=verbose) #simple yolo
    first_detector = pose_detectors.PoseColorGuidedDetector() #detector combining color detection and PifPaf
    current_path=os.getcwd()
    print(current_path)
    ReIDpath=current_path+"/src/dlav22/trackers/ReID_model.pth.tar"
    tracker=reid_tracker.ReID_Tracker(ReIDpath, 'cosine', 0.87, verbose=verbose)

    # Change the logging level
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # start streaming video from webcam
    grab = FrameGrab(mode="video")
    # label for video
    label_html = 'Capturing...'
    # initialze bounding box to empty'list' object has no attribute 'shape'
    bbox = None
    bbox_bytes=None
    count = 0 


    while(1):

        img = grab.read_cap() # This outputs BGR format

        # -----
        # create transparent overlay for bounding box
        bbox_array = np.zeros([480,640,4], dtype=np.uint8)

        ############
        # Detector #
        ############
        #first detection case => PifPaf
        nb_ref=tracker.ref_emb.nelement() 
        if nb_ref == 0:
            if verbose is True: print("running pifpaf for first detection")
            bbox_list = first_detector.predict(img)
            if bbox_list is not None:
                bbox = bbox_list[0]
                if verbose is True: print("pifpaf bbox:", bbox)
        else:
            #2nd Detection ... => Yolo
            bbox_list= detector.predict_multiple(img)
            #bbox_list=np.array(bbox)
            if verbose is True: print("yolo detection", bbox_list)
            
        ###########################################
        #   Image Cropping and preprocessing      #
        ###########################################
        #Format for cropping => crop_img = img[y:y+h, x:x+w]
        img_list=[]
        if bbox_list is not None:
            if bbox_list[0] is not None:

                if verbose is True: print("in preprocessing: ", bbox_list)
                for i in range(bbox_list.shape[0]):
                    bbox_indiv=bbox_list[i]
                    crop_img=np.array(img[int((bbox_indiv[1]-bbox_indiv[3]/2)):int((bbox_indiv[1]+bbox_indiv[3]/2)), int((bbox_indiv[0]-bbox_indiv[2]/2)):int((bbox_indiv[0]+bbox_indiv[2]/2))])
                    #to apply the normalization need a PIL image
                    # PIL RGB while CV is BGR.
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_img = Image.fromarray(crop_img)
                    tensor_img_indiv=tracker.image_preprocessing(crop_img)
                    tensor_img_indiv=torch.unsqueeze(tensor_img_indiv, 0)
                    img_list.append(tensor_img_indiv)
                tensor_img=torch.cat(img_list)

        elif bbox_list is None:
            if verbose is True: print("no detection")

        ############
        # Tracking #
        ############
        #generate embedding
        if bbox_list is not None:
            #if bbox_list != [None]:
            #generate embedding
            idx=tracker.track(tensor_img)
            if idx!=None:
                bbox=bbox_list[idx]
            #FIXME handle the case when tracker doesn't return any index
            # else:
            #     #print("empty bbox")

            else:
                bbox=None
                if verbose is True: print("in tracking no detection")

        ###################
        #  Visualization  #
        ###################
        if bbox is not None:
            if verbose is True: print("Visualization bbox:", bbox)
            #for (x,y,w,h) in bbox:
            top_left=(int(bbox[0]-bbox[2]/2), int(bbox[1]+bbox[3]/2))  #top-left corner
            bot_right= (int(bbox[0]+bbox[2]/2), int(bbox[1]-bbox[3]/2)) #bottom right corner
            bbox_array = cv2.rectangle(img,top_left,bot_right,(255,0,0),2)
            # bbox_array[:,:,3] = (img.max(axis = 2) > 0 ).astype(int) * 255
            # update bbox so next frame gets new overlay
            #bbox = bbox_bytes
        else:
            pass
                # -----
        cv2.imshow('result', img)


        #cv2.waitKey(0)    
        k = cv2.waitKey(10) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
        
        sleep(0.3)
    
    cv2.destroyAllWindows()
    del grab