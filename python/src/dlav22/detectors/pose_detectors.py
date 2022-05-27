import numpy as np
from typing import Tuple, List
from enum import Enum
from charset_normalizer import detect
import openpifpaf
from PIL import Image
import time 
import logging
import cv2
import matplotlib.pyplot as plt

import itertools

from dlav22.utils.utils import Utils, ClosedInterval,ClosedIntervalNotDirected
from dlav22.detectors.base_detector import BaseDetector
from dlav22.detectors.color_detector import ColorDetector

class PifPafKeyPoints(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class PoseDetector(BaseDetector):
    def __init__(self, cfg) -> None:
        verbose = cfg.PERCEPTION.VERBOSE
        checkpoint = cfg.DETECTOR.POSE_DETECTOR.PIFPAF_CHECKPOINT

        super().__init__(verbose)
        self.predictor = openpifpaf.Predictor(checkpoint=checkpoint, visualize_processed_image=True)

        self.detect_pose = getattr(self,cfg.DETECTOR.POSE_DETECTOR.DETECT_POSE)
    
    def predict(self, img: np.ndarray) -> list:

        bboxes, confidences = self.predict_all_bboxes(img)

        # Assume there is only one detection
        if not bboxes:
            return None
        
        bbox = bboxes[0]
        
        bbox = Utils.get_bbox_xcent_ycent_w_h_from_xy_xy(bbox)
        bboxes = np.expand_dims(bbox, axis=0)

        return bboxes

    def predict_all_bboxes(self, img: np.ndarray) -> Tuple[list, list]:

        bbox = None

        bboxes = []

        tic = time.perf_counter()
        predictions, gt_anns, image_meta = self.predictor.numpy_image(img)

        confidences = []
        if self.verbose:
            print(f"Number of people observed by PifPaf: {len(predictions)}")

        for i in range(len(predictions)):
            pred = predictions[i].data
            detected = self.check_if_desired_pose(pred)
            if detected:
                xy = pred[pred[:,2]>0.5,:2]
                if self.verbose:
                    print(f"Detected person {i+1}.")
                bbox = Utils.bounding_box_from_points(xy.astype(int))
                confidences.append(xy)
                bboxes.append(bbox)

        toc2 = time.perf_counter()
        if self.verbose:    
            print(f"Needed {round((toc2-tic)*1e3,3)}ms.")

        return bboxes, confidences

    def check_if_desired_pose(self,keypoints: np.ndarray) -> bool:

        detected = True

        relevant_keypoints = self.get_wrists_elbows_shoulders(keypoints)

        #FIXME Check confidence and exclude if necessary -> FIXME Do this nicer
        MIN_CONF = 0.4
        # print([rk[2] for rk in relevant_keypoints])
        if not all([rk[2] > MIN_CONF for rk in relevant_keypoints]):
            if self.verbose:
                print(f"Confidence too low")
            return False

        # detected = detected & self.check_if_ellbows_are_left_right_boundaries(relevant_keypoints)
        # detected = detected & self.check_if_arms_sprawled_out(relevant_keypoints)
        # detected = detected & self.check_if_wrists_at_the_head(relevant_keypoints)
        # detected = detected & self.check_if_wrists_in_the_middle(relevant_keypoints)
        if self.detect_pose is not None:
            detected = detected & self.detect_pose(relevant_keypoints)

        if self.verbose and detected:
             print(f"Detected a person in desired pose.")

        return detected

    #FIXME Update pose is different
    # check that wrists are higher than legs (if confidence for all is high)
    
    def get_wrists_elbows_shoulders(self, keypoints: List[np.ndarray]) -> List[np.ndarray]:

        left_wrist = keypoints[PifPafKeyPoints.LEFT_WRIST.value]
        right_wrist = keypoints[PifPafKeyPoints.RIGHT_WRIST.value]
        left_elbow = keypoints[PifPafKeyPoints.LEFT_ELBOW.value]
        right_elbow = keypoints[PifPafKeyPoints.RIGHT_ELBOW.value]
        left_shoulder = keypoints[PifPafKeyPoints.LEFT_SHOULDER.value]
        right_shoulder = keypoints[PifPafKeyPoints.RIGHT_SHOULDER.value]

        relevant_keypoints = [left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder]

        return relevant_keypoints


    def check_if_ellbows_are_left_right_boundaries(self, relevant_keypoints: List[np.ndarray]) -> bool:
        
        detected = False

        left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder = relevant_keypoints

        # Left side
        interval_left_vert = ClosedInterval(left_shoulder[1],left_wrist[1])
        # interval_left_hor = ClosedInterval(0,max(left_wrist[0],left_shoulder[0])) 
        interval_left_hor = ClosedInterval(min(left_wrist[0],left_shoulder[0]),5000) 

        # Right side
        interval_right_vert = ClosedInterval(right_shoulder[1],right_wrist[1])
        # interval_right_hor = ClosedInterval(max(right_wrist[0],right_shoulder[0]),500) #FIXME nicer
        interval_right_hor = ClosedInterval(0,min(right_wrist[0],right_shoulder[0]))

        # Check if desired pose:
        left_side_desired = (left_elbow[0] in interval_left_hor) and (left_elbow[1] in interval_left_vert)
        right_side_desired = (right_elbow[0] in interval_right_hor) and (right_elbow[1] in interval_right_vert)

        # hand_crossed = True
        # hand_crossed = left_wrist[0] < right_wrist[0] + 10

        if self.verbose:
            print(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
        if left_side_desired and right_side_desired:
            detected = True

        return detected

    #FIXME Do that in utils -> class 2D polytope / in area
    def check_if_wrists_in_the_middle(self,relevant_keypoints: List[np.ndarray]) -> bool:

        detected = False

        left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder = relevant_keypoints

        interval_hor = ClosedIntervalNotDirected(left_shoulder[0],right_shoulder[0]) 
        # Left side
        interval_left_vert = ClosedIntervalNotDirected(left_shoulder[1],left_wrist[1])
        # Right side
        interval_right_vert = ClosedIntervalNotDirected(right_shoulder[1],right_wrist[1])

        # Check if desired pose:
        left_side_desired = (left_wrist[0] in interval_hor) and (left_wrist[1] in interval_left_vert)
        right_side_desired = (right_wrist[0] in interval_hor) and (right_wrist[1] in interval_right_vert)

        if self.verbose:
            print(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
            logging.debug(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
            # FIXME Instead of verbose: logging.debug

        if left_side_desired and right_side_desired:
            detected = True

        return detected

    def check_if_wrists_at_the_head(self,relevant_keypoints: List[np.ndarray]) -> bool:

        detected = False

        left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder = relevant_keypoints

        interval_hor = ClosedIntervalNotDirected(left_elbow[0],right_elbow[0]) 
        # Left side
        interval_left_vert = ClosedIntervalNotDirected(left_shoulder[1],0)
        # Right side
        interval_right_vert = ClosedIntervalNotDirected(right_shoulder[1],0)
        # Check if desired pose:
        left_side_desired = (left_wrist[0] in interval_hor) and (left_wrist[1] in interval_left_vert)
        right_side_desired = (right_wrist[0] in interval_hor) and (right_wrist[1] in interval_right_vert)

        if self.verbose:
            print(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
            logging.debug(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
            # FIXME Instead of verbose: logging.debug

        if left_side_desired and right_side_desired:
            detected = True

        return detected

    def check_if_arms_sprawled_out(self,relevant_keypoints: List[np.ndarray]) -> bool:

        detected = False

        left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder = relevant_keypoints

        interval_left_hor = ClosedInterval(left_shoulder[0],left_wrist[0]) 
        interval_left_vert = ClosedInterval(left_shoulder[1],left_wrist[1])
        # Right side
        interval_right_hor = ClosedInterval(right_wrist[0],right_shoulder[0]) 
        interval_right_vert = ClosedInterval(right_shoulder[1],right_wrist[1]) # ClosedIntervalNotDirected alternative

        # Check if desired pose:
        left_side_desired = (left_elbow[0] in interval_left_hor) and (left_elbow[1] in interval_left_vert)
        right_side_desired = (right_elbow[0] in interval_right_hor) and (right_elbow[1] in interval_right_vert)

        if self.verbose:
            print(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
            logging.debug(f"left_side_desired: {left_side_desired} | right_side_desired: {right_side_desired}")
            # FIXME Instead of verbose: logging.debug

        if left_side_desired and right_side_desired:
            detected = True

        return detected


    def plot_key_points(self,img: np.ndarray) -> np.ndarray:
        predictions, gt_anns, image_meta = self.predictor.numpy_image(img)
        for i in range(len(predictions)):
            pred = predictions[i].data
            plt.scatter(pred[:,0],pred[:,1],c=pred[:,2],ec='k')

        return img

class PoseColorFusingDetector(BaseDetector):

    def __init__(self,dim_img=(200,150), verbose=False) -> None:
        super().__init__(verbose=verbose)
        self.pifpafDetect = PoseDetector(verbose=verbose)
        self.colDetect = ColorDetector(verbose=verbose,dim_img=dim_img)
        self.min_IoU_for_detection = 0.3
        

    def predict(self, image: np.ndarray) -> list:
        
        bboxes_pifpaf, confidences_pifpaf = self.pifpafDetect.predict_all_bboxes(image)
        bbox_color = self.colDetect.predict(image)

        # One system is not detecting sth
        if not bboxes_pifpaf and bbox_color is not None:
            return None
        
        # Both systems are detecting -> Use the biggest blue blob to decide which person to take
        if bboxes_pifpaf and bbox_color is not None:
            bbox_union, IoU, bbox_pifpaf, det_bbox_color = Utils.get_detection_with_max_IoU(bboxes_pifpaf,[bbox_color])
            if IoU > self.min_IoU_for_detection:
                print(f"Detected desired object at {bbox_pifpaf} with IoU = {IoU:.2f}.")
                bbox = [(bbox_pifpaf[0],bbox_pifpaf[1]),(bbox_pifpaf[2],bbox_pifpaf[3])]

            else:
                bbox = None
        else:
            bbox = None
        

        bbox = Utils.get_bbox_xcent_ycent_w_h_from_xy_xy(bbox)
        bboxes = np.expand_dims(bbox, axis=0)

        return bboxes

class PoseColorGuidedDetector(PoseColorFusingDetector):

    def __init__(self,dim_img=(200,150)) -> None:
        super().__init__(dim_img)

    def predict(self, image: np.ndarray) -> list:
        
        bbox = None
        bboxes_pifpaf, confidences_pifpaf = self.pifpafDetect.predict_all_bboxes(image)
        
        if not bboxes_pifpaf:
            return bbox

        if len(bboxes_pifpaf) > 1:
            bboxes_color = self.colDetect.predict_all_bboxes(image)
            if len(bboxes_color) > 1:
                bbox_union, IoU, bbox_pifpaf, det_bbox_color = Utils.get_detection_with_max_IoU(bboxes_pifpaf,bboxes_color)
                if IoU > self.min_IoU_for_detection:
                    print(f"Detected desired object at {bbox_pifpaf} using the color as additional feature with IoU = {IoU:.2f}.")
                    bbox = [(bbox_pifpaf[0],bbox_pifpaf[1]),(bbox_pifpaf[2],bbox_pifpaf[3])]
            else:
                # Just take the first
                bbox = bboxes_pifpaf[0]

        else:
            bbox = bboxes_pifpaf[0]
        
        bbox = Utils.get_bbox_xcent_ycent_w_h_from_xy_xy(bbox)
        bboxes = np.expand_dims(bbox, axis=0)

        return bboxes
