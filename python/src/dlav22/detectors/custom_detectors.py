
from tabnanny import verbose
from typing import Tuple, List
from enum import Enum
from charset_normalizer import detect
import numpy as np
import openpifpaf
from PIL import Image
import time 
import cv2
import matplotlib.pyplot as plt
from dlav22.utils.utils import Utils, ClosedInterval,ClosedIntervalNotDirected

import itertools

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

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


class Detector():
    def __init__(self, verbose) -> None:
        self.verbose = verbose
    
    def predict(self, image: np.ndarray) -> list:
        raise NotImplementedError("Detector Base Class does not provide a predict class.")



class PifPafDetector(Detector):
    def __init__(self, checkpoint="shufflenetv2k16", verbose=False) -> None:
        super().__init__(verbose)
        self.predictor = openpifpaf.Predictor(checkpoint=checkpoint)
    
    def predict(self, img: np.ndarray) -> list:
        
        bboxes, confidences = self.predict_all_bboxes(img)

        # Assume there is only one detection
        if not bboxes:
            return None
        
        bbox = bboxes[0]

        return bbox

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

        left_wrist = keypoints[PifPafKeyPoints.LEFT_WRIST.value]
        right_wrist = keypoints[PifPafKeyPoints.RIGHT_WRIST.value]
        left_elbow = keypoints[PifPafKeyPoints.LEFT_ELBOW.value]
        right_elbow = keypoints[PifPafKeyPoints.RIGHT_ELBOW.value]
        left_shoulder = keypoints[PifPafKeyPoints.LEFT_SHOULDER.value]
        right_shoulder = keypoints[PifPafKeyPoints.RIGHT_SHOULDER.value]

        relevant_keypoints = [left_wrist,right_wrist,left_elbow,right_elbow,left_shoulder,right_shoulder]

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
        detected = detected & self.check_if_wrists_in_the_middle(relevant_keypoints)

        if self.verbose and detected:
             print(f"Detected a person in desired pose.")

        return detected
    


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


class ColorDetector(Detector):
    def __init__(self, verbose=False, dim_img=(200,150)) -> None: #FIXME infer that from input dim
        super().__init__(verbose)
        config = {'color': 'blue'}
        config['min_detection_area'] = 0.01 * dim_img[0]*dim_img[1] #FIXME Put img size here -> adaptiveley
        self.config = config
        

    def get_contour_areas(self, contours):
        all_areas= []
        for cnt in contours:
            area= cv2.contourArea(cnt)
            all_areas.append(area)
        return all_areas
    
    def predict_all_bboxes(self, image, img_color_format="bgr"):

        tic = time.perf_counter()

        image_ratio_wh = image.shape[0] / image.shape[1]
        
        if img_color_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_hsv=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        if self.config['color'] == 'red':
            lower_red = np.array([0,100,20])
            upper_red = np.array([2,255,255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
            lower_red = np.array([130,100,20])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            mask = mask0 + mask1
        elif self.config['color'] == 'blue':
            lower_blue = np.array([100, 35, 100])
            upper_blue = np.array([170, 255, 255])
            mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

        nb_iterations = 1
        mask = cv2.dilate(mask, None, iterations = nb_iterations)
        mask = cv2.erode(mask, None, iterations = nb_iterations)

        # set my output img to zero everywhere except my mask
        output_img = image.copy()
        output_img[np.where(mask==0)] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.verbose:
            cv2.drawContours(image, contours, -1,(255,0,255),3)

        # It's probably not so robust because it depends on the ratio of the original image 
        # -> FIXME We have this info, we can use it...
        # only deciding depending on the area might be problematic if there are large blue areads in the background
        # We could als only consider the center of the img because the robot will be usually oriented there
        
        # cont_x,cont_y,cont_w,cont_h = cv2.boundingRect(contours[0])
        # ratio_like_shirt = image_ratio_wh * cont_h > cont_w

        if len(contours) > 0:
                
            contours = sorted(contours, key=cv2.contourArea, reverse= True)
            area = cv2.contourArea(contour=contours[0])
            detection_area_large_enough = area > self.config['min_detection_area']

            if not detection_area_large_enough: # and ratio_like_shirt:
                #print(f"Nothing detected.")
                return []
            else:
                if self.verbose:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0), 5)
                    cv2.imshow('mask', mask)
                    cv2.imshow('result', image)
                    cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0), 5)
                bboxes = []

                i_bb = 0
                # Only one area considered here
                while cv2.contourArea(contour=contours[i_bb]) > self.config['min_detection_area'] and i_bb + 1 < len(contours):
                    x, y, w, h = cv2.boundingRect(contours[i_bb])
                    bbox = [(x,y),(x+w,y+h)] 
                    bboxes.append(bbox)
                    i_bb += 1

                toc = time.perf_counter()
                # print(f"Time elapsed: {toc-tic:0.4f}s")
                return bboxes
        else:
            #print(f"Nothing detected.")
            # time.sleep(1)
            return []


    def predict(self, image, img_color_format="bgr"):
        
        bboxes = self.predict_all_bboxes(image=image, img_color_format=img_color_format)
        
        # Simples implementation: Get largest bbox
        if not bboxes:
            return None
        
        bbox = bboxes[0]

        return bbox


class PoseColorFusingDetector(Detector):

    def __init__(self,dim_img=(200,150), verbose=False) -> None:
        super().__init__(verbose=verbose)
        self.pifpafDetect = PifPafDetector(verbose=verbose)
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
    


if __name__ == "__main__":
    pil_im = Image.open("images/test_color_detector_blue.jpg")
    img = np.asarray(pil_im)

    coldetect = ColorDetector(verbose=True)

    bbox = coldetect.predict(img)
    if bbox is not None:
        print(bbox[0])
        print(bbox[1])
        cv2.rectangle(img,bbox[0],bbox[1],(0,255,0),2)
    plt.imshow(img)
    plt.show()
