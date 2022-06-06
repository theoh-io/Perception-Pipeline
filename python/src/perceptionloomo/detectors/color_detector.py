import cv2
import numpy as np
import time

from perceptionloomo.detectors.base_detector import BaseDetector

class ColorDetector(BaseDetector):
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

