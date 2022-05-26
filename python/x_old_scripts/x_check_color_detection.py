from time import sleep
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_contour_areas(contours):
    all_areas= []
    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

def get_bbox_from_image(image, config):
    tic = time.perf_counter()
    img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([0,100,20])
    # upper_red = np.array([2,255,255])
    # mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # lower_red = np.array([130,100,20])
    # upper_red = np.array([180,255,255])

    # mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # mask = mask0 + mask1

    lower_blue = np.array([100, 50, 100])
    upper_blue = np.array([170, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    nb_iterations = 1
    mask = cv2.dilate(mask, None, iterations = nb_iterations)
    mask = cv2.erode(mask, None, iterations = nb_iterations)

    # set my output img to zero everywhere except my mask
    output_img = image.copy()
    output_img[np.where(mask==0)] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse= True)

    if config['plot']:
        cv2.drawContours(image, contours, -1,(255,0,255),3)
    #print('Total number of contours detected: ' + str(len(contours)))
    
    #first_contour = cv2.drawContours(image, contours, 0,(255,0,255),3)
    # FIXME Do this more elaborate.
    # - Check shape and orientation
    # - get a t-shirt with two colors
    # - Include a tracker

    if len(contours) > 0:
        area = cv2.contourArea(contour=contours[0])
        #print(area)
        if area < 500:
            print(f"Nothing detected.")
            sleep(1)
            return None
    else:
        print(f"Nothing detected.")
        sleep(1)
        return None

    x, y, w, h = cv2.boundingRect(contours[0])
    bbox = [(x,y),(x+w,y+h)]

    if config['plot']:
        cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0), 5)
        cv2.imshow('mask', mask)
        cv2.imshow('result', image)
        cv2.rectangle(image,(x,y), (x+w,y+h), (255,0,0), 5)

    toc = time.perf_counter()
    print(f"Time elapsed: {toc-tic:0.4f}s")
    return bbox


if __name__ == "__main__":

    config = {'plot': True}


    cap = cv2.VideoCapture(0)

    while(1):

        _, image = cap.read()
        if get_bbox_from_image(image, config) is None:
            continue

        #cv2.waitKey(0)    
        k = cv2.waitKey(10) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
        
        sleep(0.3)
    
    cv2.destroyAllWindows()
    cap.release()

    # pil_im = Image.open("test4color_detector.jpg")
    # img = np.asarray(pil_im)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # #img = cv2.imread('test4color_detector.jpg')
    # #img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB

    # bbox = get_bbox_from_image(img, config)
    # print(bbox is None)

    # if bbox is not None:
    #     print(bbox[0])
    #     print(bbox[1])
    #     cv2.rectangle(img,bbox[0],bbox[1],(0,255,0),2)
    # plt.imshow(img)
    # plt.show()