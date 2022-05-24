import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple
import itertools
import logging

class Utils():
    @staticmethod
    def get_scaled_img(img: np.ndarray, scale_percent: float=20) -> np.ndarray:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    @staticmethod
    def get_resized_img(img: np.ndarray, new_dim: Tuple[int,int]=(200,140)):
        return cv2.resize(img, new_dim, interpolation = cv2.INTER_AREA)
    
    # FIXME Do it consistent! Always use one representation for bboxes... -> Maybe x1,y1,x2,y2
    @staticmethod
    def bounding_box_from_points(points):
        x_coordinates, y_coordinates = zip(*points)
        return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

    @staticmethod
    def get_bbox_xyxy_from_xy_wh(bbox):
        return [bbox[0][0],bbox[0][1],bbox[0][0] + bbox[1][0],bbox[0][1] + bbox[1][1]]
    
    @staticmethod
    def get_bbox_xyxy_from_xy_xy(bbox):
        return [bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]
    

    @staticmethod
    def get_bbox_as_union_from_two_bboxes_xyxy(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return xA,yA,xB,yB

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        boxA = Utils.get_bbox_xyxy_from_xy_wh(boxA)
        boxB = Utils.get_bbox_xyxy_from_xy_wh(boxB)
        xA,yA,xB,yB = Utils.get_bbox_as_union_from_two_bboxes_xyxy(boxA, boxB)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @staticmethod
    def get_detection_with_max_IoU(bboxesA: list, bboxesB: list) -> Tuple[np.ndarray, float]:
        
        all_combinations = list(itertools.product(bboxesA, bboxesB))
        
        #candidates.sort(key=lambda x: (x[0] - ref[0]) ** 2 + (x[1] - ref[1]) ** 2)  
        # FIXME Do that nice!
        best_iou_metric = 0
        ic_best = -1
        for ic in range(len(all_combinations)):
            comb = all_combinations[ic]
            iou_metric = Utils.bb_intersection_over_union(comb[0],comb[1])
            if iou_metric >= best_iou_metric: #FIXME Not true
                ic_best = ic
                best_iou_metric = iou_metric
    
        # print("ic_best",ic_best)
        det_bbox_A = Utils.get_bbox_xyxy_from_xy_xy(all_combinations[ic_best][0])
        det_bbox_B = Utils.get_bbox_xyxy_from_xy_xy(all_combinations[ic_best][1])
        detected_bbox = list(Utils.get_bbox_as_union_from_two_bboxes_xyxy(det_bbox_A,det_bbox_B)) # FIXME This is in some way wrong... -> use utils.BBox class for all bbox operations
        
        return detected_bbox, best_iou_metric, det_bbox_A, det_bbox_B
    
    @staticmethod
    def get_bbox_xcent_ycent_w_h_from_xy_xy(bbox):
        #original format: [(top_left), (bot-right)]
        bbox=[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
        x_center=(bbox[0]+bbox[2])/2
        y_center=(bbox[1]+bbox[3])/2
        width=bbox[2]-bbox[0]
        height=bbox[3]-bbox[1]
        bbox=[x_center, y_center, width, height]
        bbox=np.array(bbox).astype(int)
        return bbox

    @staticmethod
    def get_bbox_tlwh_from_xcent_ycent_w_h(img_xc_yc_w_h):
        x_cent, y_cent, w, h = img_xc_yc_w_h
        x_top_left = x_cent - w//2
        y_top_left = y_cent - h//2
        return np.array([x_top_left, y_top_left, w, h])

    @staticmethod
    def get_xyah_from_tlwh(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @staticmethod
    def get_xyah_from_xc_yc_wh(xywh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = xywh.copy()
        ret[2] /= ret[3]
        return ret

class FrameGrab:
    def __init__(self, mode: str ="webcam", video: str = "Loomo/Demo3/theo_Indoor.avi") -> None:
        self.cap = None
        if mode == "webcam": #FIXME Do it as enum
            self.cap = cv2.VideoCapture(0) 
        elif mode == "video":
            FOLDER = "../Benchmark/"
            # self.cap = cv2.VideoCapture(FILDER + "Loomo/video.avi")
            self.cap = cv2.VideoCapture(FOLDER + video)
        
    def read_cap(self) -> np.ndarray:
        success, image =  self.cap.read()
        if not success:
            logging.warning("Reading was not successful.")
            image = None
        return image
    
    def __del__(self):
        self.cap.release()
        print('Released cap.')

        
class BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def bottom(self):
        return self.y + self.h
    
    def right(self):
        return self.x + self.w
    
    def area(self):
        return self.w * self.h

    def union(self, b):
        posX = min(self.x, b.x)
        posY = min(self.y, b.y)
        
        return BBox(posX, posY, max(self.right(), b.right()) - posX, max(self.bottom(), b.bottom()) - posY)
    
    def intersection(self, b):
        posX = max(self.x, b.x)
        posY = max(self.y, b.y)
        
        candidate = BBox(posX, posY, min(self.right(), b.right()) - posX, min(self.bottom(), b.bottom()) - posY)
        if candidate.w > 0 and candidate.h > 0:
            return candidate
        return BBox(0, 0, 0, 0)
    
    def ratio(self, b):
        return self.intersection(b).area() / self.union(b).area()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Interval(ABC):
    def __init__(self, min, max):
        self._min = min
        self._max = max

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def asdict(self):
        return {'min': self.min, 'max': self.max}

    def astuple(self):
        return self.min, self.max

    @abstractmethod
    def __contains__(self, item):
        pass

    @abstractmethod
    def __str__(self):
        pass

class OpenInterval(Interval):
    def __init__(self, min, max):
        super().__init__(min, max)

    def __contains__(self, item):
        return self.min < item < self.max

    def __str__(self):
        return f"({self.min}, {self.max})"

class ClosedInterval(Interval):
    def __init__(self, min, max):
        super().__init__(min, max)

    def __contains__(self, item):
        return self.min <= item <= self.max

    def __str__(self):
        return f"[{self.min}, {self.max}]"

class ClosedIntervalNotDirected(Interval):
    def __init__(self, min, max):
        super().__init__(min, max)

    def __contains__(self, item):
        return self.min <= item <= self.max or self.max <= item <= self.min

    def __str__(self):
        return f"[{self.min}, {self.max}]"