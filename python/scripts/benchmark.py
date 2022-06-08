import os
import glob
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
from perceptionloomo.utils.utils import Utils

def init_parser():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    description="Evaluating performance of Single person Tracker given ground truth and detection"
    )
    parser.add_argument('-gt','--ground-truth', default='None',
                    help='Path of the ground_truth bbox')
    parser.add_argument('-det','--detections', default='None',
                    help='Path of the detected bbox for a given algorithm')
    parser.add_argument('-iou','--iou-thresh', default=0.5, type=float,
                    help='IoU threshold to consider a detection as correct')
    parser.add_argument('-v', '--verbose', action='store_true',
                    help=('Silent Mode if Verbose is False'))
    args = parser.parse_args()
    return args


def IoU(df_det, df_gt):
    global verbose
    list_iou=[]
    nb_frame=df_det.shape[0]
    if verbose is True: print("size detection :", nb_frame)
    if verbose is True: print("size gt :", df_gt.shape[0])

    if nb_frame != df_gt.shape[0]:
        print("error in dimensions")
    for i in range(nb_frame):
        #if verbose is True: print(i)
        det_bbox=df_det.iloc[i,:]
        truthbbox=df_gt.iloc[i,:]
        #convert format from [xcenter, ycenter, width, height] to top left, bottom right
        boxA=Utils.bbox_xcentycentwh_to_x1y1x2y2(truthbbox)
        #if verbose is True: print(f"gt bbox: {truthbbox}")
        #if verbose is True: print(f"new gt bbox: {boxA}")
        boxB=Utils.bbox_xcentycentwh_to_x1y1x2y2(det_bbox)
        #if verbose is True: print(f"detection bbox: {det_bbox}")
        #if verbose is True: print(f"new detection bbox: {boxB}")
        #boxA=[truthbbox[0]-truthbbox[2]/2, truthbbox[1]+truthbbox[3]/2, truthbbox[0]+truthbbox[2]/2, truthbbox[1]-truthbbox[3]/2]
        #boxB=[det_bbox[0]-det_bbox[2]/2, det_bbox[1]+det_bbox[3]/2, det_bbox[0]+det_bbox[2]/2, det_bbox[1]-det_bbox[3]/2]
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        #if verbose is True: print(f"xA: {xA}, xB: {xB}, yA: {yA},yB: {yB}")
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        #if verbose is True: print(f"interArea: {interArea}")
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * ( boxA[3]- boxA[1]  + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        #if verbose is True: print(f"Iou value:{iou}")
        #Fix IoU > 1
        if iou>1:
            iou=1
        list_iou.append(iou)
    # return the intersection over union value
    #if verbose is True: print("iou value:", iou)
    return list_iou

def precision_recall(iou_list, nb_det, nb_gt, thresh_iou):
    global verbose
    #precision = True positives / all detections
    #recall = True Positives / all ground truth
    #define true positive as iou> thresh
    nb_frame=len(iou_list)
    if verbose is True: print("size iou list :",nb_frame)
    true_pos=0
    false_pos=0
    for i in range(nb_frame):
        if iou_list[i]>= thresh_iou:
            true_pos+=1
        else:
            false_pos+=1

    if verbose is True: print(f"nb of true positives :{true_pos}, nb false pos {false_pos}, nb_det:{nb_det}, nb_gt:{nb_gt}")
    precision=true_pos/nb_det
    recall=true_pos/nb_gt
    return precision, recall

# def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
#     """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
#     are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
#     if NP == 0:
#         return {
#             "precision": None,
#             "recall": None,
#             "AP": None,
#             "interpolated precision": None,
#             "interpolated recall": None,
#             "total positives": None,
#             "TP": None,
#             "FP": None
#         }

#     # by default evaluate on 101 recall levels
#     if recall_thresholds is None:
#         recall_thresholds = np.linspace(0.0,
#                                         1.00,
#                                         int(np.round((1.00 - 0.0) / 0.01)) + 1,
#                                         endpoint=True)

#     # sort in descending score order
#     inds = np.argsort(-scores, kind="stable")

#     scores = scores[inds]
#     matched = matched[inds]

#     tp = np.cumsum(matched)
#     fp = np.cumsum(~matched)

#     rc = tp / NP
#     pr = tp / (tp + fp)

#     # make precision monotonically decreasing
#     i_pr = np.maximum.accumulate(pr[::-1])[::-1]

#     rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
#     n_recalls = len(recall_thresholds)

#     # get interpolated precision values at the evaluation thresholds
#     i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

#     return {
#         "precision": pr,
#         "recall": rc,
#         "AP": np.mean(i_pr),
#         "interpolated precision": i_pr,
#         "interpolated recall": recall_thresholds,
#         "total positives": NP,
#         "TP": tp[-1] if len(tp) != 0 else 0,
#         "FP": fp[-1] if len(fp) != 0 else 0
#     }



def count_det(df):
    det_count=0
    for i in range(df.shape[0]):
        det=df.iloc[i,:].to_numpy()
        if det.any():
            det_count+=1
            
    return det_count


if __name__ == "__main__":
  
    args=init_parser()
    path_gt=args.ground_truth
    path_results=args.detections
    verbose = args.verbose
    #thresh_iou=args.iou_thresh
    precision_list=np.array([])
    recall_list=np.array([])
    #thresh_iou_list=[0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_iou_list=[0.5]
    for thresh_iou in thresh_iou_list:
        #list_results=[]
        name_col=["x_center", "y_center", "width", "height"]

        #read data into dataframe and get list of missdetection
        df_det=pd.read_csv(path_results, header=None, comment='#')
        df_gt=pd.read_csv(path_gt, header=None)
        df_det.columns=name_col
        df_gt.columns=name_col
        if verbose is True: print(df_det)
        if verbose is True: print(df_gt)
        nb_det=count_det(df_det)
        if verbose is True: print("nb_det: ", nb_det)
        nb_gt=count_det(df_gt)

        #Replace with built-in functions
        list_iou=IoU(df_det, df_gt)
        if verbose is True: print(list_iou)
        precision, recall = precision_recall(list_iou, nb_det, nb_gt, thresh_iou)

        print("precision:", precision)
        print("recall :", recall )
        precision_list=np.append(precision_list, precision)
        recall_list=np.append(recall_list, recall)
        #print("mean IoU :", np.mean(np.asarray(list_iou), axis=0))
    
    print(f"list of precision scores for different iou thresh{precision_list}")
    print(f"list of recall scores for different iou thresh{recall_list}")
    
    # # plot
    # fig, ax = plt.subplots()

    # ax.plot(recall_list, precision_list, linewidth=0.5)

    # #ax.set(xlim=(0, 8), xticks=np.arange(1, 8),ylim=(0, 8), yticks=np.arange(1, 8))

    # plt.show()


