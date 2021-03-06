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
    parser.add_argument('-s', '--single', action='store_true',
                    help=('single file Mode'))
    parser.add_argument('-x1y1', action='store_true',
                    help=('convert gt from x1w1wh to xcenter, ycenter,w, h'))
            

    args = parser.parse_args()
    return args

def read_data(path_det, path_gt):

    name_col=["x_center", "y_center", "width", "height"]
    #read data into dataframe and get list of missdetection
    df_det=pd.read_csv(path_det, header=None, comment='#')
    df_gt=pd.read_csv(path_gt, header=None,  delimiter='\t')
    df_det.columns=name_col
    df_gt.columns=name_col
    # if verbose is True: print(df_det)
    # if verbose is True: print(df_gt)
    return df_gt, df_det 

def IoU(df_det, df_gt):
    global verbose
    list_iou=[]
    nb_frame=df_det.shape[0]
    #if verbose is True: 
    print(f"size detection : {nb_frame}, size gt : {df_gt.shape[0]}")
    if nb_frame != df_gt.shape[0]:
        print("error in dimensions")
        return None
    for i in range(nb_frame):
        #if verbose is True: print(i)
        det_bbox=df_det.iloc[i,:]
        truthbbox=df_gt.iloc[i,:]
        if truthbbox.to_numpy().any() :
            #convert format from [xcenter, ycenter, width, height] to top left, bottom right
            boxA=Utils.bbox_xcentycentwh_to_x1y1x2y2(truthbbox)
            boxB=Utils.bbox_xcentycentwh_to_x1y1x2y2(det_bbox)
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
            if iou>1:
                iou=1
            list_iou.append(iou)
        else:
            list_iou.append(0)

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
        elif iou_list[i] != 0:
            false_pos+=1

    if verbose is True: print(f"nb of true positives :{true_pos}, nb false pos {false_pos}, nb_det:{nb_det}, nb_gt:{nb_gt}")
    precision=true_pos/nb_det
    recall=true_pos/nb_gt
    return precision, recall



def count_det(df):
    det_count=0
    for i in range(df.shape[0]):
        det=df.iloc[i,:].to_numpy()
        if det.any():
            det_count+=1
            
    return det_count

def single_result(path_results, path_gt, thresh_iou=0.5, precision_list=np.array([]), recall_list=np.array([]), x1y1_gt=False):
    df_gt, df_det=read_data(path_results, path_gt)
    if x1y1_gt:
        df_gt=df_gt.apply(Utils.bbox_x1y1wh_to_xcentycentwh, axis=1)
    nb_det=count_det(df_det)
    nb_gt=count_det(df_gt)
    if verbose is True: print(f"det count:  {nb_det}, gt count: {nb_gt}")

    #Replace with built-in functions
    list_iou=IoU(df_det, df_gt)
    if list_iou is not None:
        #if verbose is True: print(list_iou)
        precision, recall = precision_recall(list_iou, nb_det, nb_gt, thresh_iou)

        if verbose is True: print("precision:", precision)
        if verbose is True: print("recall :", recall )
        if precision_list.size and recall_list.size !=0:
            precision_list=np.append(precision_list, precision)
            recall_list=np.append(recall_list, recall)
            #print("mean IoU :", np.mean(np.asarray(list_iou), axis=0))
            return precision_list, recall_list

        return precision, recall
    else:
        return None, None
    

def filter_files(folder_path, filter, sep, nb_split=1):
    list_files=os.listdir(folder_path)
    remove_list=[]
    for i in range(len(list_files)):
        try:
            name, type=list_files[i].rsplit(sep, nb_split)
            if type != filter:
                remove_list.append(list_files[i])
        except:
            remove_list.append(list_files[i])
        
    for j in range(len(remove_list)):
        list_files.remove(remove_list[j])
    list_files.sort()
    return list_files


def average_results(path_folder_det, path_folder_gt, iou_thresh):
    #list all the files in the folder
    #check if need to sort
    #for loop based on the number of files
    #compute performance for each video and store in list
    #average perf
    #do it for multiple thresh iou values
    #verbose=True
    list_det=filter_files(path_folder_det, "prediction.txt", "_")
    if verbose is True: print(list_det)
    list_gt=filter_files(path_folder_gt, "gt.txt", "_")
    if verbose is True: print(list_gt)
    precision_across_vids=[]
    recall_across_vids=[]
    for vid in range(len(list_det)):
        path_det=path_folder_det+"/"+list_det[vid]
        path_gt=path_folder_gt+"/"+list_gt[vid]
        precision, recall= single_result(path_det, path_gt, thresh_iou)
        if verbose is True:
            print(f"for vid nb{vid+1}, pr={precision}, recall={recall}")
        if precision and recall is not None:
            precision_across_vids.append(precision)
            recall_across_vids.append(recall)
    #print(precision_across_vids)
    av_precision=np.mean(np.asarray(precision_across_vids), axis=0)
    av_recall=np.mean(np.asarray(recall_across_vids), axis=0)
    print(f"AP@{iou_thresh} Average precision = {av_precision}, Average recall={av_recall}")




if __name__ == "__main__":
    print(f" Corrent WD is: {os.getcwd()}")
    args=init_parser()
    path_gt=args.ground_truth
    path_results=args.detections
    verbose = args.verbose
    single_file=args.single
    x1y1_gt=args.x1y1
    precision_list=np.array([])
    recall_list=np.array([])
    #thresh_iou_list=[0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_iou_list=[0.01, 0.5, 0.75, 0.9]
    #thresh_iou_list=[0.5]
    for thresh_iou in thresh_iou_list:
        if single_file:
            precision, recall =single_result(path_results, path_gt, thresh_iou, precision_list, recall_list, x1y1_gt)
            precision_list=np.append(precision_list, precision)
            recall_list=np.append(recall_list, recall)
        else:
            average_results(path_results, path_gt, thresh_iou)
        
        
    # print(f"list of precision scores for different iou thresh{precision_list}")
    # print(f"list of recall scores for different iou thresh{recall_list}")
    
    # # plot
    # fig, ax = plt.subplots()

    # ax.plot(recall_list, precision_list, linewidth=0.5)

    # #ax.set(xlim=(0, 8), xticks=np.arange(1, 8),ylim=(0, 8), yticks=np.arange(1, 8))

    # plt.show()


