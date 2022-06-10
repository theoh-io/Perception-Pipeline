import time
import cv2
import numpy as np
import time
import os 
if str(os.getcwd())[-7:] == "scripts":
    os.chdir("..")

print(f"Current WD: {os.getcwd()}")


from perceptionloomo.utils.utils import FrameGrab
from perceptionloomo.perception import perception
from perceptionloomo.utils.utils import Utils
from perceptionloomo.deep_sort.utils.parser import get_config

if __name__ == "__main__":
    detector = perception.DetectorG16()
    # start video stream
    path_benchmark=detector.cfg.PERCEPTION.BENCHMARK_FILE
    path_groundtruth=detector.cfg.PERCEPTION.GROUNDTRUTH
    df_gt=Utils.load_groundtruth(path_groundtruth)
    print(f"Using: {path_benchmark} as input")
    grab = FrameGrab(mode="img", path=path_benchmark)

    bboxes_to_save = []
    elapsed_time_list=[]
    frame_number=0
    while(True):
        img = grab.read_cap()
        if img is None:
            print("Stop reading.")
            break
        tic = time.perf_counter()
        #Call to the algorithm
        bbox = detector.forward(img)
        toc = time.perf_counter()
        print(f"Elapsed time for whole fordward pass: {(toc-tic)*1e3:.1f}ms")
        #record bbox and elapsed time
        bboxes_to_save.append(bbox)
        elapsed_time_list.append((toc-tic)*1e3)
        #Visualization
        #Utils.visualization(img, bbox, (255, 0, 0), 2)
        if df_gt is not None:
            truth=df_gt[frame_number]
            truth=Utils.bbox_x1y1wh_to_xcentycentwh(truth)
            Utils.visualization(img, truth, (0, 255, 0), 1)

        print("bbox:", bbox)
        #To get result before the end quit the program with q instead of Ctrl+C
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
        frame_number+=1

    average_forward_time=np.mean(elapsed_time_list)
    print(f"Average time for a forward pass is {average_forward_time:.1f}ms")
    Utils.save_results(detector, bboxes_to_save)

    




    cv2.destroyAllWindows()
    del grab
